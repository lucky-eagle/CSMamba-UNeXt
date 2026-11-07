"""Reproducible training and evaluation for the manuscript's three protocols."""

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import ManifestDataset, read_manifest, seed_worker
from losses import BCEDiceLoss
from metrics import (
    StandardSegmentationMetrics,
    boundary_f1_from_probabilities,
    per_image_iou_dice_from_logits,
)
from model_factory import MODEL_NAME, SUPPORTED_MODELS, build_model, split_outputs


ROOT = Path(__file__).resolve().parent
DEFAULT_SPLITS = ROOT / "splits"
EXPECTED_COUNTS = {
    "clinicdb": {"train": 490, "val": 61, "test": 61},
    "kvasir": {"train": 800, "val": 100, "test": 100},
    "combined": {"train": 1290, "val": 161, "test": 161},
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def append_csv(path, row):
    path = Path(path)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def manifest_paths(splits, protocol):
    base = Path(splits) / protocol
    return {split: base / f"{split}.csv" for split in ("train", "val", "test")}


def validate_protocol_splits(splits, protocol, strict_counts=True):
    paths = manifest_paths(splits, protocol)
    rows = {name: read_manifest(path) for name, path in paths.items()}
    ids = {name: {row["sample_id"] for row in values} for name, values in rows.items()}
    if ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"]:
        raise ValueError(f"Overlapping samples in {protocol} split manifests")
    if strict_counts:
        actual = {name: len(values) for name, values in rows.items()}
        if actual != EXPECTED_COUNTS[protocol]:
            raise ValueError(
                f"{protocol} counts must be {EXPECTED_COUNTS[protocol]}, got {actual}. "
                "Regenerate the fixed manifests before training."
            )
    if protocol == "clinicdb":
        groups = {name: {row["group"] for row in values} for name, values in rows.items()}
        if groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"]:
            raise ValueError("CVC-ClinicDB video sequences cross split boundaries")
    if protocol == "combined":
        datasets = {row["dataset"] for values in rows.values() for row in values}
        if datasets != {"clinicdb", "kvasir"}:
            raise ValueError(f"Pooled protocol must contain clinicdb and kvasir, got {sorted(datasets)}")
    return paths, rows


def build_loader(manifest, args, train=False):
    dataset = ManifestDataset(
        manifest=manifest,
        data_root=args.data_root,
        size=args.size,
        train=train,
        limit=args.limit,
    )
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size if train else args.eval_batch_size,
        shuffle=train,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=args.workers > 0,
    )


def segmentation_loss(criterion, outputs, masks, auxiliary_weight=0.4):
    main_logits, auxiliary_logits = split_outputs(outputs)
    loss = criterion(main_logits, masks)
    if auxiliary_logits is not None:
        loss = loss + auxiliary_weight * criterion(auxiliary_logits, masks)
    return loss, main_logits


def train_epoch(model, loader, optimizer, scaler, criterion, device, amp, grad_clip, auxiliary_weight):
    model.train()
    total_loss = 0.0
    count = 0
    amp_enabled = bool(amp and device.type == "cuda")
    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            loss, _ = segmentation_loss(criterion, model(images), masks, auxiliary_weight)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach()) * images.shape[0]
        count += images.shape[0]
    if not count:
        raise RuntimeError("Training loader produced no samples")
    return total_loss / count


@torch.no_grad()
def evaluate_validation(model, loader, criterion, device, auxiliary_weight):
    model.eval()
    loss_sum = iou_sum = dice_sum = 0.0
    count = 0
    for images, masks, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        loss, main_logits = segmentation_loss(criterion, model(images), masks, auxiliary_weight)
        iou, dice = per_image_iou_dice_from_logits(main_logits, masks)
        batch = images.shape[0]
        loss_sum += float(loss) * batch
        iou_sum += float(iou.sum())
        dice_sum += float(dice.sum())
        count += batch
    if not count:
        raise RuntimeError("Validation loader produced no samples")
    return {"loss": loss_sum / count, "mean_iou": iou_sum / count, "mean_dice": dice_sum / count}


@torch.no_grad()
def evaluate_full(model, loader, device):
    model.eval()
    meter = StandardSegmentationMetrics()
    per_image = []
    boundary_f1_sum = 0.0
    boundary_tolerance_pixels = None
    for images, masks, sample_ids in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        main_logits, _ = split_outputs(model(images))
        probability = torch.sigmoid(main_logits)
        meter.update(probability, masks)
        iou, dice = per_image_iou_dice_from_logits(main_logits, masks)
        mae = torch.abs(probability - masks).flatten(1).mean(1)
        boundary_f1, tolerance_pixels = boundary_f1_from_probabilities(probability, masks)
        boundary_tolerance_pixels = tolerance_pixels
        boundary_f1_sum += float(boundary_f1.sum())
        for index, sample_id in enumerate(sample_ids):
            per_image.append({
                "sample_id": sample_id,
                "iou": float(iou[index]),
                "dice": float(dice[index]),
                "boundary_f1": float(boundary_f1[index]),
                "mae": float(mae[index]),
            })
    if not per_image:
        raise RuntimeError("Evaluation loader produced no samples")
    accumulated = meter.result()
    metrics = {
        "mean_iou": accumulated["mean_iou"],
        "mean_dice": accumulated["mean_dice"],
        "boundary_f1": boundary_f1_sum / len(per_image),
        "f_beta_omega": accumulated["f_beta_omega"],
        "s_alpha": accumulated["s_alpha"],
        "e_phi_max": accumulated["e_phi_max"],
        "mae": accumulated["mae"],
        "boundary_tolerance_pixels": boundary_tolerance_pixels,
    }
    return metrics, per_image


def checkpoint_payload(model, optimizer, scheduler, scaler, epoch, best_iou, bad_epochs, resolved):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_val_miou": best_iou,
        "epochs_without_improvement": bad_epochs,
        "resolved_config": resolved,
    }


def load_checkpoint_model(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    resolved = checkpoint.get("resolved_config", {})
    model = build_model(resolved).to(device)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)
    return model, checkpoint


def train_command(args):
    seed_everything(args.seed)
    paths, rows = validate_protocol_splits(args.splits, args.protocol, not args.allow_nonstandard_counts)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dir = args.output / (args.run_name or args.protocol)
    run_dir.mkdir(parents=True, exist_ok=True)
    copied_dir = run_dir / "manifests"
    copied_dir.mkdir(exist_ok=True)
    copied_paths = {}
    for split, source in paths.items():
        destination = copied_dir / f"{split}.csv"
        shutil.copy2(source, destination)
        copied_paths[split] = destination
    manifest_hashes = {name: sha256_file(path) for name, path in copied_paths.items()}
    resolved = {
        "model_name": args.model_name,
        "protocol": args.protocol,
        "seed": args.seed,
        "size": args.size,
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "epochs": args.epochs,
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "betas": [0.9, 0.999],
        "weight_decay": args.weight_decay,
        "scheduler": "CosineAnnealingLR",
        "t_max": args.epochs,
        "min_lr": args.min_lr,
        "warmup_epochs": 0,
        "early_stopping_patience": args.patience,
        "selection_metric": "validation mean IoU (image-wise macro average)",
        "amp": args.amp,
        "grad_clip_norm": args.grad_clip,
        "auxiliary_loss_weight": args.auxiliary_weight,
        "drop_rate": 0.2,
        "base_drop_rate": 0.0,
        "base_drop_path_rate": 0.0,
        "csmamba_drop_path_rate": 0.1,
        "drop_path_rate": 0.1,
        "attention_dropout": 0.0,
        "pretrained": False,
        "split_counts": {name: len(values) for name, values in rows.items()},
        "manifest_sha256": manifest_hashes,
        "git_commit": git_commit(),
    }
    write_json(run_dir / "config.resolved.json", resolved)
    model = build_model(resolved).to(device)
    criterion = BCEDiceLoss(smooth=1.0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_lr
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    train_loader = build_loader(copied_paths["train"], args, train=True)
    val_loader = build_loader(copied_paths["val"], args, train=False)
    start_epoch, best_iou, bad_epochs = 1, -math.inf, 0
    best_epoch = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        previous = checkpoint.get("resolved_config", {})
        if previous.get("manifest_sha256") != manifest_hashes:
            raise ValueError("Resume checkpoint uses different split manifests")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_iou = checkpoint["best_val_miou"]
        bad_epochs = checkpoint["epochs_without_improvement"]
    log_path = run_dir / "training_log.csv"
    final_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        began = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler, criterion, device,
            args.amp, args.grad_clip, args.auxiliary_weight,
        )
        validation = evaluate_validation(
            model, val_loader, criterion, device, args.auxiliary_weight
        )
        if not math.isfinite(validation["mean_iou"]):
            raise RuntimeError("Validation mIoU is not finite")
        improved = validation["mean_iou"] > best_iou + args.min_delta
        if improved:
            best_iou = validation["mean_iou"]
            bad_epochs = 0
            best_epoch = epoch
        else:
            bad_epochs += 1
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        payload = checkpoint_payload(
            model, optimizer, scheduler, scaler, epoch, best_iou, bad_epochs, resolved
        )
        torch.save(payload, run_dir / "last.pt")
        if improved:
            torch.save(payload, run_dir / "best_miou.pt")
        row = {
            "epoch": epoch,
            "lr": current_lr,
            "seconds": round(time.time() - began, 2),
            "train_loss": train_loss,
            "val_loss": validation["loss"],
            "val_mean_iou": validation["mean_iou"],
            "val_mean_dice": validation["mean_dice"],
            "best_val_mean_iou": best_iou,
            "epochs_without_improvement": bad_epochs,
        }
        append_csv(log_path, row)
        print(
            f"[{args.protocol}] epoch={epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_mIoU={validation['mean_iou']:.4f} "
            f"val_mDice={validation['mean_dice']:.4f} best={best_iou:.4f} "
            f"patience={bad_epochs}/{args.patience}",
            flush=True,
        )
        final_epoch = epoch
        if bad_epochs >= args.patience:
            print(f"Early stopping after {args.patience} consecutive non-improving epochs")
            break
    if not (run_dir / "best_miou.pt").is_file():
        raise RuntimeError("Training ended without a best checkpoint")
    best_checkpoint = torch.load(run_dir / "best_miou.pt", map_location="cpu", weights_only=False)
    best_epoch = int(best_checkpoint["epoch"])
    metadata = {
        "protocol": args.protocol,
        "model_name": args.model_name,
        "best_epoch": best_epoch,
        "best_validation_miou": float(best_checkpoint["best_val_miou"]),
        "stopped_epoch": final_epoch,
        "checkpoint": "best_miou.pt",
        "checkpoint_sha256": sha256_file(run_dir / "best_miou.pt"),
        "manifest_sha256": manifest_hashes,
        "split_counts": resolved["split_counts"],
        "reserved_test_policy": (
            "stored for protocol completeness and not evaluated"
            if args.protocol == "combined" else "evaluated only by an explicit evaluation command"
        ),
    }
    write_json(run_dir / "checkpoint_metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def write_per_image(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "iou", "dice", "boundary_f1", "mae"],
        )
        writer.writeheader()
        writer.writerows(rows)


def evaluate_command(args):
    seed_everything(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint = load_checkpoint_model(args.checkpoint, device)
    resolved = checkpoint.get("resolved_config", {})
    args.size = int(resolved.get("size", args.size))
    checkpoint_protocol = resolved.get("protocol")
    checkpoint_hash = sha256_file(args.checkpoint)
    results = {}
    for specification in args.target:
        name, separator, manifest_text = specification.partition("=")
        if not separator or not name or not manifest_text:
            raise ValueError(f"Target must use NAME=MANIFEST format: {specification}")
        manifest = Path(manifest_text)
        manifest_hash = sha256_file(manifest)
        reserved_hash = resolved.get("manifest_sha256", {}).get("test")
        if checkpoint_protocol == "combined" and (
            name.lower() in {"combined", "pooled", "reserved"}
            or (reserved_hash and manifest_hash == reserved_hash)
        ):
            raise ValueError(
                "The pooled reserved test subset is not a reported endpoint in the manuscript. "
                "Evaluate the fixed pooled checkpoint only on CVC-300, CVC-ColonDB and ETIS."
            )
        loader = build_loader(manifest, args, train=False)
        metrics, per_image = evaluate_full(model, loader, device)
        results[name] = {
            "metrics": metrics,
            "samples": len(per_image),
            "manifest": str(manifest),
            "manifest_sha256": manifest_hash,
            "checkpoint_sha256": checkpoint_hash,
        }
        write_per_image(args.output_dir / f"{name}_per_image.csv", per_image)
        print(name, json.dumps(results[name], ensure_ascii=False), flush=True)
    report = {
        "model_name": resolved.get("model_name", MODEL_NAME),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_protocol": checkpoint_protocol,
        "results": results,
    }
    write_json(args.output_dir / "summary.json", report)


def smoke_command(args):
    seed_everything(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model({"model_name": args.model_name, "size": args.size}).to(device)
    criterion = BCEDiceLoss(smooth=1.0)
    loader = build_loader(args.manifest, args, train=True)
    images, masks, _ = next(iter(loader))
    images, masks = images.to(device), masks.to(device)
    loss, main_logits = segmentation_loss(criterion, model(images), masks, 0.4)
    iou, dice = per_image_iou_dice_from_logits(main_logits, masks)
    print(json.dumps({
        "loss": float(loss.detach()), "mean_iou": float(iou.mean()),
        "mean_dice": float(dice.mean()), "output_shape": list(main_logits.shape),
    }, indent=2))


def add_common(parser):
    parser.add_argument("--data-root", type=Path, default=ROOT / "inputs")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--device")
    parser.add_argument("--limit", type=int, help="debug only; never use for reported results")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="train one protocol-specific model instance")
    add_common(train)
    train.add_argument("--protocol", choices=tuple(EXPECTED_COUNTS), required=True)
    train.add_argument("--model-name", choices=SUPPORTED_MODELS, default=MODEL_NAME)
    train.add_argument("--run-name")
    train.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    train.add_argument("--output", type=Path, default=ROOT / "experiments")
    train.add_argument("--epochs", type=int, default=200)
    train.add_argument("--patience", type=int, default=20)
    train.add_argument("--min-delta", type=float, default=0.0)
    train.add_argument("--lr", type=float, default=5e-4)
    train.add_argument("--min-lr", type=float, default=1e-5)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--grad-clip", type=float, default=1.0)
    train.add_argument("--auxiliary-weight", type=float, default=0.4)
    train.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--resume", type=Path)
    train.add_argument(
        "--allow-nonstandard-counts", action="store_true",
        help="debug only; disables the manuscript count check",
    )
    train.set_defaults(func=train_command)

    evaluate = subparsers.add_parser("evaluate", help="evaluate one fixed best checkpoint")
    add_common(evaluate)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument(
        "--target", action="append", required=True,
        help="NAME=MANIFEST; repeat for the three external datasets",
    )
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.set_defaults(func=evaluate_command)

    smoke = subparsers.add_parser("smoke", help="one forward/loss pass for installation checks")
    add_common(smoke)
    smoke.add_argument("--model-name", choices=SUPPORTED_MODELS, default=MODEL_NAME)
    smoke.add_argument("--manifest", type=Path, required=True)
    smoke.set_defaults(func=smoke_command)
    return parser.parse_args(argv)


def main(argv=None):
    arguments = parse_args(argv)
    arguments.func(arguments)


if __name__ == "__main__":
    main()
