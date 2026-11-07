"""Create the fixed split manifests used by the manuscript protocols."""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# CVC-ClinicDB's 612 numbered frames grouped by the source video sequences.
CVC_CLINICDB_SEQUENCE_RANGES = (
    (1, 25), (26, 50), (51, 67), (68, 78), (79, 103), (104, 126),
    (127, 151), (152, 177), (178, 199), (200, 205), (206, 227),
    (228, 252), (253, 277), (278, 297), (298, 317), (318, 342),
    (343, 363), (364, 383), (384, 408), (409, 428), (429, 447),
    (448, 466), (467, 478), (479, 503), (504, 528), (529, 546),
    (547, 571), (572, 591), (592, 612),
)


def relative_text(path, data_root):
    return Path(path).resolve().relative_to(Path(data_root).resolve()).as_posix()


def find_mask_dir(root):
    direct = root / "masks"
    class_zero = direct / "0"
    if class_zero.is_dir():
        return class_zero
    return direct


def paired_samples(dataset_name, root, data_root, allow_unpaired=False):
    root = Path(root)
    image_dir = root / "images"
    mask_dir = find_mask_dir(root)
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Expected images/ and masks/ under {root}")
    images = {
        path.stem: path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    masks = {
        path.stem: path for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    if images.keys() != masks.keys() and not allow_unpaired:
        raise ValueError(
            f"Unpaired files in {root}; missing masks={sorted(images.keys() - masks.keys())[:5]}, "
            f"missing images={sorted(masks.keys() - images.keys())[:5]}"
        )
    paired_ids = sorted(images.keys() & masks.keys())
    return [{
        "sample_id": f"{dataset_name}::{stem}",
        "dataset": dataset_name,
        "group": f"{dataset_name}::{stem}",
        "image": relative_text(images[stem], data_root),
        "mask": relative_text(masks[stem], data_root),
    } for stem in paired_ids]


def add_clinicdb_groups(samples):
    for row in samples:
        try:
            frame = int(row["sample_id"].split("::", 1)[1])
        except ValueError as error:
            raise ValueError(
                "CVC-ClinicDB grouped splitting expects official numeric frame names."
            ) from error
        matches = [
            index for index, (start, end) in enumerate(CVC_CLINICDB_SEQUENCE_RANGES, 1)
            if start <= frame <= end
        ]
        if len(matches) != 1:
            raise ValueError(f"No unique CVC-ClinicDB sequence for frame {frame}")
        row["group"] = f"clinicdb_sequence::{matches[0]:02d}"
    return samples


def target_counts(total):
    train = round(total * 0.8)
    val = round(total * 0.1)
    return train, val, total - train - val


def split_image_level(samples, seed):
    shuffled = sorted(samples, key=lambda row: row["sample_id"])
    random.Random(seed).shuffle(shuffled)
    train_count, val_count, _ = target_counts(len(shuffled))
    return (
        shuffled[:train_count],
        shuffled[train_count:train_count + val_count],
        shuffled[train_count + val_count:],
    )


def split_grouped_exact(samples, seed):
    grouped = defaultdict(list)
    for row in samples:
        grouped[row["group"]].append(row)
    train_target, val_target, test_target = target_counts(len(samples))
    items = sorted(grouped.items())
    random.Random(seed).shuffle(items)
    states = {(0, 0): ((), ())}
    for group, rows in items:
        size = len(rows)
        updated = dict(states)
        for (val_count, test_count), (val_groups, test_groups) in states.items():
            if val_count + size <= val_target:
                updated.setdefault(
                    (val_count + size, test_count), (val_groups + (group,), test_groups)
                )
            if test_count + size <= test_target:
                updated.setdefault(
                    (val_count, test_count + size), (val_groups, test_groups + (group,))
                )
        states = updated
    selection = states.get((val_target, test_target))
    if selection is None:
        raise ValueError(
            f"No group-preserving split reaches exact counts "
            f"{train_target}/{val_target}/{test_target}"
        )
    val_groups, test_groups = map(set, selection)
    train, val, test = [], [], []
    for row in samples:
        if row["group"] in val_groups:
            val.append(row)
        elif row["group"] in test_groups:
            test.append(row)
        else:
            train.append(row)
    return train, val, test


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["sample_id", "dataset", "group", "image", "mask"]
        )
        writer.writeheader()
        writer.writerows(rows)


def write_clinicdb_sequence_map(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: int(row["sample_id"].split("::", 1)[1]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "sequence_id"])
        writer.writeheader()
        writer.writerows(
            {"sample_id": row["sample_id"], "sequence_id": row["group"]}
            for row in ordered
        )


def validate_and_write(output, protocol, split_tuple):
    names = ("train", "val", "test")
    split_rows = dict(zip(names, split_tuple))
    sets = {name: {row["sample_id"] for row in rows} for name, rows in split_rows.items()}
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise ValueError(f"Overlap detected in {protocol}")
    for name, rows in split_rows.items():
        write_manifest(output / protocol / f"{name}.csv", rows)
    return {
        "counts": {name: len(rows) for name, rows in split_rows.items()},
        "groups": {name: len({row["group"] for row in rows}) for name, rows in split_rows.items()},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("inputs"))
    parser.add_argument("--clinicdb-root", type=Path, default=Path("inputs/CVC-ClinicDB"))
    parser.add_argument("--kvasir-root", type=Path, default=Path("inputs/Kvasir"))
    parser.add_argument(
        "--cvc300-root", type=Path,
        default=Path("inputs/polyp/TestDataset/CVC-300"),
    )
    parser.add_argument(
        "--colondb-root", type=Path,
        default=Path("inputs/polyp/TestDataset/CVC-ColonDB"),
    )
    parser.add_argument(
        "--etis-root", type=Path,
        default=Path("inputs/polyp/TestDataset/ETIS-LaribPolypDB"),
    )
    parser.add_argument("--output", type=Path, default=Path("splits"))
    parser.add_argument(
        "--clinicdb-map-output", type=Path,
        default=Path("metadata/cvc_clinicdb_sequence_map.csv"),
    )
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--allow-unpaired", action="store_true",
        help="ignore extra files that do not have an image/mask partner",
    )
    args = parser.parse_args(argv)

    clinicdb = add_clinicdb_groups(
        paired_samples(
            "clinicdb", args.clinicdb_root, args.data_root, args.allow_unpaired
        )
    )
    kvasir = paired_samples(
        "kvasir", args.kvasir_root, args.data_root, args.allow_unpaired
    )
    if len(clinicdb) != 612 or len(kvasir) != 1000:
        raise ValueError(
            f"Expected 612 CVC-ClinicDB and 1000 Kvasir-SEG images, got "
            f"{len(clinicdb)} and {len(kvasir)}"
        )
    clinic_splits = split_grouped_exact(clinicdb, args.seed)
    write_clinicdb_sequence_map(args.clinicdb_map_output, clinicdb)
    kvasir_splits = split_image_level(kvasir, args.seed)
    pooled_splits = split_image_level(clinicdb + kvasir, args.seed)
    summary = {
        "seed": args.seed,
        "protocols": {
            "clinicdb": validate_and_write(args.output, "clinicdb", clinic_splits),
            "kvasir": validate_and_write(args.output, "kvasir", kvasir_splits),
            "combined": validate_and_write(args.output, "combined", pooled_splits),
        },
        "pooled_reserved_test_policy": "manifest only; not evaluated or reported",
    }
    external_roots = {
        "cvc-300": args.cvc300_root,
        "cvc-colondb": args.colondb_root,
        "etis": args.etis_root,
    }
    summary["external"] = {}
    for name, root in external_roots.items():
        rows = paired_samples(name, root, args.data_root, args.allow_unpaired)
        write_manifest(args.output / "external" / f"{name}.csv", rows)
        summary["external"][name] = len(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
