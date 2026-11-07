"""Run single-image inference with a validation-selected checkpoint."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataset import preprocess_rgb_image
from model_factory import build_model, split_outputs


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be within [0, 1]")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint.get("resolved_config", {})
    model = build_model(config).to(device)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)
    model.eval()
    with Image.open(args.image) as handle:
        image = handle.convert("RGB")
        original_size = image.size
        tensor = preprocess_rgb_image(image, int(config.get("size", 256))).unsqueeze(0)
    with torch.no_grad():
        main_logits, _ = split_outputs(model(tensor.to(device)))
        probability = torch.sigmoid(main_logits)[0, 0].cpu().numpy()
    mask = Image.fromarray((probability >= args.threshold).astype(np.uint8) * 255)
    mask = mask.resize(original_size, Image.Resampling.NEAREST)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mask.save(args.output, format="PNG")
    print(f"Saved binary mask to {args.output}")


if __name__ == "__main__":
    main()
