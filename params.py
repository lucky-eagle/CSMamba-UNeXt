"""Report trainable parameters and FLOPs for one 256x256 RGB input."""

import argparse

import torch
from thop import profile

from model_factory import MODEL_NAME, SUPPORTED_MODELS, build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", choices=SUPPORTED_MODELS, default=MODEL_NAME)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    model = build_model({"model_name": args.model_name, "size": args.size}).to(device).eval()
    sample = torch.zeros(1, 3, args.size, args.size, device=device)
    macs, _ = profile(model, inputs=(sample,), verbose=False)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Parameters: {parameters / 1e6:.3f} M")
    print(f"MACs: {macs / 1e9:.3f} G")
    print(f"FLOPs (2 x MACs convention): {2 * macs / 1e9:.3f} G")
    print("Note: profiler support for selective-scan custom operations can affect FLOP totals.")


if __name__ == "__main__":
    main()
