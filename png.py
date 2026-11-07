"""Convert images in a directory to PNG without hard-coded local paths."""

import argparse
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    converted = 0
    for source in sorted(args.input_dir.iterdir()):
        if not source.is_file() or source.suffix.lower() not in {".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            continue
        destination = args.output_dir / f"{source.stem}.png"
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {destination}; pass --overwrite")
        with Image.open(source) as image:
            image.save(destination, "PNG")
        converted += 1
    print(f"Converted {converted} images")


if __name__ == "__main__":
    main()
