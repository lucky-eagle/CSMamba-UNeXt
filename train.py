"""Compatibility entry point for protocol-specific training."""

import sys

from experiment import main


if __name__ == "__main__":
    main(["train", *sys.argv[1:]])
