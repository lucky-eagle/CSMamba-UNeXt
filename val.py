"""Compatibility entry point for fixed-checkpoint evaluation."""

import sys

from experiment import main


if __name__ == "__main__":
    main(["evaluate", *sys.argv[1:]])
