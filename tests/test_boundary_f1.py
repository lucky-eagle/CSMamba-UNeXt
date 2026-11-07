"""Small regression checks for the manuscript Boundary-F1 definition.

Run with ``python test_boundary_f1.py`` inside the project environment.
"""

import torch

from metrics import boundary_f1_from_probabilities


def main():
    target = torch.zeros(2, 1, 256, 256)
    target[0, 0, 48:112, 64:128] = 1
    target[1, 0, 20:44, 20:44] = 1

    identical, tolerance = boundary_f1_from_probabilities(target, target)
    assert tolerance == 3
    assert torch.allclose(identical, torch.ones_like(identical))

    missed, _ = boundary_f1_from_probabilities(torch.zeros_like(target), target)
    assert torch.equal(missed, torch.zeros_like(missed))
    print("Boundary-F1 regression checks passed (256x256 tolerance = 3 pixels).")


if __name__ == "__main__":
    main()
