"""Explicit commands preserve the separation between development and testing."""

import argparse
from pathlib import Path

from .data import load_split
from .explore import explore


def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST 1/7 assignment experiments")
    parser.add_argument("stage", choices=["explore"])
    parser.add_argument("--data", type=Path, default=Path("../mnist_1_7_subsets-1.npz"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.stage == "explore":
        images, labels = load_split(args.data, "train")
        print(explore(images, labels, args.output / "exploration"))
