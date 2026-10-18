"""Explicit commands preserve the separation between development and testing."""

import argparse
from pathlib import Path

from .data import load_split
from .explore import explore


def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST 1/7 assignment experiments")
    parser.add_argument("stage", choices=["explore", "train", "evaluate", "cluster"])
    parser.add_argument("--data", type=Path, default=Path("../mnist_1_7_subsets-1.npz"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.stage == "explore":
        images, labels = load_split(args.data, "train")
        print(explore(images, labels, args.output / "exploration"))
    elif args.stage == "train":
        from .train import train
        images, labels = load_split(args.data, "train")
        train(images, labels, args.output / "training")
    elif args.stage == "evaluate":
        from .evaluate import evaluate
        if not (args.output / "training" / "selection.json").exists():
            parser.error("Run training and freeze model selection before test evaluation")
        images, labels = load_split(args.data, "test")
        print(evaluate(images, labels, args.output / "training", args.output / "evaluation"))
    elif args.stage == "cluster":
        from .cluster import cluster
        if not (args.output / "evaluation" / "test_metrics.json").exists():
            parser.error("Complete frozen-model test evaluation before clustering")
        images, labels = load_split(args.data, "test")
        print(cluster(images, labels, args.output / "clustering"))
