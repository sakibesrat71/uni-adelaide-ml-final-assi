"""Training-set statistics and visual inspection."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import flatten


def save_figure(fig, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def explore(images: np.ndarray, labels: np.ndarray, output: Path) -> dict:
    """Summarise only training data and save labelled diagnostic figures."""
    output.mkdir(parents=True, exist_ok=True)
    x = flatten(images)
    stats = {
        "n_samples": len(images), "n_features": x.shape[1],
        "image_shape": list(images.shape[1:]), "dtype": str(images.dtype),
        "class_counts": {str(d): int((labels == d).sum()) for d in [1, 7]},
        "pixel_min": int(images.min()), "pixel_max": int(images.max()),
        "missing_values": int((~np.isfinite(x)).sum()),
        "duplicate_images": int(len(x) - len(np.unique(x, axis=0))),
        "constant_pixels": int((np.ptp(x, axis=0) == 0).sum()),
        "zero_pixel_fraction": float((x == 0).mean()),
        "class_statistics": {},
    }
    for digit in [1, 7]:
        subset = x[labels == digit]
        stats["class_statistics"][str(digit)] = {
            "mean_intensity": float(subset.mean()),
            "mean_image_intensity_sd": float(subset.mean(axis=1).std(ddof=1)),
            "mean_nonzero_pixels": float((subset > 0).sum(axis=1).mean()),
        }
    (output / "training_summary.json").write_text(json.dumps(stats, indent=2))

    fig, axes = plt.subplots(2, 8, figsize=(10, 3))
    for row, digit in enumerate([1, 7]):
        for ax, index in zip(axes[row], np.flatnonzero(labels == digit)[:8]):
            ax.imshow(images[index], cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"Digit {digit}, index {index}", fontsize=8)
            ax.axis("off")
    fig.suptitle("Training examples: handwriting variation")
    save_figure(fig, output / "training_examples.png")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    means = [images[labels == digit].mean(axis=0) for digit in [1, 7]]
    for ax, mean, digit in zip(axes[:2], means, [1, 7]):
        im = ax.imshow(mean, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Mean training image: {digit}")
        fig.colorbar(im, ax=ax, fraction=0.046, label="Pixel intensity")
    diff = means[1] - means[0]
    im = axes[2].imshow(diff, cmap="RdBu_r", vmin=-255, vmax=255)
    axes[2].set_title("Mean difference: 7 minus 1")
    fig.colorbar(im, ax=axes[2], fraction=0.046, label="Intensity difference")
    for ax in axes:
        ax.set_xlabel("Pixel column")
        ax.set_ylabel("Pixel row")
    fig.tight_layout()
    save_figure(fig, output / "training_mean_images.png")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    axes[0].bar(["1", "7"], [int((labels == d).sum()) for d in [1, 7]])
    axes[0].set(xlabel="Digit", ylabel="Training images", title="Class balance")
    for digit in [1, 7]:
        subset = x[labels == digit]
        axes[1].hist(subset.mean(axis=1), bins=12, alpha=0.6, label=str(digit))
        axes[2].hist((subset > 0).sum(axis=1), bins=12, alpha=0.6, label=str(digit))
    axes[1].set(xlabel="Mean intensity / 255", ylabel="Images", title="Image intensity")
    axes[2].set(xlabel="Nonzero pixels per image", ylabel="Images", title="Stroke extent")
    axes[1].legend(title="Digit")
    axes[2].legend(title="Digit")
    fig.tight_layout()
    save_figure(fig, output / "training_distributions.png")
    return stats
