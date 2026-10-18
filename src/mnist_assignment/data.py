"""Load only the split requested by each experiment stage."""

from pathlib import Path

import numpy as np


def load_split(path: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Validate grayscale images and labels without opening the other split."""
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    with np.load(path, allow_pickle=False) as archive:
        images = archive[f"{split}_data"]
        labels = archive[f"{split}_targets"]
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError("Images must have shape (n, 28, 28)")
    if labels.shape != (len(images),) or not np.array_equal(np.unique(labels), [1, 7]):
        raise ValueError("Labels must align with images and contain both 1 and 7")
    if not np.isfinite(images).all() or images.min() < 0 or images.max() > 255:
        raise ValueError("Expected finite pixel intensities between 0 and 255")
    return images, labels


def flatten(images: np.ndarray) -> np.ndarray:
    """Use the fixed intensity scale; no population statistics are estimated."""
    return images.reshape(len(images), -1).astype(np.float64) / 255.0


def training_fingerprint(images: np.ndarray, labels: np.ndarray) -> str:
    """Identify the training data used by a saved experiment."""
    import hashlib

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(images).tobytes())
    digest.update(np.ascontiguousarray(labels).tobytes())
    return digest.hexdigest()
