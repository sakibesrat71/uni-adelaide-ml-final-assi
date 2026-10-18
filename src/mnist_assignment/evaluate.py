"""Evaluate frozen classifiers once, without changing their parameters."""

import hashlib
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             balanced_accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

from .data import flatten
from .explore import save_figure


def wilson_interval(successes: int, total: int) -> list[float]:
    """Approximate 95% binomial interval for held-out accuracy."""
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total**2)) / denominator
    return [float(centre - radius), float(centre + radius)]


def evaluate(images: np.ndarray, labels: np.ndarray, training: Path, output: Path) -> dict:
    """Report both preselected models; the development winner remains unchanged."""
    if (output / "test_metrics.json").exists():
        raise FileExistsError("Test evaluation already exists. Inspect it instead of rerunning selection.")
    selection_path = training / "selection.json"
    selection = json.loads(selection_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    result = {"selected_model": selection["selected_model"], "n_test": len(images),
              "selection_sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest(), "models": {}}
    x = flatten(images)
    for name in selection["models"]:
        model = joblib.load(training / f"{name}.joblib")
        prediction = model.predict(x)
        scores = model.decision_function(x)
        result["models"][name] = {
            "accuracy": accuracy_score(labels, prediction),
            "accuracy_wilson_95": wilson_interval(int((prediction == labels).sum()), len(labels)),
            "balanced_accuracy": balanced_accuracy_score(labels, prediction),
            "f1_digit7": f1_score(labels, prediction, pos_label=7),
            "precision_digit7": precision_score(labels, prediction, pos_label=7, zero_division=0),
            "recall_digit7": recall_score(labels, prediction, pos_label=7),
            "roc_auc": roc_auc_score(labels == 7, scores),
            "confusion_matrix_labels_1_7": confusion_matrix(labels, prediction, labels=[1, 7]).tolist(),
            "classification_report": classification_report(labels, prediction, output_dict=True),
        }
        np.savez_compressed(output / f"{name}_predictions.npz", truth=labels, prediction=prediction, score=scores)
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        ConfusionMatrixDisplay.from_predictions(labels, prediction, labels=[1, 7], ax=axes[0], colorbar=False)
        RocCurveDisplay.from_predictions(labels == 7, scores, ax=axes[1], name=name)
        axes[0].set_title(f"{name}: test confusion matrix")
        axes[1].set_title("Test ROC curve (digit 7 positive)")
        fig.tight_layout()
        save_figure(fig, output / f"{name}_test.png")
        errors = np.flatnonzero(prediction != labels)
        if len(errors):
            count = min(24, len(errors))
            fig, axes = plt.subplots(int(np.ceil(count / 6)), 6, figsize=(10, 2 * int(np.ceil(count / 6))), squeeze=False)
            for ax in axes.flat:
                ax.axis("off")
            for ax, index in zip(axes.flat, errors[:count]):
                ax.imshow(images[index], cmap="gray", vmin=0, vmax=255)
                ax.set_title(f"#{index}: {labels[index]} -> {prediction[index]}", fontsize=8)
            fig.suptitle(f"{name}: test errors (first {count} of {len(errors)})")
            save_figure(fig, output / f"{name}_errors.png")
    (output / "test_metrics.json").write_text(json.dumps(result, indent=2))
    return result
