"""Select regularised linear and nonlinear classifiers using training data only."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data import flatten, training_fingerprint
from .explore import save_figure

SEED = 42
SCORING = {
    "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy",
    "f1": make_scorer(f1_score, pos_label=7), "roc_auc": "roc_auc",
}


def model_specs() -> dict:
    """Compare common intensity scaling with optional fold-fitted standardisation."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=5000, solver="lbfgs", random_state=SEED),
            {"model__C": [0.001, 0.01, 0.1, 1, 10, 100],
             "scale": ["passthrough", StandardScaler()]},
        ),
        "rbf_svm": (
            SVC(kernel="rbf", random_state=SEED),
            {"model__C": [0.01, 0.1, 1, 10, 100],
             "model__gamma": [0.001, 0.01, 0.1, "scale"],
             "scale": ["passthrough", StandardScaler()]},
        ),
    }


def train(images: np.ndarray, labels: np.ndarray, output: Path) -> dict:
    """Tune both models on identical repeated stratified folds and freeze selection."""
    output.mkdir(parents=True, exist_ok=True)
    x = flatten(images)
    cv = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5,
                                    random_state=SEED).split(x, labels))
    (output / "cv_splits.json").write_text(json.dumps([
        {"train": a.tolist(), "validation": b.tolist()} for a, b in cv
    ]))
    summary = {"seed": SEED, "selection_metric": "accuracy", "cv": "5-fold stratified, 5 repeats",
               "training_fingerprint": training_fingerprint(images, labels),
               "numpy_version": np.__version__, "sklearn_version": sklearn.__version__, "models": {}}
    fitted = {}
    for name, (estimator, grid) in model_specs().items():
        pipeline = Pipeline([("remove_constant", VarianceThreshold()),
                             ("scale", "passthrough"), ("model", estimator)])
        search = GridSearchCV(pipeline, grid, scoring=SCORING, refit="accuracy",
                              cv=cv, n_jobs=1, return_train_score=True, error_score="raise")
        search.fit(x, labels)
        table = pd.DataFrame(search.cv_results_)
        table.to_csv(output / f"{name}_cv.csv", index=False)
        row = table.iloc[search.best_index_]
        summary["models"][name] = {
            "best_params": {k: str(v) if k == "scale" else v for k, v in search.best_params_.items()},
            "cv_metrics": {metric: {"mean": float(row[f"mean_test_{metric}"]),
                                      "fold_sd": float(row[f"std_test_{metric}"])} for metric in SCORING},
            "training_accuracy": float(row["mean_train_accuracy"]),
            "candidate_count": len(table),
        }
        fitted[name] = search.best_estimator_
        joblib.dump(search.best_estimator_, output / f"{name}.joblib")
        plot_search(table, name, output)
        print(name, summary["models"][name], flush=True)
    # Ties retain the first model, favouring the simpler linear classifier.
    selected = max(summary["models"], key=lambda name: summary["models"][name]["cv_metrics"]["accuracy"]["mean"])
    summary["selected_model"] = selected
    joblib.dump(fitted[selected], output / "selected_model.joblib")
    (output / "selection.json").write_text(json.dumps(summary, indent=2))
    return summary


def plot_search(table: pd.DataFrame, name: str, output: Path) -> None:
    """Display all configurations without selecting curves using test results."""
    fig, ax = plt.subplots(figsize=(9, 5))
    grouping = ["param_scale"] + (["param_model__gamma"] if "param_model__gamma" in table else [])
    for key, group in table.groupby(grouping, dropna=False):
        group = group.sort_values("param_model__C")
        label = ", ".join(map(str, key if isinstance(key, tuple) else (key,)))
        ax.errorbar(group["param_model__C"].astype(float), group["mean_test_accuracy"],
                    yerr=group["std_test_accuracy"], marker="o", capsize=2, label=label)
    ax.set(xscale="log", xlabel="C (inverse regularisation strength)", ylabel="Validation accuracy",
           title=f"{name}: mean accuracy and fold SD (25 folds)")
    ax.legend(title="Scaling, gamma" if len(grouping) == 2 else "Scaling", fontsize=7)
    ax.grid(alpha=0.2)
    save_figure(fig, output / f"{name}_tuning.png")
