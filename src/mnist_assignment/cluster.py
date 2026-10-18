"""Explore handwriting styles among test images labelled seven."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from .data import flatten
from .explore import save_figure


def montage(images: np.ndarray, titles: list[str], path: Path, heading: str) -> None:
    columns = min(6 if len(images) <= 12 else 10, len(images))
    rows = int(np.ceil(len(images) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.6, rows * 1.7), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, image, title in zip(axes.flat, images, titles):
        ax.imshow(image.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=8)
    fig.suptitle(heading)
    fig.tight_layout()
    save_figure(fig, path)


def cluster(images: np.ndarray, labels: np.ndarray, output: Path) -> dict:
    """Fit k=100 first, then select k in 2..99 using mean silhouette."""
    output.mkdir(parents=True, exist_ok=True)
    source_indices = np.flatnonzero(labels == 7)
    x = flatten(images[source_indices])
    if len(np.unique(x, axis=0)) <= 100:
        raise ValueError("Need more than 100 distinct digit-seven images")
    baseline = KMeans(n_clusters=100, n_init=20, random_state=42).fit(x)
    counts = np.bincount(baseline.labels_, minlength=100)
    montage(baseline.cluster_centers_, [f"C{i}, n={n}" for i, n in enumerate(counts)],
            output / "centres_k100.png", "Test digit 7: initial 100 cluster centres")
    rows = []
    best_model = None
    best_score = -np.inf
    for k in range(2, 100):
        model = KMeans(n_clusters=k, n_init=20, random_state=42).fit(x)
        score = silhouette_score(x, model.labels_)
        rows.append({"k": k, "silhouette": score, "inertia": model.inertia_,
                     "min_cluster_size": int(np.bincount(model.labels_, minlength=k).min())})
        if score > best_score:
            best_model, best_score = model, score
        if k % 10 == 0:
            print(f"Clustering search: completed k={k}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(output / "cluster_search.csv", index=False)
    baseline_silhouette = silhouette_score(x, baseline.labels_)
    model = best_model
    sizes = np.bincount(model.labels_, minlength=model.n_clusters)
    distances = model.transform(x)
    representatives = [int(np.flatnonzero(model.labels_ == c)[
        np.argmin(distances[model.labels_ == c, c])]) for c in range(model.n_clusters)]
    np.savez_compressed(output / "clusters.npz", source_indices=source_indices,
                        labels=model.labels_, centres=model.cluster_centers_,
                        baseline_labels=baseline.labels_, baseline_centres=baseline.cluster_centers_)
    titles = [f"C{i}, n={n}" for i, n in enumerate(sizes)]
    montage(model.cluster_centers_, titles, output / "selected_centres.png", "Selected cluster centres: test digit 7")
    montage(x[representatives], titles, output / "selected_representatives.png", "Nearest member of each selected cluster")
    for c in range(model.n_clusters):
        members = np.flatnonzero(model.labels_ == c)
        ordered = members[np.argsort(distances[members, c])]
        chosen = np.unique(np.linspace(0, len(ordered) - 1, min(12, len(ordered))).astype(int))
        examples = ordered[chosen]
        montage(x[examples], [f"Test #{source_indices[i]}" for i in examples],
                output / f"cluster_{c}_examples.png", f"Cluster {c}: members from nearest to farthest")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(table.k, table.silhouette)
    axes[0].axvline(model.n_clusters, color="red", linestyle="--", label=f"Selected k={model.n_clusters}")
    axes[0].set(xlabel="Number of clusters", ylabel="Mean silhouette", title="Separation in 784-pixel space")
    axes[0].legend()
    axes[1].plot(table.k, table.inertia)
    axes[1].scatter([100], [baseline.inertia_], label="Initial k=100")
    axes[1].set(xlabel="Number of clusters", ylabel="Within-cluster sum of squares", title="Inertia / elbow diagnostic")
    axes[1].legend()
    fig.tight_layout()
    save_figure(fig, output / "cluster_selection.png")
    pca = PCA(n_components=2, svd_solver="full").fit(x)
    coordinates = pca.transform(x)
    fig, ax = plt.subplots(figsize=(7, 5))
    for c in range(model.n_clusters):
        ax.scatter(*coordinates[model.labels_ == c].T, s=12, alpha=0.6, label=f"C{c}")
    ax.set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
           ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", title="Display only: PCA projection of digit-seven clusters")
    ax.legend(fontsize=7)
    save_figure(fig, output / "cluster_projection.png")
    summary = {"n_digit7": len(x), "initial_k": 100, "initial_inertia": baseline.inertia_,
               "initial_silhouette": baseline_silhouette,
               "initial_singletons": int((counts == 1).sum()), "selected_k": model.n_clusters,
               "selected_silhouette": best_score, "selected_inertia": model.inertia_,
               "selection_rule": "Maximum mean Euclidean silhouette among every integer k from 2 to 99; ties prefer smaller k",
               "n_init": 20, "seed": 42, "cluster_sizes": sizes.tolist(),
               "representative_test_indices": source_indices[representatives].tolist(),
               "display_pca_explained_variance": pca.explained_variance_ratio_.tolist()}
    (output / "clustering_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
