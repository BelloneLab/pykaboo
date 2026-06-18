"""Cluster EmbTCN SSL embeddings into unsupervised behavioral motifs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .config import load_config
from .storage import read_table, write_table


def embedding_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if str(col).startswith("embedding_")]


def compute_embedding_projection(
    x: np.ndarray,
    n_components: int,
    random_state: int,
) -> np.ndarray:
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=30,
            min_dist=0.1,
            random_state=random_state,
        )
        return reducer.fit_transform(x)
    except Exception:
        n = min(n_components, x.shape[1], max(x.shape[0] - 1, 1))
        if n <= 0:
            return np.zeros((x.shape[0], n_components), dtype=np.float32)
        reduced = PCA(n_components=n, random_state=random_state).fit_transform(x)
        if n < n_components:
            pad = np.zeros((reduced.shape[0], n_components - n), dtype=reduced.dtype)
            reduced = np.concatenate([reduced, pad], axis=1)
        return reduced


def cluster_projection(
    projection: np.ndarray,
    method: str,
    n_clusters: int,
    min_cluster_size: int,
    random_state: int,
) -> np.ndarray:
    if len(projection) == 0:
        return np.asarray([], dtype=int)
    if method == "hdbscan":
        try:
            import hdbscan  # type: ignore

            return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(projection)
        except Exception:
            method = "kmeans"
    if method == "dbscan":
        return DBSCAN(eps=0.8, min_samples=max(min_cluster_size // 10, 5)).fit_predict(projection)
    if method == "agglomerative":
        return AgglomerativeClustering(n_clusters=n_clusters).fit_predict(projection)
    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(projection)


def cluster_purity(labels: np.ndarray, truth: np.ndarray) -> float:
    total = 0
    correct = 0
    for cluster_id in np.unique(labels):
        if cluster_id < 0:
            continue
        mask = labels == cluster_id
        if not mask.any():
            continue
        values, counts = np.unique(truth[mask], return_counts=True)
        total += int(mask.sum())
        correct += int(counts.max()) if len(values) else 0
    return correct / total if total else 0.0


def score_clusters(df: pd.DataFrame, cluster_col: str, label_col: str | None) -> dict:
    if not label_col or label_col not in df.columns:
        return {}
    valid = df[label_col].notna() & (df[label_col].astype(str) != "")
    valid &= df[cluster_col] >= 0
    if valid.sum() < 2:
        return {}
    truth = df.loc[valid, label_col].astype(str).to_numpy()
    labels = df.loc[valid, cluster_col].to_numpy()
    return {
        "nmi": float(normalized_mutual_info_score(truth, labels)),
        "ari": float(adjusted_rand_score(truth, labels)),
        "purity": float(cluster_purity(labels, truth)),
        "scored_frames": int(valid.sum()),
    }


def cluster_embeddings(
    embeddings_path: str | Path,
    output_dir: str | Path,
    method: str = "hdbscan",
    n_clusters: int = 20,
    min_cluster_size: int = 100,
    n_components: int = 2,
    label_col: str | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    df = read_table(embeddings_path)
    cols = embedding_columns(df)
    if not cols:
        raise RuntimeError(f"No embedding columns found in {embeddings_path}.")
    x = df[cols].to_numpy(dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    projection = compute_embedding_projection(x, n_components, random_state)
    clusters = cluster_projection(
        projection,
        method=method,
        n_clusters=n_clusters,
        min_cluster_size=min_cluster_size,
        random_state=random_state,
    )
    out = df.drop(columns=cols).copy()
    for dim in range(projection.shape[1]):
        out[f"umap_{dim}"] = projection[:, dim]
    out["motif_id"] = clusters.astype(int)
    metrics = score_clusters(out, "motif_id", label_col)
    metrics.update(
        {
            "num_frames": int(len(out)),
            "num_motifs": int(len(set(clusters.tolist()) - {-1})),
            "noise_frames": int((clusters < 0).sum()),
            "method": method,
        }
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_table(out, output_dir / "ssl_motifs.parquet")
    pd.DataFrame([metrics]).to_csv(output_dir / "ssl_cluster_metrics.csv", index=False)
    summary = (
        out.groupby("motif_id")
        .agg(
            frames=("motif_id", "size"),
            mean_fault_score=("fault_score", "mean") if "fault_score" in out.columns else ("motif_id", "size"),
        )
        .reset_index()
        .sort_values("frames", ascending=False)
    )
    write_table(summary, output_dir / "ssl_motif_summary.csv")
    return out, metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cluster EmbTCN SSL embeddings.")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--method", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    method = args.method or config.clustering.method
    _out, metrics = cluster_embeddings(
        args.embeddings,
        args.output_dir,
        method=method,
        n_clusters=config.clustering.n_clusters,
        min_cluster_size=config.clustering.min_cluster_size,
        n_components=config.clustering.umap_components,
        label_col=args.label_column,
        random_state=config.project.seed,
    )
    print(metrics, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
