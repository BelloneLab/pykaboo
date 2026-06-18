"""Unsupervised motif discovery on exported embeddings.

UMAP reduces the embedding dimensionality for visualization, and a density-based
or centroid-based clusterer groups frames into motifs. HDBSCAN is preferred when
installed; otherwise KMeans, Gaussian mixtures, DBSCAN, or agglomerative
clustering from scikit-learn are used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ClusteringConfig, load_config
from .storage import read_table, write_table

EMBED_PREFIXES = ("embedding_", "mean_", "std_", "max_")
META_COLUMNS = {
    "video_id",
    "frame_idx",
    "subject_id",
    "object_id",
    "predicted_behavior",
    "confidence",
    "behavior_id",
    "start_frame",
    "end_frame",
    "duration_frames",
}


def embedding_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = [
        c
        for c in df.columns
        if c not in META_COLUMNS and c.startswith(EMBED_PREFIXES)
    ]
    if not cols:
        cols = [
            c
            for c in df.columns
            if c not in META_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
        ]
    return df[cols].to_numpy(dtype=np.float64), cols


def reduce_umap(matrix: np.ndarray, n_components: int = 2, seed: int = 42) -> np.ndarray:
    try:
        import umap

        reducer = umap.UMAP(
            n_components=n_components, random_state=seed, n_neighbors=15, min_dist=0.1
        )
        return reducer.fit_transform(matrix)
    except Exception:
        from sklearn.decomposition import PCA

        n = min(n_components, matrix.shape[1])
        return PCA(n_components=n, random_state=seed).fit_transform(matrix)


def run_clustering(matrix: np.ndarray, config: ClusteringConfig) -> np.ndarray:
    method = config.method.lower()
    if method == "hdbscan":
        try:
            import hdbscan

            clusterer = hdbscan.HDBSCAN(min_cluster_size=config.min_cluster_size)
            return clusterer.fit_predict(matrix)
        except Exception:
            method = "kmeans"
    if method == "kmeans":
        from sklearn.cluster import KMeans

        return KMeans(n_clusters=config.n_clusters, n_init=10, random_state=42).fit_predict(
            matrix
        )
    if method == "gmm":
        from sklearn.mixture import GaussianMixture

        return GaussianMixture(
            n_components=config.n_clusters, random_state=42
        ).fit_predict(matrix)
    if method == "dbscan":
        from sklearn.cluster import DBSCAN

        return DBSCAN(min_samples=max(config.min_cluster_size // 4, 4)).fit_predict(
            matrix
        )
    from sklearn.cluster import AgglomerativeClustering

    return AgglomerativeClustering(n_clusters=config.n_clusters).fit_predict(matrix)


def cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster_id, group in df.groupby("cluster"):
        behaviors = group.get("predicted_behavior")
        dominant = behaviors.mode().iloc[0] if behaviors is not None and not behaviors.empty else ""
        purity = (
            float((behaviors == dominant).mean())
            if behaviors is not None and not behaviors.empty
            else 0.0
        )
        rows.append(
            {
                "cluster": int(cluster_id),
                "num_frames": int(len(group)),
                "dominant_behavior": dominant,
                "label_purity": purity,
                "videos": group["video_id"].nunique() if "video_id" in group else 0,
                "subjects": group["subject_id"].nunique()
                if "subject_id" in group
                else 0,
            }
        )
    return pd.DataFrame(rows).sort_values("num_frames", ascending=False)


def cluster_embeddings(
    df: pd.DataFrame, config: ClusteringConfig, output_dir: str | Path
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix, _ = embedding_matrix(df)
    coords = reduce_umap(matrix, config.umap_components)
    labels = run_clustering(matrix, config)

    df = df.copy()
    df["cluster"] = labels
    for i in range(coords.shape[1]):
        df[f"umap_{i}"] = coords[:, i]

    outputs: dict[str, Path] = {}
    outputs["clusters"] = write_table(df, output_dir / "clusters.parquet")
    outputs["clusters_csv"] = write_table(df, output_dir / "clusters.csv")
    umap_df = pd.DataFrame(coords, columns=[f"umap_{i}" for i in range(coords.shape[1])])
    umap_df["cluster"] = labels
    outputs["umap"] = write_table(umap_df, output_dir / "umap_coordinates.csv")
    outputs["summary"] = write_table(
        cluster_summary(df), output_dir / "cluster_summary.csv"
    )
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cluster behavior embeddings.")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--method", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.method:
        config.clustering.method = args.method
    df = read_table(args.embeddings)
    outputs = cluster_embeddings(df, config.clustering, args.output_dir)
    print(f"Wrote clustering outputs: {[str(p) for p in outputs.values()]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
