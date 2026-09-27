"""Embedding + agglomerative clustering + purity (label consistency only)."""

from clustering.agglomerative import ClusterResult, Clusterer, agglomerative_fit_predict
from clustering.embed import case_text, embed_cases, hash_embed
from clustering.purity import cluster_purity, cluster_size_distribution, compression_ratio

__all__ = [
    "Clusterer",
    "ClusterResult",
    "agglomerative_fit_predict",
    "hash_embed",
    "embed_cases",
    "case_text",
    "cluster_purity",
    "cluster_size_distribution",
    "compression_ratio",
]
