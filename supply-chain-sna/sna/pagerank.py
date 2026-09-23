"""
Supply Chain SNA — PageRank
PHASE 8.5: Measures structural importance in directed networks.

PageRank models a random walk on the directed graph: following supply-chain
edges with probability alpha, and teleporting with probability (1-alpha).
Nodes receiving many inbound links from important nodes rank higher.

This is particularly suitable for supply-chain directed graphs where
eigenvector centrality may not converge.
"""
from __future__ import annotations

import logging

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def compute_pagerank(
    G: nx.DiGraph,
    alpha: float = 0.85,
    max_iter: int = 1000,
    tol: float = 1e-6,
    weight: str | None = "weight",
) -> pd.DataFrame:
    """Compute PageRank for all nodes.

    Args:
        G: Directed supply-chain graph.
        alpha: Damping factor (probability of following an edge). Default 0.85.
        max_iter: Maximum iterations.
        tol: Convergence tolerance.
        weight: Edge attribute to use as weight. None = unweighted.

    Returns:
        DataFrame with columns: node, pagerank
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["node", "pagerank"])

    logger.info("Computing PageRank (alpha=%.2f, weighted=%s)…", alpha, weight is not None)

    pr = nx.pagerank(G, alpha=alpha, max_iter=max_iter, tol=tol, weight=weight)

    df = (
        pd.DataFrame(list(pr.items()), columns=["node", "pagerank"])
        .sort_values("pagerank", ascending=False)
        .reset_index(drop=True)
    )
    logger.info(
        "PageRank: max=%.6f (node=%s), sum=%.4f",
        df["pagerank"].max() if len(df) > 0 else 0,
        df.iloc[0]["node"] if len(df) > 0 else "N/A",
        df["pagerank"].sum() if len(df) > 0 else 0,
    )
    return df
