"""
Supply Chain SNA — Betweenness Centrality
PHASE 8.2: Identifies bridge/intermediary organizations.

An organization with high betweenness lies on many shortest paths between
other node pairs. In supply chains, this indicates a bottleneck or critical
intermediary that goods or information must flow through.

Note: Normalized betweenness is used so values are comparable across
different network sizes. Computed on the directed graph to preserve
supply-chain flow direction.
"""
from __future__ import annotations

import logging

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def compute_betweenness(
    G: nx.DiGraph,
    normalized: bool = True,
    weight: str | None = None,
) -> pd.DataFrame:
    """Compute betweenness centrality for all nodes.

    Args:
        G: Directed supply-chain graph.
        normalized: If True, normalizes by 1/((n-1)(n-2)) for directed graphs.
        weight: If specified, uses edge attribute for weighted shortest paths.
            For betweenness, weight represents path cost (lower = faster).
            Pass None for unweighted.

    Returns:
        DataFrame with columns: node, betweenness
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["node", "betweenness"])

    logger.info("Computing betweenness centrality (normalized=%s)…", normalized)

    bc = nx.betweenness_centrality(G, normalized=normalized, weight=weight)

    df = (
        pd.DataFrame(list(bc.items()), columns=["node", "betweenness"])
        .sort_values("betweenness", ascending=False)
        .reset_index(drop=True)
    )
    logger.info(
        "Betweenness: max=%.6f (node=%s), mean=%.6f",
        df["betweenness"].max() if len(df) > 0 else 0,
        df.iloc[0]["node"] if len(df) > 0 else "N/A",
        df["betweenness"].mean() if len(df) > 0 else 0,
    )
    return df
