"""
Supply Chain SNA — Closeness Centrality
PHASE 8.3: Identifies organizations structurally close to the rest of the network.

High closeness means shorter average path distances to all other nodes.
In supply chains, this can indicate organizations that can reach many
others quickly (e.g., for information propagation or product distribution).

Methodology note:
  NetworkX closeness_centrality with Wasserman-Faust normalization is used.
  This handles disconnected graphs by normalizing by the fraction of nodes
  reachable, preventing artificially low values for isolated nodes.
"""
from __future__ import annotations

import logging

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def compute_closeness(G: nx.DiGraph) -> pd.DataFrame:
    """Compute closeness centrality for all nodes.

    Uses NetworkX closeness_centrality with Wasserman-Faust normalization,
    which handles disconnected graphs appropriately.

    Args:
        G: Directed supply-chain graph.

    Returns:
        DataFrame with columns: node, closeness
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["node", "closeness"])

    logger.info("Computing closeness centrality (Wasserman-Faust, directed)…")

    # NetworkX closeness_centrality handles disconnected directed graphs
    # via Wasserman-Faust normalization
    cc = nx.closeness_centrality(G)

    df = (
        pd.DataFrame(list(cc.items()), columns=["node", "closeness"])
        .sort_values("closeness", ascending=False)
        .reset_index(drop=True)
    )
    logger.info(
        "Closeness: max=%.4f (node=%s), mean=%.4f",
        df["closeness"].max() if len(df) > 0 else 0,
        df.iloc[0]["node"] if len(df) > 0 else "N/A",
        df["closeness"].mean() if len(df) > 0 else 0,
    )
    return df
