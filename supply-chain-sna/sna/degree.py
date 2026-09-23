"""
Supply Chain SNA — Degree Centrality
PHASE 8.1: Computes in-degree, out-degree, and total degree for all nodes.

Interpretation:
  - High in-degree: many organizations supply this node (popular buyer/recipient)
  - High out-degree: this organization supplies many others (major supplier/distributor)
  - Total degree: overall connectivity regardless of direction
"""
from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def compute_degree(G: nx.DiGraph) -> pd.DataFrame:
    """Compute degree centrality for all nodes.

    Args:
        G: Directed supply-chain graph.

    Returns:
        DataFrame with columns:
            node, in_degree, out_degree, total_degree,
            in_degree_centrality, out_degree_centrality
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["node", "in_degree", "out_degree", "total_degree",
                                     "in_degree_centrality", "out_degree_centrality"])

    in_deg_centrality = nx.in_degree_centrality(G)
    out_deg_centrality = nx.out_degree_centrality(G)

    records = []
    for node in G.nodes():
        in_d = G.in_degree(node)
        out_d = G.out_degree(node)
        records.append(
            {
                "node": node,
                "in_degree": in_d,
                "out_degree": out_d,
                "total_degree": in_d + out_d,
                "in_degree_centrality": in_deg_centrality.get(node, 0.0),
                "out_degree_centrality": out_deg_centrality.get(node, 0.0),
            }
        )

    df = pd.DataFrame(records).sort_values("total_degree", ascending=False).reset_index(drop=True)
    logger.info("Degree centrality computed for %d nodes", len(df))
    return df
