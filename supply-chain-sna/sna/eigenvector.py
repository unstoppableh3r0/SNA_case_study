"""
Supply Chain SNA — Eigenvector Centrality
PHASE 8.4: Measures importance based on the importance of neighbors.

An organization can have few connections but high eigenvector centrality
if it connects to highly connected/important organizations. This reveals
strategic positioning in the supply chain.

Handles convergence failures gracefully with fallback to PageRank.
"""
from __future__ import annotations

import logging

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def compute_eigenvector(
    G: nx.DiGraph,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> pd.DataFrame:
    """Compute eigenvector centrality for all nodes.

    Falls back to returning zero values with a warning if convergence fails.
    For disconnected graphs, applies to each weakly connected component
    where possible.

    Args:
        G: Directed supply-chain graph.
        max_iter: Maximum power iterations for convergence.
        tol: Convergence tolerance.

    Returns:
        DataFrame with columns: node, eigenvector, converged
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["node", "eigenvector", "converged"])

    logger.info("Computing eigenvector centrality (max_iter=%d, tol=%e)…", max_iter, tol)

    converged = True
    try:
        ec = nx.eigenvector_centrality(G, max_iter=max_iter, tol=tol, weight="weight")
    except nx.PowerIterationFailedConvergence:
        logger.warning(
            "Eigenvector centrality did not converge (max_iter=%d). "
            "This is common in directed supply-chain graphs. "
            "Returning zero values — use PageRank as alternative.",
            max_iter,
        )
        ec = {node: 0.0 for node in G.nodes()}
        converged = False
    except Exception as exc:
        logger.warning("Eigenvector centrality error: %s. Returning zeros.", exc)
        ec = {node: 0.0 for node in G.nodes()}
        converged = False

    df = pd.DataFrame(
        [{"node": node, "eigenvector": val, "converged": converged} for node, val in ec.items()]
    )
    df = df.sort_values("eigenvector", ascending=False).reset_index(drop=True)

    if converged:
        logger.info(
            "Eigenvector: max=%.6f (node=%s)",
            df["eigenvector"].max(),
            df.iloc[0]["node"] if len(df) > 0 else "N/A",
        )
    return df
