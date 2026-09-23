"""
Supply Chain SNA — K-Core Analysis
PHASE 11: Identifies the coreness of each node.

K-core decomposition finds the maximal subgraph where every node has
at least k neighbors. Nodes with high core numbers are deeply embedded
in the network. In supply chains, high-core nodes may be critical
infrastructure.

Methodology note:
  K-core is applied on the undirected projection since NetworkX's
  core_number() requires an undirected graph. This is documented.
"""
from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def compute_kcore(G: nx.DiGraph) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute k-core decomposition for the supply-chain network.

    Applied on undirected projection of the directed graph.

    Args:
        G: Directed supply-chain graph.

    Returns:
        Tuple of:
            - DataFrame with columns: node, core_number
            - Statistics dictionary
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["node", "core_number"]), {}

    U = G.to_undirected()
    logger.info("Computing k-core decomposition (on undirected projection)…")

    core_numbers = nx.core_number(U)

    df = pd.DataFrame(
        [{"node": node, "core_number": cnum} for node, cnum in core_numbers.items()]
    )
    df = df.sort_values("core_number", ascending=False).reset_index(drop=True)

    max_core = df["core_number"].max() if len(df) > 0 else 0
    core_distribution = df["core_number"].value_counts().sort_index().to_dict()

    stats: dict[str, Any] = {
        "max_core_number": int(max_core),
        "core_distribution": {int(k): int(v) for k, v in core_distribution.items()},
        "num_nodes_in_max_core": int((df["core_number"] == max_core).sum()),
        "note": "Applied on undirected projection of DiGraph",
    }

    logger.info(
        "K-core: max_core=%d, nodes_in_max_core=%d",
        stats["max_core_number"],
        stats["num_nodes_in_max_core"],
    )
    return df, stats
