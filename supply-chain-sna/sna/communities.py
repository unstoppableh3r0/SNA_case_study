"""
Supply Chain SNA — Community Detection
PHASE 10: Detects and evaluates supply-chain communities.

Primary algorithm: Louvain (applied on undirected projection).
Secondary: Greedy modularity (for comparison).

Ground-truth evaluation uses Adjusted Rand Index (ARI) and
Normalized Mutual Information (NMI) where planted labels exist.
"""
from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def compute_communities(
    G: nx.DiGraph,
    algorithm: str = "louvain",
    resolution: float = 1.0,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Detect communities in the supply-chain network.

    Note: Louvain and greedy modularity require an undirected graph.
    The directed graph is projected to undirected before community detection.
    This is documented as a methodological choice.

    Args:
        G: Directed supply-chain graph.
        algorithm: "louvain" or "greedy_modularity".
        resolution: Resolution parameter for Louvain (higher = more communities).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of:
            - DataFrame with columns: node, community_id
            - Dictionary with community statistics
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["node", "community_id"]), {}

    # Convert to undirected for community detection
    U = G.to_undirected()

    logger.info("Detecting communities (algorithm=%s, undirected projection)…", algorithm)

    if algorithm == "louvain":
        communities = _louvain_communities(U, resolution=resolution, seed=random_state)
    elif algorithm == "greedy_modularity":
        communities = _greedy_modularity_communities(U)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use 'louvain' or 'greedy_modularity'")

    # Build node → community_id mapping
    node_community: dict[str, int] = {}
    for comm_id, comm_nodes in enumerate(communities):
        for node in comm_nodes:
            node_community[node] = comm_id

    df = pd.DataFrame(
        [{"node": node, "community_id": cid} for node, cid in node_community.items()]
    )

    # Community statistics
    community_sizes = df["community_id"].value_counts().to_dict()

    # Modularity
    try:
        modularity = nx.community.modularity(U, communities)
    except Exception as exc:
        logger.warning("Could not compute modularity: %s", exc)
        modularity = None

    # Inter-community edges
    inter_edges = sum(
        1
        for u, v in U.edges()
        if node_community.get(u) != node_community.get(v)
    )

    stats: dict[str, Any] = {
        "algorithm": algorithm,
        "num_communities": len(communities),
        "community_sizes": community_sizes,
        "modularity": modularity,
        "inter_community_edges": inter_edges,
        "total_edges": U.number_of_edges(),
        "inter_community_fraction": inter_edges / max(1, U.number_of_edges()),
        "note": "Applied on undirected projection of directed supply-chain graph",
    }

    logger.info(
        "Communities: %d detected, modularity=%.4f",
        stats["num_communities"],
        modularity or 0.0,
    )
    return df, stats


def _louvain_communities(
    U: nx.Graph, resolution: float = 1.0, seed: int = 42
) -> list[set]:
    """Run Louvain community detection.

    Args:
        U: Undirected graph.
        resolution: Resolution parameter.
        seed: Random seed.

    Returns:
        List of sets of node IDs, one per community.
    """
    try:
        import community as community_louvain  # python-louvain
        partition = community_louvain.best_partition(U, resolution=resolution, random_state=seed)
        communities: dict[int, set] = {}
        for node, cid in partition.items():
            communities.setdefault(cid, set()).add(node)
        return list(communities.values())
    except ImportError:
        logger.warning("python-louvain not available; falling back to NetworkX greedy modularity")
        return _greedy_modularity_communities(U)
    except Exception as exc:
        logger.warning("Louvain failed (%s); falling back to greedy modularity", exc)
        return _greedy_modularity_communities(U)


def _greedy_modularity_communities(U: nx.Graph) -> list[set]:
    """Run greedy modularity community detection.

    Args:
        U: Undirected graph.

    Returns:
        List of sets of node IDs.
    """
    result = nx.community.greedy_modularity_communities(U)
    return [set(c) for c in result]


def evaluate_community_recovery(
    detected_df: pd.DataFrame,
    ground_truth_communities: dict[str, list[str]],
) -> dict[str, Any]:
    """Evaluate how well detected communities match planted communities.

    Uses Adjusted Rand Index (ARI) and Normalized Mutual Information (NMI).

    Args:
        detected_df: DataFrame with columns: node, community_id.
        ground_truth_communities: Dict mapping region/label → list of node IDs.

    Returns:
        Dictionary with ARI, NMI, and interpretation.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    # Build ground-truth node → label mapping
    gt_labels: dict[str, str] = {}
    for label, nodes in ground_truth_communities.items():
        for node in nodes:
            gt_labels[node] = label

    # Align on common nodes
    common_nodes = [
        node
        for node in detected_df["node"].tolist()
        if node in gt_labels
    ]

    if len(common_nodes) < 2:
        return {
            "ari": None,
            "nmi": None,
            "note": "Insufficient common nodes for evaluation",
        }

    detected_map = dict(zip(detected_df["node"], detected_df["community_id"]))
    gt_node_list = [gt_labels[n] for n in common_nodes]
    det_node_list = [str(detected_map[n]) for n in common_nodes]

    ari = adjusted_rand_score(gt_node_list, det_node_list)
    nmi = normalized_mutual_info_score(gt_node_list, det_node_list)

    return {
        "ari": float(ari),
        "nmi": float(nmi),
        "common_nodes_evaluated": len(common_nodes),
        "interpretation": _interpret_ari(ari),
    }


def _interpret_ari(ari: float) -> str:
    """Return a textual interpretation of the ARI score.

    Args:
        ari: Adjusted Rand Index value.

    Returns:
        Interpretation string.
    """
    if ari >= 0.8:
        return "Excellent community recovery"
    elif ari >= 0.6:
        return "Good community recovery"
    elif ari >= 0.4:
        return "Moderate community recovery"
    elif ari >= 0.2:
        return "Weak community recovery"
    else:
        return "Poor community recovery (near random)"
