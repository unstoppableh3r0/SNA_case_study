"""
Supply Chain SNA — Network Statistics
PHASE 7: Computes network-level metrics for a directed graph.

All methodological choices (e.g., directed vs. undirected projection for
certain metrics) are explicitly documented.
"""
from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


def compute_network_statistics(G: nx.DiGraph) -> dict[str, Any]:
    """Compute comprehensive network-level statistics.

    For directed graphs, some metrics require an undirected projection.
    All such conversions are documented in the returned dict.

    Args:
        G: Directed supply-chain graph.

    Returns:
        Dictionary of network statistics.
    """
    stats: dict[str, Any] = {}

    stats["num_nodes"] = G.number_of_nodes()
    stats["num_edges"] = G.number_of_edges()
    stats["is_directed"] = True

    # ── Density ───────────────────────────────────────────────────────────────
    # Density = edges / (nodes * (nodes-1)) for directed graph
    n = G.number_of_nodes()
    e = G.number_of_edges()
    stats["density"] = nx.density(G) if n > 1 else 0.0

    # ── Degree statistics ─────────────────────────────────────────────────────
    in_degrees = [d for _, d in G.in_degree()]
    out_degrees = [d for _, d in G.out_degree()]
    total_degrees = [i + o for i, o in zip(in_degrees, out_degrees)]

    stats["avg_in_degree"] = float(np.mean(in_degrees)) if in_degrees else 0.0
    stats["avg_out_degree"] = float(np.mean(out_degrees)) if out_degrees else 0.0
    stats["avg_total_degree"] = float(np.mean(total_degrees)) if total_degrees else 0.0
    stats["max_in_degree"] = int(max(in_degrees)) if in_degrees else 0
    stats["max_out_degree"] = int(max(out_degrees)) if out_degrees else 0
    stats["degree_distribution"] = {
        "in": _degree_histogram(in_degrees),
        "out": _degree_histogram(out_degrees),
    }

    # ── Connected components ───────────────────────────────────────────────────
    # For directed graph: weakly connected components
    wcc = list(nx.weakly_connected_components(G))
    stats["weakly_connected_components"] = len(wcc)
    stats["largest_wcc_size"] = max(len(c) for c in wcc) if wcc else 0
    stats["largest_wcc_fraction"] = stats["largest_wcc_size"] / n if n > 0 else 0.0

    # Strongly connected components
    scc = list(nx.strongly_connected_components(G))
    stats["strongly_connected_components"] = len(scc)
    stats["largest_scc_size"] = max(len(c) for c in scc) if scc else 0

    # ── Clustering (undirected projection) ────────────────────────────────────
    # Note: We use the undirected projection to compute clustering because
    # supply-chain graphs are predominantly DAG-like and directed clustering
    # would return near-zero values.
    U = G.to_undirected()
    stats["avg_clustering_undirected"] = nx.average_clustering(U) if U.number_of_nodes() > 0 else 0.0
    stats["clustering_note"] = "Computed on undirected projection of DiGraph"

    # ── Path length (largest WCC only) ────────────────────────────────────────
    # Average path length is only meaningful on connected graphs.
    # We compute it on the largest WCC subgraph only.
    if stats["largest_wcc_size"] > 1:
        largest_wcc_nodes = max(wcc, key=len)
        subG = G.subgraph(largest_wcc_nodes).copy()
        if nx.is_weakly_connected(subG):
            try:
                # Use undirected subgraph for path length (reachability)
                subU = subG.to_undirected()
                if subU.number_of_nodes() <= 5000:
                    stats["avg_path_length_largest_wcc"] = nx.average_shortest_path_length(subU)
                else:
                    # Approximate with sample
                    sample_nodes = list(subU.nodes())[:500]
                    lengths = []
                    for node in sample_nodes:
                        spl = nx.single_source_shortest_path_length(subU, node)
                        lengths.extend(spl.values())
                    stats["avg_path_length_largest_wcc"] = float(np.mean(lengths)) if lengths else None
                    stats["path_length_note"] = "Approximated from 500-node sample (large graph)"
            except Exception as exc:
                logger.warning("Could not compute avg path length: %s", exc)
                stats["avg_path_length_largest_wcc"] = None
    else:
        stats["avg_path_length_largest_wcc"] = None

    # ── Network efficiency ────────────────────────────────────────────────────
    # Global efficiency: average inverse shortest path length
    # Defined on undirected graph for tractability
    try:
        if n <= 5000:
            stats["global_efficiency"] = nx.global_efficiency(U)
        else:
            # Sample-based approximation
            sample = list(U.nodes())[:500]
            subU_sample = U.subgraph(sample).copy()
            stats["global_efficiency"] = nx.global_efficiency(subU_sample)
            stats["efficiency_note"] = "Approximated from 500-node sample (large graph)"
    except Exception as exc:
        logger.warning("Could not compute global efficiency: %s", exc)
        stats["global_efficiency"] = None

    # ── Reciprocity ───────────────────────────────────────────────────────────
    # Fraction of edges (u,v) where (v,u) also exists
    stats["reciprocity"] = nx.reciprocity(G) if G.number_of_edges() > 0 else 0.0

    logger.info(
        "Network stats: %d nodes, %d edges, density=%.5f, WCC=%d",
        stats["num_nodes"],
        stats["num_edges"],
        stats["density"],
        stats["weakly_connected_components"],
    )
    return stats


def _degree_histogram(degrees: list[int]) -> dict[str, Any]:
    """Compute a degree histogram summary.

    Args:
        degrees: List of degree values.

    Returns:
        Dictionary with histogram statistics.
    """
    if not degrees:
        return {}
    arr = np.array(degrees)
    return {
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
    }
