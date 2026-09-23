"""
Supply Chain SNA — Resilience Engine
PHASE 14: Simulates node-removal experiments.

Implements four attack strategies:
  1. Random removal (repeated with multiple seeds for statistical validity)
  2. Highest-degree removal (targeted)
  3. Highest-betweenness removal (targeted)
  4. Highest-PageRank removal (targeted)

Measures per removal fraction:
  - Largest connected component fraction
  - Network efficiency (global efficiency)
  - Number of connected components
  - Network fragmentation index
"""
from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_resilience_experiments(
    G: nx.DiGraph,
    removal_fractions: list[float] | None = None,
    random_seeds: list[int] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run all resilience experiments.

    Args:
        G: Directed supply-chain graph (working copy will be made).
        removal_fractions: List of fractions of nodes to remove.
        random_seeds: Seeds for random removal experiments.

    Returns:
        Dictionary with keys:
            'random': DataFrame from random removal experiments (mean ± std)
            'degree': DataFrame from degree-targeted removal
            'betweenness': DataFrame from betweenness-targeted removal
            'pagerank': DataFrame from PageRank-targeted removal
    """
    if removal_fractions is None:
        removal_fractions = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    if random_seeds is None:
        random_seeds = [42, 123, 456, 789, 1337]

    results: dict[str, pd.DataFrame] = {}

    # ── 1. Random removal ─────────────────────────────────────────────────────
    logger.info("Running random removal experiments (%d seeds)…", len(random_seeds))
    random_results = _run_random_removal(G, removal_fractions, random_seeds)
    results["random"] = random_results

    # ── 2. Degree-targeted removal ────────────────────────────────────────────
    logger.info("Running degree-targeted removal…")
    degree_order = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)
    results["degree"] = _run_targeted_removal(G, removal_fractions, degree_order, "degree")

    # ── 3. Betweenness-targeted removal ───────────────────────────────────────
    logger.info("Running betweenness-targeted removal…")
    from sna.betweenness import compute_betweenness
    bc_df = compute_betweenness(G, normalized=True)
    bc_order = bc_df["node"].tolist()
    results["betweenness"] = _run_targeted_removal(G, removal_fractions, bc_order, "betweenness")

    # ── 4. PageRank-targeted removal ──────────────────────────────────────────
    logger.info("Running PageRank-targeted removal…")
    from sna.pagerank import compute_pagerank
    pr_df = compute_pagerank(G)
    pr_order = pr_df["node"].tolist()
    results["pagerank"] = _run_targeted_removal(G, removal_fractions, pr_order, "pagerank")

    logger.info("Resilience experiments complete")
    return results


def _run_random_removal(
    G: nx.DiGraph,
    fractions: list[float],
    seeds: list[int],
) -> pd.DataFrame:
    """Run random node removal with multiple seeds and aggregate results.

    Args:
        G: Original graph.
        fractions: Removal fractions.
        seeds: Random seeds for multiple runs.

    Returns:
        DataFrame with mean and std over seeds for each fraction.
    """
    rng = np.random.default_rng(0)
    all_runs: list[pd.DataFrame] = []

    for seed in seeds:
        seed_rng = np.random.default_rng(seed)
        nodes = list(G.nodes())
        seed_rng.shuffle(nodes)
        run_records = [_baseline_metrics(G, fraction=0.0, strategy="random", seed=seed)]

        for frac in sorted(fractions):
            n_remove = max(1, int(len(nodes) * frac))
            remove_set = set(nodes[:n_remove])
            H = G.copy()
            H.remove_nodes_from(remove_set)
            metrics = _compute_removal_metrics(H, frac, "random", seed)
            run_records.append(metrics)

        all_runs.append(pd.DataFrame(run_records))

    # Aggregate
    combined = pd.concat(all_runs, ignore_index=True)
    agg = (
        combined.groupby("fraction_removed")
        .agg(
            lcc_mean=("largest_component_fraction", "mean"),
            lcc_std=("largest_component_fraction", "std"),
            efficiency_mean=("network_efficiency", "mean"),
            efficiency_std=("network_efficiency", "std"),
            components_mean=("component_count", "mean"),
        )
        .reset_index()
    )
    agg["strategy"] = "random"
    agg["lcc_std"] = agg["lcc_std"].fillna(0.0)
    agg["efficiency_std"] = agg["efficiency_std"].fillna(0.0)
    return agg


def _run_targeted_removal(
    G: nx.DiGraph,
    fractions: list[float],
    node_order: list[str],
    strategy: str,
) -> pd.DataFrame:
    """Run targeted node removal in order of importance.

    Args:
        G: Original graph.
        fractions: Removal fractions.
        node_order: Nodes sorted by removal priority (most important first).
        strategy: Strategy name for labeling.

    Returns:
        DataFrame with metrics per fraction.
    """
    records = [_baseline_metrics(G, fraction=0.0, strategy=strategy)]

    for frac in sorted(fractions):
        n_remove = max(1, int(len(node_order) * frac))
        remove_set = set(node_order[:n_remove])
        H = G.copy()
        H.remove_nodes_from(remove_set)
        metrics = _compute_removal_metrics(H, frac, strategy)
        records.append(metrics)

    df = pd.DataFrame(records)
    df["lcc_mean"] = df["largest_component_fraction"]
    df["lcc_std"] = 0.0
    df["efficiency_mean"] = df["network_efficiency"]
    df["efficiency_std"] = 0.0
    df["components_mean"] = df["component_count"]
    return df


def _baseline_metrics(
    G: nx.DiGraph, fraction: float, strategy: str, seed: int | None = None
) -> dict[str, Any]:
    """Compute baseline metrics at 0% removal.

    Args:
        G: Graph.
        fraction: Should be 0.0 for baseline.
        strategy: Strategy label.
        seed: Optional seed label.

    Returns:
        Metrics dictionary.
    """
    return _compute_removal_metrics(G, fraction, strategy, seed)


def _compute_removal_metrics(
    H: nx.DiGraph,
    fraction: float,
    strategy: str,
    seed: int | None = None,
) -> dict[str, Any]:
    """Compute network metrics after node removal.

    Args:
        H: Graph after removals.
        fraction: Fraction of nodes removed.
        strategy: Strategy label.
        seed: Optional random seed label.

    Returns:
        Dictionary of metrics.
    """
    n = H.number_of_nodes()
    if n == 0:
        return {
            "fraction_removed": fraction,
            "strategy": strategy,
            "seed": seed,
            "num_nodes": 0,
            "largest_component_fraction": 0.0,
            "network_efficiency": 0.0,
            "component_count": 0,
            "reachability": 0.0,
        }

    wcc = list(nx.weakly_connected_components(H))
    largest_wcc = max(len(c) for c in wcc) if wcc else 0
    lcc_fraction = largest_wcc / n

    U = H.to_undirected()
    try:
        if n <= 2000:
            eff = nx.global_efficiency(U)
        else:
            # Sample for large graphs
            sample = list(U.nodes())[:500]
            sub = U.subgraph(sample).copy()
            eff = nx.global_efficiency(sub) if sub.number_of_nodes() > 1 else 0.0
    except Exception:
        eff = 0.0

    # Reachability: fraction of node pairs that can reach each other
    total_pairs = n * (n - 1)
    reachable_pairs = sum(len(c) * (len(c) - 1) for c in wcc)
    reachability = reachable_pairs / max(1, total_pairs)

    return {
        "fraction_removed": fraction,
        "strategy": strategy,
        "seed": seed,
        "num_nodes": n,
        "largest_component_fraction": lcc_fraction,
        "network_efficiency": eff,
        "component_count": len(wcc),
        "reachability": reachability,
    }
