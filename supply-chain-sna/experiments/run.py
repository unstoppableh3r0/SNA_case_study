"""
Supply Chain SNA — Experiment Runner
PHASE 16: Unified CLI for running all SNA experiments.

CLI usage:
    python -m experiments.run --experiment all
    python -m experiments.run --experiment centrality
    python -m experiments.run --experiment communities
    python -m experiments.run --experiment resilience
    python -m experiments.run --experiment temporal
    python -m experiments.run --experiment ground_truth
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Run SNA experiments.

    Args:
        argv: Optional CLI arguments for testing.
    """
    parser = argparse.ArgumentParser(description="Supply Chain SNA — Experiment Runner")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument(
        "--experiment",
        default="all",
        choices=["all", "centrality", "communities", "resilience", "temporal", "ground_truth"],
    )
    args = parser.parse_args(argv)

    from config import load_config
    config = load_config(args.config)

    data_dir = Path(config["output"]["data_dir"])
    graph_dir = Path(config["output"]["graph_dir"])
    results_dir = Path(config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data…")
    organizations = pd.read_csv(data_dir / "organizations.csv")
    transactions = pd.read_csv(data_dir / "transactions.csv")

    with (data_dir / "ground_truth.json").open() as fh:
        ground_truth = json.load(fh)

    # Load graphs
    graph_path = graph_dir / "supply_chain_graph_frequency.pkl"
    temporal_path = graph_dir / "temporal_graphs_frequency.pkl"

    if not graph_path.exists():
        logger.error("Graph not found. Run: python -m graph.build")
        sys.exit(1)

    with graph_path.open("rb") as fh:
        G = pickle.load(fh)

    if temporal_path.exists():
        with temporal_path.open("rb") as fh:
            temporal_graphs = pickle.load(fh)
    else:
        temporal_graphs = {}

    logger.info("Graph loaded: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    run_all = args.experiment == "all"

    summary: dict[str, Any] = {
        "config": args.config,
        "seed": config["seed"],
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
        "experiments_run": [],
    }

    centrality_df: pd.DataFrame | None = None
    community_df: pd.DataFrame | None = None

    # ── Centrality ────────────────────────────────────────────────────────────
    if run_all or args.experiment == "centrality":
        logger.info("=== Running Centrality Experiment ===")
        from sna.centrality import compute_all_centrality
        centrality_df, cent_stats = compute_all_centrality(G, config)
        centrality_df.to_csv(results_dir / "centrality.csv", index=False)
        with (results_dir / "centrality_stats.json").open("w") as fh:
            json.dump(cent_stats, fh, indent=2, default=str)
        logger.info("Centrality results → %s", results_dir / "centrality.csv")
        summary["experiments_run"].append("centrality")
        summary["centrality"] = {
            "top_degree_node": centrality_df.iloc[0]["node"] if len(centrality_df) > 0 else None,
            "top_betweenness_node": centrality_df.nsmallest(1, "betweenness_rank")["node"].tolist()[0] if "betweenness_rank" in centrality_df.columns and len(centrality_df) > 0 else None,
        }

    # ── Communities ───────────────────────────────────────────────────────────
    if run_all or args.experiment == "communities":
        logger.info("=== Running Community Detection Experiment ===")
        from sna.communities import compute_communities
        comm_cfg = config.get("community", {})
        community_df, comm_stats = compute_communities(
            G,
            algorithm=comm_cfg.get("algorithm", "louvain"),
            resolution=comm_cfg.get("louvain_resolution", 1.0),
            random_state=comm_cfg.get("random_state", 42),
        )
        community_df.to_csv(results_dir / "communities.csv", index=False)
        with (results_dir / "community_stats.json").open("w") as fh:
            json.dump(comm_stats, fh, indent=2, default=str)
        logger.info("Communities: %d detected", comm_stats.get("num_communities", 0))
        summary["experiments_run"].append("communities")
        summary["communities"] = {
            "num_communities": comm_stats.get("num_communities"),
            "modularity": comm_stats.get("modularity"),
        }

    # ── K-Core ────────────────────────────────────────────────────────────────
    if run_all or args.experiment == "centrality":
        logger.info("=== Running K-Core Analysis ===")
        from sna.kcore import compute_kcore
        kcore_df, kcore_stats = compute_kcore(G)
        kcore_df.to_csv(results_dir / "kcore.csv", index=False)
        with (results_dir / "kcore_stats.json").open("w") as fh:
            json.dump(kcore_stats, fh, indent=2, default=str)

    # ── Dependencies ──────────────────────────────────────────────────────────
    if run_all or args.experiment == "centrality":
        logger.info("=== Running Dependency Analysis ===")
        from sna.dependencies import analyze_dependencies
        dep_results = analyze_dependencies(G, centrality_df)
        # Save concentration table
        if isinstance(dep_results.get("upstream_concentration"), pd.DataFrame):
            dep_results["upstream_concentration"].to_csv(results_dir / "dependencies.csv", index=False)
        dep_summary = {k: v for k, v in dep_results.items() if not isinstance(v, pd.DataFrame)}
        with (results_dir / "dependency_summary.json").open("w") as fh:
            json.dump(dep_summary, fh, indent=2, default=str)
        summary["dependencies"] = {
            "single_source_count": dep_results.get("single_source_count"),
            "high_concentration_count": dep_results.get("high_concentration_count"),
        }

    # ── Network statistics ────────────────────────────────────────────────────
    if run_all:
        logger.info("=== Computing Network Statistics ===")
        from sna.network_metrics import compute_network_statistics
        net_stats = compute_network_statistics(G)
        with (results_dir / "network_statistics.json").open("w") as fh:
            json.dump(net_stats, fh, indent=2, default=str)
        summary["network_statistics"] = net_stats

    # ── Temporal ──────────────────────────────────────────────────────────────
    if (run_all or args.experiment == "temporal") and temporal_graphs:
        logger.info("=== Running Temporal Analysis ===")
        from sna.temporal import analyze_temporal_snapshots
        snapshot_df, node_time_df, temp_summary = analyze_temporal_snapshots(temporal_graphs)
        snapshot_df.to_csv(results_dir / "temporal.csv", index=False)
        node_time_df.to_csv(results_dir / "temporal_centrality.csv", index=False)
        with (results_dir / "temporal_summary.json").open("w") as fh:
            json.dump(temp_summary, fh, indent=2, default=str)
        logger.info("Temporal analysis: %d snapshots", len(snapshot_df))
        summary["experiments_run"].append("temporal")
        summary["temporal"] = temp_summary

    # ── Resilience ────────────────────────────────────────────────────────────
    if run_all or args.experiment == "resilience":
        logger.info("=== Running Resilience Experiments ===")
        from sna.resilience import run_resilience_experiments
        resil_cfg = config.get("resilience", {})
        resilience_results = run_resilience_experiments(
            G,
            removal_fractions=resil_cfg.get("removal_fractions"),
            random_seeds=resil_cfg.get("random_seeds"),
        )
        for strategy, df in resilience_results.items():
            df.to_csv(results_dir / f"resilience_{strategy}.csv", index=False)
        logger.info("Resilience experiments complete")
        summary["experiments_run"].append("resilience")

    # ── Ground Truth ─────────────────────────────────────────────────────────
    if (run_all or args.experiment == "ground_truth") and centrality_df is not None:
        logger.info("=== Running Ground-Truth Experiments ===")
        from experiments.ground_truth import run_ground_truth_experiments
        if community_df is None:
            from sna.communities import compute_communities
            comm_cfg = config.get("community", {})
            community_df, _ = compute_communities(G, algorithm=comm_cfg.get("algorithm", "louvain"))

        gt_results = run_ground_truth_experiments(
            G, centrality_df, community_df, ground_truth, config
        )
        with (results_dir / "ground_truth_results.json").open("w") as fh:
            json.dump(gt_results, fh, indent=2, default=str)
        logger.info("Ground-truth results → %s", results_dir / "ground_truth_results.json")
        summary["experiments_run"].append("ground_truth")

    # ── Save summary ──────────────────────────────────────────────────────────
    with (results_dir / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info("=== Experiment Run Complete ===")
    logger.info("Summary → %s", results_dir / "summary.json")


if __name__ == "__main__":
    main()
