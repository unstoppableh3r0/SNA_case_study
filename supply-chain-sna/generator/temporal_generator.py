"""
Supply Chain SNA — Temporal Generator
PHASE 4: Generates multi-period snapshots with realistic temporal evolution.

Simulates organizational entry/exit, relationship creation/weakening,
seasonal demand variation, and disruption events.
"""
from __future__ import annotations

import logging
import random
from datetime import datetime
from typing import Any

import pandas as pd

from generator.organization_generator import generate_organizations
from generator.network_generator import generate_network
from generator.transaction_generator import generate_transactions, _add_months

logger = logging.getLogger(__name__)


def generate_temporal_dataset(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Generate the full temporal supply-chain dataset.

    Generates organizations, the base network, and transactions for each
    time period. Applies temporal events (org entry/exit, edge changes,
    disruptions, seasonality).

    Args:
        config: Configuration dictionary.

    Returns:
        Tuple of:
            - organizations: DataFrame of all organizations (with status changes)
            - all_transactions: DataFrame of all transactions across all months
            - event_log: DataFrame of all simulated events
            - ground_truth: Ground-truth dictionary with planted structures
    """
    seed: int = config["seed"]
    rng = random.Random(seed + 999)

    n_months: int = config["network"]["months"]
    start_date = datetime.strptime(config["network"]["start_date"], "%Y-%m-%d")

    org_entry_rate: float = config["temporal"]["organization_entry_rate"]
    org_exit_rate: float = config["temporal"]["organization_exit_rate"]
    rel_new_rate: float = config["temporal"]["relationship_new_rate"]
    rel_drop_rate: float = config["temporal"]["relationship_drop_rate"]
    disruption_months: list[int] = config["temporal"]["disruption_months"]
    seasonal_peak_months: list[int] = config["temporal"]["seasonal_peak_months"]

    # Generate base organizations and network
    logger.info("Generating base organizations…")
    organizations, ground_truth = generate_organizations(config)

    logger.info("Generating base network structure…")
    base_edges, ground_truth = generate_network(organizations, ground_truth, config)

    active_orgs = set(organizations["organization_id"].tolist())
    current_edges = set(zip(base_edges["source_node"], base_edges["target_node"]))
    edge_rel_types = dict(
        zip(
            zip(base_edges["source_node"], base_edges["target_node"]),
            base_edges["relationship_type"],
        )
    )

    all_transactions: list[pd.DataFrame] = []
    event_records: list[dict[str, Any]] = []
    temporal_snapshots: list[dict[str, Any]] = []

    org_type_map = dict(zip(organizations["organization_id"], organizations["organization_type"]))

    for month_idx in range(n_months):
        month_date = _add_months(start_date, month_idx)
        month_str = month_date.strftime("%Y-%m")
        month_of_year = month_date.month

        logger.info("Processing month %s (index %d/%d)…", month_str, month_idx + 1, n_months)

        # ── Determine event multipliers ───────────────────────────────────────
        event_mults: dict[str, float] = {}

        is_disruption = (month_idx + 1) in disruption_months
        is_seasonal_peak = month_of_year in seasonal_peak_months

        if is_disruption:
            # Disrupt 10% of edges (reduce volume)
            disrupted_edges = rng.sample(list(current_edges), max(1, len(current_edges) // 10))
            for edge in disrupted_edges:
                event_mults[f"{edge[0]}→{edge[1]}"] = 0.2
            event_records.append(
                {
                    "event_id": f"EVT-{len(event_records)+1:05d}",
                    "timestamp": month_date.isoformat(),
                    "event_type": "disruption",
                    "organization_id": None,
                    "target_id": None,
                    "severity": "high",
                    "description": f"Supply disruption in month {month_str}: {len(disrupted_edges)} edges impacted",
                    "month": month_str,
                }
            )

        if is_seasonal_peak:
            # Boost all volumes
            for edge in current_edges:
                event_mults[f"{edge[0]}→{edge[1]}"] = event_mults.get(f"{edge[0]}→{edge[1]}", 1.0) * 1.5

        # ── Org exits ────────────────────────────────────────────────────────
        n_exits = max(0, round(len(active_orgs) * org_exit_rate))
        if n_exits > 0 and month_idx > 0:  # no exits in first month
            exiting = rng.sample(list(active_orgs), min(n_exits, len(active_orgs)))
            for org_id in exiting:
                active_orgs.discard(org_id)
                # Remove edges involving this org
                current_edges = {(s, t) for s, t in current_edges if s != org_id and t != org_id}
                event_records.append(
                    {
                        "event_id": f"EVT-{len(event_records)+1:05d}",
                        "timestamp": month_date.isoformat(),
                        "event_type": "organization_exit",
                        "organization_id": org_id,
                        "target_id": None,
                        "severity": "medium",
                        "description": f"{org_id} became inactive in {month_str}",
                        "month": month_str,
                    }
                )

        # ── Edge drops ───────────────────────────────────────────────────────
        n_edge_drops = max(0, round(len(current_edges) * rel_drop_rate))
        if n_edge_drops > 0 and month_idx > 0:
            dropping = rng.sample(list(current_edges), min(n_edge_drops, len(current_edges)))
            for edge in dropping:
                current_edges.discard(edge)

        # ── Build active edges for this month ────────────────────────────────
        active_edges_list = [
            {
                "source_node": s,
                "target_node": t,
                "relationship_type": edge_rel_types.get((s, t), "supply_chain_link"),
            }
            for s, t in current_edges
            if s in active_orgs and t in active_orgs
        ]
        active_edges_df = pd.DataFrame(active_edges_list) if active_edges_list else pd.DataFrame(
            columns=["source_node", "target_node", "relationship_type"]
        )

        # ── Generate transactions for this month ─────────────────────────────
        if not active_edges_df.empty:
            month_txns = generate_transactions(
                active_edges_df,
                organizations[organizations["organization_id"].isin(active_orgs)],
                config,
                month_offset=month_idx,
                event_multipliers=event_mults,
            )
            all_transactions.append(month_txns)

        # ── Record snapshot metadata ──────────────────────────────────────────
        temporal_snapshots.append(
            {
                "month": month_str,
                "month_index": month_idx,
                "active_orgs": len(active_orgs),
                "active_edges": len(active_edges_list),
                "is_disruption": is_disruption,
                "is_seasonal_peak": is_seasonal_peak,
            }
        )

        # ── New edge relationships ────────────────────────────────────────────
        n_new_edges = max(0, round(len(current_edges) * rel_new_rate))
        active_org_list = list(active_orgs)
        for _ in range(n_new_edges):
            if len(active_org_list) < 2:
                break
            src = rng.choice(active_org_list)
            src_type = org_type_map.get(src, "")
            from generator.network_generator import ALLOWED_EDGES
            allowed = ALLOWED_EDGES.get(src_type, [])
            if not allowed:
                continue
            tgt_type = rng.choice(allowed)
            candidates = [
                o for o in active_org_list
                if org_type_map.get(o, "") == tgt_type and o != src
            ]
            if candidates:
                tgt = rng.choice(candidates)
                new_edge = (src, tgt)
                if new_edge not in current_edges:
                    current_edges.add(new_edge)
                    edge_rel_types[new_edge] = "supply_chain_link"

    combined_transactions = pd.concat(all_transactions, ignore_index=True) if all_transactions else pd.DataFrame()
    event_log = pd.DataFrame(event_records) if event_records else pd.DataFrame()

    ground_truth["temporal_snapshots"] = temporal_snapshots

    logger.info(
        "Temporal generation complete: %d total transactions across %d months",
        len(combined_transactions),
        n_months,
    )

    return organizations, combined_transactions, event_log, ground_truth
