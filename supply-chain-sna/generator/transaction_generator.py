"""
Supply Chain SNA — Transaction Generator
PHASE 3: Generates transaction-level data on top of the network structure.

Each edge in the network produces multiple transactions over the time
period, with realistic variation in quantities, values, and timings.
"""
from __future__ import annotations

import logging
import random
import uuid
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Product categories and realistic value ranges per unit (USD)
PRODUCT_VALUE_RANGES: dict[str, tuple[float, float]] = {
    "Electronics_Components": (10.0, 500.0),
    "Raw_Chemicals": (2.0, 50.0),
    "Pharmaceutical_Ingredients": (50.0, 2000.0),
    "Auto_Parts": (5.0, 800.0),
    "Food_Ingredients": (0.5, 20.0),
    "Textile_Materials": (1.0, 30.0),
    "Finished_Electronics": (50.0, 1500.0),
    "Finished_Pharmaceuticals": (100.0, 5000.0),
    "Finished_Auto": (500.0, 30000.0),
    "Finished_Food": (2.0, 50.0),
    "Finished_Textiles": (5.0, 100.0),
}

PRODUCT_UNITS: dict[str, str] = {
    "Electronics_Components": "units",
    "Raw_Chemicals": "kg",
    "Pharmaceutical_Ingredients": "kg",
    "Auto_Parts": "units",
    "Food_Ingredients": "kg",
    "Textile_Materials": "meters",
    "Finished_Electronics": "units",
    "Finished_Pharmaceuticals": "units",
    "Finished_Auto": "units",
    "Finished_Food": "kg",
    "Finished_Textiles": "meters",
}


def generate_transactions(
    edges_df: pd.DataFrame,
    organizations: pd.DataFrame,
    config: dict[str, Any],
    month_offset: int = 0,
    event_multipliers: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Generate transaction records for a given time period.

    Args:
        edges_df: DataFrame of supply-chain edges.
        organizations: DataFrame of organizations.
        config: Configuration dictionary.
        month_offset: Months since the start_date (0 = first month).
        event_multipliers: Optional dict mapping edge key → volume multiplier
            (used to simulate disruptions or seasonal effects).

    Returns:
        DataFrame of transactions with full schema.
    """
    seed: int = config["seed"] + month_offset + 100
    rng = random.Random(seed)

    start_date = datetime.strptime(config["network"]["start_date"], "%Y-%m-%d")
    month_start = _add_months(start_date, month_offset)
    month_end = _add_months(month_start, 1)
    products: list[str] = config.get("products", list(PRODUCT_VALUE_RANGES.keys()))
    txns_per_edge: int = config["network"].get("transactions_per_month_per_edge", 3)

    org_region: dict[str, str] = dict(
        zip(organizations["organization_id"], organizations["region"])
    )

    if event_multipliers is None:
        event_multipliers = {}

    records: list[dict[str, Any]] = []
    txn_counter = month_offset * 100_000

    for _, edge in edges_df.iterrows():
        src: str = edge["source_node"]
        tgt: str = edge["target_node"]
        rel_type: str = edge.get("relationship_type", "supply_chain_link")

        edge_key = f"{src}→{tgt}"
        volume_mult = event_multipliers.get(edge_key, 1.0)

        n_txns = max(1, round(txns_per_edge * volume_mult * rng.uniform(0.5, 2.0)))

        for _ in range(n_txns):
            product = rng.choice(products)
            lo, hi = PRODUCT_VALUE_RANGES.get(product, (1.0, 100.0))
            unit = PRODUCT_UNITS.get(product, "units")
            quantity = round(rng.uniform(50, 5000) * volume_mult, 2)
            unit_value = round(rng.uniform(lo, hi), 2)
            total_value = round(quantity * unit_value, 2)
            lead_time = rng.randint(1, 30)

            # Random timestamp within month
            day_offset = rng.uniform(0, (month_end - month_start).total_seconds())
            timestamp = month_start + timedelta(seconds=day_offset)

            txn_counter += 1
            txn_id = f"TXN-{txn_counter:08d}"

            records.append(
                {
                    "transaction_id": txn_id,
                    "timestamp": timestamp.isoformat(),
                    "source_node": src,
                    "target_node": tgt,
                    "product_id": f"PROD-{product}",
                    "product_category": product,
                    "quantity": quantity,
                    "unit": unit,
                    "transaction_value": total_value,
                    "region": org_region.get(src, "Unknown"),
                    "relationship_type": rel_type,
                    "lead_time_days": lead_time,
                    "status": "completed",
                    "month": month_start.strftime("%Y-%m"),
                }
            )

    df = pd.DataFrame(records)
    logger.info(
        "Month %d: generated %d transactions on %d edges",
        month_offset,
        len(df),
        len(edges_df),
    )
    return df


def _add_months(dt: datetime, months: int) -> datetime:
    """Add a number of months to a datetime.

    Args:
        dt: Base datetime.
        months: Number of months to add.

    Returns:
        New datetime with months added.
    """
    month = dt.month - 1 + months
    year = dt.year + month // 12
    month = month % 12 + 1
    day = min(dt.day, [31, 29 if year % 4 == 0 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
    return dt.replace(year=year, month=month, day=day)
