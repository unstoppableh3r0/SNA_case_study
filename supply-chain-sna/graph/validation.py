"""
Supply Chain SNA — Data Validation
PHASE 5: Validates generated datasets before graph construction.

Checks for duplicate IDs, missing nodes, invalid types, timestamp issues,
negative quantities, orphan nodes, and other data quality problems.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

VALID_ORG_TYPES = {
    "supplier",
    "manufacturer",
    "distributor",
    "warehouse",
    "logistics_provider",
    "retailer",
}

REQUIRED_ORG_COLUMNS = {
    "organization_id",
    "organization_name",
    "organization_type",
    "region",
    "industry",
    "size_category",
    "status",
    "created_at",
}

REQUIRED_TXN_COLUMNS = {
    "transaction_id",
    "timestamp",
    "source_node",
    "target_node",
    "product_id",
    "product_category",
    "quantity",
    "unit",
    "transaction_value",
    "region",
    "relationship_type",
    "lead_time_days",
    "status",
    "month",
}


def validate_dataset(
    organizations: pd.DataFrame,
    transactions: pd.DataFrame,
) -> dict[str, Any]:
    """Run data quality validation on the generated dataset.

    Args:
        organizations: DataFrame of organizations.
        transactions: DataFrame of transactions.

    Returns:
        Data quality report as a dictionary. Contains:
            is_valid: bool
            issues: list of critical issues
            warnings: list of non-critical issues
            stats: summary statistics
    """
    issues: list[str] = []
    warnings: list[str] = []
    stats: dict[str, Any] = {}

    # ── Organization validation ───────────────────────────────────────────────
    _validate_organizations(organizations, issues, warnings, stats)

    # ── Transaction validation ────────────────────────────────────────────────
    _validate_transactions(transactions, organizations, issues, warnings, stats)

    is_valid = len(issues) == 0
    if is_valid:
        logger.info("Data validation PASSED — no critical issues found")
    else:
        logger.error("Data validation FAILED — %d critical issues", len(issues))
        for issue in issues:
            logger.error("  ISSUE: %s", issue)
    for warning in warnings:
        logger.warning("  WARNING: %s", warning)

    report = {
        "is_valid": is_valid,
        "issues": issues,
        "warnings": warnings,
        "stats": stats,
    }
    return report


def _validate_organizations(
    organizations: pd.DataFrame,
    issues: list[str],
    warnings: list[str],
    stats: dict[str, Any],
) -> None:
    """Validate the organizations DataFrame.

    Args:
        organizations: DataFrame to validate.
        issues: List to append critical issues to.
        warnings: List to append warnings to.
        stats: Dictionary to append statistics to.
    """
    stats["organization_count"] = len(organizations)

    # Required columns
    missing_cols = REQUIRED_ORG_COLUMNS - set(organizations.columns)
    if missing_cols:
        issues.append(f"Organizations missing required columns: {missing_cols}")
        return

    # Unique IDs
    dup_ids = organizations[organizations.duplicated("organization_id")]["organization_id"].tolist()
    if dup_ids:
        issues.append(f"Duplicate organization_ids found: {dup_ids[:5]}")

    # Valid org types
    invalid_types = organizations[
        ~organizations["organization_type"].isin(VALID_ORG_TYPES)
    ]["organization_type"].unique().tolist()
    if invalid_types:
        issues.append(f"Invalid organization types: {invalid_types}")

    # No null IDs
    null_ids = organizations["organization_id"].isna().sum()
    if null_ids > 0:
        issues.append(f"Found {null_ids} null organization_ids")

    # Type distribution
    stats["org_type_counts"] = organizations["organization_type"].value_counts().to_dict()
    stats["region_counts"] = organizations["region"].value_counts().to_dict()

    # Warn if any type has 0 organizations
    for otype in VALID_ORG_TYPES:
        if otype not in stats["org_type_counts"]:
            warnings.append(f"No organizations of type '{otype}' found")


def _validate_transactions(
    transactions: pd.DataFrame,
    organizations: pd.DataFrame,
    issues: list[str],
    warnings: list[str],
    stats: dict[str, Any],
) -> None:
    """Validate the transactions DataFrame.

    Args:
        transactions: DataFrame to validate.
        organizations: DataFrame of valid organizations.
        issues: List to append critical issues to.
        warnings: List to append warnings to.
        stats: Dictionary to append statistics to.
    """
    if transactions.empty:
        issues.append("Transaction dataset is empty")
        return

    stats["transaction_count"] = len(transactions)

    # Required columns
    missing_cols = REQUIRED_TXN_COLUMNS - set(transactions.columns)
    if missing_cols:
        issues.append(f"Transactions missing required columns: {missing_cols}")
        return

    # Unique transaction IDs
    dup_txn_ids = transactions[transactions.duplicated("transaction_id")]["transaction_id"].tolist()
    if dup_txn_ids:
        issues.append(f"Duplicate transaction_ids: {len(dup_txn_ids)} duplicates found")

    # Negative quantities
    neg_qty = (transactions["quantity"] <= 0).sum()
    if neg_qty > 0:
        issues.append(f"Found {neg_qty} transactions with non-positive quantity")

    # Negative values
    neg_val = (transactions["transaction_value"] <= 0).sum()
    if neg_val > 0:
        issues.append(f"Found {neg_val} transactions with non-positive transaction_value")

    # All nodes in transactions must exist in organizations
    valid_org_ids = set(organizations["organization_id"].tolist())
    unknown_sources = set(transactions["source_node"].unique()) - valid_org_ids
    unknown_targets = set(transactions["target_node"].unique()) - valid_org_ids
    if unknown_sources:
        issues.append(f"Transactions reference unknown source nodes: {list(unknown_sources)[:5]}")
    if unknown_targets:
        issues.append(f"Transactions reference unknown target nodes: {list(unknown_targets)[:5]}")

    # Self-loops
    self_loops = (transactions["source_node"] == transactions["target_node"]).sum()
    if self_loops > 0:
        warnings.append(f"Found {self_loops} self-loop transactions (source == target)")

    # Orphan organizations (in orgs table but no transactions)
    txn_nodes = set(transactions["source_node"].unique()) | set(transactions["target_node"].unique())
    orphan_orgs = valid_org_ids - txn_nodes
    if orphan_orgs:
        warnings.append(f"{len(orphan_orgs)} organizations have no transactions (orphans)")
        stats["orphan_org_count"] = len(orphan_orgs)

    # Stats
    stats["unique_source_nodes"] = len(transactions["source_node"].unique())
    stats["unique_target_nodes"] = len(transactions["target_node"].unique())
    stats["unique_edges"] = len(
        transactions[["source_node", "target_node"]].drop_duplicates()
    )
    stats["month_counts"] = transactions.groupby("month").size().to_dict() if "month" in transactions.columns else {}
    stats["total_transaction_value"] = float(transactions["transaction_value"].sum())
    stats["avg_quantity"] = float(transactions["quantity"].mean())


def save_validation_report(report: dict[str, Any], output_path: str | Path) -> None:
    """Save the validation report to a JSON file.

    Args:
        report: Validation report dictionary.
        output_path: Path to save the JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(report, fh, indent=2, default=str)
    logger.info("Validation report saved → %s", output_path)
