#!/usr/bin/env bash
# Supply Chain SNA — Automated Execution Pipeline
# PHASE 21: Runs the entire generation, analysis, and reporting pipeline.
# Usage: ./run_pipeline.sh [config_file]

set -e

CONFIG=${1:-config/default.yaml}

echo "============================================================"
echo " SUPPLY CHAIN SNA PIPELINE"
echo " Configuration: $CONFIG"
echo "============================================================"

# Ensure output directories exist
mkdir -p data/synthetic data/processed reports/results reports/figures reports/tables

echo -e "\n[1/4] Generating Data..."
.venv/bin/python3 -m generator.generate --config "$CONFIG"

echo -e "\n[2/4] Building Graph..."
.venv/bin/python3 -m graph.build --config "$CONFIG"

echo -e "\n[3/4] Running Experiments..."
.venv/bin/python3 -m experiments.run --config "$CONFIG" --experiment all

echo -e "\n[4/4] Generating Reports..."
.venv/bin/python3 -m reports.generate --config "$CONFIG"

echo -e "\nPipeline completed successfully! Check the 'reports' directory."
