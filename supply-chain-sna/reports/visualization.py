"""
Supply Chain SNA — Visualization Engine
PHASE 17: Generates reusable plots for all SNA analyses.

All plots use actual computed data — no hard-coded values.
Saves PNG (and SVG where appropriate) to the reports/figures directory.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


def save_figure(fig: go.Figure, path: Path, formats: list[str] | None = None) -> None:
    """Save a Plotly figure in one or more formats.

    Args:
        fig: Plotly figure.
        path: Output path (without extension).
        formats: List of formats. Defaults to ["png"].
    """
    if formats is None:
        formats = ["png"]
    for fmt in formats:
        out_path = path.with_suffix(f".{fmt}")
        try:
            if fmt == "html":
                fig.write_html(str(out_path))
            else:
                fig.write_image(str(out_path))
            logger.info("Saved figure → %s", out_path)
        except Exception as exc:
            logger.warning("Could not save %s: %s", out_path, exc)


def plot_degree_distribution(
    centrality_df: pd.DataFrame,
    output_dir: Path,
) -> go.Figure:
    """Plot degree distribution (in-degree, out-degree, total).

    Args:
        centrality_df: Centrality DataFrame with degree columns.
        output_dir: Directory to save figure.

    Returns:
        Plotly figure.
    """
    fig = make_subplots(rows=1, cols=3, subplot_titles=["In-Degree", "Out-Degree", "Total Degree"])

    for i, col in enumerate(["in_degree", "out_degree", "total_degree"], 1):
        if col not in centrality_df.columns:
            continue
        fig.add_trace(
            go.Histogram(x=centrality_df[col], nbinsx=30, name=col.replace("_", " ").title()),
            row=1, col=i,
        )

    fig.update_layout(
        title="Degree Distribution of Supply-Chain Network",
        showlegend=False,
        height=400,
    )
    save_figure(fig, output_dir / "degree_distribution", ["png"])
    return fig


def plot_centrality_comparison(
    centrality_df: pd.DataFrame,
    top_n: int = 20,
    output_dir: Path | None = None,
) -> go.Figure:
    """Plot top-N nodes by each centrality metric.

    Args:
        centrality_df: Combined centrality DataFrame.
        top_n: Number of top nodes to show.
        output_dir: Directory to save figure.

    Returns:
        Plotly figure.
    """
    metrics = ["betweenness", "pagerank", "closeness", "eigenvector"]
    available = [m for m in metrics if m in centrality_df.columns]

    if not available:
        return go.Figure()

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[m.replace("_", " ").title() for m in available[:4]],
    )

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for (row, col), metric in zip(positions, available[:4]):
        top = centrality_df.nlargest(top_n, metric)
        fig.add_trace(
            go.Bar(
                x=top["node"],
                y=top[metric],
                name=metric,
                marker_color="steelblue",
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title=f"Top-{top_n} Nodes by Centrality Metric",
        height=700,
        showlegend=False,
    )

    if output_dir:
        save_figure(fig, output_dir / "centrality_comparison", ["png"])
    return fig


def plot_network_graph(
    G: nx.DiGraph,
    node_attr: str = "organization_type",
    centrality_df: pd.DataFrame | None = None,
    community_df: pd.DataFrame | None = None,
    max_nodes: int = 500,
    output_dir: Path | None = None,
    title: str = "Supply-Chain Network",
) -> go.Figure:
    """Plot an interactive network graph with node coloring.

    Args:
        G: Directed graph.
        node_attr: Node attribute to use for coloring.
        centrality_df: Optional centrality DataFrame for node sizing.
        community_df: Optional community DataFrame for coloring.
        max_nodes: Maximum nodes to render (samples if larger).
        output_dir: Directory to save figure.
        title: Figure title.

    Returns:
        Plotly figure.
    """
    # Sample if too large
    nodes = list(G.nodes())
    if len(nodes) > max_nodes:
        import random
        rng = random.Random(42)
        nodes = rng.sample(nodes, max_nodes)
        subG = G.subgraph(nodes).copy()
        logger.info("Sampled %d/%d nodes for visualization", max_nodes, len(G.nodes()))
    else:
        subG = G

    # Layout
    try:
        pos = nx.spring_layout(subG, seed=42, k=1.5 / max(1, len(nodes) ** 0.5))
    except Exception:
        pos = nx.random_layout(subG, seed=42)

    # Node colors
    if community_df is not None and not community_df.empty:
        comm_map = dict(zip(community_df["node"], community_df["community_id"]))
        node_colors = [int(comm_map.get(n, -1)) for n in subG.nodes()]
    else:
        attr_values = [subG.nodes[n].get(node_attr, "unknown") for n in subG.nodes()]
        unique_vals = list(set(attr_values))
        color_map = {v: i for i, v in enumerate(unique_vals)}
        node_colors = [int(color_map.get(v, 0)) for v in attr_values]

    # Node sizes based on degree or centrality
    if centrality_df is not None and "total_degree" in centrality_df.columns:
        deg_map = dict(zip(centrality_df["node"], centrality_df["total_degree"]))
        max_deg = max(deg_map.values()) if deg_map else 1
        node_sizes = [5 + 20 * (deg_map.get(n, 1) / max(1, max_deg)) for n in subG.nodes()]
    else:
        node_sizes = [8] * len(subG.nodes())

    # Edges
    edge_x, edge_y = [], []
    for u, v in subG.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="#cccccc"),
        hoverinfo="none",
    )

    # Nodes
    node_x = [pos[n][0] for n in subG.nodes()]
    node_y = [pos[n][1] for n in subG.nodes()]
    node_text = [
        f"{n}<br>Type: {subG.nodes[n].get('organization_type','')}<br>"
        f"Region: {subG.nodes[n].get('region','')}"
        for n in subG.nodes()
    ]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title=node_attr.replace("_", " ").title()),
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
        ),
    )

    if output_dir:
        save_figure(fig, output_dir / "network_graph", ["png"])
    return fig


def plot_community_sizes(
    community_df: pd.DataFrame,
    output_dir: Path | None = None,
) -> go.Figure:
    """Plot community size distribution.

    Args:
        community_df: Community assignment DataFrame.
        output_dir: Directory to save figure.

    Returns:
        Plotly figure.
    """
    if community_df.empty:
        return go.Figure()

    sizes = community_df["community_id"].value_counts().reset_index()
    sizes.columns = ["community_id", "size"]
    sizes = sizes.sort_values("size", ascending=False)

    fig = px.bar(
        sizes,
        x="community_id",
        y="size",
        title="Community Size Distribution",
        labels={"community_id": "Community ID", "size": "Number of Organizations"},
        color="size",
        color_continuous_scale="Blues",
    )

    if output_dir:
        save_figure(fig, output_dir / "community_sizes", ["png"])
    return fig


def plot_temporal_metrics(
    snapshot_df: pd.DataFrame,
    output_dir: Path | None = None,
) -> go.Figure:
    """Plot network evolution over time.

    Args:
        snapshot_df: Temporal snapshot DataFrame with per-month statistics.
        output_dir: Directory to save figure.

    Returns:
        Plotly figure.
    """
    if snapshot_df.empty:
        return go.Figure()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Node & Edge Count", "Network Density",
            "Weakly Connected Components", "Largest WCC Fraction"
        ],
    )

    x = snapshot_df["month"].tolist()

    fig.add_trace(go.Scatter(x=x, y=snapshot_df["num_nodes"], name="Nodes", mode="lines+markers"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=snapshot_df["num_edges"], name="Edges", mode="lines+markers"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=snapshot_df["density"], name="Density", mode="lines+markers", line=dict(color="red")), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=snapshot_df["weakly_connected_components"], name="WCC", mode="lines+markers", line=dict(color="orange")), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=snapshot_df["largest_wcc_fraction"], name="LCC Frac", mode="lines+markers", line=dict(color="green")), row=2, col=2)

    fig.update_layout(title="Temporal Network Evolution", height=600, showlegend=True)

    if output_dir:
        save_figure(fig, output_dir / "temporal_metrics", ["png"])
    return fig


def plot_resilience_curves(
    resilience_results: dict[str, pd.DataFrame],
    output_dir: Path | None = None,
) -> go.Figure:
    """Plot network degradation curves for all attack strategies.

    Args:
        resilience_results: Dict strategy → DataFrame with resilience metrics.
        output_dir: Directory to save figure.

    Returns:
        Plotly figure.
    """
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Largest Component Fraction", "Network Efficiency"])

    colors = {"random": "gray", "degree": "red", "betweenness": "orange", "pagerank": "blue"}

    for strategy, df in resilience_results.items():
        color = colors.get(strategy, "black")
        x = df["fraction_removed"].tolist()

        if "lcc_mean" in df.columns:
            y_lcc = df["lcc_mean"].tolist()
            y_eff = df["efficiency_mean"].tolist() if "efficiency_mean" in df.columns else []
        else:
            y_lcc = df["largest_component_fraction"].tolist() if "largest_component_fraction" in df.columns else []
            y_eff = df["network_efficiency"].tolist() if "network_efficiency" in df.columns else []

        fig.add_trace(
            go.Scatter(x=x, y=y_lcc, name=f"{strategy}", line=dict(color=color), mode="lines+markers"),
            row=1, col=1,
        )
        if y_eff:
            fig.add_trace(
                go.Scatter(x=x, y=y_eff, name=f"{strategy}", line=dict(color=color, dash="dot"), mode="lines+markers", showlegend=False),
                row=1, col=2,
            )

    fig.update_xaxes(title_text="Fraction Removed")
    fig.update_yaxes(title_text="Largest Component Fraction", row=1, col=1)
    fig.update_yaxes(title_text="Global Efficiency", row=1, col=2)
    fig.update_layout(title="Network Resilience Under Different Attack Strategies", height=450)

    if output_dir:
        save_figure(fig, output_dir / "resilience_curves", ["png"])
    return fig


def generate_all_figures(
    G: nx.DiGraph,
    centrality_df: pd.DataFrame,
    community_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    resilience_results: dict[str, pd.DataFrame],
    figures_dir: Path,
) -> None:
    """Generate and save all required figures.

    Args:
        G: Directed supply-chain graph.
        centrality_df: Centrality DataFrame.
        community_df: Community DataFrame.
        snapshot_df: Temporal snapshot DataFrame.
        resilience_results: Resilience experiment results.
        figures_dir: Directory to save figures.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Generating all figures → %s", figures_dir)

    plot_network_graph(G, centrality_df=centrality_df, output_dir=figures_dir)
    plot_degree_distribution(centrality_df, output_dir=figures_dir)
    plot_centrality_comparison(centrality_df, output_dir=figures_dir)
    plot_community_sizes(community_df, output_dir=figures_dir)
    plot_temporal_metrics(snapshot_df, output_dir=figures_dir)
    plot_resilience_curves(resilience_results, output_dir=figures_dir)

    # Community-colored network
    if not community_df.empty:
        plot_network_graph(
            G,
            community_df=community_df,
            output_dir=figures_dir,
            title="Supply-Chain Network — Community Coloring",
        )
        # Rename to avoid overwrite
        src = figures_dir / "network_graph.png"
        dst = figures_dir / "network_graph_community.png"
        if src.exists() and not dst.exists():
            src.rename(dst)

    logger.info("All figures generated")
