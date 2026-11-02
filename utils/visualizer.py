"""
Interactive Data Visualization Module using Plotly.
Modern Dark Navy & Light Blue styling for AI/ML Research Dashboards.
"""
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Theme Colors
NAVY = "#0B192C"
NAVY_LIGHT = "#1E3E62"
BLUE_ACCENT = "#008DDA"
BLUE_CYAN = "#41B06E"
LIGHT_BG = "#F8FAFC"
GRID_COLOR = "#E2E8F0"


def apply_research_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Applies modern academic/commercial research styling to Plotly figures."""
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>" if title else "",
            "font": {"size": 16, "color": NAVY, "family": "Inter, Segoe UI, sans-serif"},
            "x": 0.02,
            "xanchor": "left"
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        margin={"l": 40, "r": 30, "t": 50, "b": 40},
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#334155", "size": 12},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "bgcolor": "rgba(255,255,255,0.85)"
        },
        xaxis=dict(
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            showline=True,
            linecolor="#CBD5E1"
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            showline=True,
            linecolor="#CBD5E1"
        )
    )
    return fig


def plot_numerical_distribution(real_df: pd.DataFrame, synth_df: pd.DataFrame, column: str) -> go.Figure:
    """Plots overlaid histograms/KDE curves comparing real vs synthetic numerical data."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=real_df[column].dropna(),
        name="Real Data",
        opacity=0.65,
        marker_color="#1E3E62",
        histnorm="probability density"
    ))

    fig.add_trace(go.Histogram(
        x=synth_df[column].dropna(),
        name="Synthetic (CTGAN)",
        opacity=0.65,
        marker_color="#008DDA",
        histnorm="probability density"
    ))

    fig.update_layout(
        barmode="overlay",
        xaxis_title=f"{column} Values",
        yaxis_title="Probability Density"
    )
    return apply_research_theme(fig, f"Distribution Comparison: {column}")


def plot_categorical_distribution(real_df: pd.DataFrame, synth_df: pd.DataFrame, column: str) -> go.Figure:
    """Plots side-by-side bar charts comparing relative category frequencies."""
    r_counts = real_df[column].value_counts(normalize=True).reset_index()
    r_counts.columns = [column, "Proportion"]
    r_counts["Source"] = "Real Data"

    s_counts = synth_df[column].value_counts(normalize=True).reset_index()
    s_counts.columns = [column, "Proportion"]
    s_counts["Source"] = "Synthetic (CTGAN)"

    combined = pd.concat([r_counts, s_counts], axis=0)

    fig = px.bar(
        combined,
        x=column,
        y="Proportion",
        color="Source",
        barmode="group",
        color_discrete_map={"Real Data": "#1E3E62", "Synthetic (CTGAN)": "#008DDA"}
    )
    return apply_research_theme(fig, f"Categorical Frequency: {column}")


def plot_correlation_heatmaps(real_df: pd.DataFrame, synth_df: pd.DataFrame, num_cols: List[str]) -> Tuple[go.Figure, go.Figure, go.Figure]:
    """Plots real, synthetic, and absolute error correlation heatmaps."""
    real_corr = real_df[num_cols].corr()
    synth_corr = synth_df[num_cols].corr()
    diff_corr = (real_corr - synth_corr).abs()

    fig_real = go.Figure(data=go.Heatmap(
        z=real_corr.values,
        x=num_cols,
        y=num_cols,
        colorscale="Blues",
        zmin=-1,
        zmax=1,
        text=np.round(real_corr.values, 2),
        texttemplate="%{text}",
        showscale=True
    ))
    apply_research_theme(fig_real, "Real Data Correlation Matrix")

    fig_synth = go.Figure(data=go.Heatmap(
        z=synth_corr.values,
        x=num_cols,
        y=num_cols,
        colorscale="Blues",
        zmin=-1,
        zmax=1,
        text=np.round(synth_corr.values, 2),
        texttemplate="%{text}",
        showscale=True
    ))
    apply_research_theme(fig_synth, "Synthetic Data Correlation Matrix")

    fig_diff = go.Figure(data=go.Heatmap(
        z=diff_corr.values,
        x=num_cols,
        y=num_cols,
        colorscale="Reds",
        zmin=0,
        zmax=0.5,
        text=np.round(diff_corr.values, 2),
        texttemplate="%{text}",
        showscale=True
    ))
    apply_research_theme(fig_diff, "Correlation Absolute Difference (|Real - Synth|)")

    return fig_real, fig_synth, fig_diff


def plot_pca_projection(real_df: pd.DataFrame, synth_df: pd.DataFrame, num_cols: List[str], max_points: int = 400) -> Optional[go.Figure]:
    """
    Projects Real and Synthetic datasets onto 2D PCA space to inspect manifold overlap.
    """
    if len(num_cols) < 2:
        return None

    r_sub = real_df[num_cols].dropna()
    s_sub = synth_df[num_cols].dropna()

    if len(r_sub) > max_points:
        r_sub = r_sub.sample(max_points, random_state=42)
    if len(s_sub) > max_points:
        s_sub = s_sub.sample(max_points, random_state=42)

    combined = pd.concat([r_sub, s_sub], axis=0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(combined)

    pca = PCA(n_components=2)
    proj = pca.fit_transform(scaled)

    var_1 = round(pca.explained_variance_ratio_[0] * 100, 1)
    var_2 = round(pca.explained_variance_ratio_[1] * 100, 1)

    n_r = len(r_sub)
    plot_df = pd.DataFrame({
        "PC1": proj[:, 0],
        "PC2": proj[:, 1],
        "Type": ["Real Data"] * n_r + ["Synthetic (CTGAN)"] * (len(proj) - n_r)
    })

    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="Type",
        color_discrete_map={"Real Data": "#1E3E62", "Synthetic (CTGAN)": "#008DDA"},
        opacity=0.75
    )
    fig.update_traces(marker=dict(size=7, line=dict(width=0.8, color="#FFFFFF")))
    fig.update_layout(
        xaxis_title=f"Principal Component 1 ({var_1}% var)",
        yaxis_title=f"Principal Component 2 ({var_2}% var)"
    )
    return apply_research_theme(fig, "2D PCA Latent Manifold Projection")


def plot_dcr_histogram(dcr_values: List[float]) -> go.Figure:
    """Plots histogram of Distance to Closest Record (DCR) for privacy analysis."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=dcr_values,
        nbinsx=30,
        marker_color="#1E3E62",
        opacity=0.8
    ))
    fig.update_layout(
        xaxis_title="Normalized Distance to Closest Training Record",
        yaxis_title="Synthetic Record Count"
    )
    return apply_research_theme(fig, "Distance to Closest Record (DCR) Privacy Distribution")
