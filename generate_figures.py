#!/usr/bin/env python3
"""
Generate Showcase Figures
==========================
Produces publication-quality figures demonstrating the full analytical
output of the Lead Response Time Analyzer.

Outputs saved to ./figures/
"""

import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import CHART_TEMPLATE, CHART_COLOR_SEQUENCE, BUCKET_COLORS
from analysis.preprocessing import create_response_buckets
from analysis.descriptive import (
    calculate_close_rates, calculate_cross_tabulation,
    calculate_rep_performance
)
from analysis.statistical_tests import run_chi_square_test, run_pairwise_comparisons
from analysis.regression import run_logistic_regression, compare_models

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Shared layout defaults
LAYOUT_DEFAULTS = dict(
    template=CHART_TEMPLATE,
    font=dict(family="Inter, Arial, sans-serif", size=13),
    margin=dict(l=60, r=30, t=70, b=60),
)

WIDTH = 900
HEIGHT = 500


def save(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.write_image(path, width=WIDTH, height=HEIGHT, scale=2)
    print(f"  Saved {name}")


def load_data():
    df = pd.read_csv("sample_lead_data.csv", parse_dates=["lead_time", "first_response_time"])
    buckets, _ = create_response_buckets(df["response_time_mins"])
    df["response_bucket"] = buckets
    return df


# ============================================================================
# FIGURE 1: Close Rate by Response Bucket (hero chart)
# ============================================================================
def fig_close_rate_bars(df):
    cr = calculate_close_rates(df)
    colors = CHART_COLOR_SEQUENCE[:len(cr)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cr["bucket"].astype(str),
        y=cr["close_rate"] * 100,
        text=cr["close_rate"].apply(lambda x: f"{x*100:.1f}%"),
        textposition="outside",
        marker_color=colors,
        error_y=dict(
            type="data", symmetric=False,
            array=(cr["ci_upper"] - cr["close_rate"]) * 100,
            arrayminus=(cr["close_rate"] - cr["ci_lower"]) * 100,
            color="rgba(0,0,0,0.3)", thickness=2, width=6
        ),
        customdata=cr[["n_leads", "n_orders"]].values,
        hovertemplate="<b>%{x}</b><br>Close Rate: %{y:.1f}%<br>Leads: %{customdata[0]:,}<br>Orders: %{customdata[1]:,}<extra></extra>",
    ))

    # Annotation: fastest vs slowest
    fastest_rate = cr.iloc[0]["close_rate"] * 100
    slowest_rate = cr.iloc[-1]["close_rate"] * 100
    lift = fastest_rate / slowest_rate

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="Close Rate by Response Time Bucket", font=dict(size=22)),
        xaxis_title="Response Time Bucket",
        yaxis_title="Close Rate (%)",
        yaxis=dict(range=[0, fastest_rate * 1.4], ticksuffix="%"),
        showlegend=False,
        annotations=[dict(
            x=0.5, y=1.02, xref="paper", yref="paper",
            text=f"Fastest bucket closes at {lift:.1f}x the rate of the slowest",
            showarrow=False, font=dict(size=13, color="gray")
        )]
    )
    save(fig, "01_close_rate_by_bucket.png")


# ============================================================================
# FIGURE 2: Sample Size Distribution
# ============================================================================
def fig_sample_sizes(df):
    cr = calculate_close_rates(df)
    colors = CHART_COLOR_SEQUENCE[:len(cr)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cr["bucket"].astype(str),
        y=cr["n_leads"],
        text=cr["n_leads"].apply(lambda x: f"{x:,}"),
        textposition="outside",
        marker_color=colors,
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="Lead Volume by Response Time Bucket", font=dict(size=22)),
        xaxis_title="Response Time Bucket",
        yaxis_title="Number of Leads",
        showlegend=False,
    )
    save(fig, "02_sample_sizes.png")


# ============================================================================
# FIGURE 3: Response Time Distribution
# ============================================================================
def fig_response_distribution(df):
    rt = df["response_time_mins"].dropna()
    cap = min(rt.quantile(0.98), 180)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=rt[rt <= cap], nbinsx=60,
        marker_color="steelblue", opacity=0.85,
        hovertemplate="%{x:.0f} min: %{y:,} leads<extra></extra>",
    ))

    for bound, label in [(15, "15 min"), (30, "30 min"), (60, "60 min")]:
        fig.add_vline(x=bound, line_dash="dash", line_color="red", line_width=2,
                      annotation_text=label, annotation_position="top")

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="Response Time Distribution with Bucket Boundaries", font=dict(size=22)),
        xaxis_title="Response Time (minutes)",
        yaxis_title="Number of Leads",
        xaxis=dict(range=[0, cap]),
    )
    save(fig, "03_response_time_distribution.png")


# ============================================================================
# FIGURE 4: Heatmap — Close Rate by Source × Response Bucket
# ============================================================================
def fig_heatmap(df):
    _, close_rate_matrix, _ = calculate_cross_tabulation(df)
    z = close_rate_matrix.values * 100

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=close_rate_matrix.columns.tolist(),
        y=close_rate_matrix.index.astype(str).tolist(),
        colorscale="RdYlGn",
        text=np.round(z, 1),
        texttemplate="%{text:.1f}%",
        textfont=dict(size=11),
        colorbar=dict(title="Close %", ticksuffix="%"),
        hovertemplate="Response: %{y}<br>Source: %{x}<br>Close Rate: %{z:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="Close Rate: Lead Source × Response Speed", font=dict(size=22)),
        xaxis_title="Lead Source",
        yaxis_title="Response Time Bucket",
        height=480,
    )
    save(fig, "04_heatmap_source_x_bucket.png")


# ============================================================================
# FIGURE 5: Forest Plot — Odds Ratios from Regression
# ============================================================================
def fig_forest_plot(df):
    result = run_logistic_regression(df, include_lead_source=True)
    or_df = result.odds_ratios

    if or_df.empty:
        print("  Skipping forest plot — regression returned no odds ratios.")
        return

    fig = go.Figure()
    fig.add_vline(x=1, line_dash="dash", line_color="gray", line_width=1.5,
                  annotation_text="No Effect (OR=1)", annotation_position="top")

    for _, row in or_df.iterrows():
        color = "#2ECC71" if row["odds_ratio"] > 1 else "#E74C3C"
        fig.add_trace(go.Scatter(
            x=[row["odds_ratio"]], y=[row["bucket"]],
            mode="markers",
            marker=dict(size=14, color=color, line=dict(width=1, color="white")),
            error_x=dict(
                type="data", symmetric=False,
                array=[row["ci_upper"] - row["odds_ratio"]],
                arrayminus=[row["odds_ratio"] - row["ci_lower"]],
                color=color, thickness=2.5, width=8
            ),
            name=row["bucket"],
            hovertemplate=f"<b>{row['bucket']}</b><br>OR: {row['odds_ratio']:.2f}<br>95% CI: [{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]<extra></extra>",
        ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="Odds Ratios vs Slowest Bucket (Controlled for Lead Source)", font=dict(size=20)),
        xaxis_title="Odds Ratio",
        yaxis_title="Response Time Bucket",
        showlegend=False,
        xaxis=dict(type="log"),
        height=380,
        annotations=[dict(
            x=0.5, y=-0.18, xref="paper", yref="paper",
            text=f"Reference: {result.reference_bucket} | Pseudo R²: {result.pseudo_r_squared:.3f} | n = {result.n_observations:,}",
            showarrow=False, font=dict(size=11, color="gray")
        )]
    )
    save(fig, "05_forest_plot_odds_ratios.png")


# ============================================================================
# FIGURE 6: Sales Rep Scatter — Speed vs Close Rate
# ============================================================================
def fig_rep_scatter(df):
    rep_df = calculate_rep_performance(df)
    if rep_df.empty:
        return

    corr = rep_df["median_response_mins"].corr(rep_df["close_rate"])

    fig = px.scatter(
        rep_df, x="median_response_mins", y="close_rate",
        size="n_leads", hover_name="sales_rep",
        color="close_rate", color_continuous_scale="RdYlGn",
        labels={"median_response_mins": "Median Response Time (min)", "close_rate": "Close Rate", "n_leads": "Leads Handled"},
    )

    # Trend line
    z = np.polyfit(rep_df["median_response_mins"], rep_df["close_rate"], 1)
    p = np.poly1d(z)
    x_range = np.linspace(rep_df["median_response_mins"].min(), rep_df["median_response_mins"].max(), 100)
    fig.add_trace(go.Scatter(
        x=x_range, y=p(x_range), mode="lines",
        line=dict(color="gray", dash="dash", width=2),
        name="Trend", showlegend=True
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text=f"Sales Rep: Response Speed vs Close Rate (r = {corr:.2f})", font=dict(size=20)),
        yaxis=dict(tickformat=".0%"),
    )
    save(fig, "06_rep_scatter.png")


# ============================================================================
# FIGURE 7: Weekly Trend — Close Rate Over Time by Bucket
# ============================================================================
def fig_weekly_trend(df):
    df2 = df.copy()
    df2["week"] = df2["lead_time"].dt.to_period("W").dt.start_time

    weekly = df2.groupby(["week", "response_bucket"], observed=True).agg(
        close_rate=("ordered", "mean"),
        n_leads=("ordered", "count")
    ).reset_index()

    fig = px.line(
        weekly, x="week", y="close_rate",
        color="response_bucket",
        color_discrete_sequence=CHART_COLOR_SEQUENCE,
        labels={"week": "Week", "close_rate": "Close Rate", "response_bucket": "Response Time"},
    )
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="Close Rate Over Time by Response Speed", font=dict(size=22)),
        yaxis=dict(tickformat=".0%"),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.8)", bordercolor="lightgray", borderwidth=1),
    )
    save(fig, "07_weekly_trend.png")


# ============================================================================
# FIGURE 8: Pairwise Z-Test Matrix
# ============================================================================
def fig_pairwise_matrix(df):
    buckets = sorted(df["response_bucket"].dropna().unique(), key=str)
    n = len(buckets)

    # Build matrix of p-values
    p_matrix = np.ones((n, n))
    diff_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            g1 = df[df["response_bucket"] == buckets[i]]
            g2 = df[df["response_bucket"] == buckets[j]]
            p1, p2 = g1["ordered"].mean(), g2["ordered"].mean()
            diff_matrix[i][j] = (p1 - p2) * 100

            from statsmodels.stats.proportion import proportions_ztest
            count = np.array([g1["ordered"].sum(), g2["ordered"].sum()])
            nobs = np.array([len(g1), len(g2)])
            _, pval = proportions_ztest(count, nobs)
            p_matrix[i][j] = pval

    # Text annotations
    text = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append("—")
            else:
                sig = "***" if p_matrix[i][j] < 0.001 else "**" if p_matrix[i][j] < 0.01 else "*" if p_matrix[i][j] < 0.05 else "ns"
                row.append(f"{diff_matrix[i][j]:+.1f}pp\n({sig})")
        text.append(row)

    bucket_labels = [str(b) for b in buckets]

    fig = go.Figure(data=go.Heatmap(
        z=diff_matrix,
        x=bucket_labels, y=bucket_labels,
        colorscale="RdBu", zmid=0,
        text=text, texttemplate="%{text}",
        textfont=dict(size=12),
        colorbar=dict(title="Diff (pp)"),
        hovertemplate="Row: %{y}<br>Col: %{x}<br>Diff: %{z:.1f}pp<extra></extra>",
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="Pairwise Close Rate Differences (Z-Test Significance)", font=dict(size=20)),
        xaxis_title="Bucket (Column)",
        yaxis_title="Bucket (Row)",
        height=520,
        annotations=[dict(
            x=0.5, y=-0.15, xref="paper", yref="paper",
            text="Values = row − column (percentage points). *** p<0.001, ** p<0.01, * p<0.05, ns = not significant",
            showarrow=False, font=dict(size=11, color="gray")
        )]
    )
    save(fig, "08_pairwise_ztest_matrix.png")


# ============================================================================
# FIGURE 9: Model Comparison — With vs Without Controls
# ============================================================================
def fig_model_comparison(df):
    comparison = compare_models(df)
    m_simple = comparison["model_simple"]
    m_ctrl = comparison["model_controlled"]

    if m_simple.odds_ratios.empty or m_ctrl.odds_ratios.empty:
        print("  Skipping model comparison — missing data.")
        return

    fig = go.Figure()
    buckets = m_simple.odds_ratios["bucket"].tolist()
    x = list(range(len(buckets)))

    fig.add_trace(go.Bar(
        x=[i - 0.17 for i in x],
        y=m_simple.odds_ratios["odds_ratio"],
        width=0.3,
        name="Without Controls",
        marker_color="#3498DB",
        error_y=dict(
            type="data", symmetric=False,
            array=(m_simple.odds_ratios["ci_upper"] - m_simple.odds_ratios["odds_ratio"]).tolist(),
            arrayminus=(m_simple.odds_ratios["odds_ratio"] - m_simple.odds_ratios["ci_lower"]).tolist(),
        ),
        text=m_simple.odds_ratios["odds_ratio"].apply(lambda x: f"{x:.2f}x"),
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=[i + 0.17 for i in x],
        y=m_ctrl.odds_ratios["odds_ratio"],
        width=0.3,
        name="With Lead Source Control",
        marker_color="#E67E22",
        error_y=dict(
            type="data", symmetric=False,
            array=(m_ctrl.odds_ratios["ci_upper"] - m_ctrl.odds_ratios["odds_ratio"]).tolist(),
            arrayminus=(m_ctrl.odds_ratios["odds_ratio"] - m_ctrl.odds_ratios["ci_lower"]).tolist(),
        ),
        text=m_ctrl.odds_ratios["odds_ratio"].apply(lambda x: f"{x:.2f}x"),
        textposition="outside",
    ))

    fig.add_hline(y=1, line_dash="dash", line_color="gray")

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(text="Effect of Response Time: With vs Without Controls", font=dict(size=20)),
        xaxis=dict(tickvals=x, ticktext=buckets, title="Response Time Bucket"),
        yaxis_title="Odds Ratio (vs Slowest Bucket)",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        barmode="group",
    )
    save(fig, "09_model_comparison.png")


# ============================================================================
# FIGURE 10: Executive Summary Dashboard (multi-panel)
# ============================================================================
def fig_executive_dashboard(df):
    cr = calculate_close_rates(df)
    colors = CHART_COLOR_SEQUENCE[:len(cr)]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Close Rate by Response Bucket",
            "Lead Volume by Bucket",
            "Response Time Distribution",
            "Weekly Close Rate Trend"
        ],
        vertical_spacing=0.15, horizontal_spacing=0.1,
    )

    # Panel 1: Close rates
    fig.add_trace(go.Bar(
        x=cr["bucket"].astype(str), y=cr["close_rate"] * 100,
        marker_color=colors, showlegend=False,
        text=cr["close_rate"].apply(lambda x: f"{x*100:.1f}%"),
        textposition="outside",
    ), row=1, col=1)

    # Panel 2: Sample sizes
    fig.add_trace(go.Bar(
        x=cr["bucket"].astype(str), y=cr["n_leads"],
        marker_color=colors, showlegend=False,
        text=cr["n_leads"].apply(lambda x: f"{x:,}"),
        textposition="outside",
    ), row=1, col=2)

    # Panel 3: Histogram
    rt = df["response_time_mins"].dropna()
    cap = min(rt.quantile(0.98), 180)
    fig.add_trace(go.Histogram(
        x=rt[rt <= cap], nbinsx=40,
        marker_color="steelblue", opacity=0.8, showlegend=False,
    ), row=2, col=1)

    # Panel 4: Weekly trend
    df2 = df.copy()
    df2["week"] = df2["lead_time"].dt.to_period("W").dt.start_time
    weekly_overall = df2.groupby("week")["ordered"].mean().reset_index()
    fig.add_trace(go.Scatter(
        x=weekly_overall["week"], y=weekly_overall["ordered"] * 100,
        mode="lines+markers", line=dict(color="#3498DB", width=2),
        marker=dict(size=5), showlegend=False,
    ), row=2, col=2)

    fig.update_yaxes(title_text="Close Rate (%)", ticksuffix="%", row=1, col=1)
    fig.update_yaxes(title_text="Leads", row=1, col=2)
    fig.update_yaxes(title_text="Leads", row=2, col=1)
    fig.update_yaxes(title_text="Close Rate (%)", ticksuffix="%", row=2, col=2)
    fig.update_xaxes(range=[0, cap], row=2, col=1)

    total_leads = len(df)
    total_orders = df["ordered"].sum()
    overall_rate = df["ordered"].mean() * 100

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=dict(
            text=f"Executive Summary — {total_leads:,} Leads | {total_orders:,} Orders | {overall_rate:.1f}% Close Rate",
            font=dict(size=20)
        ),
        height=700,
    )
    save(fig, "10_executive_dashboard.png")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("Loading data...")
    df = load_data()
    print(f"  {len(df):,} leads loaded\n")

    print("Generating figures...")
    fig_close_rate_bars(df)
    fig_sample_sizes(df)
    fig_response_distribution(df)
    fig_heatmap(df)
    fig_forest_plot(df)
    fig_rep_scatter(df)
    fig_weekly_trend(df)
    fig_pairwise_matrix(df)
    fig_model_comparison(df)
    fig_executive_dashboard(df)

    print(f"\nAll figures saved to: {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
