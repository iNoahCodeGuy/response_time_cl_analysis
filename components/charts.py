# =============================================================================
# Charts Module
# =============================================================================
# This module creates all visualizations for the Response Time Analyzer.
#
# WHY THIS MODULE EXISTS:
# -----------------------
# Good visualizations make data understandable. Each chart here serves
# a specific purpose in telling the response time story.
#
# DESIGN PRINCIPLES:
# ------------------
# 1. Use consistent colors (green = fast = good)
# 2. Always show uncertainty (confidence intervals, error bars)
# 3. Make charts interactive (Plotly)
# 4. Include clear titles and labels
#
# MAIN FUNCTIONS:
# ---------------
# - create_close_rate_chart(): Main close rate by bucket bar chart
# - create_funnel_chart(): Lead to order conversion funnel
# - create_heatmap(): Lead source x response bucket heatmap
# - create_forest_plot(): Odds ratios from regression
# - create_rep_scatter(): Rep speed vs close rate
# =============================================================================

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional, List
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import BUCKET_COLORS, CHART_TEMPLATE, CHART_COLOR_SEQUENCE


def create_close_rate_chart(
    close_rates_df: pd.DataFrame,
    title: str = "Close Rate by Response Time",
    show_ci: bool = True
) -> go.Figure:
    """
    Create a bar chart of close rates by response time bucket.
    
    WHY THIS CHART:
    ---------------
    This is the main visual for our analysis. It answers:
    "Do leads that get faster responses have higher close rates?"
    
    DESIGN CHOICES:
    ---------------
    - Bars are colored green-to-red (fast=green, slow=red)
    - Error bars show 95% confidence intervals
    - Close rate is shown both as height and as text on bars
    
    PARAMETERS:
    -----------
    close_rates_df : pd.DataFrame
        DataFrame with columns: bucket, n_leads, close_rate, ci_lower, ci_upper
    title : str
        Chart title
    show_ci : bool
        Whether to show confidence interval error bars
        
    RETURNS:
    --------
    go.Figure
        Plotly figure object
        
    EXAMPLE:
    --------
    >>> fig = create_close_rate_chart(close_rates)
    >>> st.plotly_chart(fig)
    """
    # Prepare data
    df = close_rates_df.copy()
    df['bucket_str'] = df['bucket'].astype(str)
    df['close_rate_pct'] = df['close_rate'] * 100
    
    # Create color mapping based on bucket order
    colors = CHART_COLOR_SEQUENCE[:len(df)]
    
    # Create the figure
    fig = go.Figure()
    
    # Add bar trace
    fig.add_trace(go.Bar(
        x=df['bucket_str'],
        y=df['close_rate_pct'],
        text=df['close_rate_pct'].apply(lambda x: f'{x:.1f}%'),
        textposition='outside',
        marker_color=colors,
        name='Close Rate',
        hovertemplate=(
            '<b>%{x}</b><br>' +
            'Close Rate: %{y:.1f}%<br>' +
            'Leads: %{customdata[0]:,}<br>' +
            'Orders: %{customdata[1]:,}<extra></extra>'
        ),
        customdata=df[['n_leads', 'n_orders']].values
    ))
    
    # Add error bars for confidence intervals
    if show_ci and 'ci_lower' in df.columns and 'ci_upper' in df.columns:
        fig.update_traces(
            error_y=dict(
                type='data',
                symmetric=False,
                array=(df['ci_upper'] - df['close_rate']) * 100,
                arrayminus=(df['close_rate'] - df['ci_lower']) * 100,
                color='rgba(0,0,0,0.3)',
                thickness=2,
                width=4
            )
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=20)
        ),
        xaxis_title='Response Time Bucket',
        yaxis_title='Close Rate (%)',
        template=CHART_TEMPLATE,
        showlegend=False,
        height=400,
        yaxis=dict(
            range=[0, max(df['close_rate_pct']) * 1.3],  # Add headroom for text
            ticksuffix='%'
        )
    )
    
    return fig


def create_sample_size_chart(close_rates_df: pd.DataFrame) -> go.Figure:
    """
    Create a bar chart showing sample sizes per bucket.
    
    WHY THIS CHART:
    ---------------
    Users need to see if buckets have enough data.
    Small buckets = unreliable estimates.
    """
    df = close_rates_df.copy()
    df['bucket_str'] = df['bucket'].astype(str)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['bucket_str'],
        y=df['n_leads'],
        text=df['n_leads'].apply(lambda x: f'{x:,}'),
        textposition='outside',
        marker_color='steelblue',
        name='Number of Leads'
    ))
    
    fig.update_layout(
        title='Sample Size by Response Bucket',
        xaxis_title='Response Time Bucket',
        yaxis_title='Number of Leads',
        template=CHART_TEMPLATE,
        showlegend=False,
        height=300
    )
    
    return fig


def create_funnel_chart(
    close_rates_df: pd.DataFrame,
    title: str = "Lead Conversion Funnel by Response Time"
) -> go.Figure:
    """
    Create a funnel chart showing leads → orders by bucket.
    
    WHY THIS CHART:
    ---------------
    Funnels are intuitive for showing conversion.
    Each bucket gets its own mini-funnel.
    
    DESIGN CHOICES:
    ---------------
    - Side-by-side funnels for each bucket
    - Shows both absolute numbers and conversion rate
    """
    df = close_rates_df.copy()
    
    # Create subplots - one funnel per bucket
    n_buckets = len(df)
    
    fig = make_subplots(
        rows=1, cols=n_buckets,
        subplot_titles=df['bucket'].astype(str).tolist(),
        specs=[[{'type': 'funnel'}] * n_buckets]
    )
    
    colors = CHART_COLOR_SEQUENCE[:n_buckets]
    
    for i, (_, row) in enumerate(df.iterrows()):
        fig.add_trace(
            go.Funnel(
                y=['Leads', 'Orders'],
                x=[row['n_leads'], row['n_orders']],
                textinfo='value+percent initial',
                marker=dict(color=[colors[i], colors[i]]),
                connector=dict(line=dict(color='gray', width=1)),
                name=str(row['bucket'])
            ),
            row=1, col=i+1
        )
    
    fig.update_layout(
        title=title,
        template=CHART_TEMPLATE,
        height=350,
        showlegend=False
    )
    
    return fig


def create_heatmap(
    crosstab_df: pd.DataFrame,
    title: str = "Close Rate by Response Time and Lead Source",
    value_format: str = '.1%'
) -> go.Figure:
    """
    Create a heatmap showing close rates across two dimensions.
    
    WHY THIS CHART:
    ---------------
    Shows if response time effects vary by lead source.
    Darker colors = higher close rates.
    
    PARAMETERS:
    -----------
    crosstab_df : pd.DataFrame
        Pivot table with close rates (rows=buckets, cols=sources)
    title : str
        Chart title
    value_format : str
        Format for displayed values
    """
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=crosstab_df.values * 100,  # Convert to percentage
        x=crosstab_df.columns.tolist(),
        y=crosstab_df.index.astype(str).tolist(),
        colorscale='RdYlGn',
        text=np.round(crosstab_df.values * 100, 1),
        texttemplate='%{text:.1f}%',
        textfont=dict(size=12),
        hovertemplate=(
            'Response: %{y}<br>' +
            'Source: %{x}<br>' +
            'Close Rate: %{z:.1f}%<extra></extra>'
        ),
        colorbar=dict(
            title='Close Rate %',
            ticksuffix='%'
        )
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Lead Source',
        yaxis_title='Response Time Bucket',
        template=CHART_TEMPLATE,
        height=400
    )
    
    return fig


def create_forest_plot(
    odds_ratios: pd.DataFrame,
    reference_bucket: str = '60+ min',
    title: str = "Odds Ratios by Response Time (vs Slowest Bucket)"
) -> go.Figure:
    """
    Create a forest plot showing odds ratios from regression.
    
    WHY THIS CHART:
    ---------------
    Forest plots are the standard way to show effect sizes with uncertainty.
    Each bucket shows how many times more likely to close vs reference.
    
    DESIGN CHOICES:
    ---------------
    - Vertical line at OR=1 (no effect)
    - Points show estimate, whiskers show 95% CI
    - Color indicates direction of effect
    
    PARAMETERS:
    -----------
    odds_ratios : pd.DataFrame
        DataFrame with columns: bucket, odds_ratio, ci_lower, ci_upper
    reference_bucket : str
        The reference category (usually slowest bucket)
    """
    df = odds_ratios.copy()
    
    # Create figure
    fig = go.Figure()
    
    # Add reference line at OR=1
    fig.add_vline(
        x=1, 
        line_dash="dash", 
        line_color="gray",
        annotation_text="No Effect",
        annotation_position="top"
    )
    
    # Add points and error bars for each bucket
    for i, row in df.iterrows():
        color = 'green' if row['odds_ratio'] > 1 else 'red'
        
        fig.add_trace(go.Scatter(
            x=[row['odds_ratio']],
            y=[row['bucket']],
            mode='markers',
            marker=dict(size=12, color=color),
            error_x=dict(
                type='data',
                symmetric=False,
                array=[row['ci_upper'] - row['odds_ratio']],
                arrayminus=[row['odds_ratio'] - row['ci_lower']],
                color=color,
                thickness=2,
                width=6
            ),
            name=row['bucket'],
            hovertemplate=(
                f"<b>{row['bucket']}</b><br>" +
                f"Odds Ratio: {row['odds_ratio']:.2f}<br>" +
                f"95% CI: [{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]" +
                "<extra></extra>"
            )
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title=f'Odds Ratio (vs {reference_bucket})',
        yaxis_title='Response Time Bucket',
        template=CHART_TEMPLATE,
        height=300,
        showlegend=False,
        xaxis=dict(type='log')  # Log scale for odds ratios
    )
    
    return fig


def create_rep_scatter(
    rep_df: pd.DataFrame,
    title: str = "Sales Rep: Response Speed vs Close Rate"
) -> go.Figure:
    """
    Create a scatter plot of rep response time vs close rate.
    
    WHY THIS CHART:
    ---------------
    Shows if fast-responding reps are also high closers.
    This reveals potential confounding.
    
    INTERPRETATION:
    ---------------
    - Negative correlation: Fast reps close more (confounding likely)
    - No correlation: Speed and skill are independent
    - Positive correlation: Slow reps close more (unusual, investigate)
    
    PARAMETERS:
    -----------
    rep_df : pd.DataFrame
        DataFrame with columns: sales_rep, median_response_mins, close_rate, n_leads
    """
    df = rep_df.copy()
    
    # Calculate correlation
    corr = df['median_response_mins'].corr(df['close_rate'])
    
    # Create scatter plot
    fig = px.scatter(
        df,
        x='median_response_mins',
        y='close_rate',
        size='n_leads',
        hover_name='sales_rep',
        color='close_rate',
        color_continuous_scale='RdYlGn',
        labels={
            'median_response_mins': 'Median Response Time (min)',
            'close_rate': 'Close Rate',
            'n_leads': 'Number of Leads'
        },
        title=f"{title}<br><sup>Correlation: r = {corr:.2f}</sup>"
    )
    
    # Add trend line
    z = np.polyfit(df['median_response_mins'], df['close_rate'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['median_response_mins'].min(), df['median_response_mins'].max(), 100)
    
    fig.add_trace(go.Scatter(
        x=x_line,
        y=p(x_line),
        mode='lines',
        line=dict(color='gray', dash='dash'),
        name='Trend Line',
        showlegend=True
    ))
    
    fig.update_layout(
        template=CHART_TEMPLATE,
        height=400,
        yaxis=dict(tickformat='.0%')
    )
    
    return fig


def create_response_time_distribution(
    df: pd.DataFrame,
    title: str = "Response Time Distribution"
) -> go.Figure:
    """
    Create a histogram of response times with bucket boundaries.
    
    WHY THIS CHART:
    ---------------
    Shows the distribution of response times.
    Helps users understand how their buckets divide the data.
    """
    # Create histogram
    fig = go.Figure()
    
    # Add histogram
    response_times = df['response_time_mins'].dropna()
    
    fig.add_trace(go.Histogram(
        x=response_times,
        nbinsx=50,
        marker_color='steelblue',
        name='Response Times',
        hovertemplate='%{x:.0f} min: %{y} leads<extra></extra>'
    ))
    
    # Add vertical lines for bucket boundaries
    bucket_boundaries = [15, 30, 60]
    for bound in bucket_boundaries:
        fig.add_vline(
            x=bound,
            line_dash='dash',
            line_color='red',
            annotation_text=f'{bound} min',
            annotation_position='top'
        )
    
    fig.update_layout(
        title=title,
        xaxis_title='Response Time (minutes)',
        yaxis_title='Number of Leads',
        template=CHART_TEMPLATE,
        height=350,
        xaxis=dict(range=[0, min(response_times.quantile(0.99), 180)])
    )
    
    return fig


def create_time_series_chart(
    df: pd.DataFrame,
    date_col: str = 'lead_time',
    title: str = "Close Rate Over Time by Response Speed"
) -> go.Figure:
    """
    Create a time series of close rates by response bucket.
    
    WHY THIS CHART:
    ---------------
    Shows if the response time effect is consistent over time
    or if it varies (e.g., seasonally).
    """
    # Ensure datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Create weekly aggregation
    df['week'] = df[date_col].dt.to_period('W').dt.start_time
    
    # Calculate close rate by week and bucket
    weekly = df.groupby(['week', 'response_bucket']).agg(
        close_rate=('ordered', 'mean'),
        n_leads=('ordered', 'count')
    ).reset_index()
    
    # Create line chart
    fig = px.line(
        weekly,
        x='week',
        y='close_rate',
        color='response_bucket',
        color_discrete_sequence=CHART_COLOR_SEQUENCE,
        labels={
            'week': 'Week',
            'close_rate': 'Close Rate',
            'response_bucket': 'Response Time'
        },
        title=title
    )
    
    fig.update_layout(
        template=CHART_TEMPLATE,
        height=400,
        yaxis=dict(tickformat='.0%'),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

