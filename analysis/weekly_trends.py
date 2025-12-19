# =============================================================================
# Week-Over-Week Analysis Module
# =============================================================================
# This module analyzes how metrics change from week to week.
#
# WHY THIS MODULE EXISTS:
# -----------------------
# Business decisions need context over time. Key questions this answers:
# 1. Is our response time getting better or worse?
# 2. Are close rates improving as we respond faster?
# 3. Are there seasonal patterns we should know about?
#
# HOW TO READ THE OUTPUT:
# -----------------------
# The module returns a WeeklyAnalysis object containing:
# - weekly_stats: DataFrame with one row per week
# - trends: Dictionary of trend metrics (slope, direction, etc.)
# - insights: Plain-English summary of what's happening
#
# JUNIOR DEVELOPER TIPS:
# ----------------------
# 1. "Week" = 7-day period, starting Monday
# 2. "WoW" = Week-over-Week (comparing this week to last week)
# 3. "Trend" = overall direction over multiple weeks (improving/declining)
# =============================================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# DATA STRUCTURES
# =============================================================================
# Using dataclass makes the return value clear and well-documented

@dataclass
class WeeklyAnalysis:
    """
    Container for all week-over-week analysis results.
    
    WHAT'S INSIDE:
    --------------
    weekly_stats : DataFrame
        One row per week with all key metrics
    trends : dict
        Summary of overall trends (e.g., is response time improving?)
    insights : list
        Plain-English insights about what's happening
    weeks_analyzed : int
        Number of weeks in the analysis
    """
    weekly_stats: pd.DataFrame
    trends: Dict[str, Any]
    insights: List[str]
    weeks_analyzed: int


# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

def analyze_weekly_trends(
    df: pd.DataFrame,
    date_column: str = 'lead_time'
) -> WeeklyAnalysis:
    """
    Perform complete week-over-week analysis on the dataset.
    
    WHY THIS FUNCTION:
    ------------------
    This is the main entry point for weekly trend analysis.
    Call this function to get a full picture of how metrics
    are changing over time.
    
    HOW IT WORKS:
    -------------
    1. Groups data by week (Monday-Sunday)
    2. Calculates key metrics for each week
    3. Computes week-over-week changes
    4. Identifies overall trends
    5. Generates plain-English insights
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        Preprocessed DataFrame with at least these columns:
        - lead_time (datetime): When the lead arrived
        - ordered (bool): Whether the lead converted
        - response_time_mins (float): Response time in minutes
    date_column : str
        Name of the datetime column to use (default: 'lead_time')
        
    RETURNS:
    --------
    WeeklyAnalysis
        Object containing:
        - weekly_stats: DataFrame with metrics per week
        - trends: Summary of overall trends
        - insights: Plain-English takeaways
        - weeks_analyzed: Number of weeks
        
    EXAMPLE:
    --------
    >>> analysis = analyze_weekly_trends(df)
    >>> print(f"Analyzed {analysis.weeks_analyzed} weeks")
    >>> print(analysis.weekly_stats)
    >>> for insight in analysis.insights:
    ...     print(f"• {insight}")
    """
    # =========================================================================
    # STEP 1: Validate Input Data
    # =========================================================================
    # Make sure we have the columns we need
    
    required_columns = ['ordered']
    if date_column not in df.columns:
        raise ValueError(
            f"Date column '{date_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")
    
    # =========================================================================
    # STEP 2: Extract Week Information
    # =========================================================================
    # We'll add a 'week_start' column that groups data by week
    
    df_weekly = df.copy()
    
    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df_weekly[date_column]):
        df_weekly[date_column] = pd.to_datetime(df_weekly[date_column])
    
    # Get the Monday of each week (week_start)
    # Example: if lead_time is Wednesday Jan 15, week_start is Monday Jan 13
    df_weekly['week_start'] = df_weekly[date_column].dt.to_period('W-MON').dt.start_time
    
    # =========================================================================
    # STEP 3: Calculate Weekly Metrics
    # =========================================================================
    # Aggregate data by week to get key metrics
    
    weekly_stats = _calculate_weekly_metrics(df_weekly)
    
    # =========================================================================
    # STEP 4: Calculate Week-over-Week Changes
    # =========================================================================
    # Compare each week to the previous week
    
    weekly_stats = _add_wow_changes(weekly_stats)
    
    # =========================================================================
    # STEP 5: Calculate Overall Trends
    # =========================================================================
    # Look at the big picture: are things improving or declining?
    
    trends = _calculate_trends(weekly_stats)
    
    # =========================================================================
    # STEP 6: Generate Insights
    # =========================================================================
    # Convert numbers into plain-English takeaways
    
    insights = _generate_weekly_insights(weekly_stats, trends)
    
    # =========================================================================
    # STEP 7: Package and Return Results
    # =========================================================================
    
    return WeeklyAnalysis(
        weekly_stats=weekly_stats,
        trends=trends,
        insights=insights,
        weeks_analyzed=len(weekly_stats)
    )


def _calculate_weekly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate key metrics for each week.
    
    WHAT THIS CALCULATES:
    ---------------------
    For each week:
    - n_leads: How many leads came in
    - n_orders: How many converted
    - close_rate: Conversion rate (orders / leads)
    - median_response_mins: Typical response time
    - mean_response_mins: Average response time
    - fast_response_pct: % of leads responded to in < 15 minutes
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        DataFrame with 'week_start' column added
        
    RETURNS:
    --------
    pd.DataFrame
        One row per week with all metrics
    """
    # Group by week and calculate aggregates
    weekly = df.groupby('week_start').agg(
        n_leads=('ordered', 'count'),
        n_orders=('ordered', 'sum'),
        close_rate=('ordered', 'mean')
    ).reset_index()
    
    # Add response time metrics if available
    if 'response_time_mins' in df.columns:
        response_agg = df.groupby('week_start').agg(
            median_response_mins=('response_time_mins', 'median'),
            mean_response_mins=('response_time_mins', 'mean')
        ).reset_index()
        
        weekly = weekly.merge(response_agg, on='week_start')
        
        # Calculate fast response percentage (< 15 minutes)
        fast_pct = df.groupby('week_start').apply(
            lambda x: (x['response_time_mins'] <= 15).mean() * 100,
            include_groups=False
        ).reset_index(name='fast_response_pct')
        
        weekly = weekly.merge(fast_pct, on='week_start')
    
    # Sort by date (oldest first)
    weekly = weekly.sort_values('week_start').reset_index(drop=True)
    
    # Add week number for display (Week 1, Week 2, etc.)
    weekly['week_number'] = range(1, len(weekly) + 1)
    
    # Add formatted week label
    weekly['week_label'] = weekly['week_start'].dt.strftime('%b %d')
    
    return weekly


def _add_wow_changes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add week-over-week change columns.
    
    WHY THIS MATTERS:
    -----------------
    Week-over-week changes show momentum. Is the metric:
    - Improving this week vs last week?
    - Declining?
    - Staying stable?
    
    WHAT THIS ADDS:
    ---------------
    For each metric, adds a _change column showing:
    - Absolute change (this week - last week)
    - Percentage change (for rates)
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        Weekly metrics DataFrame
        
    RETURNS:
    --------
    pd.DataFrame
        Same DataFrame with _change columns added
    """
    result = df.copy()
    
    # Calculate changes for numeric columns
    # .shift(1) gets the previous week's value
    
    # Close rate change (percentage points)
    result['close_rate_change'] = (
        result['close_rate'] - result['close_rate'].shift(1)
    ) * 100  # Convert to percentage points
    
    # Lead volume change (percentage)
    result['n_leads_change_pct'] = (
        (result['n_leads'] - result['n_leads'].shift(1)) / 
        result['n_leads'].shift(1) * 100
    )
    
    # Response time change (if available)
    if 'median_response_mins' in result.columns:
        result['response_time_change'] = (
            result['median_response_mins'] - 
            result['median_response_mins'].shift(1)
        )
        
        # Fast response rate change
        if 'fast_response_pct' in result.columns:
            result['fast_response_change'] = (
                result['fast_response_pct'] - 
                result['fast_response_pct'].shift(1)
            )
    
    return result


def _calculate_trends(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate overall trends across all weeks.
    
    WHY THIS MATTERS:
    -----------------
    Individual week-over-week changes can be noisy.
    Trends look at the overall direction over multiple weeks.
    
    WHAT THIS CALCULATES:
    ---------------------
    For each metric:
    - direction: 'improving', 'declining', or 'stable'
    - slope: How much the metric changes per week
    - first_week_value: Starting point
    - last_week_value: Ending point
    - total_change: Overall change from start to end
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        Weekly metrics DataFrame
        
    RETURNS:
    --------
    Dict[str, Any]
        Trend information for each metric
    """
    trends = {}
    
    # Need at least 2 weeks to calculate trends
    if len(df) < 2:
        return {'error': 'Need at least 2 weeks of data for trend analysis'}
    
    # Close rate trend
    trends['close_rate'] = _get_metric_trend(
        df['close_rate'].values,
        metric_name='close_rate',
        is_percentage=True,
        higher_is_better=True
    )
    
    # Response time trend (if available)
    if 'median_response_mins' in df.columns:
        trends['response_time'] = _get_metric_trend(
            df['median_response_mins'].values,
            metric_name='response_time',
            is_percentage=False,
            higher_is_better=False  # Lower response time is better
        )
    
    # Fast response rate trend (if available)
    if 'fast_response_pct' in df.columns:
        trends['fast_response_pct'] = _get_metric_trend(
            df['fast_response_pct'].values,
            metric_name='fast_response_pct',
            is_percentage=True,
            higher_is_better=True
        )
    
    # Lead volume trend
    trends['lead_volume'] = _get_metric_trend(
        df['n_leads'].values,
        metric_name='lead_volume',
        is_percentage=False,
        higher_is_better=True  # Generally want more leads
    )
    
    return trends


def _get_metric_trend(
    values: np.ndarray,
    metric_name: str,
    is_percentage: bool,
    higher_is_better: bool
) -> Dict[str, Any]:
    """
    Calculate trend for a single metric.
    
    HOW TREND DIRECTION IS DETERMINED:
    -----------------------------------
    1. Calculate slope using linear regression
    2. If slope is positive and higher_is_better=True → 'improving'
    3. If slope is negative and higher_is_better=False → 'improving'
    4. If change is < 5% of the mean → 'stable'
    
    PARAMETERS:
    -----------
    values : np.ndarray
        Array of metric values (one per week)
    metric_name : str
        Name of the metric (for documentation)
    is_percentage : bool
        Whether the metric is a percentage (affects formatting)
    higher_is_better : bool
        Whether higher values are better (affects direction label)
        
    RETURNS:
    --------
    Dict[str, Any]
        Trend information including direction, slope, etc.
    """
    # Handle missing values
    valid_values = values[~np.isnan(values)]
    if len(valid_values) < 2:
        return {'direction': 'unknown', 'reason': 'insufficient data'}
    
    # Calculate linear regression slope
    # x = week number (0, 1, 2, ...)
    # y = metric values
    x = np.arange(len(valid_values))
    slope, intercept = np.polyfit(x, valid_values, 1)
    
    # Calculate total change
    first_value = valid_values[0]
    last_value = valid_values[-1]
    total_change = last_value - first_value
    
    # Determine if change is significant (> 5% of mean)
    mean_value = np.mean(valid_values)
    change_threshold = abs(mean_value * 0.05)  # 5% of mean
    
    if abs(total_change) < change_threshold:
        direction = 'stable'
    elif (slope > 0 and higher_is_better) or (slope < 0 and not higher_is_better):
        direction = 'improving'
    else:
        direction = 'declining'
    
    return {
        'direction': direction,
        'slope_per_week': slope,
        'first_week_value': first_value,
        'last_week_value': last_value,
        'total_change': total_change,
        'is_percentage': is_percentage,
        'higher_is_better': higher_is_better
    }


def _generate_weekly_insights(
    weekly_stats: pd.DataFrame,
    trends: Dict[str, Any]
) -> List[str]:
    """
    Generate plain-English insights from weekly analysis.
    
    WHY THIS MATTERS:
    -----------------
    Numbers tell the story, but insights make it actionable.
    This function converts statistics into sentences anyone can understand.
    
    WHAT IT GENERATES:
    ------------------
    1. Overall trend summary (improving/declining/stable)
    2. Best week highlight
    3. Recent momentum (last 2 weeks)
    4. Actionable recommendations
    
    PARAMETERS:
    -----------
    weekly_stats : pd.DataFrame
        Weekly metrics DataFrame
    trends : Dict[str, Any]
        Trend analysis results
        
    RETURNS:
    --------
    List[str]
        List of insight strings
    """
    insights = []
    
    # Handle case with too little data
    if len(weekly_stats) < 2:
        insights.append(
            "⚠️ Only 1 week of data available. "
            "Add more weeks to see trends."
        )
        return insights
    
    # =========================================================================
    # INSIGHT 1: Overall Close Rate Trend
    # =========================================================================
    if 'close_rate' in trends:
        cr_trend = trends['close_rate']
        if cr_trend['direction'] == 'improving':
            insights.append(
                f"📈 Close rate is improving! "
                f"Started at {cr_trend['first_week_value']*100:.1f}% "
                f"and now at {cr_trend['last_week_value']*100:.1f}%."
            )
        elif cr_trend['direction'] == 'declining':
            insights.append(
                f"📉 Close rate is declining. "
                f"Started at {cr_trend['first_week_value']*100:.1f}% "
                f"but now at {cr_trend['last_week_value']*100:.1f}%."
            )
        else:
            insights.append(
                f"📊 Close rate is stable around {cr_trend['last_week_value']*100:.1f}%."
            )
    
    # =========================================================================
    # INSIGHT 2: Response Time Trend
    # =========================================================================
    if 'response_time' in trends:
        rt_trend = trends['response_time']
        if rt_trend['direction'] == 'improving':
            insights.append(
                f"⚡ Response time is getting faster! "
                f"Down from {rt_trend['first_week_value']:.0f} to "
                f"{rt_trend['last_week_value']:.0f} minutes median."
            )
        elif rt_trend['direction'] == 'declining':
            insights.append(
                f"🐢 Response time is slowing down. "
                f"Was {rt_trend['first_week_value']:.0f} minutes, "
                f"now {rt_trend['last_week_value']:.0f} minutes median."
            )
    
    # =========================================================================
    # INSIGHT 3: Best Week Highlight
    # =========================================================================
    best_week_idx = weekly_stats['close_rate'].idxmax()
    best_week = weekly_stats.loc[best_week_idx]
    insights.append(
        f"🏆 Best week was {best_week['week_label']} "
        f"with {best_week['close_rate']*100:.1f}% close rate "
        f"({best_week['n_orders']:.0f} orders from {best_week['n_leads']:.0f} leads)."
    )
    
    # =========================================================================
    # INSIGHT 4: Most Recent Week Performance
    # =========================================================================
    most_recent = weekly_stats.iloc[-1]
    if 'close_rate_change' in most_recent and not pd.isna(most_recent['close_rate_change']):
        change = most_recent['close_rate_change']
        if change > 0:
            insights.append(
                f"📊 This week's close rate is up {change:.1f} percentage points "
                f"compared to last week."
            )
        elif change < 0:
            insights.append(
                f"📊 This week's close rate is down {abs(change):.1f} percentage points "
                f"compared to last week."
            )
    
    return insights


# =============================================================================
# HELPER FUNCTIONS FOR DISPLAY
# =============================================================================

def format_weekly_stats_for_display(weekly_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Format weekly stats DataFrame for nice display in the UI.
    
    WHAT THIS DOES:
    ---------------
    - Renames columns to human-friendly names
    - Formats percentages nicely
    - Adds color indicators for changes
    
    PARAMETERS:
    -----------
    weekly_stats : pd.DataFrame
        Raw weekly stats from analyze_weekly_trends()
        
    RETURNS:
    --------
    pd.DataFrame
        Formatted DataFrame ready for display
        
    EXAMPLE:
    --------
    >>> analysis = analyze_weekly_trends(df)
    >>> display_df = format_weekly_stats_for_display(analysis.weekly_stats)
    >>> st.dataframe(display_df)
    """
    display_df = weekly_stats.copy()
    
    # Format close rate as percentage
    display_df['Close Rate'] = display_df['close_rate'].apply(
        lambda x: f"{x*100:.1f}%"
    )
    
    # Format response time
    if 'median_response_mins' in display_df.columns:
        display_df['Median Response'] = display_df['median_response_mins'].apply(
            lambda x: f"{x:.0f} min"
        )
    
    # Format close rate change with arrow indicator
    if 'close_rate_change' in display_df.columns:
        def format_change(x):
            if pd.isna(x):
                return "—"
            elif x > 0:
                return f"↑ +{x:.1f}pp"
            elif x < 0:
                return f"↓ {x:.1f}pp"
            else:
                return "→ 0"
        
        display_df['WoW Change'] = display_df['close_rate_change'].apply(format_change)
    
    # Select and rename columns for display
    columns_to_show = {
        'week_label': 'Week',
        'n_leads': 'Leads',
        'n_orders': 'Orders',
        'Close Rate': 'Close Rate',
        'WoW Change': 'WoW Change'
    }
    
    if 'Median Response' in display_df.columns:
        columns_to_show['Median Response'] = 'Response Time'
    
    # Filter to only columns that exist
    columns_to_show = {k: v for k, v in columns_to_show.items() if k in display_df.columns}
    
    result = display_df[list(columns_to_show.keys())].rename(columns=columns_to_show)
    
    return result


def get_wow_comparison(weekly_stats: pd.DataFrame) -> Dict[str, Any]:
    """
    Get detailed comparison between the most recent week and the previous week.
    
    WHY THIS MATTERS:
    -----------------
    Executives often want to know "how did this week compare to last week?"
    This function provides that specific comparison.
    
    PARAMETERS:
    -----------
    weekly_stats : pd.DataFrame
        Weekly stats from analyze_weekly_trends()
        
    RETURNS:
    --------
    Dict[str, Any]
        Comparison data including:
        - this_week: Current week metrics
        - last_week: Previous week metrics
        - changes: Dictionary of changes for each metric
        
    EXAMPLE:
    --------
    >>> comparison = get_wow_comparison(analysis.weekly_stats)
    >>> print(f"Close rate: {comparison['this_week']['close_rate']:.1%}")
    >>> print(f"Change: {comparison['changes']['close_rate_pp']:.1f}pp")
    """
    if len(weekly_stats) < 2:
        return {'error': 'Need at least 2 weeks for comparison'}
    
    this_week = weekly_stats.iloc[-1]
    last_week = weekly_stats.iloc[-2]
    
    return {
        'this_week': {
            'week_label': this_week['week_label'],
            'n_leads': int(this_week['n_leads']),
            'n_orders': int(this_week['n_orders']),
            'close_rate': this_week['close_rate'],
            'median_response_mins': this_week.get('median_response_mins')
        },
        'last_week': {
            'week_label': last_week['week_label'],
            'n_leads': int(last_week['n_leads']),
            'n_orders': int(last_week['n_orders']),
            'close_rate': last_week['close_rate'],
            'median_response_mins': last_week.get('median_response_mins')
        },
        'changes': {
            'close_rate_pp': (this_week['close_rate'] - last_week['close_rate']) * 100,
            'n_leads_pct': (this_week['n_leads'] - last_week['n_leads']) / last_week['n_leads'] * 100,
            'n_orders_diff': int(this_week['n_orders']) - int(last_week['n_orders']),
            'response_time_diff': (
                this_week.get('median_response_mins', 0) - 
                last_week.get('median_response_mins', 0)
            ) if 'median_response_mins' in this_week else None
        }
    }


# =============================================================================
# FOR TESTING THIS MODULE DIRECTLY
# =============================================================================

if __name__ == "__main__":
    """
    Run this file directly to test the weekly trends analysis.
    
    Usage:
        python analysis/weekly_trends.py
    """
    import sys
    import os
    
    # Add parent directory for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data.sample_generator import generate_sample_data
    
    print("Generating sample data...")
    df = generate_sample_data(n_weeks=8)
    
    print(f"\nAnalyzing {len(df)} leads across 8 weeks...")
    analysis = analyze_weekly_trends(df)
    
    print(f"\n{'='*60}")
    print("WEEKLY STATISTICS")
    print('='*60)
    
    display_df = format_weekly_stats_for_display(analysis.weekly_stats)
    print(display_df.to_string(index=False))
    
    print(f"\n{'='*60}")
    print("TRENDS")
    print('='*60)
    
    for metric, trend in analysis.trends.items():
        if isinstance(trend, dict) and 'direction' in trend:
            print(f"  {metric}: {trend['direction']}")
    
    print(f"\n{'='*60}")
    print("INSIGHTS")
    print('='*60)
    
    for insight in analysis.insights:
        print(f"  {insight}")
    
    print(f"\n✅ Analysis complete!")

