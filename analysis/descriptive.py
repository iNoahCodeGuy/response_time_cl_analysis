# =============================================================================
# Descriptive Statistics Module
# =============================================================================
# This module calculates summary statistics for the lead data.
#
# WHY THIS MODULE EXISTS:
# -----------------------
# Before running statistical tests, we need to understand our data:
# - How many leads are in each response time bucket?
# - What's the close rate for each bucket?
# - How do close rates vary by lead source and sales rep?
#
# These descriptive stats tell the story and set up the hypothesis tests.
#
# MAIN FUNCTIONS:
# ---------------
# - calculate_summary_stats(): Overall data summary
# - calculate_close_rates(): Close rates by response bucket
# - calculate_cross_tabulation(): Lead source x response bucket analysis
# =============================================================================

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from scipy import stats


def calculate_summary_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate overall summary statistics for the dataset.
    
    WHY THIS MATTERS:
    -----------------
    Summary stats give us the lay of the land. Before we dig into
    response time effects, we need to know:
    - How big is our dataset?
    - What's our overall close rate?
    - How long does it typically take to respond?
    
    WHAT WE CALCULATE:
    ------------------
    1. Dataset size (total leads, total orders)
    2. Overall close rate
    3. Response time distribution (mean, median, percentiles)
    4. Breakdowns by lead source and sales rep
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        Preprocessed DataFrame with response_bucket and ordered columns
        
    RETURNS:
    --------
    Dict[str, Any]
        Comprehensive summary statistics
        
    EXAMPLE:
    --------
    >>> summary = calculate_summary_stats(df)
    >>> print(f"Total leads: {summary['n_leads']:,}")
    >>> print(f"Close rate: {summary['overall_close_rate']:.1%}")
    """
    summary = {}
    
    # =========================================================================
    # SECTION 1: Basic counts
    # =========================================================================
    summary['n_leads'] = len(df)
    summary['n_orders'] = df['ordered'].sum()
    summary['overall_close_rate'] = df['ordered'].mean()
    
    # =========================================================================
    # SECTION 2: Response time statistics
    # =========================================================================
    # Get response time in minutes (use cleaned if available)
    if 'response_time_mins_cleaned' in df.columns:
        response_times = df['response_time_mins_cleaned']
    else:
        response_times = df['response_time_mins']
    
    valid_response = response_times.dropna()
    
    summary['response_time'] = {
        'mean': valid_response.mean(),
        'median': valid_response.median(),
        'std': valid_response.std(),
        'min': valid_response.min(),
        'max': valid_response.max(),
        'p25': valid_response.quantile(0.25),
        'p75': valid_response.quantile(0.75),
        'p90': valid_response.quantile(0.90),
        'p95': valid_response.quantile(0.95)
    }
    
    # =========================================================================
    # SECTION 3: Response bucket distribution
    # =========================================================================
    bucket_counts = df['response_bucket'].value_counts().sort_index()
    summary['bucket_distribution'] = {
        'counts': bucket_counts.to_dict(),
        'percentages': (bucket_counts / len(df) * 100).to_dict()
    }
    
    # =========================================================================
    # SECTION 4: Lead source breakdown
    # =========================================================================
    if 'lead_source' in df.columns:
        source_stats = df.groupby('lead_source').agg(
            n_leads=('ordered', 'count'),
            n_orders=('ordered', 'sum'),
            close_rate=('ordered', 'mean')
        ).sort_values('n_leads', ascending=False)
        
        summary['by_lead_source'] = source_stats.to_dict('index')
    
    # =========================================================================
    # SECTION 5: Sales rep breakdown
    # =========================================================================
    if 'sales_rep' in df.columns:
        rep_stats = df.groupby('sales_rep').agg(
            n_leads=('ordered', 'count'),
            n_orders=('ordered', 'sum'),
            close_rate=('ordered', 'mean'),
            median_response_mins=('response_time_mins', 'median')
        ).sort_values('n_leads', ascending=False)
        
        summary['by_sales_rep'] = rep_stats.to_dict('index')
        summary['n_reps'] = df['sales_rep'].nunique()
    
    return summary


def calculate_close_rates(
    df: pd.DataFrame,
    include_confidence_intervals: bool = True,
    confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    Calculate close rates by response time bucket with confidence intervals.
    
    WHY THIS MATTERS:
    -----------------
    This is the heart of our analysis. We want to know:
    "Do leads that get fast responses close at higher rates?"
    
    WHAT WE CALCULATE:
    ------------------
    For each response bucket:
    - Number of leads
    - Number of orders (closes)
    - Close rate (orders / leads)
    - 95% confidence interval for the close rate
    
    HOW CONFIDENCE INTERVALS WORK:
    ------------------------------
    The confidence interval tells us the range where the "true" close rate
    likely falls. A 95% CI means: if we repeated this analysis many times,
    95% of the CIs would contain the true rate.
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        Preprocessed DataFrame
    include_confidence_intervals : bool
        Whether to calculate CIs (slightly slower)
    confidence_level : float
        Confidence level for intervals (default 0.95 = 95%)
        
    RETURNS:
    --------
    pd.DataFrame
        Close rates by bucket with columns:
        - bucket: Response time bucket
        - n_leads: Number of leads
        - n_orders: Number of orders
        - close_rate: Close rate (0-1)
        - close_rate_pct: Close rate as percentage string
        - ci_lower: Lower bound of CI
        - ci_upper: Upper bound of CI
        
    EXAMPLE:
    --------
    >>> close_rates = calculate_close_rates(df)
    >>> print(close_rates)
           bucket  n_leads  n_orders  close_rate close_rate_pct  ci_lower  ci_upper
    0   0-15 min     4521       452      0.100           10.0%     0.091     0.109
    1  15-30 min     2103       168      0.080            8.0%     0.069     0.092
    """
    # Group by response bucket
    grouped = df.groupby('response_bucket', observed=True).agg(
        n_leads=('ordered', 'count'),
        n_orders=('ordered', 'sum')
    ).reset_index()
    
    # Rename bucket column
    grouped = grouped.rename(columns={'response_bucket': 'bucket'})
    
    # Calculate close rate
    grouped['close_rate'] = grouped['n_orders'] / grouped['n_leads']
    grouped['close_rate_pct'] = grouped['close_rate'].apply(lambda x: f"{x*100:.1f}%")
    
    # Calculate confidence intervals using Wilson score interval
    # This is more accurate than normal approximation for proportions
    if include_confidence_intervals:
        z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        
        ci_lower = []
        ci_upper = []
        
        for _, row in grouped.iterrows():
            n = row['n_leads']
            p = row['close_rate']
            
            # Wilson score interval formula
            denominator = 1 + z**2 / n
            center = (p + z**2 / (2 * n)) / denominator
            spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
            
            ci_lower.append(max(0, center - spread))
            ci_upper.append(min(1, center + spread))
        
        grouped['ci_lower'] = ci_lower
        grouped['ci_upper'] = ci_upper
        grouped['ci_lower_pct'] = grouped['ci_lower'].apply(lambda x: f"{x*100:.1f}%")
        grouped['ci_upper_pct'] = grouped['ci_upper'].apply(lambda x: f"{x*100:.1f}%")
    
    return grouped


def calculate_cross_tabulation(
    df: pd.DataFrame,
    row_var: str = 'response_bucket',
    col_var: str = 'lead_source'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate cross-tabulation of two categorical variables.
    
    WHY THIS MATTERS:
    -----------------
    We need to check if response time effects vary by lead source.
    For example:
    - Does fast response matter more for website leads vs referrals?
    - Are some lead sources systematically slower to respond to?
    
    WHAT WE CALCULATE:
    ------------------
    1. Count matrix: How many leads in each cell
    2. Close rate matrix: Close rate in each cell
    3. Response time matrix: Median response time in each cell
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        Preprocessed DataFrame
    row_var : str
        Variable for rows (default: response_bucket)
    col_var : str
        Variable for columns (default: lead_source)
        
    RETURNS:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        - Count matrix
        - Close rate matrix
        - Response time matrix
        
    EXAMPLE:
    --------
    >>> counts, rates, times = calculate_cross_tabulation(df)
    >>> print(rates)
    lead_source    Dealer Referral  Website Form  Phone Call
    response_bucket
    0-15 min                 0.18          0.08        0.14
    15-30 min                0.15          0.06        0.11
    """
    # Count matrix
    count_matrix = pd.crosstab(
        df[row_var],
        df[col_var],
        margins=True,
        margins_name='Total'
    )
    
    # Close rate matrix
    # Create a pivot table with close rate
    close_rate_matrix = df.pivot_table(
        values='ordered',
        index=row_var,
        columns=col_var,
        aggfunc='mean'
    )
    
    # Response time matrix (if looking at lead source as column)
    if 'response_time_mins' in df.columns:
        response_time_matrix = df.pivot_table(
            values='response_time_mins',
            index=row_var if row_var != 'response_bucket' else col_var,
            columns=col_var if row_var != 'response_bucket' else row_var,
            aggfunc='median'
        )
    else:
        response_time_matrix = pd.DataFrame()
    
    return count_matrix, close_rate_matrix, response_time_matrix


def calculate_rep_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate performance metrics for each sales rep.
    
    WHY THIS MATTERS:
    -----------------
    We need to understand if response time effects are confounded
    by rep performance. Key questions:
    - Are fast responders also high closers?
    - Is there correlation between speed and skill?
    
    WHAT WE CALCULATE:
    ------------------
    For each rep:
    - Number of leads handled
    - Close rate
    - Median response time
    - Fast response rate (% under 15 mins)
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        Preprocessed DataFrame
        
    RETURNS:
    --------
    pd.DataFrame
        Rep performance metrics
    """
    if 'sales_rep' not in df.columns:
        return pd.DataFrame()
    
    # Calculate per-rep metrics
    rep_stats = df.groupby('sales_rep').agg(
        n_leads=('ordered', 'count'),
        n_orders=('ordered', 'sum'),
        close_rate=('ordered', 'mean'),
        median_response_mins=('response_time_mins', 'median'),
        mean_response_mins=('response_time_mins', 'mean')
    ).reset_index()
    
    # Calculate fast response rate (% under 15 mins)
    fast_response = df.groupby('sales_rep').apply(
        lambda x: (x['response_time_mins'] <= 15).mean()
    ).reset_index(name='fast_response_rate')
    
    rep_stats = rep_stats.merge(fast_response, on='sales_rep')
    
    # Sort by close rate
    rep_stats = rep_stats.sort_values('close_rate', ascending=False)
    
    # Calculate correlation between speed and close rate
    corr = rep_stats['median_response_mins'].corr(rep_stats['close_rate'])
    rep_stats.attrs['speed_close_correlation'] = corr
    
    return rep_stats


def generate_descriptive_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a complete descriptive analysis package.
    
    WHY THIS MATTERS:
    -----------------
    This is the main entry point for descriptive analysis.
    It runs all descriptive functions and packages results together.
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        Preprocessed DataFrame
        
    RETURNS:
    --------
    Dict[str, Any]
        Complete descriptive analysis results
    """
    results = {}
    
    # Overall summary
    results['summary_stats'] = calculate_summary_stats(df)
    
    # Close rates by bucket
    results['close_rates'] = calculate_close_rates(df)
    
    # Cross-tabulation
    counts, rates, times = calculate_cross_tabulation(df)
    results['crosstab'] = {
        'counts': counts,
        'close_rates': rates,
        'response_times': times
    }
    
    # Rep performance
    results['rep_performance'] = calculate_rep_performance(df)
    
    # Key insights
    results['insights'] = generate_insights(results)
    
    return results


def generate_insights(results: Dict[str, Any]) -> List[str]:
    """
    Generate plain-English insights from descriptive statistics.
    
    WHY THIS MATTERS:
    -----------------
    Raw numbers don't tell a story. This function converts
    statistics into actionable insights.
    
    PARAMETERS:
    -----------
    results : Dict[str, Any]
        Results from descriptive analysis
        
    RETURNS:
    --------
    List[str]
        List of insight statements
    """
    insights = []
    
    # Insight 1: Compare fastest vs slowest bucket
    close_rates = results.get('close_rates')
    if close_rates is not None and len(close_rates) > 1:
        fastest = close_rates.iloc[0]
        slowest = close_rates.iloc[-1]
        
        rate_diff = fastest['close_rate'] - slowest['close_rate']
        relative_diff = rate_diff / slowest['close_rate'] * 100 if slowest['close_rate'] > 0 else 0
        
        insights.append(
            f"Leads in the fastest response bucket ({fastest['bucket']}) have a "
            f"{fastest['close_rate']*100:.1f}% close rate, compared to "
            f"{slowest['close_rate']*100:.1f}% for the slowest bucket ({slowest['bucket']}). "
            f"That's a {abs(rate_diff)*100:.1f} percentage point difference "
            f"({abs(relative_diff):.0f}% {'higher' if rate_diff > 0 else 'lower'})."
        )
    
    # Insight 2: Rep correlation
    rep_perf = results.get('rep_performance')
    if rep_perf is not None and len(rep_perf) > 0:
        corr = rep_perf.attrs.get('speed_close_correlation', None)
        if corr is not None:
            if corr < -0.3:
                insights.append(
                    f"There's a negative correlation (r={corr:.2f}) between response time "
                    f"and close rate across reps. Faster reps tend to close more deals. "
                    f"This could indicate confounding - we need to control for rep skill."
                )
            elif corr > 0.3:
                insights.append(
                    f"Interestingly, there's a positive correlation (r={corr:.2f}) between "
                    f"response time and close rate. Slower responding reps close more. "
                    f"This warrants deeper investigation."
                )
    
    # Insight 3: Sample size adequacy
    summary = results.get('summary_stats', {})
    n_leads = summary.get('n_leads', 0)
    
    if n_leads < 1000:
        insights.append(
            f"With {n_leads:,} leads, the sample size is relatively small. "
            f"Consider collecting more data for robust conclusions."
        )
    elif n_leads > 10000:
        insights.append(
            f"With {n_leads:,} leads, you have strong statistical power to detect "
            f"even small differences in close rates."
        )
    
    return insights

