# =============================================================================
# Preprocessing Module
# =============================================================================
# This module handles data preprocessing before statistical analysis.
#
# WHY THIS MODULE EXISTS:
# -----------------------
# Raw data needs to be transformed before we can analyze it:
# 1. Calculate response time from timestamps
# 2. Create response time buckets (0-15 min, 15-30 min, etc.)
# 3. Handle missing values and outliers
# 4. Validate data quality
#
# MAIN FUNCTIONS:
# ---------------
# - calculate_response_time(): Compute time between lead and first response
# - create_response_buckets(): Categorize response times into buckets
# - preprocess_data(): Run all preprocessing steps
# =============================================================================

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DEFAULT_BUCKETS, DEFAULT_BUCKET_LABELS


def calculate_response_time(
    df: pd.DataFrame, 
    lead_time_col: str = 'lead_time',
    response_time_col: str = 'response_time'
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Calculate the time between lead arrival and first response.
    
    WHY THIS MATTERS:
    -----------------
    Response time is our key independent variable. This is what we're 
    investigating - does responding faster lead to more sales?
    
    HOW IT WORKS:
    -------------
    1. Take the first response timestamp
    2. Subtract the lead arrival timestamp
    3. Convert to minutes for easier interpretation
    4. Track any issues (negative times, missing values)
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        DataFrame containing lead and response time columns
    lead_time_col : str
        Name of the column with lead arrival timestamps
    response_time_col : str
        Name of the column with first response timestamps
        
    RETURNS:
    --------
    Tuple[pd.Series, Dict[str, Any]]
        - Series of response times in minutes
        - Dictionary with diagnostic information
        
    EXAMPLE:
    --------
    >>> response_mins, diagnostics = calculate_response_time(df)
    >>> print(f"Average response time: {response_mins.mean():.1f} minutes")
    Average response time: 23.4 minutes
    >>> print(f"Missing values: {diagnostics['n_missing']}")
    Missing values: 15
    """
    # Ensure columns exist
    if lead_time_col not in df.columns:
        raise ValueError(f"Lead time column '{lead_time_col}' not found in DataFrame")
    if response_time_col not in df.columns:
        raise ValueError(f"Response time column '{response_time_col}' not found in DataFrame")
    
    # Ensure datetime types
    lead_times = pd.to_datetime(df[lead_time_col], errors='coerce')
    response_times = pd.to_datetime(df[response_time_col], errors='coerce')
    
    # Calculate difference in minutes
    # Timedelta divided by Timedelta gives a float
    time_diff = (response_times - lead_times)
    response_mins = time_diff.dt.total_seconds() / 60
    
    # Collect diagnostics
    diagnostics = {
        'n_total': len(df),
        'n_missing': response_mins.isna().sum(),
        'n_negative': (response_mins < 0).sum(),
        'n_zero': (response_mins == 0).sum(),
        'n_valid': ((response_mins >= 0) & response_mins.notna()).sum(),
        'min_value': response_mins.min() if response_mins.notna().any() else None,
        'max_value': response_mins.max() if response_mins.notna().any() else None,
        'median_value': response_mins.median() if response_mins.notna().any() else None,
        'mean_value': response_mins.mean() if response_mins.notna().any() else None
    }
    
    # Add warnings
    diagnostics['warnings'] = []
    
    if diagnostics['n_missing'] > 0:
        pct = diagnostics['n_missing'] / diagnostics['n_total'] * 100
        diagnostics['warnings'].append(
            f"{diagnostics['n_missing']} rows ({pct:.1f}%) have missing response times"
        )
    
    if diagnostics['n_negative'] > 0:
        diagnostics['warnings'].append(
            f"{diagnostics['n_negative']} rows have negative response times "
            "(response before lead arrived?)"
        )
    
    if diagnostics['max_value'] and diagnostics['max_value'] > 24 * 60:
        n_over_day = (response_mins > 24 * 60).sum()
        diagnostics['warnings'].append(
            f"{n_over_day} responses took more than 24 hours"
        )
    
    return response_mins, diagnostics


def create_response_buckets(
    response_mins: pd.Series,
    bucket_boundaries: List[float] = None,
    bucket_labels: List[str] = None
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Categorize response times into buckets.
    
    WHY THIS MATTERS:
    -----------------
    Continuous response times are hard to compare statistically.
    Buckets let us ask: "Do leads that get a response within 15 minutes
    close at a higher rate than those who wait over an hour?"
    
    HOW IT WORKS:
    -------------
    1. Take response times in minutes
    2. Use pd.cut to assign each to a bucket
    3. Return both the bucket labels and a summary table
    
    PARAMETERS:
    -----------
    response_mins : pd.Series
        Response times in minutes
    bucket_boundaries : List[float], optional
        List of bucket boundaries (default: [0, 15, 30, 60, inf])
    bucket_labels : List[str], optional
        Labels for each bucket (default: ['0-15 min', '15-30 min', ...])
        
    RETURNS:
    --------
    Tuple[pd.Series, pd.DataFrame]
        - Series of bucket labels for each row
        - Summary DataFrame showing count per bucket
        
    EXAMPLE:
    --------
    >>> buckets, summary = create_response_buckets(response_mins)
    >>> print(summary)
           bucket  count  percentage
    0   0-15 min   4521       45.2%
    1  15-30 min   2103       21.0%
    2  30-60 min   1876       18.8%
    3    60+ min   1500       15.0%
    """
    # Use defaults if not provided
    if bucket_boundaries is None:
        bucket_boundaries = DEFAULT_BUCKETS
    if bucket_labels is None:
        bucket_labels = DEFAULT_BUCKET_LABELS
    
    # Validate inputs
    if len(bucket_labels) != len(bucket_boundaries) - 1:
        raise ValueError(
            f"Number of labels ({len(bucket_labels)}) must be one less than "
            f"number of boundaries ({len(bucket_boundaries)})"
        )
    
    # Create buckets using pd.cut
    # pd.cut assigns each value to a bucket based on boundaries
    buckets = pd.cut(
        response_mins,
        bins=bucket_boundaries,
        labels=bucket_labels,
        include_lowest=True,
        right=True  # Intervals are (left, right]
    )
    
    # Create summary table
    bucket_counts = buckets.value_counts().sort_index()
    summary = pd.DataFrame({
        'bucket': bucket_counts.index.astype(str),
        'count': bucket_counts.values,
        'percentage': (bucket_counts.values / len(response_mins) * 100).round(1)
    })
    
    # Add percentage string
    summary['percentage_str'] = summary['percentage'].apply(lambda x: f"{x:.1f}%")
    
    return buckets, summary


def handle_outliers(
    response_mins: pd.Series,
    method: str = 'cap',
    lower_percentile: float = 0,
    upper_percentile: float = 99
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Handle outliers in response time data.
    
    WHY THIS MATTERS:
    -----------------
    Outliers can skew statistics:
    - A single 1-week response time can dramatically increase the mean
    - Negative response times indicate data quality issues
    
    This function handles outliers by either capping or removing them.
    
    HOW IT WORKS:
    -------------
    1. Identify outliers based on percentiles
    2. Either cap them to the threshold or remove them
    3. Return the cleaned data with diagnostics
    
    PARAMETERS:
    -----------
    response_mins : pd.Series
        Response times in minutes
    method : str
        'cap' to cap outliers at thresholds, 'remove' to drop them
    lower_percentile : float
        Lower percentile threshold (default 0 = no lower cap)
    upper_percentile : float
        Upper percentile threshold (default 99)
        
    RETURNS:
    --------
    Tuple[pd.Series, Dict[str, Any]]
        - Cleaned response times
        - Diagnostics about what was changed
        
    EXAMPLE:
    --------
    >>> cleaned, info = handle_outliers(response_mins, method='cap')
    >>> print(f"Capped {info['n_upper_outliers']} values above {info['upper_threshold']:.0f} mins")
    Capped 50 values above 180 mins
    """
    # Calculate thresholds
    lower_threshold = response_mins.quantile(lower_percentile / 100)
    upper_threshold = response_mins.quantile(upper_percentile / 100)
    
    # Identify outliers
    n_lower = (response_mins < lower_threshold).sum()
    n_upper = (response_mins > upper_threshold).sum()
    n_negative = (response_mins < 0).sum()
    
    diagnostics = {
        'lower_threshold': lower_threshold,
        'upper_threshold': upper_threshold,
        'n_lower_outliers': n_lower,
        'n_upper_outliers': n_upper,
        'n_negative': n_negative,
        'method': method
    }
    
    # Apply method
    if method == 'cap':
        cleaned = response_mins.clip(lower=max(0, lower_threshold), upper=upper_threshold)
        diagnostics['action'] = 'Capped outliers at thresholds'
    elif method == 'remove':
        mask = (response_mins >= lower_threshold) & (response_mins <= upper_threshold) & (response_mins >= 0)
        cleaned = response_mins[mask]
        diagnostics['n_removed'] = len(response_mins) - len(cleaned)
        diagnostics['action'] = 'Removed outliers'
    else:
        raise ValueError(f"Unknown method: {method}. Use 'cap' or 'remove'.")
    
    return cleaned, diagnostics


def preprocess_data(
    df: pd.DataFrame,
    bucket_boundaries: List[float] = None,
    bucket_labels: List[str] = None,
    handle_outliers_method: Optional[str] = 'cap'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run all preprocessing steps on the data.
    
    WHY THIS MATTERS:
    -----------------
    This is the main entry point for preprocessing. It runs all steps
    in the correct order and returns a clean, analysis-ready DataFrame.
    
    HOW IT WORKS:
    -------------
    1. Calculate response times
    2. Handle outliers (if specified)
    3. Create response time buckets
    4. Add all new columns to the DataFrame
    5. Return the enhanced DataFrame with diagnostics
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        DataFrame with standardized column names (from ColumnMapper)
    bucket_boundaries : List[float], optional
        Custom bucket boundaries
    bucket_labels : List[str], optional
        Custom bucket labels
    handle_outliers_method : str, optional
        'cap', 'remove', or None to skip outlier handling
        
    RETURNS:
    --------
    Tuple[pd.DataFrame, Dict[str, Any]]
        - Preprocessed DataFrame with new columns
        - Comprehensive diagnostics dictionary
        
    EXAMPLE:
    --------
    >>> preprocessed_df, diagnostics = preprocess_data(df)
    >>> print(f"Added columns: {preprocessed_df.columns.tolist()}")
    >>> print(f"Warnings: {diagnostics['warnings']}")
    """
    # Make a copy to avoid modifying original
    result = df.copy()
    all_diagnostics = {}
    all_warnings = []
    
    # Step 1: Calculate response time
    response_mins, response_diag = calculate_response_time(result)
    all_diagnostics['response_time'] = response_diag
    all_warnings.extend(response_diag['warnings'])
    
    result['response_time_mins'] = response_mins
    
    # Step 2: Handle outliers (if specified)
    if handle_outliers_method:
        cleaned_response, outlier_diag = handle_outliers(
            response_mins, 
            method=handle_outliers_method
        )
        all_diagnostics['outliers'] = outlier_diag
        result['response_time_mins_cleaned'] = cleaned_response
        
        # Use cleaned values for bucketing
        bucket_input = cleaned_response
    else:
        bucket_input = response_mins
    
    # Step 3: Create response time buckets
    buckets, bucket_summary = create_response_buckets(
        bucket_input,
        bucket_boundaries=bucket_boundaries,
        bucket_labels=bucket_labels
    )
    
    result['response_bucket'] = buckets
    all_diagnostics['buckets'] = {
        'summary': bucket_summary.to_dict('records'),
        'n_per_bucket': bucket_summary['count'].tolist()
    }
    
    # Check for small buckets (statistical power warning)
    from config.settings import MIN_SAMPLE_SIZE_WARNING, MIN_SAMPLE_SIZE_ERROR
    
    for _, row in bucket_summary.iterrows():
        if row['count'] < MIN_SAMPLE_SIZE_ERROR:
            all_warnings.append(
                f"Bucket '{row['bucket']}' has only {row['count']} leads - "
                f"results may be unreliable"
            )
        elif row['count'] < MIN_SAMPLE_SIZE_WARNING:
            all_warnings.append(
                f"Bucket '{row['bucket']}' has {row['count']} leads - "
                f"consider using wider buckets"
            )
    
    # Step 4: Calculate some derived metrics
    result['response_time_hours'] = result['response_time_mins'] / 60
    
    # Step 5: Add data quality flags
    result['has_valid_response'] = (
        result['response_time_mins'].notna() & 
        (result['response_time_mins'] >= 0)
    )
    
    # Compile final diagnostics
    all_diagnostics['summary'] = {
        'n_rows': len(result),
        'n_with_valid_response': result['has_valid_response'].sum(),
        'pct_with_valid_response': result['has_valid_response'].mean() * 100,
        'overall_close_rate': result['ordered'].mean() * 100,
        'n_closed': result['ordered'].sum()
    }
    all_diagnostics['warnings'] = all_warnings
    
    return result, all_diagnostics


def get_preprocessing_summary(diagnostics: Dict[str, Any]) -> str:
    """
    Generate a human-readable summary of preprocessing results.
    
    WHY THIS MATTERS:
    -----------------
    Users need to understand what happened to their data.
    This function creates a plain-English summary.
    
    PARAMETERS:
    -----------
    diagnostics : Dict[str, Any]
        Diagnostics from preprocess_data()
        
    RETURNS:
    --------
    str
        Human-readable summary
    """
    summary = diagnostics.get('summary', {})
    
    text = f"""
### Data Preprocessing Complete

**Dataset Overview:**
- Total rows: {summary.get('n_rows', 'N/A'):,}
- Valid response times: {summary.get('n_with_valid_response', 'N/A'):,} ({summary.get('pct_with_valid_response', 0):.1f}%)
- Overall close rate: {summary.get('overall_close_rate', 0):.1f}%
- Total orders: {summary.get('n_closed', 'N/A'):,}

**Response Time Statistics:**
"""
    
    rt_diag = diagnostics.get('response_time', {})
    if rt_diag:
        text += f"""
- Median response time: {rt_diag.get('median_value', 0):.1f} minutes
- Mean response time: {rt_diag.get('mean_value', 0):.1f} minutes
- Range: {rt_diag.get('min_value', 0):.1f} to {rt_diag.get('max_value', 0):.1f} minutes
"""
    
    # Add warnings
    warnings = diagnostics.get('warnings', [])
    if warnings:
        text += "\n**Warnings:**\n"
        for warning in warnings:
            text += f"- ⚠️ {warning}\n"
    
    return text

