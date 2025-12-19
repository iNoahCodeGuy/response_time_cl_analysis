# =============================================================================
# Weeks of Data Analyzer
# =============================================================================
# This module analyzes how many weeks of data are in an uploaded file.
#
# WHY THIS MATTERS:
# -----------------
# Response time analysis works best with 4-8 weeks of data because:
#   1. Weekly patterns: Response times vary by day (Monday vs Saturday)
#   2. Sample size: Each bucket needs enough leads for reliable statistics
#   3. Stability: More weeks = more stable close rate estimates
#
# WHAT THIS MODULE DOES:
# ----------------------
#   1. Detects date columns in the uploaded data
#   2. Calculates how many weeks the data spans
#   3. Provides recommendations based on the data period
#
# HOW TO USE:
# -----------
#   from data.weeks_analyzer import analyze_weeks_of_data
#   result = analyze_weeks_of_data(df)
#   print(f"Data covers {result['weeks']} weeks")
# =============================================================================

import pandas as pd
from typing import Dict, Any, Optional, Tuple
from datetime import datetime


# =============================================================================
# CONSTANTS - Easy to find and modify!
# =============================================================================

# These thresholds define what we consider "too few" or "too many" weeks
MIN_WEEKS_WARNING = 2      # Below this, we show a warning
MIN_WEEKS_IDEAL = 4        # This is the start of the ideal range
MAX_WEEKS_IDEAL = 12       # Above this, we suggest checking for changes

# What percentage of values must parse as dates to consider it a date column
DATE_DETECTION_THRESHOLD = 0.8  # 80%


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def analyze_weeks_of_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze how many weeks of data are in a DataFrame.
    
    HOW IT WORKS:
    -------------
    1. Look through columns to find ones that contain dates
    2. Calculate the date range (earliest to latest)
    3. Convert days to weeks
    4. Generate a recommendation based on the number of weeks
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        The uploaded data (before any preprocessing)
    
    RETURNS:
    --------
    dict with these keys:
        - 'weeks': float or None (number of weeks, e.g., 6.5)
        - 'date_range': tuple (min_date, max_date) or None
        - 'date_column': str or None (which column we used)
        - 'recommendation': dict with recommendation details
    
    EXAMPLE:
    --------
    >>> result = analyze_weeks_of_data(my_dataframe)
    >>> if result['weeks']:
    ...     print(f"Your data spans {result['weeks']} weeks")
    ...     print(result['recommendation']['message'])
    """
    
    # Step 1: Find date columns
    # -------------------------
    # We need to detect which column(s) contain dates
    date_column = _find_date_column(df)
    
    if date_column is None:
        # Could not find any date columns
        return {
            'weeks': None,
            'date_range': None,
            'date_column': None,
            'recommendation': _create_no_dates_recommendation()
        }
    
    # Step 2: Calculate date range
    # ----------------------------
    # Parse the dates and find min/max
    date_range = _calculate_date_range(df, date_column)
    
    if date_range is None:
        return {
            'weeks': None,
            'date_range': None,
            'date_column': date_column,
            'recommendation': _create_no_dates_recommendation()
        }
    
    # Step 3: Calculate weeks
    # -----------------------
    # Convert the date range to number of weeks
    min_date, max_date = date_range
    days = (max_date - min_date).days
    weeks = max(1, round(days / 7, 1))  # At least 1 week, rounded to 1 decimal
    
    # Step 4: Generate recommendation
    # --------------------------------
    recommendation = _create_recommendation(weeks)
    
    return {
        'weeks': weeks,
        'date_range': date_range,
        'date_column': date_column,
        'recommendation': recommendation
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
# These are "private" functions (start with _) that do the actual work.
# Breaking the logic into small functions makes it easier to understand and test.

def _find_date_column(df: pd.DataFrame) -> Optional[str]:
    """
    Find the first column that contains dates.
    
    HOW IT WORKS:
    -------------
    For each column, we try to parse a sample of values as dates.
    If most values parse successfully, we consider it a date column.
    
    WHY CHECK A SAMPLE?
    -------------------
    Checking every value would be slow for large files.
    100 values is enough to be confident about the column type.
    """
    
    for column_name in df.columns:
        # Get a sample of non-null values (up to 100)
        sample_values = df[column_name].dropna().head(100)
        
        if len(sample_values) == 0:
            continue  # Skip empty columns
        
        try:
            # Try to parse as dates
            # errors='coerce' means invalid dates become NaT (Not a Time)
            parsed_dates = pd.to_datetime(sample_values, errors='coerce')
            
            # Calculate what percentage parsed successfully
            success_rate = parsed_dates.notna().mean()
            
            # If enough values parsed as dates, this is probably a date column!
            if success_rate >= DATE_DETECTION_THRESHOLD:
                return column_name
                
        except Exception:
            # If parsing fails entirely, just skip this column
            continue
    
    # No date column found
    return None


def _calculate_date_range(
    df: pd.DataFrame, 
    date_column: str
) -> Optional[Tuple[datetime, datetime]]:
    """
    Calculate the min and max dates in a column.
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        The data
    date_column : str
        Name of the column containing dates
    
    RETURNS:
    --------
    Tuple of (min_date, max_date) or None if parsing fails
    """
    try:
        # Parse the entire column as dates
        dates = pd.to_datetime(df[date_column], errors='coerce')
        
        # Remove any values that didn't parse
        valid_dates = dates.dropna()
        
        if len(valid_dates) == 0:
            return None
        
        # Return the range
        return (valid_dates.min(), valid_dates.max())
        
    except Exception:
        return None


def _create_recommendation(weeks: float) -> Dict[str, Any]:
    """
    Create a recommendation based on the number of weeks.
    
    This is where we apply our business logic about what's
    "too little", "just right", or "a lot" of data.
    
    PARAMETERS:
    -----------
    weeks : float
        Number of weeks of data
    
    RETURNS:
    --------
    dict with:
        - type: 'success', 'warning', or 'info'
        - icon: emoji for visual indication
        - title: short title
        - message: detailed explanation
        - action: suggested next step (or None)
    """
    
    # Case 1: Very little data (less than 2 weeks)
    # =============================================
    if weeks < MIN_WEEKS_WARNING:
        return {
            'type': 'warning',
            'icon': '⚠️',
            'title': 'Limited Data Period',
            'message': (
                f"Your data covers only **{weeks:.1f} week(s)**. "
                f"For robust statistical analysis, we recommend **{MIN_WEEKS_IDEAL}-{MAX_WEEKS_IDEAL} weeks** "
                f"of data to account for weekly patterns and ensure sufficient sample sizes."
            ),
            'action': "Consider uploading additional weeks of data for more reliable results."
        }
    
    # Case 2: Acceptable but not ideal (2-4 weeks)
    # =============================================
    elif weeks < MIN_WEEKS_IDEAL:
        return {
            'type': 'info',
            'icon': 'ℹ️',
            'title': 'Moderate Data Period',
            'message': (
                f"Your data covers **{weeks:.1f} weeks**. "
                f"This is acceptable, but **{MIN_WEEKS_IDEAL}-{MAX_WEEKS_IDEAL} weeks** "
                f"would provide more stable estimates."
            ),
            'action': "Results should be interpreted with some caution."
        }
    
    # Case 3: Ideal range (4-12 weeks)
    # =================================
    elif weeks <= MAX_WEEKS_IDEAL:
        return {
            'type': 'success',
            'icon': '✅',
            'title': 'Good Data Period',
            'message': (
                f"Your data covers **{weeks:.1f} weeks**. "
                f"This is an ideal range for response time analysis!"
            ),
            'action': None  # No action needed - data looks good!
        }
    
    # Case 4: Very long period (more than 12 weeks)
    # ==============================================
    else:
        months = weeks / 4
        return {
            'type': 'info',
            'icon': 'ℹ️',
            'title': 'Extended Data Period',
            'message': (
                f"Your data covers **{weeks:.1f} weeks** ({months:.1f} months). "
                f"While more data is generally good, consider whether market conditions, "
                f"processes, or team composition changed significantly over this period."
            ),
            'action': "If significant changes occurred, consider analyzing more recent data separately."
        }


def _create_no_dates_recommendation() -> Dict[str, Any]:
    """
    Create a recommendation when we couldn't detect dates.
    
    This might happen if:
    - The file has no date columns
    - The dates are in an unusual format we can't parse
    - The date columns have unusual names
    """
    return {
        'type': 'info',
        'icon': 'ℹ️',
        'title': 'Could Not Detect Dates',
        'message': (
            "We couldn't automatically detect date columns in your data. "
            "The weeks-of-data recommendation requires a date column like "
            "'lead_time' or 'created_at'."
        ),
        'action': "Proceed to column mapping where you can specify your date columns."
    }


# =============================================================================
# DISPLAY GUIDANCE (what users should know about data periods)
# =============================================================================

WEEKS_GUIDANCE = """
**Optimal Data Period: 4-8 weeks**

| Weeks | Assessment | Why? |
|-------|------------|------|
| < 2 | ⚠️ Too short | May miss weekly patterns, small sample per bucket |
| 2-4 | ℹ️ Acceptable | Use with caution, limited statistical power |
| 4-8 | ✅ Ideal | Captures weekly patterns, sufficient samples |
| 8-12 | ✅ Good | More stable estimates, check for trend changes |
| > 12 | ℹ️ Consider filtering | Market/process changes may confound results |

**Why this matters for response time analysis:**

1. **Weekly patterns**: Response times vary by day (Monday vs Saturday)
2. **Sample size**: Each bucket needs ~30+ leads for reliable statistics  
3. **Stability**: More weeks = more stable close rate estimates
4. **Relevance**: Very old data may not reflect current performance
"""

