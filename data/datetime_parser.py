# =============================================================================
# DateTime Parser Module
# =============================================================================
# This module handles parsing datetime columns that may be in various formats.
#
# WHY THIS MODULE EXISTS:
# -----------------------
# Date/time data comes in many formats:
# - ISO: 2024-01-15 14:30:00
# - US: 01/15/2024 2:30 PM
# - European: 15/01/2024 14:30
# - Excel serial numbers: 45306.604166667
#
# This module auto-detects the format and converts to proper datetime objects.
#
# MAIN FUNCTIONS:
# ---------------
# - detect_datetime_format(): Guess the format from sample values
# - parse_datetime_column(): Convert a column to datetime
# =============================================================================

import pandas as pd
import numpy as np
from dateutil import parser as date_parser
from typing import Tuple, Optional, List
from datetime import datetime


# Common datetime formats we try to detect
# Listed in order of preference (most specific first)
DATETIME_FORMATS = [
    # ISO formats (most reliable)
    '%Y-%m-%d %H:%M:%S',
    '%Y-%m-%dT%H:%M:%S',
    '%Y-%m-%d %H:%M:%S.%f',
    '%Y-%m-%dT%H:%M:%S.%f',
    '%Y-%m-%d',
    
    # US formats
    '%m/%d/%Y %I:%M:%S %p',
    '%m/%d/%Y %I:%M %p',
    '%m/%d/%Y %H:%M:%S',
    '%m/%d/%Y %H:%M',
    '%m/%d/%Y',
    '%m-%d-%Y %H:%M:%S',
    '%m-%d-%Y',
    
    # European formats
    '%d/%m/%Y %H:%M:%S',
    '%d/%m/%Y %H:%M',
    '%d/%m/%Y',
    '%d-%m-%Y %H:%M:%S',
    '%d-%m-%Y',
]


def is_excel_serial_date(value) -> bool:
    """
    Check if a value looks like an Excel serial date number.
    
    WHAT IS AN EXCEL SERIAL DATE?
    -----------------------------
    Excel stores dates as numbers. For example:
    - 1 = January 1, 1900
    - 45306 = January 15, 2024
    - Decimal part represents time (0.5 = noon)
    
    PARAMETERS:
    -----------
    value : any
        A value to check
        
    RETURNS:
    --------
    bool
        True if it looks like an Excel serial date
    """
    try:
        num = float(value)
        # Excel dates are typically between 1 (1900-01-01) and 60000+ (future)
        # We check for reasonable date range
        return 1 <= num <= 100000
    except (ValueError, TypeError):
        return False


def convert_excel_serial_to_datetime(serial: float) -> datetime:
    """
    Convert an Excel serial date number to a Python datetime.
    
    HOW IT WORKS:
    -------------
    Excel's epoch starts at December 30, 1899 (not January 1, 1900 due to a 
    historical bug in Lotus 1-2-3 that Excel preserved for compatibility).
    
    We add the serial number as days to the epoch to get the date.
    The decimal portion represents the time of day.
    
    PARAMETERS:
    -----------
    serial : float
        Excel serial date number
        
    RETURNS:
    --------
    datetime
        Converted datetime object
        
    EXAMPLE:
    --------
    >>> convert_excel_serial_to_datetime(45306.604166667)
    datetime(2024, 1, 15, 14, 30, 0)
    """
    # Excel's epoch (adjusted for the leap year bug)
    excel_epoch = datetime(1899, 12, 30)
    
    # Add days to epoch
    return excel_epoch + pd.Timedelta(days=serial)


def detect_datetime_format(series: pd.Series, sample_size: int = 10) -> Tuple[str, float]:
    """
    Detect the datetime format used in a pandas Series.
    
    WHY THIS MATTERS:
    -----------------
    Knowing the format allows us to parse dates correctly and quickly.
    Auto-detection saves users from having to specify the format manually.
    
    HOW IT WORKS:
    -------------
    1. Take a sample of non-null values
    2. Try each known format on the sample
    3. Return the format that parses the most values successfully
    
    PARAMETERS:
    -----------
    series : pd.Series
        The column to analyze
    sample_size : int
        Number of values to sample for format detection
        
    RETURNS:
    --------
    Tuple[str, float]
        - Format string (or 'excel_serial' or 'auto' for dateutil)
        - Confidence score (0-1) based on parse success rate
        
    EXAMPLE:
    --------
    >>> format_str, confidence = detect_datetime_format(df['lead_time'])
    >>> print(f"Detected format: {format_str} (confidence: {confidence:.0%})")
    Detected format: %Y-%m-%d %H:%M:%S (confidence: 100%)
    """
    # Get sample of non-null values
    non_null = series.dropna()
    if len(non_null) == 0:
        return 'auto', 0.0
    
    sample = non_null.head(sample_size)
    n_samples = len(sample)
    
    # First, check if it's already datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return 'already_datetime', 1.0
    
    # Check for Excel serial dates
    excel_count = sum(1 for v in sample if is_excel_serial_date(v))
    if excel_count == n_samples:
        return 'excel_serial', 1.0
    
    # Try each format
    best_format = 'auto'
    best_score = 0.0
    
    for fmt in DATETIME_FORMATS:
        success_count = 0
        for value in sample:
            try:
                datetime.strptime(str(value), fmt)
                success_count += 1
            except ValueError:
                pass
        
        score = success_count / n_samples
        if score > best_score:
            best_score = score
            best_format = fmt
    
    # If no format worked well, fall back to auto-detection
    if best_score < 0.5:
        # Try dateutil parser on sample
        dateutil_count = 0
        for value in sample:
            try:
                date_parser.parse(str(value))
                dateutil_count += 1
            except:
                pass
        
        dateutil_score = dateutil_count / n_samples
        if dateutil_score > best_score:
            return 'auto', dateutil_score
    
    return best_format, best_score


def parse_datetime_column(
    df: pd.DataFrame, 
    column_name: str,
    format_override: Optional[str] = None
) -> Tuple[pd.Series, bool, str]:
    """
    Parse a column to datetime, handling various formats.
    
    WHY THIS MATTERS:
    -----------------
    Response time analysis requires proper datetime objects so we can
    calculate time differences. This function handles the messy reality
    of user data.
    
    HOW IT WORKS:
    -------------
    1. If format is specified, use that
    2. Otherwise, auto-detect the format
    3. Apply appropriate parsing based on detected format
    4. Return parsed column with success/failure info
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        The DataFrame containing the column
    column_name : str
        Name of the column to parse
    format_override : str, optional
        If provided, use this format instead of auto-detecting
        
    RETURNS:
    --------
    Tuple[pd.Series, bool, str]
        - Parsed datetime Series
        - Success flag (True if parsing worked)
        - Message describing what happened
        
    EXAMPLE:
    --------
    >>> parsed_col, success, msg = parse_datetime_column(df, 'lead_time')
    >>> if success:
    ...     df['lead_time'] = parsed_col
    ... else:
    ...     st.error(msg)
    """
    if column_name not in df.columns:
        return pd.Series(dtype='datetime64[ns]'), False, f"Column '{column_name}' not found"
    
    series = df[column_name]
    
    # If already datetime, return as-is
    if pd.api.types.is_datetime64_any_dtype(series):
        return series, True, "Column is already in datetime format"
    
    # Detect or use provided format
    if format_override:
        detected_format = format_override
        confidence = 1.0
    else:
        detected_format, confidence = detect_datetime_format(series)
    
    # Parse based on detected format
    try:
        if detected_format == 'already_datetime':
            return series, True, "Column is already in datetime format"
        
        elif detected_format == 'excel_serial':
            # Convert Excel serial numbers
            parsed = series.apply(
                lambda x: convert_excel_serial_to_datetime(float(x)) 
                if pd.notna(x) else pd.NaT
            )
            return parsed, True, f"Converted from Excel serial numbers (confidence: {confidence:.0%})"
        
        elif detected_format == 'auto':
            # Use dateutil for flexible parsing
            parsed = series.apply(
                lambda x: date_parser.parse(str(x)) if pd.notna(x) else pd.NaT
            )
            return parsed, True, f"Parsed using flexible date detection (confidence: {confidence:.0%})"
        
        else:
            # Use specific format string
            parsed = pd.to_datetime(series, format=detected_format, errors='coerce')
            
            # Check how many parsed successfully
            success_rate = parsed.notna().sum() / len(series)
            
            if success_rate < 0.5:
                # Fall back to auto-detection
                parsed = pd.to_datetime(series, infer_datetime_format=True, errors='coerce')
                return parsed, True, f"Used automatic date detection (some values may not have parsed)"
            
            return parsed, True, f"Parsed using format '{detected_format}' (confidence: {confidence:.0%})"
    
    except Exception as e:
        # Last resort: try pandas default parsing
        try:
            parsed = pd.to_datetime(series, errors='coerce')
            success_rate = parsed.notna().sum() / len(series)
            
            if success_rate > 0.5:
                return parsed, True, f"Parsed {success_rate:.0%} of values using automatic detection"
            else:
                return parsed, False, f"Only {success_rate:.0%} of values could be parsed as dates"
        
        except Exception as e2:
            return pd.Series(dtype='datetime64[ns]'), False, f"Failed to parse dates: {str(e2)}"


def get_datetime_summary(series: pd.Series) -> dict:
    """
    Get summary statistics for a datetime column.
    
    WHY THIS MATTERS:
    -----------------
    Helps users verify that dates parsed correctly by showing the range
    and any potential issues (like dates in the future or distant past).
    
    PARAMETERS:
    -----------
    series : pd.Series
        A datetime Series
        
    RETURNS:
    --------
    dict
        Summary statistics including min, max, null count, and warnings
    """
    if not pd.api.types.is_datetime64_any_dtype(series):
        return {'error': 'Column is not datetime type'}
    
    valid = series.dropna()
    
    summary = {
        'total_count': len(series),
        'valid_count': len(valid),
        'null_count': series.isna().sum(),
        'null_percent': series.isna().mean() * 100,
        'min_date': valid.min() if len(valid) > 0 else None,
        'max_date': valid.max() if len(valid) > 0 else None,
        'warnings': []
    }
    
    # Check for suspicious dates
    now = pd.Timestamp.now()
    
    if summary['min_date'] and summary['min_date'] < pd.Timestamp('1990-01-01'):
        summary['warnings'].append(f"Some dates are before 1990 (earliest: {summary['min_date']})")
    
    if summary['max_date'] and summary['max_date'] > now + pd.Timedelta(days=365):
        summary['warnings'].append(f"Some dates are more than a year in the future")
    
    if summary['null_percent'] > 10:
        summary['warnings'].append(f"{summary['null_percent']:.1f}% of values are missing")
    
    return summary

