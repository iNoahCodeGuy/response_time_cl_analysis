# =============================================================================
# Data Loader Module
# =============================================================================
# This module handles loading data files (CSV and Excel) into pandas DataFrames.
#
# WHY THIS MODULE EXISTS:
# -----------------------
# Users may upload data in different formats. This module provides a unified
# interface for loading data regardless of file type, with proper validation
# and error handling.
#
# MAIN FUNCTIONS:
# ---------------
# - load_file(): Load a CSV or Excel file into a DataFrame
# - validate_file(): Check if file meets our requirements
# =============================================================================

import pandas as pd
import streamlit as st
from typing import Tuple, Optional, List
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import MAX_FILE_SIZE_MB, ALLOWED_FILE_TYPES, PREVIEW_ROWS


def validate_file(uploaded_file) -> Tuple[bool, str]:
    """
    Validate an uploaded file before processing.
    
    WHY THIS MATTERS:
    -----------------
    We need to catch problems early before attempting to load large files.
    This saves time and provides clear error messages to users.
    
    WHAT WE CHECK:
    --------------
    1. File is not None (something was actually uploaded)
    2. File extension is allowed (CSV or Excel)
    3. File size is within limits
    
    PARAMETERS:
    -----------
    uploaded_file : streamlit.UploadedFile
        The file object from st.file_uploader()
        
    RETURNS:
    --------
    Tuple[bool, str]
        - First element: True if valid, False if invalid
        - Second element: Error message if invalid, empty string if valid
        
    EXAMPLE:
    --------
    >>> is_valid, error_msg = validate_file(uploaded_file)
    >>> if not is_valid:
    ...     st.error(error_msg)
    """
    # Check 1: File exists
    if uploaded_file is None:
        return False, "No file uploaded. Please upload a CSV or Excel file."
    
    # Check 2: File extension is allowed
    file_name = uploaded_file.name.lower()
    file_extension = file_name.split('.')[-1]
    
    if file_extension not in ALLOWED_FILE_TYPES:
        allowed_str = ', '.join(ALLOWED_FILE_TYPES)
        return False, f"Invalid file type '.{file_extension}'. Allowed types: {allowed_str}"
    
    # Check 3: File size is within limits
    # uploaded_file.size is in bytes, convert to MB
    file_size_mb = uploaded_file.size / (1024 * 1024)
    
    if file_size_mb > MAX_FILE_SIZE_MB:
        return False, f"File too large ({file_size_mb:.1f} MB). Maximum size: {MAX_FILE_SIZE_MB} MB"
    
    # All checks passed
    return True, ""


def load_file(uploaded_file) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Load a CSV or Excel file into a pandas DataFrame.
    
    WHY THIS MATTERS:
    -----------------
    This is the entry point for user data. We need to handle different file
    formats gracefully and provide helpful error messages when things go wrong.
    
    HOW IT WORKS:
    -------------
    1. Validate the file first (type, size)
    2. Detect file type from extension
    3. Use appropriate pandas reader
    4. Return DataFrame or error message
    
    PARAMETERS:
    -----------
    uploaded_file : streamlit.UploadedFile
        The file object from st.file_uploader()
        
    RETURNS:
    --------
    Tuple[Optional[pd.DataFrame], str]
        - First element: DataFrame if successful, None if failed
        - Second element: Error message if failed, empty string if successful
        
    EXAMPLE:
    --------
    >>> df, error = load_file(uploaded_file)
    >>> if df is not None:
    ...     st.write(f"Loaded {len(df)} rows")
    ... else:
    ...     st.error(error)
    """
    # Step 1: Validate the file
    is_valid, error_msg = validate_file(uploaded_file)
    if not is_valid:
        return None, error_msg
    
    # Step 2: Determine file type
    file_name = uploaded_file.name.lower()
    is_excel = file_name.endswith('.xlsx') or file_name.endswith('.xls')
    
    # Step 3: Load the file
    try:
        if is_excel:
            # For Excel files, we use openpyxl engine
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        else:
            # For CSV files, try to auto-detect encoding
            df = pd.read_csv(uploaded_file)
        
        # Step 4: Basic validation of loaded data
        if df.empty:
            return None, "The file is empty. Please upload a file with data."
        
        if len(df.columns) < 3:
            return None, "The file has fewer than 3 columns. Please check the file format."
        
        return df, ""
        
    except pd.errors.EmptyDataError:
        return None, "The file is empty or has no parseable data."
    
    except pd.errors.ParserError as e:
        return None, f"Error parsing file: {str(e)}. Please check the file format."
    
    except Exception as e:
        return None, f"Unexpected error loading file: {str(e)}"


def get_column_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary information about each column in the DataFrame.
    
    WHY THIS MATTERS:
    -----------------
    Helps users understand their data and choose the right column mappings.
    Shows data types, missing values, and sample values.
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        The loaded data
        
    RETURNS:
    --------
    pd.DataFrame
        Summary table with column info
        
    EXAMPLE:
    --------
    >>> info = get_column_info(df)
    >>> st.dataframe(info)
    """
    info_data = []
    
    for col in df.columns:
        # Get sample values (first 3 non-null values)
        non_null = df[col].dropna()
        sample_values = non_null.head(3).tolist() if len(non_null) > 0 else []
        sample_str = ', '.join(str(v)[:20] for v in sample_values)
        
        info_data.append({
            'Column': col,
            'Type': str(df[col].dtype),
            'Non-Null Count': df[col].notna().sum(),
            'Missing %': f"{df[col].isna().mean() * 100:.1f}%",
            'Sample Values': sample_str
        })
    
    return pd.DataFrame(info_data)


def preview_data(df: pd.DataFrame, n_rows: int = PREVIEW_ROWS) -> pd.DataFrame:
    """
    Get a preview of the data for display.
    
    WHY THIS MATTERS:
    -----------------
    Users need to see their data before proceeding to confirm it loaded correctly.
    We show a small preview to avoid overwhelming the UI.
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        The full dataset
    n_rows : int
        Number of rows to show (default from settings)
        
    RETURNS:
    --------
    pd.DataFrame
        First n_rows of data
    """
    return df.head(n_rows)

