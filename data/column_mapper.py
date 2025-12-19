# =============================================================================
# Column Mapper Module
# =============================================================================
# This module handles mapping user's column names to our expected format.
#
# WHY THIS MODULE EXISTS:
# -----------------------
# Different users have different column names for the same data:
# - One user might call it "lead_time", another "created_at"
# - One might have "sales_rep", another "agent_name"
#
# This module:
# 1. Tries to auto-detect the right columns based on common names
# 2. Lets users manually map columns if auto-detection fails
# 3. Validates that mapped columns have the right data types
#
# MAIN CLASSES:
# -------------
# - ColumnMapper: Handles all column mapping logic
# =============================================================================

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import EXPECTED_COLUMNS, POSITIVE_OUTCOME_VALUES


class ColumnMapper:
    """
    Handles mapping user columns to expected column names.
    
    WHY THIS CLASS EXISTS:
    ----------------------
    Encapsulates all the logic for:
    1. Auto-detecting columns based on common names
    2. Validating column data types
    3. Converting outcome column to boolean
    4. Storing and retrieving mappings
    
    HOW TO USE:
    -----------
    >>> mapper = ColumnMapper(df)
    >>> mapper.auto_detect()
    >>> if mapper.is_complete():
    ...     mapped_df = mapper.apply_mapping()
    ... else:
    ...     # Show UI to complete mapping
    ...     missing = mapper.get_missing_columns()
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the ColumnMapper with a DataFrame.
        
        PARAMETERS:
        -----------
        df : pd.DataFrame
            The user's data to map
        """
        self.df = df
        self.available_columns = list(df.columns)
        
        # Dictionary mapping our expected columns to user's columns
        # Example: {'lead_time': 'created_at', 'sales_rep': 'agent_name'}
        self.mapping: Dict[str, Optional[str]] = {
            col: None for col in EXPECTED_COLUMNS.keys()
        }
        
        # Store any detected outcome values that map to True/False
        self.outcome_value_mapping: Dict[Any, bool] = {}
    
    def auto_detect(self) -> Dict[str, Optional[str]]:
        """
        Attempt to automatically detect column mappings.
        
        HOW IT WORKS:
        -------------
        For each expected column, we check if any of the user's columns
        match common names. The first match wins.
        
        IMPORTANT: Once a user column is assigned to an expected column,
        it won't be assigned again. This prevents issues like 'sales_rep'
        being matched to 'ordered' because 'sale' is a substring of 'sales_rep'.
        
        RETURNS:
        --------
        Dict[str, Optional[str]]
            The current mapping (some values may still be None if not detected)
            
        EXAMPLE:
        --------
        >>> mapper = ColumnMapper(df)
        >>> detected = mapper.auto_detect()
        >>> print(detected)
        {'lead_time': 'created_at', 'response_time': None, ...}
        """
        # Track which user columns have already been assigned
        # This prevents the same column being used for multiple expected columns
        used_columns = set()
        
        for expected_col, config in EXPECTED_COLUMNS.items():
            common_names = config['common_names']
            
            # Check each available column against common names
            for user_col in self.available_columns:
                # Skip columns that have already been assigned to another expected column
                if user_col in used_columns:
                    continue
                    
                user_col_lower = user_col.lower().strip()
                
                # Check for exact or close match
                for common_name in common_names:
                    if user_col_lower == common_name.lower():
                        self.mapping[expected_col] = user_col
                        used_columns.add(user_col)
                        break
                    
                    # Also check if user column contains the common name
                    # e.g., "lead_created_date" should match "lead_created"
                    if common_name.lower() in user_col_lower:
                        self.mapping[expected_col] = user_col
                        used_columns.add(user_col)
                        break
                
                if self.mapping[expected_col] is not None:
                    break
        
        return self.mapping.copy()
    
    def set_mapping(self, expected_col: str, user_col: str) -> bool:
        """
        Manually set a column mapping.
        
        PARAMETERS:
        -----------
        expected_col : str
            One of our expected column names (e.g., 'lead_time')
        user_col : str
            The user's column name to map to
            
        RETURNS:
        --------
        bool
            True if mapping was set successfully, False if invalid
        """
        if expected_col not in EXPECTED_COLUMNS:
            return False
        
        if user_col not in self.available_columns:
            return False
        
        self.mapping[expected_col] = user_col
        return True
    
    def get_missing_columns(self) -> List[str]:
        """
        Get list of expected columns that haven't been mapped yet.
        
        RETURNS:
        --------
        List[str]
            Names of columns that still need to be mapped
            
        EXAMPLE:
        --------
        >>> missing = mapper.get_missing_columns()
        >>> print(f"Still need to map: {missing}")
        Still need to map: ['response_time', 'ordered']
        """
        missing = []
        for expected_col, config in EXPECTED_COLUMNS.items():
            if config['required'] and self.mapping[expected_col] is None:
                missing.append(expected_col)
        return missing
    
    def is_complete(self) -> bool:
        """
        Check if all required columns have been mapped.
        
        RETURNS:
        --------
        bool
            True if all required columns are mapped
        """
        return len(self.get_missing_columns()) == 0
    
    def get_mapping_status(self) -> Dict[str, Dict]:
        """
        Get detailed status of all column mappings.
        
        WHY THIS MATTERS:
        -----------------
        Used by the UI to show users which columns are mapped,
        which are missing, and what each column means.
        
        RETURNS:
        --------
        Dict[str, Dict]
            Status info for each expected column
        """
        status = {}
        
        for expected_col, config in EXPECTED_COLUMNS.items():
            mapped_to = self.mapping[expected_col]
            
            status[expected_col] = {
                'description': config['description'],
                'type': config['type'],
                'required': config['required'],
                'mapped_to': mapped_to,
                'is_mapped': mapped_to is not None
            }
            
            # If mapped, include sample values
            if mapped_to is not None:
                sample = self.df[mapped_to].dropna().head(3).tolist()
                status[expected_col]['sample_values'] = sample
        
        return status
    
    def detect_outcome_values(self) -> Dict[Any, bool]:
        """
        Detect what values in the outcome column represent True/False.
        
        WHY THIS MATTERS:
        -----------------
        The 'ordered' column might contain:
        - Boolean: True/False
        - Integer: 1/0
        - String: 'Yes'/'No', 'Ordered'/'Lost', 'Won'/'Lost'
        
        We need to figure out which values mean "customer ordered"
        
        RETURNS:
        --------
        Dict[Any, bool]
            Mapping from original values to True/False
            
        EXAMPLE:
        --------
        >>> outcome_map = mapper.detect_outcome_values()
        >>> print(outcome_map)
        {'Won': True, 'Lost': False, 'Pending': False}
        """
        ordered_col = self.mapping.get('ordered')
        
        if ordered_col is None:
            return {}
        
        unique_values = self.df[ordered_col].dropna().unique()
        
        # Check each unique value against our known positive values
        value_mapping = {}
        
        for val in unique_values:
            # Convert numpy types to Python primitives for consistent dict keys
            # This ensures dict lookup works correctly during apply_mapping
            if hasattr(val, 'item'):
                key = val.item()  # Convert numpy scalar to Python primitive
            else:
                key = val
            
            # Check if this value indicates a positive outcome
            # First try with the original value (for string matching)
            is_positive = val in POSITIVE_OUTCOME_VALUES
            
            # Also check with the converted key (for numeric types)
            if not is_positive and key != val:
                is_positive = key in POSITIVE_OUTCOME_VALUES
            
            # Also check string representations
            if not is_positive and isinstance(val, str):
                is_positive = val.lower().strip() in [
                    str(v).lower() for v in POSITIVE_OUTCOME_VALUES
                ]
            
            value_mapping[key] = is_positive
        
        self.outcome_value_mapping = value_mapping
        return value_mapping
    
    def apply_mapping(self) -> pd.DataFrame:
        """
        Apply the column mapping to create a standardized DataFrame.
        
        WHY THIS MATTERS:
        -----------------
        The rest of the analysis expects columns with specific names.
        This function creates a new DataFrame with our standard names.
        
        RETURNS:
        --------
        pd.DataFrame
            New DataFrame with standardized column names and types
            
        RAISES:
        -------
        ValueError
            If required columns are not mapped
            
        EXAMPLE:
        --------
        >>> mapper.auto_detect()
        >>> if mapper.is_complete():
        ...     standardized_df = mapper.apply_mapping()
        ...     print(standardized_df.columns)
        Index(['lead_time', 'response_time', 'lead_source', 'sales_rep', 'ordered'])
        """
        if not self.is_complete():
            missing = self.get_missing_columns()
            raise ValueError(f"Missing required columns: {missing}")
        
        # Create new DataFrame with our standard column names
        result = pd.DataFrame()
        
        for expected_col, user_col in self.mapping.items():
            if user_col is not None:
                result[expected_col] = self.df[user_col].copy()
        
        # Convert outcome column to boolean
        if 'ordered' in result.columns:
            # Check if already boolean - if so, just ensure it's proper boolean dtype
            if result['ordered'].dtype == bool:
                result['ordered'] = result['ordered'].astype(bool)
            else:
                # Check if it's already numeric 0/1 - if so, convert directly
                unique_vals = result['ordered'].dropna().unique()
                numeric_vals = [v for v in unique_vals if isinstance(v, (int, float, np.integer, np.floating))]
                
                if len(numeric_vals) == len(unique_vals) and set(numeric_vals) <= {0, 1}:
                    # Already 0/1 numeric values - convert directly to boolean
                    result['ordered'] = (result['ordered'] != 0).astype(bool)
                else:
                    # Need mapping - detect outcome values if not already done
                    if not self.outcome_value_mapping:
                        self.detect_outcome_values()
                    
                    # Map to boolean - convert numpy types to Python primitives for lookup
                    def get_outcome_value(x):
                        # Convert numpy scalar to Python primitive for consistent lookup
                        if pd.isna(x):
                            return False
                        if hasattr(x, 'item'):
                            lookup_key = x.item()  # numpy.int64 -> int, numpy.float64 -> float
                        else:
                            lookup_key = x
                        
                        # Try lookup with converted key
                        return self.outcome_value_mapping.get(lookup_key, False)
                    
                    result['ordered'] = result['ordered'].map(get_outcome_value).astype(bool)
        
        return result
    
    def validate_mapping(self) -> Tuple[bool, List[str]]:
        """
        Validate that mapped columns have appropriate data types.
        
        WHY THIS MATTERS:
        -----------------
        Even if columns are mapped, they might have wrong data types:
        - Datetime columns that are actually strings
        - Numeric columns that are actually text
        
        RETURNS:
        --------
        Tuple[bool, List[str]]
            - True if all validations pass, False otherwise
            - List of validation error messages
        """
        errors = []
        
        for expected_col, config in EXPECTED_COLUMNS.items():
            user_col = self.mapping[expected_col]
            
            if user_col is None:
                continue
            
            expected_type = config['type']
            series = self.df[user_col]
            
            # Check datetime columns
            if expected_type == 'datetime':
                if not pd.api.types.is_datetime64_any_dtype(series):
                    # Check if it can be parsed as datetime
                    try:
                        pd.to_datetime(series.head(10))
                    except:
                        errors.append(
                            f"Column '{user_col}' (mapped to {expected_col}) "
                            f"doesn't appear to contain dates. Sample: {series.head(1).tolist()}"
                        )
            
            # Check boolean columns
            elif expected_type == 'boolean':
                unique_count = series.nunique()
                if unique_count > 10:
                    errors.append(
                        f"Column '{user_col}' (mapped to {expected_col}) has {unique_count} "
                        f"unique values. Expected a boolean-like column with few unique values."
                    )
        
        return len(errors) == 0, errors


def suggest_column_mapping(
    df: pd.DataFrame, 
    expected_col: str
) -> List[Tuple[str, float]]:
    """
    Suggest possible mappings for an expected column, ranked by confidence.
    
    WHY THIS MATTERS:
    -----------------
    When auto-detection fails, we can still help users by ranking
    the available columns by how likely they are to be the right match.
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        The user's data
    expected_col : str
        The expected column we're trying to map (e.g., 'lead_time')
        
    RETURNS:
    --------
    List[Tuple[str, float]]
        List of (column_name, confidence_score) tuples, sorted by score
        
    EXAMPLE:
    --------
    >>> suggestions = suggest_column_mapping(df, 'lead_time')
    >>> for col, score in suggestions[:3]:
    ...     print(f"{col}: {score:.0%} confidence")
    created_at: 80% confidence
    timestamp: 60% confidence
    date: 40% confidence
    """
    if expected_col not in EXPECTED_COLUMNS:
        return []
    
    config = EXPECTED_COLUMNS[expected_col]
    common_names = [n.lower() for n in config['common_names']]
    expected_type = config['type']
    
    suggestions = []
    
    for col in df.columns:
        col_lower = col.lower().strip()
        score = 0.0
        
        # Check name similarity
        if col_lower in common_names:
            score += 0.6
        else:
            # Partial match
            for common in common_names:
                if common in col_lower or col_lower in common:
                    score += 0.3
                    break
        
        # Check data type compatibility
        series = df[col]
        
        if expected_type == 'datetime':
            if pd.api.types.is_datetime64_any_dtype(series):
                score += 0.4
            else:
                # Try to parse
                try:
                    pd.to_datetime(series.head(5))
                    score += 0.2
                except:
                    pass
        
        elif expected_type == 'categorical':
            n_unique = series.nunique()
            n_total = len(series)
            # Good ratio of unique to total for categorical
            if n_unique < n_total * 0.1:  # Less than 10% unique
                score += 0.2
        
        elif expected_type == 'boolean':
            if series.nunique() <= 5:  # Few unique values
                score += 0.3
        
        if score > 0:
            suggestions.append((col, score))
    
    # Sort by score descending
    suggestions.sort(key=lambda x: x[1], reverse=True)
    
    return suggestions

