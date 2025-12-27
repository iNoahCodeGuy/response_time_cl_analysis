# =============================================================================
# Column Mapping UI Component
# =============================================================================
# This component creates the interface for mapping user columns to expected format.
#
# WHY THIS COMPONENT EXISTS:
# --------------------------
# Users have different column names. We need them to tell us which of their
# columns corresponds to each piece of data we need.
#
# This component:
# 1. Attempts auto-detection first
# 2. Shows what was detected
# 3. Lets users correct any mistakes
# 4. Validates the mapping before proceeding
#
# MAIN FUNCTIONS:
# ---------------
# - render_column_mapping(): Main mapping interface
# - render_outcome_value_mapping(): Map outcome values to True/False
# =============================================================================

import streamlit as st
import pandas as pd
from typing import Optional, Dict, Any, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.column_mapper import ColumnMapper, suggest_column_mapping
from data.datetime_parser import parse_datetime_column, get_datetime_summary
from config.settings import EXPECTED_COLUMNS


def render_column_mapping(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], bool]:
    """
    Render the column mapping interface.
    
    WHY THIS FUNCTION:
    ------------------
    Guides users through mapping their columns to our expected format.
    Uses auto-detection as a starting point, then lets users adjust.
    
    WHAT IT DISPLAYS:
    -----------------
    1. Auto-detection results
    2. Dropdown for each expected column
    3. Sample values from selected columns
    4. Validation status
    5. Outcome value mapping (for boolean conversion)
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        The uploaded data
        
    RETURNS:
    --------
    Tuple[Optional[pd.DataFrame], bool]
        - Mapped DataFrame (None if not complete)
        - Whether mapping is complete and valid
        
    EXAMPLE:
    --------
    >>> mapped_df, is_valid = render_column_mapping(df)
    >>> if is_valid:
    ...     proceed_to_analysis(mapped_df)
    """
    st.header("🔗 Column Mapping")
    
    st.markdown("""
    We need to know which of your columns contains each piece of information.
    We've tried to auto-detect them - please verify and correct if needed.
    """)
    
    # =========================================================================
    # INITIALIZE MAPPER
    # =========================================================================
    # Check if we already have a mapper in session state
    if 'column_mapper' not in st.session_state or st.session_state.get('mapper_df_id') != id(df):
        mapper = ColumnMapper(df)
        mapper.auto_detect()
        st.session_state['column_mapper'] = mapper
        st.session_state['mapper_df_id'] = id(df)
    else:
        mapper = st.session_state['column_mapper']
    
    # =========================================================================
    # MAPPING INTERFACE
    # =========================================================================
    st.markdown("### Select your columns")
    
    # Create a nice layout for column selection
    all_columns = [''] + list(df.columns)  # Add empty option
    
    mapping_complete = True
    
    for expected_col, config in EXPECTED_COLUMNS.items():
        col1, col2, col3 = st.columns([2, 3, 2])
        
        with col1:
            # Label with required indicator
            required_marker = "**" if config['required'] else ""
            st.markdown(f"{required_marker}{config['description']}{required_marker}")
            st.caption(f"Type: {config['type']}")
        
        with col2:
            # Get current mapping or auto-detected value
            current_value = mapper.mapping.get(expected_col, '')
            if current_value is None:
                current_value = ''
            
            # Try to find index
            try:
                default_index = all_columns.index(current_value) if current_value else 0
            except ValueError:
                default_index = 0
            
            # Create selectbox
            selected = st.selectbox(
                f"Select column for {expected_col}",
                options=all_columns,
                index=default_index,
                key=f"mapping_{expected_col}",
                label_visibility="collapsed"
            )
            
            # Update mapper
            if selected and selected != '':
                mapper.set_mapping(expected_col, selected)
            elif config['required']:
                mapper.mapping[expected_col] = None
        
        with col3:
            # Show sample values
            mapped_col = mapper.mapping.get(expected_col)
            if mapped_col and mapped_col in df.columns:
                sample = df[mapped_col].dropna().head(2).tolist()
                sample_str = ', '.join(str(v)[:25] for v in sample)
                st.caption(f"Sample: {sample_str}")
                
                # Show detection status
                if mapper.mapping.get(expected_col):
                    # Note: Streamlit requires actual emoji characters (like "✅") for icons,
                    # not text symbols like "✓". Using empty message since the success box
                    # already provides visual feedback.
                    st.success("", icon="✅")
            else:
                if config['required']:
                    st.warning("Required", icon="⚠️")
                    mapping_complete = False
                else:
                    st.info("Optional", icon="ℹ️")
    
    # =========================================================================
    # DATETIME PARSING PREVIEW
    # =========================================================================
    lead_time_col = mapper.mapping.get('lead_time')
    response_time_col = mapper.mapping.get('response_time')
    
    if lead_time_col and response_time_col:
        with st.expander("🕐 Date/Time Parsing Preview"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Lead Time Column:**")
                parsed_lead, success, msg = parse_datetime_column(df, lead_time_col)
                if success:
                    summary = get_datetime_summary(parsed_lead)
                    st.success(f"✓ {msg}")
                    st.caption(f"Range: {summary['min_date']} to {summary['max_date']}")
                else:
                    st.error(f"✗ {msg}")
                    mapping_complete = False
            
            with col2:
                st.markdown("**Response Time Column:**")
                parsed_response, success, msg = parse_datetime_column(df, response_time_col)
                if success:
                    summary = get_datetime_summary(parsed_response)
                    st.success(f"✓ {msg}")
                    st.caption(f"Range: {summary['min_date']} to {summary['max_date']}")
                else:
                    st.error(f"✗ {msg}")
                    mapping_complete = False
    
    # =========================================================================
    # OUTCOME VALUE MAPPING
    # =========================================================================
    ordered_col = mapper.mapping.get('ordered')
    
    if ordered_col and ordered_col in df.columns:
        st.markdown("---")
        st.markdown("### Order Outcome Mapping")
        st.markdown(f"Column **{ordered_col}** has these values. Which indicate an order?")
        
        unique_values = df[ordered_col].dropna().unique()[:10]  # Limit to 10 values
        
        # Detect outcome values
        outcome_map = mapper.detect_outcome_values()
        
        # Let user verify/modify
        positive_values = st.multiselect(
            "Select values that mean 'customer ordered':",
            options=[str(v) for v in unique_values],
            default=[str(v) for v, is_positive in outcome_map.items() if is_positive],
            help="Select all values that indicate the customer placed an order"
        )
        
        # Update the outcome mapping
        # Convert numpy types to Python primitives for consistent dict lookup
        mapper.outcome_value_mapping = {}
        for v in unique_values:
            # Convert numpy scalar to Python primitive
            key = v.item() if hasattr(v, 'item') else v
            mapper.outcome_value_mapping[key] = str(v) in positive_values
        
        # #region agent log
        import json
        log_path = "/Users/noahdelacalzada/response_time_cl_investigation/debug.log"
        with open(log_path, "a") as f:
            f.write(json.dumps({"location": "mapping_ui.py:outcome_mapping_update", "message": "updating outcome_value_mapping from multiselect", "data": {"positive_values": positive_values, "unique_values": [str(v) for v in unique_values], "new_mapping": {str(k): v for k, v in mapper.outcome_value_mapping.items()}}, "hypothesisId": "B", "timestamp": __import__("time").time()}) + "\n")
        # #endregion
        
        # Show preview
        n_positive = sum(1 for v in df[ordered_col] if str(v) in positive_values)
        total = len(df[ordered_col].dropna())
        
        st.info(f"ℹ️ Based on your selection: {n_positive:,} orders out of {total:,} leads ({n_positive/total*100:.1f}% close rate)")
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    st.markdown("---")
    
    # Check if mapping is complete
    missing = mapper.get_missing_columns()
    
    # Separate required vs optional missing columns
    missing_required = [
        col for col in missing 
        if EXPECTED_COLUMNS[col]['required']
    ]
    missing_optional = [
        col for col in missing 
        if not EXPECTED_COLUMNS[col]['required']
    ]
    
    if missing_required:
        st.error(f"❌ Please map the following required columns: {', '.join(missing_required)}")
        mapping_complete = False
    
    if missing_optional:
        st.warning(f"⚠️ Optional columns not mapped: {', '.join(missing_optional)}")
        st.info("""
        **Note:** Your analysis will still run, but will be less rigorous:
        - Without **lead_source**: Cannot control for differences in lead quality by source
        - Without **sales_rep**: Cannot control for differences in rep skill/performance
        - Statistical tests will still work, but results may be confounded by these factors
        """)
    
    # Validate the mapping
    if mapping_complete:
        is_valid, errors = mapper.validate_mapping()
        
        if errors:
            for error in errors:
                st.warning(f"⚠️ {error}")
    
    # =========================================================================
    # APPLY MAPPING
    # =========================================================================
    if mapping_complete:
        try:
            mapped_df = mapper.apply_mapping()
            
            # Parse datetime columns
            for col in ['lead_time', 'response_time']:
                if col in mapped_df.columns:
                    parsed, success, msg = parse_datetime_column(mapped_df, col)
                    if success:
                        mapped_df[col] = parsed
            
            st.success("✅ Column mapping complete! Ready for analysis.")
            
            # Store in session state
            st.session_state['mapped_df'] = mapped_df
            
            return mapped_df, True
            
        except Exception as e:
            st.error(f"❌ Error applying mapping: {str(e)}")
            return None, False
    
    return None, False


def render_mapping_summary(mapper: ColumnMapper) -> None:
    """
    Display a summary of the current column mapping.
    
    PARAMETERS:
    -----------
    mapper : ColumnMapper
        The column mapper with current mappings
    """
    st.markdown("### Mapping Summary")
    
    status = mapper.get_mapping_status()
    
    for expected_col, info in status.items():
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write(info['description'])
        
        with col2:
            if info['is_mapped']:
                st.success(f"✓ {info['mapped_to']}")
            elif EXPECTED_COLUMNS[expected_col]['required']:
                st.error("✗ Not mapped (required)")
            else:
                st.warning("○ Not mapped (optional)")


def render_column_suggestion_help(df: pd.DataFrame, expected_col: str) -> None:
    """
    Display suggestions for a specific column mapping.
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        The user's data
    expected_col : str
        The expected column we're trying to map
    """
    suggestions = suggest_column_mapping(df, expected_col)
    
    if suggestions:
        st.markdown("**Suggested columns:**")
        for col, score in suggestions[:3]:
            st.write(f"- {col} ({score:.0%} confidence)")
    else:
        st.info("No strong matches found. Please select manually.")

