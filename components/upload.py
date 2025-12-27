# =============================================================================
# Upload Component
# =============================================================================
# This component handles file upload with a styled drag-and-drop interface.
#
# WHY THIS COMPONENT EXISTS:
# --------------------------
# The upload experience sets the tone for the entire app.
# We want it to be:
# 1. Clear and welcoming
# 2. Provide helpful feedback
# 3. Show data preview immediately
# 4. Offer sample data for demos
#
# MAIN FUNCTIONS:
# ---------------
# - render_upload_section(): Main upload interface
# - render_data_preview(): Show preview of uploaded data
# =============================================================================

import streamlit as st
import pandas as pd
from typing import Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.loader import load_file, validate_file, get_column_info, preview_data
from data.sample_generator import generate_sample_data, get_sample_data_summary
from data.weeks_analyzer import analyze_weeks_of_data, WEEKS_GUIDANCE
from config.settings import ALLOWED_FILE_TYPES, MAX_FILE_SIZE_MB


def render_upload_section() -> Tuple[Optional[pd.DataFrame], str]:
    """
    Render the file upload section with drag-and-drop interface.
    
    WHY THIS FUNCTION:
    ------------------
    Creates an engaging, helpful upload experience.
    Offers both file upload and sample data options.
    
    WHAT IT DISPLAYS:
    -----------------
    1. Welcome message explaining what to upload
    2. File upload widget with styling
    3. Alternative: Sample data button
    4. Data preview after upload
    5. Column information table
    
    RETURNS:
    --------
    Tuple[Optional[pd.DataFrame], str]
        - DataFrame if data was loaded (None otherwise)
        - Source of data ('upload', 'sample', or '')
        
    EXAMPLE:
    --------
    >>> df, source = render_upload_section()
    >>> if df is not None:
    ...     st.write(f"Loaded {len(df)} rows from {source}")
    """
    # =========================================================================
    # HEADER
    # =========================================================================
    st.header("📁 Upload Your Data")
    
    st.markdown("""
    Upload your lead data to analyze response time impact on close rates.
    
    **Required columns:**
    - Lead arrival time (when the lead came in)
    - First response time (when someone first responded)
    - Order outcome (did the customer order?)
    
    **Optional columns (for more rigorous analysis):**
    - Lead source (where the lead came from) - enables controlling for lead quality differences
    - Sales rep (who handled the lead) - enables controlling for rep skill differences
    """)
    
    # =========================================================================
    # UPLOAD OPTIONS
    # =========================================================================
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Main file uploader
        st.markdown(
            f"""
            <div style="
                border: 2px dashed #1E88E5;
                border-radius: 10px;
                padding: 20px;
                text-align: center;
                margin-bottom: 20px;
            ">
                <p style="color: #1E88E5; font-size: 18px;">
                    📂 Drag and drop your file here
                </p>
                <p style="color: #666; font-size: 14px;">
                    Supported: {', '.join(ALLOWED_FILE_TYPES).upper()} (max {MAX_FILE_SIZE_MB}MB)
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=ALLOWED_FILE_TYPES,
            help="Upload a CSV or Excel file with your lead data",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Or try with sample data:**")
        
        # Let user choose how many weeks of sample data to generate
        # This helps them understand how data volume affects analysis
        n_weeks = st.slider(
            "Weeks of data",
            min_value=2,
            max_value=16,
            value=8,
            step=1,
            help="More weeks = more data for trend analysis. 4-8 weeks is recommended."
        )
        
        use_sample = st.button(
            "📊 Load Sample Data",
            help=f"Generate {n_weeks} weeks of realistic sample data",
            use_container_width=True
        )
    
    # =========================================================================
    # PROCESS UPLOAD OR SAMPLE
    # =========================================================================
    df = None
    source = ''
    
    # Handle sample data request
    if use_sample:
        # Clear any previously cached data to ensure fresh generation
        if 'uploaded_data' in st.session_state:
            del st.session_state['uploaded_data']
        
        with st.spinner(f"Generating {n_weeks} weeks of sample data..."):
            df = generate_sample_data(n_weeks=n_weeks)
            st.session_state['uploaded_data'] = df
            st.session_state['data_source'] = 'sample'
            st.session_state['sample_n_weeks'] = n_weeks  # Store for reference
            source = 'sample'
        
        st.success(f"✅ Generated {len(df):,} sample leads over {n_weeks} weeks!")
        
        # Show sample data info
        with st.expander("ℹ️ About the sample data"):
            st.markdown(f"""
            This sample data simulates a realistic car dealership scenario:
            
            - **{len(df):,} leads** over **{n_weeks} weeks** 
              {"✅ (ideal data period!)" if 4 <= n_weeks <= 8 else ""}
            - **5 lead sources** with different close rates
            - **12 sales reps** with varying skill levels
            - **Realistic confounding**: faster reps also tend to close more
            
            ### 📅 How data periods work
            
            In practice, you would upload your lead data from your CRM or database.
            Each file typically contains data from a specific time period.
            
            **For best results, upload 4-8 weeks of data.** This ensures:
            - Enough leads in each response time bucket
            - Weekly patterns are captured (weekdays vs weekends)
            - Statistical tests have sufficient power
            
            {"This sample data demonstrates what ideal analysis looks like!" if 4 <= n_weeks <= 8 else "Try adjusting the weeks slider to see how data volume affects analysis."}
            """)
    
    # Handle file upload
    elif uploaded_file is not None:
        # Validate
        is_valid, error_msg = validate_file(uploaded_file)
        
        if not is_valid:
            st.error(f"❌ {error_msg}")
            return None, ''
        
        # Load
        with st.spinner("Loading your data..."):
            df, error_msg = load_file(uploaded_file)
        
        if df is None:
            st.error(f"❌ {error_msg}")
            return None, ''
        
        st.session_state['uploaded_data'] = df
        st.session_state['data_source'] = 'upload'
        source = 'upload'
        
        st.success(f"✅ Loaded {len(df):,} rows from {uploaded_file.name}")
    
    # Check session state for previously loaded data
    elif 'uploaded_data' in st.session_state:
        df = st.session_state['uploaded_data']
        source = st.session_state.get('data_source', 'unknown')
    
    # =========================================================================
    # DATA PREVIEW
    # =========================================================================
    if df is not None:
        render_data_preview(df, source)
    
    return df, source


def render_data_preview(df: pd.DataFrame, source: str = '') -> None:
    """
    Render a preview of the uploaded/sample data.
    
    WHAT THIS SHOWS:
    ----------------
    1. Quick stats (rows, columns, source, memory)
    2. Weeks of data analysis with recommendations
    3. First 5 rows preview
    4. Column information
    5. Data quality checks
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        The loaded data
    source : str
        Where the data came from ('upload' or 'sample')
    """
    st.markdown("---")
    st.subheader("📋 Data Preview")
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Total Columns", len(df.columns))
    col3.metric("Source", source.title() if source else "Unknown")
    
    # Memory usage
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    col4.metric("Memory Usage", f"{memory_mb:.1f} MB")
    
    # =========================================================================
    # WEEKS OF DATA ANALYSIS
    # =========================================================================
    # This is important! We analyze how many weeks the data covers and give
    # recommendations about whether it's enough for reliable analysis.
    render_weeks_analysis(df)
    
    # Data preview table
    st.markdown("**First 5 rows:**")
    st.dataframe(
        preview_data(df, 5),
        use_container_width=True,
        hide_index=True
    )
    
    # Column information
    with st.expander("📊 Column Information", expanded=False):
        col_info = get_column_info(df)
        st.dataframe(
            col_info,
            use_container_width=True,
            hide_index=True
        )
    
    # Data quality checks
    with st.expander("🔍 Data Quality Check", expanded=False):
        render_data_quality_check(df)


def render_weeks_analysis(df: pd.DataFrame) -> None:
    """
    Analyze and display how many weeks of data are in the upload.
    
    WHY THIS MATTERS:
    -----------------
    Response time analysis works best with 4-8 weeks of data.
    This function tells users:
    - How many weeks their data covers
    - Whether that's enough for reliable analysis
    - What to do if they need more data
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        The uploaded data (before preprocessing)
    """
    # Step 1: Analyze the weeks
    # -------------------------
    weeks_info = analyze_weeks_of_data(df)
    
    # If we couldn't detect weeks, just show a brief note and move on
    if weeks_info['weeks'] is None:
        # Don't show anything alarming - dates will be mapped in the next step
        return
    
    # Step 2: Display the results
    # ---------------------------
    st.markdown("### 📅 Data Time Period")
    
    # Show the date range in a readable format
    if weeks_info['date_range']:
        min_date, max_date = weeks_info['date_range']
        st.markdown(
            f"**Date Range:** {min_date.strftime('%B %d, %Y')} → "
            f"{max_date.strftime('%B %d, %Y')} "
            f"({weeks_info['weeks']:.1f} weeks)"
        )
    
    # Step 3: Show the recommendation
    # --------------------------------
    recommendation = weeks_info['recommendation']
    
    # Use different Streamlit widgets based on recommendation type
    if recommendation['type'] == 'success':
        st.success(
            f"{recommendation['icon']} **{recommendation['title']}**: "
            f"{recommendation['message']}"
        )
    elif recommendation['type'] == 'warning':
        st.warning(
            f"{recommendation['icon']} **{recommendation['title']}**: "
            f"{recommendation['message']}"
        )
        # If there's an action, show it in an info box
        if recommendation.get('action'):
            st.info(f"**💡 Suggestion:** {recommendation['action']}")
    else:  # 'info' type
        st.info(
            f"{recommendation['icon']} **{recommendation['title']}**: "
            f"{recommendation['message']}"
        )
        if recommendation.get('action'):
            st.caption(recommendation['action'])
    
    # Step 4: Provide guidance on data periods (in an expander)
    # ---------------------------------------------------------
    with st.expander("📖 Why does the data period matter?"):
        st.markdown(WEEKS_GUIDANCE)


def render_data_quality_check(df: pd.DataFrame) -> None:
    """
    Display data quality checks and warnings.
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        The loaded data
    """
    issues = []
    
    # Check for missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    
    if len(missing_cols) > 0:
        st.markdown("**Missing Values:**")
        for col, count in missing_cols.items():
            pct = count / len(df) * 100
            if pct > 10:
                st.warning(f"⚠️ {col}: {count:,} missing ({pct:.1f}%)")
            else:
                st.info(f"ℹ️ {col}: {count:,} missing ({pct:.1f}%)")
            issues.append(f"{col} has missing values")
    else:
        st.success("✅ No missing values found!")
    
    # Check for duplicates
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        pct = n_duplicates / len(df) * 100
        st.warning(f"⚠️ Found {n_duplicates:,} duplicate rows ({pct:.1f}%)")
        issues.append("Duplicate rows found")
    else:
        st.success("✅ No duplicate rows found!")
    
    # Summary
    if len(issues) == 0:
        st.success("✅ Data looks good! Ready for analysis.")
    else:
        st.info(f"ℹ️ Found {len(issues)} potential issues. These may affect analysis accuracy.")


def render_upload_help() -> None:
    """
    Display help information about the upload process.
    """
    with st.expander("❓ Help: Preparing Your Data"):
        st.markdown("""
        ### How to prepare your data
        
        Your data should include these columns (names can vary):
        
        **Required columns:**
        
        | What We Need | Example Column Names |
        |--------------|---------------------|
        | Lead arrival time | `created_at`, `lead_time`, `timestamp`, `received_time` |
        | First response time | `first_response`, `replied_at`, `contacted_at`, `first_contact_time` |
        | Order outcome | `ordered`, `sold`, `converted` |
        
        **Optional columns (for more rigorous analysis):**
        
        | What We Need | Example Column Names |
        |--------------|---------------------|
        | Lead source | `source`, `channel`, `lead_source` |
        | Sales rep | `rep`, `agent`, `salesperson` |
        
        ### Date formats we support
        
        - ISO: `2024-01-15 14:30:00`
        - US: `01/15/2024 2:30 PM`
        - European: `15/01/2024 14:30`
        - Excel serial numbers
        
        ### Outcome column formats
        
        We can interpret:
        - Boolean: `True`/`False`, `Yes`/`No`
        - Numeric: `1`/`0`
        - Text: `Ordered`, `Sold`, `Won`, `Converted`
        
        ### Tips for best results
        
        1. **More data is better**: Aim for at least 1,000 leads
        2. **Include all leads**: Don't pre-filter based on outcome
        3. **Check timestamps**: Make sure they're in the right timezone
        4. **Consistent formatting**: Use consistent column formats
        """)

