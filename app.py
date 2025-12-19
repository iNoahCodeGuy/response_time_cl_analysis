# =============================================================================
# Response Time Analyzer - Main Application
# =============================================================================
# This is the main entry point for the Streamlit application.
#
# HOW TO RUN:
# -----------
# streamlit run app.py
#
# WHAT THIS APP DOES:
# -------------------
# 1. Upload lead data (CSV/Excel) or use sample data
# 2. Map columns to expected format
# 3. Analyze impact of response time on close rates
# 4. Present results with step-by-step explanations
#
# APPLICATION FLOW:
# -----------------
# 1. Welcome/Upload → 2. Column Mapping → 3. Settings → 4. Analysis → 5. Results
#
# DESIGNED FOR:
# -------------
# - Non-technical business users who want to understand the math
# - Data analysts who want rigorous statistical analysis
# - Sales managers who want actionable insights
# =============================================================================

import streamlit as st
import pandas as pd
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import components
from components.upload import render_upload_section
from components.mapping_ui import render_column_mapping
from components.settings_panel import render_settings_panel, render_settings_summary
from components.results_dashboard import render_results_dashboard

# Import analysis modules
from analysis.preprocessing import preprocess_data, get_preprocessing_summary

# Import config
from config.settings import APP_TITLE, APP_ICON, APP_DESCRIPTION


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    
    /* Step indicator styling */
    .step-indicator {
        display: flex;
        justify-content: space-between;
        margin-bottom: 2rem;
    }
    
    .step-item {
        text-align: center;
        flex: 1;
    }
    
    .step-number {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background-color: #E0E0E0;
        color: #666;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
    }
    
    .step-number.active {
        background-color: #1E88E5;
        color: white;
    }
    
    .step-number.completed {
        background-color: #4CAF50;
        color: white;
    }
    
    /* Card styling */
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
def initialize_session_state():
    """
    Initialize all session state variables.
    
    WHY THIS FUNCTION:
    ------------------
    Streamlit reruns the script on every interaction. Session state
    persists data across reruns. We initialize all variables here
    to avoid KeyError issues.
    """
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    
    if 'mapped_data' not in st.session_state:
        st.session_state.mapped_data = None
    
    if 'preprocessed_data' not in st.session_state:
        st.session_state.preprocessed_data = None
    
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False


# =============================================================================
# NAVIGATION
# =============================================================================
def render_step_indicator():
    """
    Render the step indicator showing progress through the app.
    """
    steps = ['Upload', 'Map Columns', 'Settings', 'Analyze', 'Results']
    current = st.session_state.current_step
    
    cols = st.columns(len(steps))
    
    for i, (col, step_name) in enumerate(zip(cols, steps), 1):
        with col:
            if i < current:
                st.markdown(f"✅ **{step_name}**")
            elif i == current:
                st.markdown(f"▶️ **{step_name}**")
            else:
                st.markdown(f"○ {step_name}")


def go_to_step(step_number: int):
    """Navigate to a specific step."""
    st.session_state.current_step = step_number


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    """
    Main application entry point.
    
    FLOW:
    -----
    1. Initialize session state
    2. Render sidebar settings
    3. Show appropriate step based on current_step
    4. Handle navigation between steps
    """
    # Initialize session state
    initialize_session_state()
    
    # =========================================================================
    # SIDEBAR - Settings Panel
    # =========================================================================
    with st.sidebar:
        settings = render_settings_panel()
    
    # =========================================================================
    # MAIN CONTENT
    # =========================================================================
    
    # App title
    st.markdown(f"<h1 class='main-title'>{APP_ICON} {APP_TITLE}</h1>", unsafe_allow_html=True)
    st.markdown(APP_DESCRIPTION)
    
    # Step indicator
    render_step_indicator()
    st.markdown("---")
    
    # =========================================================================
    # STEP 1: UPLOAD DATA
    # =========================================================================
    if st.session_state.current_step == 1:
        df, source = render_upload_section()
        
        if df is not None:
            st.session_state.uploaded_data = df
            st.session_state.data_source = source
            
            # Navigation button
            if st.button("Next: Map Columns →", type="primary", use_container_width=True):
                go_to_step(2)
                st.rerun()
    
    # =========================================================================
    # STEP 2: COLUMN MAPPING
    # =========================================================================
    elif st.session_state.current_step == 2:
        if st.session_state.uploaded_data is None:
            st.warning("Please upload data first.")
            if st.button("← Back to Upload"):
                go_to_step(1)
                st.rerun()
        else:
            mapped_df, is_complete = render_column_mapping(st.session_state.uploaded_data)
            
            if is_complete and mapped_df is not None:
                st.session_state.mapped_data = mapped_df
                
                # Navigation
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("← Back to Upload"):
                        go_to_step(1)
                        st.rerun()
                with col2:
                    if st.button("Next: Review Settings →", type="primary", use_container_width=True):
                        go_to_step(3)
                        st.rerun()
            else:
                if st.button("← Back to Upload"):
                    go_to_step(1)
                    st.rerun()
    
    # =========================================================================
    # STEP 3: SETTINGS REVIEW
    # =========================================================================
    elif st.session_state.current_step == 3:
        if st.session_state.mapped_data is None:
            st.warning("Please complete column mapping first.")
            if st.button("← Back to Mapping"):
                go_to_step(2)
                st.rerun()
        else:
            st.header("⚙️ Review Settings")
            st.markdown("Review your analysis settings before running. You can adjust these in the sidebar.")
            
            render_settings_summary()
            
            # Data summary
            st.markdown("### Data Summary")
            df = st.session_state.mapped_data
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Leads", f"{len(df):,}")
            col2.metric("Lead Sources", df['lead_source'].nunique())
            col3.metric("Sales Reps", df['sales_rep'].nunique())
            col4.metric("Orders", f"{df['ordered'].sum():,}")
            
            # Navigation
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("← Back to Mapping"):
                    go_to_step(2)
                    st.rerun()
            with col2:
                if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
                    go_to_step(4)
                    st.rerun()
    
    # =========================================================================
    # STEP 4: RUN ANALYSIS
    # =========================================================================
    elif st.session_state.current_step == 4:
        if st.session_state.mapped_data is None:
            st.warning("Please complete previous steps first.")
            if st.button("← Start Over"):
                go_to_step(1)
                st.rerun()
        else:
            st.header("⏳ Running Analysis...")
            
            # Preprocess data
            with st.spinner("Preprocessing data..."):
                preprocessed_df, diagnostics = preprocess_data(
                    st.session_state.mapped_data,
                    bucket_boundaries=settings['bucket_boundaries'],
                    bucket_labels=settings['bucket_labels']
                )
                
                st.session_state.preprocessed_data = preprocessed_df
                st.session_state.preprocessing_diagnostics = diagnostics
            
            # Show preprocessing summary
            st.markdown(get_preprocessing_summary(diagnostics))
            
            # Show any warnings
            if diagnostics.get('warnings'):
                for warning in diagnostics['warnings']:
                    st.warning(warning)
            
            st.success("✅ Preprocessing complete!")
            
            # Auto-proceed to results
            st.session_state.analysis_complete = True
            go_to_step(5)
            st.rerun()
    
    # =========================================================================
    # STEP 5: RESULTS
    # =========================================================================
    elif st.session_state.current_step == 5:
        if st.session_state.preprocessed_data is None:
            st.warning("Please run the analysis first.")
            if st.button("← Start Over"):
                go_to_step(1)
                st.rerun()
        else:
            # Back button
            if st.button("← Run New Analysis"):
                # Clear session state for new analysis
                st.session_state.uploaded_data = None
                st.session_state.mapped_data = None
                st.session_state.preprocessed_data = None
                st.session_state.analysis_complete = False
                go_to_step(1)
                st.rerun()
            
            # Render results dashboard
            render_results_dashboard(
                st.session_state.preprocessed_data,
                settings
            )


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()

