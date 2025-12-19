# =============================================================================
# Settings Panel Component
# =============================================================================
# This component renders the sidebar settings for the analysis.
#
# WHY THIS COMPONENT EXISTS:
# --------------------------
# Users need to configure their analysis:
# - Choose analysis mode (Standard vs Advanced)
# - Set response time bucket boundaries
# - Choose significance level
# - Other preferences
#
# MAIN FUNCTIONS:
# ---------------
# - render_settings_panel(): Render all settings in the sidebar
# - get_current_settings(): Get the current settings from session state
# =============================================================================

import streamlit as st
from typing import Dict, Any, List
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    ANALYSIS_MODES, 
    DEFAULT_BUCKETS, 
    DEFAULT_BUCKET_LABELS,
    BUCKET_PRESETS,
    DEFAULT_ALPHA,
    DEFAULT_CONFIDENCE_LEVEL
)


def initialize_settings():
    """
    Initialize settings in session state if not already present.
    
    WHY THIS MATTERS:
    -----------------
    Session state persists settings across Streamlit reruns.
    This function ensures all settings have default values.
    """
    # Analysis mode
    if 'analysis_mode' not in st.session_state:
        st.session_state.analysis_mode = 'standard'
    
    # Bucket configuration
    if 'bucket_preset' not in st.session_state:
        st.session_state.bucket_preset = 'standard'
    
    if 'custom_buckets' not in st.session_state:
        st.session_state.custom_buckets = DEFAULT_BUCKETS.copy()
    
    if 'custom_bucket_labels' not in st.session_state:
        st.session_state.custom_bucket_labels = DEFAULT_BUCKET_LABELS.copy()
    
    # Statistical settings
    if 'alpha_level' not in st.session_state:
        st.session_state.alpha_level = DEFAULT_ALPHA
    
    if 'confidence_level' not in st.session_state:
        st.session_state.confidence_level = DEFAULT_CONFIDENCE_LEVEL


def render_settings_panel() -> Dict[str, Any]:
    """
    Render the settings panel in the Streamlit sidebar.
    
    WHY THIS MATTERS:
    -----------------
    The sidebar provides a clean, always-visible location for settings.
    Users can adjust settings without scrolling through the main content.
    
    WHAT'S INCLUDED:
    ----------------
    1. Analysis Mode Toggle (Standard vs Advanced)
    2. Response Time Bucket Configuration
    3. Statistical Settings (alpha, confidence level)
    4. Display Preferences
    
    RETURNS:
    --------
    Dict[str, Any]
        Dictionary of all current settings
        
    EXAMPLE:
    --------
    >>> settings = render_settings_panel()
    >>> if settings['analysis_mode'] == 'advanced':
    ...     run_advanced_analysis()
    """
    # Initialize settings if needed
    initialize_settings()
    
    st.sidebar.title("⚙️ Settings")
    
    # =========================================================================
    # SECTION 1: Analysis Mode
    # =========================================================================
    st.sidebar.header("Analysis Mode")
    
    # Create nice cards for each mode
    mode_options = list(ANALYSIS_MODES.keys())
    mode_labels = [
        f"{ANALYSIS_MODES[m]['icon']} {ANALYSIS_MODES[m]['name']}" 
        for m in mode_options
    ]
    
    selected_index = mode_options.index(st.session_state.analysis_mode)
    
    selected_mode = st.sidebar.radio(
        "Choose analysis depth:",
        options=mode_options,
        format_func=lambda x: f"{ANALYSIS_MODES[x]['icon']} {ANALYSIS_MODES[x]['name']}",
        index=selected_index,
        help="Standard mode covers essential tests. Advanced mode adds controls for sales rep effects."
    )
    
    st.session_state.analysis_mode = selected_mode
    
    # Show description of selected mode
    mode_info = ANALYSIS_MODES[selected_mode]
    st.sidebar.caption(mode_info['description'])
    
    # Show what tests are included
    with st.sidebar.expander("Tests included in this mode"):
        for test in mode_info['tests']:
            test_name = test.replace('_', ' ').title()
            st.write(f"• {test_name}")
    
    st.sidebar.divider()
    
    # =========================================================================
    # SECTION 2: Response Time Buckets
    # =========================================================================
    st.sidebar.header("Response Time Buckets")
    
    st.sidebar.caption(
        "How should we group response times? This affects how we categorize "
        "leads as 'fast' or 'slow' responders."
    )
    
    # Preset selection
    preset_options = list(BUCKET_PRESETS.keys()) + ['custom']
    preset_labels = {
        'standard': '📊 Standard (0-15, 15-30, 30-60, 60+ min)',
        'aggressive': '⚡ Aggressive (0-5, 5-15, 15-30, 30+ min)',
        'relaxed': '🕐 Relaxed (0-30, 30-60, 1-2hr, 2+ hr)',
        'custom': '✏️ Custom'
    }
    
    selected_preset = st.sidebar.selectbox(
        "Bucket preset:",
        options=preset_options,
        format_func=lambda x: preset_labels.get(x, x),
        index=preset_options.index(st.session_state.bucket_preset),
        help="Choose a preset or define custom bucket boundaries"
    )
    
    st.session_state.bucket_preset = selected_preset
    
    # Custom bucket configuration
    if selected_preset == 'custom':
        st.sidebar.caption("Define custom bucket boundaries (in minutes):")
        
        # Number of buckets
        n_buckets = st.sidebar.number_input(
            "Number of buckets:",
            min_value=2,
            max_value=6,
            value=4,
            help="How many response time groups to create"
        )
        
        # Bucket boundaries
        boundaries = [0]
        labels = []
        
        for i in range(n_buckets - 1):
            bound = st.sidebar.number_input(
                f"Bucket {i+1} upper limit (min):",
                min_value=boundaries[-1] + 1,
                max_value=480,
                value=min(boundaries[-1] + 15, 480),
                key=f"bucket_bound_{i}"
            )
            boundaries.append(bound)
            
            # Create label
            if i == 0:
                labels.append(f"0-{bound} min")
            else:
                labels.append(f"{boundaries[-2]}-{bound} min")
        
        # Final bucket
        boundaries.append(float('inf'))
        labels.append(f"{boundaries[-2]}+ min")
        
        st.session_state.custom_buckets = boundaries
        st.session_state.custom_bucket_labels = labels
    
    else:
        # Use preset
        preset = BUCKET_PRESETS[selected_preset]
        st.session_state.custom_buckets = preset['boundaries']
        st.session_state.custom_bucket_labels = preset['labels']
    
    st.sidebar.divider()
    
    # =========================================================================
    # SECTION 3: Statistical Settings
    # =========================================================================
    st.sidebar.header("Statistical Settings")
    
    # Significance level
    alpha = st.sidebar.select_slider(
        "Significance level (α):",
        options=[0.01, 0.05, 0.10],
        value=st.session_state.alpha_level,
        format_func=lambda x: f"{x:.0%}",
        help="The threshold for declaring results 'statistically significant'. "
             "Lower = more stringent. 5% is standard."
    )
    st.session_state.alpha_level = alpha
    
    # Explain significance level
    st.sidebar.caption(
        f"Results with p-value < {alpha:.0%} will be considered significant. "
        f"This means there's less than a {alpha:.0%} chance the observed "
        f"differences are due to random chance."
    )
    
    # Confidence level
    conf_level = 1 - alpha
    st.session_state.confidence_level = conf_level
    
    st.sidebar.caption(f"Confidence intervals will be at the {conf_level:.0%} level.")
    
    st.sidebar.divider()
    
    # =========================================================================
    # SECTION 4: Display Preferences
    # =========================================================================
    # This section lets users choose how much detail they want to see.
    # Currently we offer one option: showing the mathematical formulas.
    # The math is always available in expanders, this just controls defaults.
    
    st.sidebar.header("Display Preferences")
    
    show_math = st.sidebar.checkbox(
        "Show mathematical formulas",
        value=True,
        help="Display LaTeX formulas explaining each calculation"
    )
    
    # Store in session state so other components can access it
    st.session_state.show_math = show_math
    
    # =========================================================================
    # Return all settings
    # =========================================================================
    return get_current_settings()


def get_current_settings() -> Dict[str, Any]:
    """
    Get all current settings from session state.
    
    WHY THIS MATTERS:
    -----------------
    Provides a single function to retrieve all settings.
    Useful for passing settings to analysis functions.
    
    RETURNS:
    --------
    Dict[str, Any]
        All current settings as a dictionary
    """
    # Initialize if needed
    initialize_settings()
    
    # Return all settings as a dictionary
    # This makes it easy to pass settings to analysis functions
    return {
        'analysis_mode': st.session_state.analysis_mode,
        'bucket_boundaries': st.session_state.custom_buckets,
        'bucket_labels': st.session_state.custom_bucket_labels,
        'alpha_level': st.session_state.alpha_level,
        'confidence_level': st.session_state.confidence_level,
        'show_math': st.session_state.get('show_math', True)
    }


def render_settings_summary() -> None:
    """
    Display a summary of current settings in the main area.
    
    WHY THIS MATTERS:
    -----------------
    Users can see at a glance what settings are being used
    without needing to check the sidebar.
    """
    settings = get_current_settings()
    
    mode = ANALYSIS_MODES[settings['analysis_mode']]
    
    cols = st.columns(4)
    
    with cols[0]:
        st.metric("Analysis Mode", mode['name'])
    
    with cols[1]:
        st.metric("Response Buckets", len(settings['bucket_labels']))
    
    with cols[2]:
        st.metric("Significance Level", f"{settings['alpha_level']:.0%}")
    
    with cols[3]:
        st.metric("Confidence Level", f"{settings['confidence_level']:.0%}")

