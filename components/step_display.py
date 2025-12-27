# =============================================================================
# Step Display Component
# =============================================================================
# This component creates consistent, educational step-by-step displays.
#
# WHY THIS COMPONENT EXISTS:
# --------------------------
# Each analysis step needs to show:
# 1. What we're calculating (in plain English)
# 2. Why it matters (business context)
# 3. The result (with interpretation)
# 4. Optional: the math (for those who want it)
#
# This component ensures consistent formatting across all steps.
#
# MAIN FUNCTIONS:
# ---------------
# - display_step(): Main step display with all sections
# - display_result_card(): Show a key result prominently
# - display_significance_badge(): Show if result is significant
# =============================================================================

import streamlit as st
from typing import Dict, Any, Optional, List
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from explanations.templates import get_explanation
from explanations.formulas import get_formula, render_formula_with_explanation


def display_step(
    step_number: int,
    title: str,
    analysis_type: str,
    result: Dict[str, Any],
    is_significant: bool = None,
    p_value: float = None,
    show_math: bool = True,
    additional_content: callable = None
) -> None:
    """
    Display a complete analysis step with explanation and results.
    
    WHY THIS FUNCTION:
    ------------------
    Creates a consistent, educational display for each analysis step.
    Combines explanation, results, and interpretation in one place.
    
    WHAT IT DISPLAYS:
    -----------------
    1. Step header with number and title
    2. What we're doing (plain English explanation)
    3. The result (key metrics, charts, tables)
    4. Interpretation (what does this mean?)
    5. Optional: The math (in an expander)
    
    PARAMETERS:
    -----------
    step_number : int
        Step number (1, 2, 3, ...)
    title : str
        Step title
    analysis_type : str
        Key for explanation templates (e.g., 'chi_square')
    result : Dict[str, Any]
        Results from the analysis
    is_significant : bool, optional
        Whether the result was significant
    p_value : float, optional
        P-value from the test
    show_math : bool
        Whether to show the math section
    additional_content : callable, optional
        Function that renders additional content (charts, tables)
        
    EXAMPLE:
    --------
    >>> display_step(
    ...     step_number=2,
    ...     title="Chi-Square Test",
    ...     analysis_type='chi_square',
    ...     result={'statistic': 15.3, 'p_value': 0.002},
    ...     is_significant=True,
    ...     p_value=0.002
    ... )
    """
    # Get explanation template
    explanation = get_explanation(analysis_type, is_significant, p_value)
    
    # Step container with visual separator
    st.markdown("---")
    
    # =========================================================================
    # HEADER
    # =========================================================================
    col1, col2 = st.columns([0.1, 0.9])
    
    with col1:
        # Step number badge
        st.markdown(
            f"""
            <div style="
                background-color: #1E88E5;
                color: white;
                width: 40px;
                height: 40px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                font-size: 18px;
            ">
                {step_number}
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.subheader(title)
        
        # Significance badge if applicable
        if is_significant is not None:
            display_significance_badge(is_significant, p_value)
    
    # =========================================================================
    # WHAT WE'RE DOING
    # =========================================================================
    with st.expander("📖 What is this test?", expanded=False):
        if 'what_it_does' in explanation:
            st.markdown(explanation['what_it_does'])
        
        if 'why_it_matters' in explanation:
            st.markdown("**Why it matters:**")
            st.markdown(explanation['why_it_matters'])
        
        if 'analogy' in explanation:
            st.info(f"**Analogy:** {explanation['analogy']}")
    
    # =========================================================================
    # RESULTS
    # =========================================================================
    st.markdown("#### Results")
    
    # Key metrics
    if 'statistic' in result:
        display_result_card(
            "Test Statistic",
            f"{result['statistic']:.2f}",
            subtitle=result.get('statistic_name', '')
        )
    
    if p_value is not None:
        display_result_card(
            "P-Value",
            f"{p_value:.4f}",
            subtitle="< 0.05 is significant",
            highlight=p_value < 0.05
        )
    
    # Additional content (charts, tables, etc.)
    if additional_content:
        additional_content()
    
    # =========================================================================
    # INTERPRETATION
    # =========================================================================
    st.markdown("#### What This Means")
    
    if 'interpretation' in explanation:
        if is_significant:
            st.success(explanation['interpretation'])
        else:
            st.warning(explanation['interpretation'])
    
    if 'p_value_context' in explanation:
        st.caption(explanation['p_value_context'])
    
    # =========================================================================
    # THE MATH (optional)
    # =========================================================================
    if show_math:
        formula = get_formula(analysis_type)
        
        if formula and 'formula' in formula:
            with st.expander("🔢 Show the math", expanded=False):
                st.markdown(f"**{formula.get('name', 'Formula')}:**")
                st.latex(formula['formula'])
                
                if formula.get('components'):
                    st.markdown("**Where:**")
                    for symbol, meaning in formula['components'].items():
                        st.markdown(f"- {symbol} = {meaning}")
                
                if formula.get('intuition'):
                    st.caption(formula['intuition'])
    
    # =========================================================================
    # CAVEATS
    # =========================================================================
    if 'caveats' in explanation and explanation['caveats']:
        with st.expander("⚠️ Important caveats", expanded=False):
            for caveat in explanation['caveats']:
                st.markdown(f"- {caveat}")


def display_result_card(
    title: str,
    value: str,
    subtitle: str = "",
    highlight: bool = False,
    delta: str = None,
    delta_color: str = "normal"
) -> None:
    """
    Display a key result in a visually prominent card.
    
    PARAMETERS:
    -----------
    title : str
        What this metric is
    value : str
        The value to display
    subtitle : str
        Additional context
    highlight : bool
        Whether to highlight (for significant results)
    delta : str
        Change indicator (e.g., "+5%")
    delta_color : str
        Color for delta ("normal", "inverse", "off")
    """
    if delta:
        st.metric(
            label=title,
            value=value,
            delta=delta,
            delta_color=delta_color,
            help=subtitle
        )
    else:
        st.metric(
            label=title,
            value=value,
            help=subtitle
        )


def display_significance_badge(is_significant: bool, p_value: float = None) -> None:
    """
    Display a badge indicating statistical significance.
    
    PARAMETERS:
    -----------
    is_significant : bool
        Whether the result is significant
    p_value : float, optional
        The p-value (shown in tooltip)
    """
    if is_significant:
        st.markdown(
            f"""
            <span style="
                background-color: #4CAF50;
                color: white;
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: bold;
            ">
                ✓ SIGNIFICANT
            </span>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <span style="
                background-color: #9E9E9E;
                color: white;
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 12px;
                font-weight: bold;
            ">
                NOT SIGNIFICANT
            </span>
            """,
            unsafe_allow_html=True
        )


def display_progress_indicator(
    current_step: int,
    total_steps: int,
    step_names: List[str]
) -> None:
    """
    Display a progress indicator showing which step we're on.
    
    PARAMETERS:
    -----------
    current_step : int
        Current step number (1-indexed)
    total_steps : int
        Total number of steps
    step_names : List[str]
        Names of each step
    """
    progress = current_step / total_steps
    
    st.progress(progress)
    
    # Step indicators
    cols = st.columns(total_steps)
    
    for i, (col, name) in enumerate(zip(cols, step_names)):
        step_num = i + 1
        
        with col:
            if step_num < current_step:
                # Completed
                st.markdown(f"✓ **{name}**")
            elif step_num == current_step:
                # Current
                st.markdown(f"➤ **{name}**")
            else:
                # Upcoming
                st.markdown(f"○ {name}")


def display_key_insight(
    insight: str,
    insight_type: str = "info"
) -> None:
    """
    Display a key insight prominently.
    
    PARAMETERS:
    -----------
    insight : str
        The insight text
    insight_type : str
        One of "info", "success", "warning", "error"
    """
    if insight_type == "success":
        st.success(f"💡 **Key Insight:** {insight}")
    elif insight_type == "warning":
        st.warning(f"⚠️ **Caution:** {insight}")
    elif insight_type == "error":
        st.error(f"🚨 **Warning:** {insight}")
    else:
        st.info(f"💡 **Key Insight:** {insight}")


def display_comparison_table(
    bucket1: str,
    bucket2: str,
    metrics: Dict[str, tuple],
    title: str = "Comparison"
) -> None:
    """
    Display a comparison table between two buckets.
    
    PARAMETERS:
    -----------
    bucket1, bucket2 : str
        Names of the buckets being compared
    metrics : Dict[str, tuple]
        Dictionary of metric names to (value1, value2) tuples
    title : str
        Table title
    """
    st.markdown(f"**{title}**")
    
    cols = st.columns([2, 1, 1, 1])
    
    # Header
    cols[0].markdown("**Metric**")
    cols[1].markdown(f"**{bucket1}**")
    cols[2].markdown(f"**{bucket2}**")
    cols[3].markdown("**Difference**")
    
    # Rows
    for metric_name, (val1, val2) in metrics.items():
        cols = st.columns([2, 1, 1, 1])
        cols[0].write(metric_name)
        cols[1].write(f"{val1}")
        cols[2].write(f"{val2}")
        
        # Calculate difference
        try:
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                diff = val1 - val2
                if diff > 0:
                    cols[3].markdown(f":green[+{diff:.2f}]")
                elif diff < 0:
                    cols[3].markdown(f":red[{diff:.2f}]")
                else:
                    cols[3].write("0")
            else:
                cols[3].write("-")
        except:
            cols[3].write("-")


def display_sample_size_warning(n: int, min_recommended: int = 30) -> None:
    """
    Display a warning if sample size is small.
    
    PARAMETERS:
    -----------
    n : int
        Actual sample size
    min_recommended : int
        Minimum recommended sample size
    """
    if n < min_recommended:
        st.warning(
            f"⚠️ **Small sample size ({n:,})**: Results may be unreliable. "
            f"We recommend at least {min_recommended:,} observations for robust analysis."
        )


def display_limitations_section() -> None:
    """
    Display a section explaining limitations of the analysis.
    """
    with st.expander("⚠️ Limitations of This Analysis", expanded=False):
        st.markdown("""
        ### What This Analysis Cannot Tell You
        
        **1. Causation vs Correlation**
        - This is observational data, not a controlled experiment
        - We can identify associations, but proving causation requires more
        - This analysis has limitations in establishing causation
        
        **2. Unmeasured Confounders**
        - We control for lead source and (optionally) sales rep
        - But other factors may still confound results:
          - Time of day
          - Lead quality signals we don't measure
          - Market conditions
        
        **3. External Validity**
        - Results are specific to your data and time period
        - May not generalize to different markets or conditions
        
        **4. Selection Bias**
        - We only analyze leads that received responses
        - Leads that were never contacted may be different
        
        ### Recommendations for Stronger Conclusions
        
        1. **Acknowledge limitations**: This observational analysis cannot definitively establish causation
        2. **Collect more covariates**: Time of day, lead score, etc.
        3. **Replicate over time**: See if effects are consistent
        4. **Check for dose-response**: Does faster = better continuously?
        """)

