# =============================================================================
# Results Dashboard Component
# =============================================================================
# This component creates the final results dashboard showing all analysis output.
#
# WHY THIS COMPONENT EXISTS:
# --------------------------
# After running all the analysis, we need to present results in a clear,
# actionable format. This component:
# 1. Shows the key headline findings
# 2. Walks through each analysis step
# 3. Provides actionable recommendations
# 4. Allows exporting results
#
# MAIN FUNCTIONS:
# ---------------
# - render_results_dashboard(): Main dashboard
# - render_executive_summary(): Key takeaways for decision makers
# - render_analysis_steps(): Step-by-step analysis with explanations
# =============================================================================

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from components.step_display import (
    display_step, display_key_insight, display_significance_badge,
    display_limitations_section, display_progress_indicator
)
from components.charts import (
    create_close_rate_chart, create_sample_size_chart, create_heatmap,
    create_forest_plot, create_rep_scatter, create_response_time_distribution
)
from analysis.descriptive import calculate_close_rates, calculate_rep_performance
from analysis.statistical_tests import run_all_statistical_tests
from analysis.regression import run_logistic_regression, get_model_summary_table
from analysis.weekly_trends import (
    analyze_weekly_trends, 
    format_weekly_stats_for_display,
    get_wow_comparison,
    get_available_weeks,
    run_weekly_statistical_analysis,
    compare_two_weeks
)
from explanations.explainers import (
    get_p_value_explanation,
    render_p_value_explainer,
    get_odds_ratio_explanation,
    render_odds_ratio_table,
    get_ci_explanation,
    render_ci_explainer,
    render_percentage_points_explainer,
    render_chi_square_walkthrough,
    render_regression_explainer,
    render_sample_size_guidance,
    render_bucket_sample_sizes,
    render_key_finding,
    format_wow_change,
    get_step_bridge,
    generate_chi_square_worked_example,
    render_chi_square_worked_example,
    generate_proportion_test_worked_example,
    generate_week_analysis_story,
    generate_week_comparison_story,
    get_week_educational_context,
    render_week_analysis_educational_intro,
    generate_weekly_chi_square_worked_example,
    render_weekly_chi_square_worked_example,
    render_weekly_close_rate_calculations,
    render_weekly_proportion_test_calculations
)
from explanations.verification_panels import (
    render_chi_square_verification,
    render_z_test_verification,
    render_regression_verification,
    render_effect_size_verification,
    render_ci_verification,
    render_bucketing_verification
)
from data.export import create_verification_csv
from explanations.common import (
    detect_non_monotonic_pattern,
    render_non_monotonic_pattern_explanation,
    render_minimal_sample_warning,
    render_ci_significance_contradiction,
    render_interaction_effect_explanation,
    detect_imbalance,
    render_imbalance_warning,
    render_no_controls_explanation,
    render_extreme_rate_guidance,
    detect_narrative_scenario
)


def render_results_dashboard(
    df: pd.DataFrame,
    settings: Dict[str, Any]
) -> None:
    """
    Render the complete results dashboard.
    
    WHY THIS FUNCTION:
    ------------------
    This is the main output of the app. It presents all analysis results
    in a logical, educational flow that helps users understand and act on findings.
    
    WHAT IT DISPLAYS:
    -----------------
    1. Executive Summary (key findings)
    2. Data Overview
    3. Step-by-step analysis with explanations
    4. Recommendations
    5. Limitations
    6. Export options
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        Preprocessed DataFrame ready for analysis
    settings : Dict[str, Any]
        Analysis settings from the settings panel
    """
    st.header("📊 Analysis Results")
    
    # Get settings
    show_math = settings.get('show_math', True)
    alpha = settings.get('alpha_level', 0.05)
    
    # =========================================================================
    # RUN ALL ANALYSES
    # =========================================================================
    with st.spinner("Running analysis..."):
        # Descriptive
        close_rates = calculate_close_rates(df, confidence_level=settings.get('confidence_level', 0.95))
        
        # Statistical tests
        stat_results = run_all_statistical_tests(df, alpha)
        
        # Regression
        regression_result = run_logistic_regression(df, include_lead_source=True)
        
        # Weekly trends (if we have date data)
        weekly_analysis = None
        if 'lead_time' in df.columns:
            try:
                weekly_analysis = analyze_weekly_trends(df)
            except Exception as e:
                # Don't fail if weekly analysis can't run
                st.warning(f"Could not run weekly analysis: {e}")
    
    # =========================================================================
    # DETECT NARRATIVE SCENARIO
    # =========================================================================
    # Detect what type of story we're telling to adapt narrative accordingly
    narrative_scenario = detect_narrative_scenario(
        close_rates, 
        stat_results['chi_square'], 
        regression_result,
        p_value=stat_results['chi_square'].p_value
    )
    
    # If non-monotonic pattern, introduce it early
    if narrative_scenario.get('is_non_monotonic'):
        pattern_info = narrative_scenario.get('pattern_info', {})
        if pattern_info.get('pattern_type') == 'inverted_u':
            st.info(f"""
            **🔍 Special Pattern Detected: Optimal Response Window**
            
            Your data reveals a non-linear relationship: **{pattern_info.get('peak_bucket', 'middle buckets')}** 
            performs best, rather than "faster is always better." This suggests there may be an optimal 
            response time window. See the detailed analysis below for more insights.
            """)
        elif pattern_info.get('pattern_type') == 'u_shaped':
            st.warning(f"""
            **🔍 Special Pattern Detected: Suboptimal Middle Ground**
            
            Your data shows a U-shaped pattern where both very fast and very slow responses outperform 
            medium-speed responses. This suggests medium response times represent a "worst of both worlds" 
            scenario. See the detailed analysis below for more insights.
            """)
    
    # =========================================================================
    # EXECUTIVE SUMMARY
    # =========================================================================
    render_executive_summary(close_rates, stat_results, regression_result)
    
    # =========================================================================
    # YOUR QUESTIONS ANSWERED - Direct answers with actual data
    # =========================================================================
    st.markdown("---")
    render_questions_answered(df, close_rates, stat_results, regression_result)
    
    # =========================================================================
    # STEP-BY-STEP ANALYSIS
    # =========================================================================
    st.markdown("---")
    st.subheader("📈 Detailed Analysis")
    
    # Define steps - always include Weekly Trends if we have date data
    step_names = ['Data Overview', 'Close Rates', 'Chi-Square', 'Proportions', 'Regression']
    if weekly_analysis:
        step_names.append('📅 Weekly Trends')
    
    # Create tabs for each step
    tabs = st.tabs(step_names)
    
    # Step 1: Data Overview
    with tabs[0]:
        render_data_overview_step(df, close_rates)
    
    # Step 2: Close Rates
    with tabs[1]:
        render_close_rates_step(df, close_rates, show_math)
    
    # Step 3: Chi-Square Test
    with tabs[2]:
        render_chi_square_step(df, stat_results, show_math)
    
    # Step 4: Proportion Tests
    with tabs[3]:
        render_proportions_step(df, stat_results, show_math)
    
    # Step 5: Regression
    with tabs[4]:
        render_regression_step(df, regression_result, show_math)
    
    # Step 6: Weekly Trends (if available)
    if weekly_analysis:
        with tabs[5]:
            render_weekly_trends_step(weekly_analysis, df)
    
    # =========================================================================
    # EXPORT OPTIONS
    # =========================================================================
    st.markdown("---")
    render_export_options(df, close_rates, stat_results, regression_result)


def render_executive_summary(
    close_rates: pd.DataFrame,
    stat_results: Dict[str, Any],
    regression_result
) -> None:
    """
    Render the executive summary using methodical, first-principles reasoning.
    """
    st.subheader("📋 Executive Summary")
    
    # Calculate key metrics
    fastest_bucket = close_rates.iloc[0]
    slowest_bucket = close_rates.iloc[-1]
    rate_diff = fastest_bucket['close_rate'] - slowest_bucket['close_rate']
    rate_multiplier = fastest_bucket['close_rate'] / slowest_bucket['close_rate'] if slowest_bucket['close_rate'] > 0 else 1
    
    chi_sq = stat_results['chi_square']
    p_value = chi_sq.p_value
    p_exp = get_p_value_explanation(p_value)
    
    # =========================================================================
    # THE CENTRAL QUESTION AND CONCLUSION
    # =========================================================================
    # Detect scenario for adaptive opening
    is_reverse = rate_diff < 0
    abs_rate_diff = abs(rate_diff)
    is_exceptional = (p_value < 0.0001 and abs_rate_diff > 5) or abs_rate_diff > 10
    
    st.markdown("### The Central Question")
    
    # Adapt opening hook based on scenario
    if is_reverse and chi_sq.is_significant:
        st.markdown("""
        This analysis addresses a fundamental business question: **Does responding faster 
        to leads cause higher conversion rates?**
        
        **⚠️ Unexpected Finding:** Your data suggests something surprising — the relationship 
        between response time and conversion may not follow conventional wisdom. This analysis 
        will help you understand what your data is actually saying.
        """)
    elif is_exceptional and chi_sq.is_significant:
        st.markdown("""
        This analysis addresses a fundamental business question: **Does responding faster 
        to leads cause higher conversion rates?**
        
        **🚀 Strong Signal Detected:** Your data shows an unusually powerful pattern that 
        could have significant business implications. This analysis will explore whether 
        this effect is real and actionable.
        """)
    elif not chi_sq.is_significant:
        st.markdown("""
        This analysis addresses a fundamental business question: **Does responding faster 
        to leads cause higher conversion rates?**
        
        **📊 Data-Driven Investigation:** Your data may or may not show a clear relationship. 
        This analysis will help you understand what conclusions you can reliably draw and 
        what remains uncertain.
        """)
    else:
        st.markdown("""
        This analysis addresses a fundamental business question: **Does responding faster 
        to leads cause higher conversion rates?**
        
        The answer to this question has direct operational implications. If speed causes 
        success, investments in response time infrastructure will yield measurable returns. 
        If the correlation is spurious, such investments would be misdirected.
        """)
    
    st.markdown("---")
    st.markdown("### The Conclusion")
    
    # Detect scenario type for adaptive narrative
    is_reverse = rate_diff < 0
    abs_rate_diff = abs(rate_diff)
    is_exceptional = (p_value < 0.0001 and abs_rate_diff > 5) or abs_rate_diff > 10
    
    if chi_sq.is_significant and regression_result.is_response_time_significant:
        # Adapt opening based on effect size
        if is_exceptional and not is_reverse:
            opening = f"""
            **🚀 Exceptional Finding: Response Speed Shows Dramatic Impact**
            
            Your data reveals an unusually strong association between response speed and conversion — 
            one of the strongest effects observed in response time analysis.
            """
            tone = "success"
        elif is_reverse:
            opening = f"""
            **⚠️ Surprising Finding: Conventional Wisdom Challenged**
            
            Your data shows a counterintuitive pattern: slower responses are associated with higher 
            conversion rates. This contradicts expected behavior and requires careful investigation.
            """
            tone = "warning"
        else:
            opening = f"""
            **The evidence demonstrates a statistically significant association between response speed and conversion.**
            """
            tone = "success"
        
        if tone == "success":
            st.success(opening)
        else:
            st.warning(opening)
        
        st.markdown(f"""
        Leads receiving responses within **{fastest_bucket['bucket']}** convert at 
        **{fastest_bucket['close_rate']*100:.1f}%** — a rate **{rate_multiplier:.1f}× {'higher' if not is_reverse else 'lower'}** than 
        leads waiting **{slowest_bucket['bucket']}** ({slowest_bucket['close_rate']*100:.1f}%).
        
        This pattern is:
        - **Statistically significant** — the probability of observing this by chance is {p_exp['luck_chance']}
        - **Robust to observed confounders** — the association persists after controlling for lead source
        - **Practically meaningful** — a {abs_rate_diff*100:.1f} percentage point difference translates 
          to substantial revenue impact at scale
        
        **Recommendation:** While the observational evidence is strong, this analysis cannot definitively 
        establish causation. The association could be explained by unmeasured confounders or selection mechanisms. 
        Consider this evidence when making decisions, but acknowledge the limitations of observational data.
        """)
    elif chi_sq.is_significant and not regression_result.is_response_time_significant:
        st.warning(f"""
        **A correlation exists, but causation is uncertain.**
        
        Fast responders ({fastest_bucket['bucket']}) convert at {fastest_bucket['close_rate']*100:.1f}% 
        versus {slowest_bucket['close_rate']*100:.1f}% for slow responders ({slowest_bucket['bucket']}). 
        This difference is statistically significant.
        
        However, when we control for lead source, the effect weakens. This suggests that 
        confounding may explain part of the correlation — different lead types may receive 
        different response times *and* have different conversion rates, independent of speed.
        
        **Recommendation:** This observational analysis cannot definitively establish causation. 
        Consider the evidence when making decisions, but acknowledge that confounding may explain the relationship.
        """)
    else:
        st.info(f"""
        **Insufficient evidence to conclude that response speed affects outcomes.**
        
        The observed difference of {rate_diff*100:.1f} percentage points between fast and 
        slow responders is not statistically significant. The probability that this 
        difference arose from random sampling variation is {p_exp['luck_chance']}.
        
        This does not prove that speed has *no* effect — only that the current data 
        cannot reliably distinguish a true effect from noise.
        
        **Recommendation:** Either the effect is negligible, or additional data is needed 
        to detect it. Focus optimization efforts on higher-certainty opportunities.
        """)
    
    # =========================================================================
    # KEY METRICS
    # =========================================================================
    st.markdown("### The Evidence")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Fast Response Rate",
            f"{fastest_bucket['close_rate']*100:.1f}%",
            help=f"Conversion rate for leads receiving response within {fastest_bucket['bucket']}"
        )
        st.caption(f"Response within {fastest_bucket['bucket']}")
    
    with col2:
        st.metric(
            "Slow Response Rate", 
            f"{slowest_bucket['close_rate']*100:.1f}%",
            help=f"Conversion rate for leads waiting {slowest_bucket['bucket']} or longer"
        )
        st.caption(f"Response after {slowest_bucket['bucket']}")
    
    with col3:
        st.metric(
            "Absolute Difference",
            f"{rate_diff*100:.1f}pp",
            f"{rate_multiplier:.1f}× ratio",
            help="Percentage point difference between fastest and slowest response groups"
        )
        st.caption("Percentage points")
    
    with col4:
        st.metric(
            "Statistical Significance",
            p_exp['confidence'],
            p_exp['verdict'],
            help=f"Confidence that the association is not due to random chance (p-value: {p_value:.4f}). This measures statistical significance of the association, not causal certainty.",
            delta_color="normal" if chi_sq.is_significant else "off"
        )
        st.caption(f"Chance of luck: {p_exp['luck_chance']}")
    
    # =========================================================================
    # DETAILED BREAKDOWN
    # =========================================================================
    with st.expander("📊 Detailed Breakdown of the Evidence", expanded=True):
        st.markdown(f"""
        ### The Methodology
        
        We conducted a rigorous analysis comparing conversion rates across response time categories, 
        then applied multiple statistical techniques to assess the reliability of our findings.
        
        ### The Data
        
        | Response Speed | Conversion Rate | Sample Size | Conversions |
        |:---------------|:----------------|:------------|:------------|
        | **{fastest_bucket['bucket']}** | **{fastest_bucket['close_rate']*100:.1f}%** | {int(fastest_bucket['n_leads']):,} leads | {int(fastest_bucket['n_orders']):,} |
        | **{slowest_bucket['bucket']}** | **{slowest_bucket['close_rate']*100:.1f}%** | {int(slowest_bucket['n_leads']):,} leads | {int(slowest_bucket['n_orders']):,} |
        | **Difference** | **{abs(rate_diff)*100:.1f} pp** | — | **{int(abs(rate_diff) * fastest_bucket['n_leads'])} additional per {int(fastest_bucket['n_leads']):,}** |
        
        ### The Statistical Assessment
        
        **Primary Question: Is this pattern real or random noise?**
        
        {p_exp['emoji']} **{p_exp['verdict']}**
        
        {p_exp['plain_english']}
        
        {p_exp.get('first_principles', '')}
        
        ### The Confounding Check
        
        {"**Association persists after controlling for lead source.** We tested whether lead source differences could explain this pattern. The association between response speed and conversion remains after controlling for lead source. However, this only addresses *one* potential confounder. Unmeasured confounders (such as lead quality signals that drive prioritization, salesperson skill differences, or time-of-day effects) may still explain the relationship. This observational analysis cannot definitively establish causation." if regression_result.is_response_time_significant else "**Caution warranted.** When we control for lead source, the association weakens. Some of the apparent relationship may be attributable to confounding rather than a true causal mechanism. This observational analysis has limitations in establishing causation."}
        
        ### Causal Inference Limitations
        
        **This analysis cannot establish causation.** This is observational data, not experimental. The observed association could be explained by:
        
        - **Selection mechanisms:** Salespeople may prioritize leads that appear more likely to convert, creating a spurious correlation between speed and success
        - **Unmeasured confounders:** Lead quality signals, salesperson skill differences, or time-of-day effects that we cannot observe or control for
        - **Reverse causation:** Higher-value leads may receive faster responses, rather than speed causing higher conversion
        
        **What we can conclude:** There is a statistically significant association between response time and conversion rates that persists after controlling for lead source. This is consistent with a causal effect, but observational data alone cannot prove it.
        
        **Limitation:** This observational analysis cannot definitively establish causation. When making decisions about staffing or infrastructure investments, consider this evidence while acknowledging that unmeasured confounders may explain the relationship.
        """)


def render_questions_answered(
    df: pd.DataFrame,
    close_rates: pd.DataFrame,
    stat_results: Dict[str, Any],
    regression_result
) -> None:
    """
    Render direct answers to the user's core questions using their actual data.
    
    This section directly addresses the questions users came to answer:
    - Is it beneficial to reach out in 15 minutes?
    - Is under an hour sufficient?
    - How do you know?
    """
    st.subheader("❓ Your Questions Answered")
    
    st.markdown("""
    *Before diving into the detailed analysis, here are direct answers to the questions 
    you likely came here to answer — backed by your actual data.*
    """)
    
    # Extract key data points for fastest bucket (0-15 min)
    fastest = close_rates.iloc[0]
    fast_sales = int(fastest['n_orders'])
    fast_leads = int(fastest['n_leads'])
    fast_rate = fastest['close_rate']
    
    # Find the 15-60 min bucket (typically second)
    medium_bucket = None
    for _, row in close_rates.iterrows():
        bucket_name = row['bucket'].lower()
        if '15' in bucket_name and '60' in bucket_name:
            medium_bucket = row
            break
        elif '16' in bucket_name or '15-' in bucket_name:
            medium_bucket = row
            break
    
    # If we can't find a medium bucket, use the second one
    if medium_bucket is None and len(close_rates) > 1:
        medium_bucket = close_rates.iloc[1]
    
    # Extract slowest bucket (60+ min or last bucket)
    slowest = close_rates.iloc[-1]
    slow_sales = int(slowest['n_orders'])
    slow_leads = int(slowest['n_leads'])
    slow_rate = slowest['close_rate']
    
    # Calculate the difference
    rate_diff = fast_rate - slow_rate
    rate_multiplier = fast_rate / slow_rate if slow_rate > 0 else float('inf')
    
    # Get chi-square results for confidence
    chi_sq = stat_results['chi_square']
    p_exp = get_p_value_explanation(chi_sq.p_value)
    
    # =========================================================================
    # QUESTION 1: Is it beneficial to reach out in 15 minutes?
    # =========================================================================
    st.markdown("---")
    st.markdown("### Q: Is it beneficial to reach out within 15 minutes?")
    
    # Detect reverse effect
    is_reverse = rate_diff < 0
    abs_rate_diff = abs(rate_diff)
    
    # Handle reverse effect scenario
    if chi_sq.is_significant and is_reverse and abs_rate_diff > 0.02:
        st.warning(f"""
        **A: Surprisingly, the data suggests slower responses may be better.**
        
        ⚠️ **This is a counterintuitive finding** that contradicts conventional wisdom. 
        Your data shows that leads receiving slower responses ({slowest['bucket']}) actually 
        convert at a **higher rate** ({slow_rate*100:.1f}%) than faster responses 
        ({fast_rate*100:.1f}%).
        """)
        
        st.markdown("#### Here's what your data shows:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"**Fastest responses ({fastest['bucket']})**")
            st.markdown(f"""
            - Sales: **{fast_sales:,}**
            - Leads: **{fast_leads:,}**
            - Close rate: **{fast_rate*100:.1f}%**
            """)
        
        with col2:
            st.markdown(f"**Slowest responses ({slowest['bucket']})**")
            st.markdown(f"""
            - Sales: **{slow_sales:,}**
            - Leads: **{slow_leads:,}**
            - Close rate: **{slow_rate*100:.1f}%**
            """)
        
        with col3:
            st.markdown("**The Surprising Difference**")
            st.markdown(f"""
            - Gap: **{abs_rate_diff*100:.1f} percentage points**
            - Slower is **{slow_rate/fast_rate:.1f}× better** (unexpected!)
            - Confidence: **{p_exp['confidence']}**
            """)
        
        st.warning("""
        ⚠️ **Important:** This finding requires careful investigation before acting. The pattern 
        may be due to confounding (e.g., better leads get slower but more thoughtful responses) 
        rather than speed itself. Do NOT slow down responses based on this alone.
        """)
    
    elif chi_sq.is_significant and rate_diff > 0:
        st.success(f"""
        **A: Yes.** Your data shows a clear benefit to faster response.
        """)
        
        # Show the actual calculation
        st.markdown("#### Here's the math with your actual numbers:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"**Your fastest responders ({fastest['bucket']})**")
            st.markdown(f"""
            - Sales: **{fast_sales:,}**
            - Leads: **{fast_leads:,}**
            - Close rate: **{fast_sales:,} ÷ {fast_leads:,} = {fast_rate*100:.1f}%**
            """)
        
        with col2:
            st.markdown(f"**Your slowest responders ({slowest['bucket']})**")
            st.markdown(f"""
            - Sales: **{slow_sales:,}**
            - Leads: **{slow_leads:,}**
            - Close rate: **{slow_sales:,} ÷ {slow_leads:,} = {slow_rate*100:.1f}%**
            """)
        
        with col3:
            st.markdown("**The Difference**")
            st.markdown(f"""
            - Gap: **{rate_diff*100:.1f} percentage points**
            - Multiplier: **{rate_multiplier:.1f}× better**
            - Confidence: **{p_exp['confidence']}**
            """)
        
        # Concrete impact
        st.markdown("#### What this means in practice:")
        extra_sales_per_100 = rate_diff * 100
        st.info(f"""
        📊 **For every 100 leads**, responding within **{fastest['bucket']}** instead of 
        **{slowest['bucket']}** is associated with approximately **{extra_sales_per_100:.0f} additional sales**.
        """)
    else:
        st.warning("""
        **A: The data does not show a statistically significant benefit.**
        
        While there may be observed differences, they could be due to random variation.
        See the detailed analysis below for more information.
        """)
    
    # =========================================================================
    # QUESTION 2: Is under an hour sufficient?
    # =========================================================================
    st.markdown("---")
    st.markdown("### Q: Is responding under an hour sufficient, or do you need to be faster?")
    
    # Show the step-down in performance
    st.markdown("#### Performance by response time bucket:")
    
    # Create a simple comparison table
    comparison_data = []
    prev_rate = None
    
    for _, row in close_rates.iterrows():
        bucket_rate = row['close_rate']
        bucket_data = {
            'Response Time': row['bucket'],
            'Close Rate': f"{bucket_rate*100:.1f}%",
            'Sales': f"{int(row['n_orders']):,}",
            'Leads': f"{int(row['n_leads']):,}"
        }
        
        if prev_rate is not None:
            drop = prev_rate - bucket_rate
            bucket_data['Drop from Previous'] = f"↓ {drop*100:.1f}pp"
        else:
            bucket_data['Drop from Previous'] = "—"
        
        comparison_data.append(bucket_data)
        prev_rate = bucket_rate
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Interpretation
    if medium_bucket is not None:
        medium_rate = medium_bucket['close_rate']
        gap_fast_vs_medium = fast_rate - medium_rate
        gap_medium_vs_slow = medium_rate - slow_rate
        
        if gap_fast_vs_medium > 0.01:  # More than 1pp difference
            st.markdown(f"""
            **Interpretation:** There's a noticeable step-down at each tier:
            - **{fastest['bucket']}** → **{medium_bucket['bucket']}**: Drop of **{gap_fast_vs_medium*100:.1f}** percentage points
            - **{medium_bucket['bucket']}** → **{slowest['bucket']}**: Drop of **{gap_medium_vs_slow*100:.1f}** percentage points
            
            ⚡ **Bottom line:** Under an hour is better than over an hour, but under 15 minutes is 
            better still. Speed matters at every threshold.
            """)
        else:
            st.markdown(f"""
            **Interpretation:** The biggest gap is between under an hour and over an hour.
            
            ⚡ **Bottom line:** Getting under an hour is the most important threshold. 
            Additional speed gains show diminishing returns in your data.
            """)
    
    # =========================================================================
    # QUESTION 3: How do you know?
    # =========================================================================
    st.markdown("---")
    st.markdown("### Q: How do you know this isn't just coincidence?")
    
    st.markdown("""
    We ran **three statistical tests** to validate this finding:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        chi_status = "✅" if chi_sq.is_significant else "⚠️"
        st.markdown(f"""
        **{chi_status} Test 1: Overall Pattern**
        
        *Chi-square test*
        
        Is there ANY relationship between 
        speed and conversion?
        
        **Result:** p = {chi_sq.p_value:.4f}
        {p_exp['verdict']}
        """)
    
    with col2:
        # Get the pairwise test for fastest vs slowest
        pairwise = stat_results.get('pairwise_comparisons', [])
        key_comparison = None
        for comp in pairwise:
            if 'adjusted_p' in comp:
                key_comparison = comp
                break
        
        if key_comparison:
            pair_status = "✅" if key_comparison.get('adjusted_p', 1) < 0.05 else "⚠️"
            st.markdown(f"""
            **{pair_status} Test 2: Direct Comparison**
            
            *Z-test for proportions*
            
            Is fast specifically better 
            than slow?
            
            **Result:** Adjusted p = {key_comparison.get('adjusted_p', 'N/A'):.4f}
            {'Statistically significant' if key_comparison.get('adjusted_p', 1) < 0.05 else 'Not significant'}
            """)
        else:
            st.markdown("""
            **Test 2: Direct Comparison**
            
            *Z-test for proportions*
            
            See detailed analysis below.
            """)
    
    with col3:
        reg_status = "✅" if regression_result.is_response_time_significant else "⚠️"
        st.markdown(f"""
        **{reg_status} Test 3: Controlling for Confounds**
        
        *Logistic regression*
        
        Does the effect hold when we 
        account for lead source?
        
        **Result:** {'Significant after controls' if regression_result.is_response_time_significant else 'Not significant after controls'}
        """)
    
    st.info("""
    📚 **Want to understand the math?** Each test is explained in detail in the 
    "Detailed Analysis" section below, with step-by-step calculations using your actual data.
    """)


def render_data_overview_step(df: pd.DataFrame, close_rates: pd.DataFrame) -> None:
    """Render the data overview step with first-principles explanation."""
    # Bridge sentence to maintain narrative flow
    st.info(f"💡 **Why this step:** {get_step_bridge('data_overview')}")
    
    st.markdown("### Step 1: Establishing the Foundation")
    
    st.markdown("""
    Before any meaningful analysis can proceed, we must first understand the nature 
    and quality of our data. Statistical conclusions are only as reliable as the data 
    underlying them. This step establishes what we are working with.
    """)
    
    # Key metrics with context
    st.markdown("#### Core Dataset Characteristics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Observations", f"{len(df):,}")
        st.caption("Individual leads analyzed")
    
    with col2:
        st.metric("Positive Outcomes", f"{df['ordered'].sum():,}")
        st.caption("Leads that converted")
    
    with col3:
        overall_rate = df['ordered'].mean()
        st.metric("Baseline Conversion Rate", f"{overall_rate*100:.1f}%")
        st.caption("Overall probability of conversion")
    
    with col4:
        st.metric("Analysis Categories", len(close_rates))
        st.caption("Response time groups")
    
    # Check for extreme conversion rates
    if overall_rate > 0.90 or overall_rate < 0.05:
        st.markdown("---")
        render_extreme_rate_guidance(overall_rate, len(df))
    
    # Data quality assessment
    st.markdown("---")
    render_sample_size_guidance(len(df), "conversion rate")
    
    # =========================================================================
    # VERIFICATION PANEL - Bucketing
    # =========================================================================
    render_bucketing_verification(df)
    
    # Distribution analysis
    st.markdown("---")
    st.markdown("#### Response Time Distribution")
    
    st.markdown("""
    Understanding the distribution of response times is essential. The distribution 
    reveals both the central tendency and the variability in current practices.
    
    The histogram below shows how response times are distributed across leads. 
    The vertical dashed lines indicate the categorical boundaries used in this analysis.
    """)
    
    fig = create_response_time_distribution(df)
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpret the distribution
    median_response = df['response_time_mins'].median()
    pct_under_15 = (df['response_time_mins'] <= 15).mean() * 100
    pct_over_60 = (df['response_time_mins'] > 60).mean() * 100
    mean_response = df['response_time_mins'].mean()
    
    st.markdown(f"""
    **Distribution Summary:**
    
    | Metric | Value | Interpretation |
    |:-------|:------|:---------------|
    | **Median** | {median_response:.0f} min | Half of leads receive responses faster than this |
    | **Mean** | {mean_response:.0f} min | Arithmetic average (sensitive to outliers) |
    | **Fast responses (<15 min)** | {pct_under_15:.1f}% | Proportion in the fastest category |
    | **Slow responses (>60 min)** | {pct_over_60:.1f}% | Proportion in the slowest category |
    
    {'The distribution is right-skewed, with a long tail of very slow responses. This is typical for response time data.' if mean_response > median_response * 1.2 else 'The distribution is relatively symmetric.'}
    """)
    
    # Sample sizes with quality notes
    st.markdown("---")
    st.markdown("#### Sample Size by Category")
    
    st.markdown("""
    **A Critical Consideration:**
    
    The reliability of our estimates depends fundamentally on sample size. Categories 
    with few observations will have wide confidence intervals and unstable estimates. 
    Before proceeding, we must verify that each category contains sufficient data.
    """)
    
    fig = create_sample_size_chart(close_rates)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show data quality by bucket
    render_bucket_sample_sizes(close_rates)
    
    # Check for minimal sample sizes and warn if needed
    min_sample = close_rates['n_leads'].min()
    if min_sample < 10:
        st.markdown("---")
        render_minimal_sample_warning(int(min_sample), "one or more response time buckets")
    
    # Check for imbalanced bucket distributions
    imbalance_info = detect_imbalance(close_rates)
    if imbalance_info.get('is_imbalanced'):
        st.markdown("---")
        render_imbalance_warning(imbalance_info)


def render_close_rates_step(df: pd.DataFrame, close_rates: pd.DataFrame, show_math: bool) -> None:
    """Render the close rates step."""
    # Bridge sentence to maintain narrative flow
    st.info(f"💡 **Why this step:** {get_step_bridge('close_rates')}")
    
    st.markdown("### Step 2: Close Rates by Response Time")
    
    st.markdown("""
    **The key question:** Do leads that get faster responses have higher close rates?
    
    Let's look at the numbers for each response time bucket.
    """)
    
    # Main chart with explanation
    fig = create_close_rate_chart(close_rates)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **Reading this chart:**
    - **Bar height** = Close rate (what % of leads became customers)
    - **Color** = Goes from green (fast) to red (slow)
    - **Small lines on top** = Uncertainty range (the true rate is probably within this range)
    """)
    
    # Detect and explain non-monotonic patterns (U-shaped, inverted-U, etc.)
    pattern_info = detect_non_monotonic_pattern(close_rates)
    if pattern_info.get('is_non_monotonic'):
        st.markdown("---")
        render_non_monotonic_pattern_explanation(pattern_info)
    
    # CI Explainer
    render_ci_explainer()
    
    # =========================================================================
    # VERIFICATION PANELS - Confidence Intervals
    # =========================================================================
    # Show verification for first bucket as example
    if len(close_rates) > 0 and 'ci_lower' in close_rates.columns:
        first_bucket = close_rates.iloc[0]
        render_ci_verification(
            n=int(first_bucket['n_leads']),
            p=first_bucket['close_rate'],
            ci_lower=first_bucket['ci_lower'],
            ci_upper=first_bucket['ci_upper'],
            confidence_level=0.95
        )
    
    # Data table with human-friendly columns
    with st.expander("📊 View Detailed Data Table"):
        st.markdown("""
        Here's all the data in table form. The "Range" columns show our uncertainty — 
        the true close rate is probably somewhere between those bounds.
        """)
        
        # Create human-friendly display
        display_rows = []
        for _, row in close_rates.iterrows():
            n = row['n_leads']
            if n >= 1000:
                data_quality = "✅ Excellent data"
            elif n >= 500:
                data_quality = "✓ Good data"
            elif n >= 200:
                data_quality = "⚠️ OK data"
            else:
                data_quality = "❌ Limited data"
            
            display_rows.append({
                'Response Time': row['bucket'],
                'Total Leads': f"{row['n_leads']:,}",
                'Sales Made': f"{row['n_orders']:,}",
                'Close Rate': f"{row['close_rate']*100:.1f}%",
                'Could Be As Low As': f"{row['ci_lower']*100:.1f}%",
                'Could Be As High As': f"{row['ci_upper']*100:.1f}%",
                'Data Quality': data_quality
            })
        
        st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)
    
    # Insight with concrete numbers
    fastest = close_rates.iloc[0]
    slowest = close_rates.iloc[-1]
    
    if fastest['close_rate'] > slowest['close_rate']:
        diff = fastest['close_rate'] - slowest['close_rate']
        multiplier = fastest['close_rate'] / slowest['close_rate'] if slowest['close_rate'] > 0 else 1
        
        # Calculate concrete impact
        extra_sales_per_100 = diff * 100
        
        if diff > 0.02:  # More than 2 percentage points
            st.success(f"""
            **📈 Clear pattern: Faster responses = Higher close rates**
            
            | Response Time | Close Rate | Per 100 Leads |
            |:--------------|:-----------|:--------------|
            | **{fastest['bucket']}** (fastest) | **{fastest['close_rate']*100:.1f}%** | {int(fastest['close_rate']*100)} sales |
            | **{slowest['bucket']}** (slowest) | **{slowest['close_rate']*100:.1f}%** | {int(slowest['close_rate']*100)} sales |
            | **Difference** | **{diff*100:.1f} points** | **{int(extra_sales_per_100)} extra sales** |
            
            Fast responders are **{multiplier:.1f}x more successful** than slow responders!
            """)
        else:
            st.info(f"""
            **📊 Small difference observed**
            
            Fast responders ({fastest['bucket']}) close at {fastest['close_rate']*100:.1f}% 
            vs {slowest['close_rate']*100:.1f}% for slow responders ({slowest['bucket']}).
            
            That's a {diff*100:.1f} percentage point difference — modest but potentially meaningful 
            if it holds up to statistical testing.
            """)


def render_chi_square_step(df: pd.DataFrame, stat_results: Dict[str, Any], show_math: bool) -> None:
    """Render the chi-square test step with plain-English explanation."""
    # Bridge sentence to maintain narrative flow
    st.info(f"💡 **Why this step:** {get_step_bridge('chi_square')}")
    
    chi_sq = stat_results['chi_square']
    p_value = chi_sq.p_value
    p_exp = get_p_value_explanation(p_value)
    
    st.markdown("### Step 3: Is the Pattern Real or Random Luck?")
    
    # Plain English explanation first - conversational and clear
    st.markdown("""
    **The question on everyone's mind:**
    We saw that fast responders close more deals. But here's the thing — 
    sometimes patterns appear just by chance, like flipping heads 7 times in a row.
    
    **So we need to ask:** Is this pattern *actually real*, or did we just get lucky with this particular data?
    
    **How we check:** We use a statistical test that answers: 
    *"If response time truly had NO effect, what's the probability we'd see differences this big?"*
    """)
    
    # Show result prominently with the new explainer format
    if chi_sq.is_significant:
        st.success(f"""
        **{p_exp['emoji']} The pattern is real!**
        
        | Question | Answer |
        |:---------|:-------|
        | Chance this is random luck | **{p_exp['luck_chance']}** |
        | Our confidence level | **{p_exp['confidence']}** |
        | Verdict | **{p_exp['verdict']}** |
        
        **What this means:** {p_exp['plain_english']}
        """)
    else:
        st.warning(f"""
        **{p_exp['emoji']} We can't be sure the pattern is real.**
        
        | Question | Answer |
        |:---------|:-------|
        | Chance this is random luck | **{p_exp['luck_chance']}** |
        | Our confidence level | **{p_exp['confidence']}** |
        | Verdict | **{p_exp['verdict']}** |
        
        **What this means:** {p_exp['plain_english']}
        """)
    
    # =========================================================================
    # SAMPLE INPUT DATA - Show actual data rows used in chi-square test
    # =========================================================================
    st.markdown("---")
    st.markdown("## 📊 Sample Input Data")
    
    st.markdown("""
    Below are sample rows from your actual data that were used in the chi-square test.
    This shows the raw values before they're aggregated into the contingency table.
    """)
    
    # Show sample rows if we have the required columns
    if 'response_bucket' in df.columns and 'ordered' in df.columns:
        chi_square_df = df[['response_bucket', 'ordered']].dropna()
        
        if len(chi_square_df) > 0:
            # Show sample rows
            sample_size = min(10, len(chi_square_df))
            sample_df = chi_square_df.head(sample_size).copy()
            
            # Make it more readable
            display_sample = sample_df.copy()
            display_sample['Ordered'] = display_sample['ordered'].apply(lambda x: 'Yes' if x == 1 else 'No')
            display_sample = display_sample[['response_bucket', 'Ordered']]
            display_sample.columns = ['Response Time Bucket', 'Ordered?']
            
            st.dataframe(display_sample, use_container_width=True, hide_index=True)
            
            st.caption(f"""
            Showing {sample_size} of {len(chi_square_df):,} total rows used in the chi-square test.
            """)
            
            # Show how this data becomes the contingency table
            st.markdown("#### How This Data Becomes the Contingency Table")
            st.markdown("""
            The chi-square test groups these individual rows into a contingency table by counting 
            how many leads in each response time bucket resulted in orders vs. no orders.
            """)
            
            # Show a small example of the grouping
            example_grouped = chi_square_df.head(20).groupby(['response_bucket', 'ordered']).size().reset_index(name='Count')
            example_grouped['Ordered'] = example_grouped['ordered'].apply(lambda x: 'Yes' if x == 1 else 'No')
            example_grouped = example_grouped[['response_bucket', 'Ordered', 'Count']]
            example_grouped.columns = ['Response Time Bucket', 'Ordered?', 'Count (in first 20 rows)']
            
            st.dataframe(example_grouped, use_container_width=True, hide_index=True)
            st.caption("Example: How the first 20 rows are grouped. The full contingency table uses all rows.")
    
    # =========================================================================
    # VISIBLE WORKED EXAMPLE - Show the calculation with actual data
    # =========================================================================
    st.markdown("---")
    
    # Generate and display the worked example using actual data
    worked_example = generate_chi_square_worked_example(df)
    render_chi_square_worked_example(worked_example)
    
    # =========================================================================
    # VERIFICATION PANEL - Chi-Square
    # =========================================================================
    render_chi_square_verification(df, chi_sq)
    
    # =========================================================================
    # COMPLETE DATA TABLES - Show all raw data used in the test
    # =========================================================================
    # #region agent log
    import json
    import time as time_module
    log_path = "/Users/noahdelacalzada/response_time_cl_investigation/debug.log"
    # #endregion
    
    st.markdown("---")
    st.markdown("### 📊 Complete Data Tables Used in This Test")
    
    st.markdown("""
    Below are the complete data tables that were used to calculate the chi-square statistic.
    You can verify every number used in the calculation.
    """)
    
    # Get the details from the test result
    if hasattr(chi_sq, 'details') and chi_sq.details:
        details = chi_sq.details
        
        # #region agent log
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps({"location": "results_dashboard.py:912", "message": "chi_sq details check", "data": {"has_details": bool(details), "details_keys": list(details.keys()) if isinstance(details, dict) else "not_dict", "details_type": str(type(details))}, "hypothesisId": "D", "timestamp": time_module.time(), "sessionId": "debug-session", "runId": "run1"}) + "\n")
        except Exception as e:
            pass  # Silent fail on logging
        # #endregion
        
        # 1. Observed Contingency Table
        st.markdown("#### 1. Observed Contingency Table (What Actually Happened)")
        st.markdown("""
        This table shows the actual counts: how many leads in each response time bucket 
        resulted in orders vs. no orders.
        """)
        
        if 'contingency_table' in details:
            contingency_dict = details['contingency_table']
            
            # #region agent log
            try:
                with open(log_path, "a") as f:
                    f.write(json.dumps({"location": "results_dashboard.py:930", "message": "contingency_dict loaded", "data": {"contingency_dict_empty": not bool(contingency_dict), "contingency_dict_keys": list(contingency_dict.keys()) if isinstance(contingency_dict, dict) else "not_dict", "contingency_dict_type": str(type(contingency_dict)), "sample_item": {k: str(v)[:100] for k, v in list(contingency_dict.items())[:2]} if isinstance(contingency_dict, dict) and len(contingency_dict) > 0 else "none"}, "hypothesisId": "A", "timestamp": time_module.time(), "sessionId": "debug-session", "runId": "run1"}) + "\n")
            except Exception as e:
                pass  # Silent fail on logging
            # #endregion
            
            # Convert to DataFrame for better display
            # Handle case where dict values might be dicts or numbers
            if contingency_dict:
                # Check if dictionary is transposed (keys are outcome values, not buckets)
                # pandas .to_dict() can create {column -> {index -> value}} orientation
                first_key = next(iter(contingency_dict.keys())) if contingency_dict else None
                is_transposed = (isinstance(first_key, bool) or first_key in [0, 1, False, True]) and \
                               isinstance(contingency_dict.get(first_key), dict) and \
                               all(isinstance(k, str) for k in contingency_dict.get(first_key, {}).keys())
                
                # #region agent log
                try:
                    with open(log_path, "a") as f:
                        f.write(json.dumps({"location": "results_dashboard.py:951", "message": "checking transpose", "data": {"is_transposed": is_transposed, "first_key": str(first_key), "first_key_type": str(type(first_key))}, "hypothesisId": "B", "timestamp": time_module.time(), "sessionId": "debug-session", "runId": "run1"}) + "\n")
                except Exception as e:
                    pass
                # #endregion
                
                # Transpose if needed: convert {False: {bucket: count}, True: {bucket: count}} 
                # to {bucket: {False: count, True: count}}
                if is_transposed:
                    transposed_dict = {}
                    for outcome_val, bucket_counts in contingency_dict.items():
                        for bucket, count in bucket_counts.items():
                            if bucket not in transposed_dict:
                                transposed_dict[bucket] = {}
                            transposed_dict[bucket][outcome_val] = count
                    contingency_dict = transposed_dict
                    
                    # #region agent log
                    try:
                        with open(log_path, "a") as f:
                            f.write(json.dumps({"location": "results_dashboard.py:965", "message": "after transpose", "data": {"transposed_keys": list(contingency_dict.keys())[:5], "sample_bucket": {k: str(v) for k, v in list(contingency_dict.values())[0].items()} if contingency_dict else "none"}, "hypothesisId": "B", "timestamp": time_module.time(), "sessionId": "debug-session", "runId": "run1"}) + "\n")
                    except Exception as e:
                        pass
                    # #endregion
                
                # Convert nested dict structure to proper DataFrame
                rows = []
                for bucket, values in contingency_dict.items():
                    # #region agent log
                    try:
                        with open(log_path, "a") as f:
                            f.write(json.dumps({"location": "results_dashboard.py:943", "message": "processing bucket", "data": {"bucket": str(bucket), "values_type": str(type(values)), "is_dict": isinstance(values, dict), "values_keys": list(values.keys()) if isinstance(values, dict) else "not_dict", "values_repr": str(values)[:200]}, "hypothesisId": "B,C", "timestamp": time_module.time(), "sessionId": "debug-session", "runId": "run1"}) + "\n")
                    except Exception as e:
                        pass  # Silent fail on logging
                    # #endregion
                    if isinstance(values, dict):
                        row = {'Response Time': bucket}
                        # Handle both boolean (True/False) and numeric (0/1) keys
                        no_order_0 = values.get(0, None)
                        no_order_false = values.get(False, None)
                        order_1 = values.get(1, None)
                        order_true = values.get(True, None)
                        
                        # #region agent log
                        try:
                            with open(log_path, "a") as f:
                                f.write(json.dumps({"location": "results_dashboard.py:956", "message": "key extraction attempt", "data": {"bucket": str(bucket), "no_order_0": no_order_0, "no_order_false": no_order_false, "order_1": order_1, "order_true": order_true, "all_keys": list(values.keys()), "all_keys_types": [str(type(k)) for k in values.keys()]}, "hypothesisId": "B", "timestamp": time_module.time(), "sessionId": "debug-session", "runId": "run1"}) + "\n")
                        except Exception as e:
                            pass  # Silent fail on logging
                        # #endregion
                        
                        row['No Order'] = values.get(0, values.get(False, 0))
                        row['Order'] = values.get(1, values.get(True, 0))
                        
                        # #region agent log
                        try:
                            with open(log_path, "a") as f:
                                f.write(json.dumps({"location": "results_dashboard.py:967", "message": "row values after extraction", "data": {"bucket": str(bucket), "no_order": row['No Order'], "order": row['Order']}, "hypothesisId": "E", "timestamp": time_module.time(), "sessionId": "debug-session", "runId": "run1"}) + "\n")
                        except Exception as e:
                            pass  # Silent fail on logging
                        # #endregion
                        
                        rows.append(row)
                    else:
                        # Single value case (shouldn't happen but handle it)
                        row = {'Response Time': bucket, 'No Order': 0, 'Order': values}
                        rows.append(row)
                
                contingency_df = pd.DataFrame(rows)
                contingency_df = contingency_df.set_index('Response Time')
                
                # Ensure we have both columns
                if 'No Order' not in contingency_df.columns:
                    contingency_df['No Order'] = 0
                if 'Order' not in contingency_df.columns:
                    contingency_df['Order'] = 0
                
                # Add totals
                contingency_df['Total Leads'] = contingency_df.sum(axis=1)
                contingency_df.loc['TOTAL'] = contingency_df.sum(axis=0)
                
                st.dataframe(contingency_df, use_container_width=True)
            
            st.caption("""
            **How to read:** Each cell shows the count of leads. For example, if the 
            "0-15 min" row shows "No Order: 500, Order: 100", that means 500 leads 
            in that bucket didn't result in orders, and 100 did.
            """)
        
        # 2. Expected Frequencies Table
        st.markdown("---")
        st.markdown("#### 2. Expected Frequencies Table (What We'd Expect If Speed Didn't Matter)")
        st.markdown("""
        This table shows what we would expect to see if response time had no effect 
        on close rates. Each cell is calculated as: 
        **(Row Total × Column Total) ÷ Grand Total**
        """)
        
        if 'expected_frequencies' in details:
            expected_dict = details['expected_frequencies']
            # Handle nested dict structure
            if expected_dict:
                # Check if dictionary is transposed (same issue as contingency_table)
                first_key = next(iter(expected_dict.keys())) if expected_dict else None
                is_transposed = (isinstance(first_key, bool) or first_key in [0, 1, False, True]) and \
                               isinstance(expected_dict.get(first_key), dict) and \
                               all(isinstance(k, str) for k in expected_dict.get(first_key, {}).keys())
                
                # Transpose if needed
                if is_transposed:
                    transposed_dict = {}
                    for outcome_val, bucket_counts in expected_dict.items():
                        for bucket, count in bucket_counts.items():
                            if bucket not in transposed_dict:
                                transposed_dict[bucket] = {}
                            transposed_dict[bucket][outcome_val] = count
                    expected_dict = transposed_dict
                
                rows = []
                for bucket, values in expected_dict.items():
                    if isinstance(values, dict):
                        row = {'Response Time': bucket}
                        # Handle both boolean (True/False) and numeric (0/1) keys
                        row['Expected No Order'] = values.get(0, values.get(False, 0.0))
                        row['Expected Order'] = values.get(1, values.get(True, 0.0))
                        rows.append(row)
                    else:
                        row = {'Response Time': bucket, 'Expected No Order': 0.0, 'Expected Order': values}
                        rows.append(row)
                
                expected_df = pd.DataFrame(rows)
                expected_df = expected_df.set_index('Response Time')
                
                # Ensure we have both columns
                if 'Expected No Order' not in expected_df.columns:
                    expected_df['Expected No Order'] = 0.0
                if 'Expected Order' not in expected_df.columns:
                    expected_df['Expected Order'] = 0.0
                
                # Round for display
                expected_df = expected_df.round(1)
                
                # Add totals
                expected_df['Expected Total'] = expected_df.sum(axis=1)
                expected_df.loc['TOTAL'] = expected_df.sum(axis=0)
                
                st.dataframe(expected_df, use_container_width=True)
            
            st.caption("""
            **How to read:** These are the expected counts if response time had no effect. 
            Compare these to the observed table above to see where reality differs from expectation.
            """)
        
        # 3. Contributions to Chi-Square Table
        st.markdown("---")
        st.markdown("#### 3. Contributions to Chi-Square Statistic (Cell-by-Cell)")
        st.markdown("""
        This table shows how much each cell contributes to the overall chi-square statistic.
        Formula for each cell: **(Observed - Expected)² ÷ Expected**
        """)
        
        if 'contributions' in details:
            contributions_dict = details['contributions']
            # Handle nested dict structure
            if contributions_dict:
                # Check if dictionary is transposed (same issue)
                first_key = next(iter(contributions_dict.keys())) if contributions_dict else None
                is_transposed = (isinstance(first_key, bool) or first_key in [0, 1, False, True]) and \
                               isinstance(contributions_dict.get(first_key), dict) and \
                               all(isinstance(k, str) for k in contributions_dict.get(first_key, {}).keys())
                
                # Transpose if needed
                if is_transposed:
                    transposed_dict = {}
                    for outcome_val, bucket_counts in contributions_dict.items():
                        for bucket, count in bucket_counts.items():
                            if bucket not in transposed_dict:
                                transposed_dict[bucket] = {}
                            transposed_dict[bucket][outcome_val] = count
                    contributions_dict = transposed_dict
                
                rows = []
                for bucket, values in contributions_dict.items():
                    if isinstance(values, dict):
                        row = {'Response Time': bucket}
                        # Handle both boolean (True/False) and numeric (0/1) keys
                        row['Contribution (No Order)'] = values.get(0, values.get(False, 0.0))
                        row['Contribution (Order)'] = values.get(1, values.get(True, 0.0))
                        rows.append(row)
                    else:
                        row = {'Response Time': bucket, 'Contribution (No Order)': 0.0, 'Contribution (Order)': values}
                        rows.append(row)
                
                contributions_df = pd.DataFrame(rows)
                contributions_df = contributions_df.set_index('Response Time')
                
                # Ensure we have both columns
                if 'Contribution (No Order)' not in contributions_df.columns:
                    contributions_df['Contribution (No Order)'] = 0.0
                if 'Contribution (Order)' not in contributions_df.columns:
                    contributions_df['Contribution (Order)'] = 0.0
                
                # Round for display
                contributions_df = contributions_df.round(3)
                
                # Add row totals
                contributions_df['Total Contribution'] = contributions_df.sum(axis=1)
                contributions_df.loc['TOTAL (χ²)'] = contributions_df.sum(axis=0)
                
                st.dataframe(contributions_df, use_container_width=True)
            
            st.caption(f"""
            **How to read:** Each cell shows its contribution to the chi-square statistic. 
            The bottom right cell shows the total chi-square statistic: **{chi_sq.statistic:.2f}**
            """)
        
        # 4. Side-by-side comparison
        st.markdown("---")
        st.markdown("#### 4. Side-by-Side Comparison: Observed vs Expected")
        st.markdown("""
        This table makes it easy to compare observed and expected values side-by-side.
        """)
        
        if 'contingency_table' in details and 'expected_frequencies' in details:
            obs_dict = details['contingency_table']
            exp_dict = details['expected_frequencies']
            
            # Check and transpose both dictionaries if needed
            first_key_obs = next(iter(obs_dict.keys())) if obs_dict else None
            is_transposed_obs = (isinstance(first_key_obs, bool) or first_key_obs in [0, 1, False, True]) and \
                               isinstance(obs_dict.get(first_key_obs), dict) and \
                               all(isinstance(k, str) for k in obs_dict.get(first_key_obs, {}).keys())
            
            if is_transposed_obs:
                transposed_obs = {}
                for outcome_val, bucket_counts in obs_dict.items():
                    for bucket, count in bucket_counts.items():
                        if bucket not in transposed_obs:
                            transposed_obs[bucket] = {}
                        transposed_obs[bucket][outcome_val] = count
                obs_dict = transposed_obs
            
            first_key_exp = next(iter(exp_dict.keys())) if exp_dict else None
            is_transposed_exp = (isinstance(first_key_exp, bool) or first_key_exp in [0, 1, False, True]) and \
                               isinstance(exp_dict.get(first_key_exp), dict) and \
                               all(isinstance(k, str) for k in exp_dict.get(first_key_exp, {}).keys())
            
            if is_transposed_exp:
                transposed_exp = {}
                for outcome_val, bucket_counts in exp_dict.items():
                    for bucket, count in bucket_counts.items():
                        if bucket not in transposed_exp:
                            transposed_exp[bucket] = {}
                        transposed_exp[bucket][outcome_val] = count
                exp_dict = transposed_exp
            
            comparison_rows = []
            for bucket in obs_dict.keys():
                # Handle observed values
                obs_values = obs_dict[bucket]
                if isinstance(obs_values, dict):
                    # Handle both boolean (True/False) and numeric (0/1) keys
                    obs_no_order = obs_values.get(0, obs_values.get(False, 0))
                    obs_order = obs_values.get(1, obs_values.get(True, 0))
                else:
                    obs_no_order = 0
                    obs_order = obs_values if isinstance(obs_values, (int, float)) else 0
                
                # Handle expected values
                exp_values = exp_dict.get(bucket, {})
                if isinstance(exp_values, dict):
                    # Handle both boolean (True/False) and numeric (0/1) keys
                    exp_no_order = exp_values.get(0, exp_values.get(False, 0.0))
                    exp_order = exp_values.get(1, exp_values.get(True, 0.0))
                else:
                    exp_no_order = 0.0
                    exp_order = exp_values if isinstance(exp_values, (int, float)) else 0.0
                
                diff_no_order = obs_no_order - exp_no_order
                diff_order = obs_order - exp_order
                
                comparison_rows.append({
                    'Response Time': bucket,
                    'Observed (No Order)': f"{int(obs_no_order):,}",
                    'Expected (No Order)': f"{exp_no_order:.1f}",
                    'Difference': f"{diff_no_order:+.1f}",
                    'Observed (Order)': f"{int(obs_order):,}",
                    'Expected (Order)': f"{exp_order:.1f}",
                    'Difference': f"{diff_order:+.1f}"
                })
            
            comparison_df = pd.DataFrame(comparison_rows)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Technical details with step-by-step walkthrough
    with st.expander("🔢 Technical Details (Chi-Square Test)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chi-Square Statistic", f"{chi_sq.statistic:.2f}")
            st.caption("Higher = Results differ more from 'no effect'")
        with col2:
            st.metric("P-Value", f"{p_value:.4f}")
            st.caption(f"{p_exp['emoji']} {p_exp['luck_chance']} chance of being random")
        
        st.markdown("---")
        st.markdown("#### What these numbers actually mean")
        
        st.markdown(f"""
        **Chi-Square Statistic ({chi_sq.statistic:.2f}):**
        
        Think of this as a "surprise score." We compare what actually happened to what 
        we'd expect if response time didn't matter at all.
        
        - A score of **0** would mean: "Reality matched our 'no effect' prediction perfectly"
        - A score of **{chi_sq.statistic:.2f}** means: "Reality is {'very' if chi_sq.statistic > 20 else 'somewhat'} different from what we'd expect if there were no effect"
        
        **P-Value ({p_value:.4f}):**
        
        This answers: "If response time truly had no effect, what are the odds we'd 
        get a chi-square score of {chi_sq.statistic:.2f} or higher just by chance?"
        
        - Answer: **{p_exp['luck_chance']}**
        - Translation: {p_exp['trust_level']}
        """)
        
        if show_math:
            st.markdown("---")
            st.markdown("#### The Formula")
            st.latex(r"\chi^2 = \sum \frac{(O - E)^2}{E}")
            
            st.markdown("""
            **Breaking down each symbol:**
            
            | Symbol | Name | What it means |
            |:-------|:-----|:--------------|
            | χ² | Chi-Square | The "surprise score" we're calculating |
            | Σ | Sum | Add up all the values |
            | O | Observed | What actually happened (e.g., 419 sales in 0-15 min bucket) |
            | E | Expected | What we'd expect if there were no effect (e.g., 314 sales) |
            | (O-E)² | Squared difference | How far off we were, squared so negatives don't cancel positives |
            
            **Why this works:**
            - If O and E are close → small contribution → pattern might be random
            - If O and E are far apart → large contribution → pattern is likely real
            - Adding across all buckets gives us one overall "surprise score"
            """)


def render_proportions_step(df: pd.DataFrame, stat_results: Dict[str, Any], show_math: bool) -> None:
    """Render the proportions comparison step."""
    # Bridge sentence to maintain narrative flow
    st.info(f"💡 **Why this step:** {get_step_bridge('proportions')}")
    
    st.markdown("### Step 4: Head-to-Head Comparisons")
    
    st.markdown("""
    **What we're doing here:**
    The chi-square test told us *something* is going on. Now we look at specific 
    matchups: "Is 0-15 min *really* different from 15-30 min?" and so on.
    
    This helps identify exactly where the biggest differences are.
    """)
    
    # Add percentage points explainer at the top
    render_percentage_points_explainer()
    
    # =========================================================================
    # SAMPLE INPUT DATA - Show actual data rows used in proportion tests
    # =========================================================================
    st.markdown("---")
    st.markdown("## 📊 Sample Input Data")
    
    st.markdown("""
    Below are sample rows from your actual data that were used in the proportion z-tests.
    This shows the raw values for each bucket before calculating close rates.
    """)
    
    # Show sample rows if we have the required columns
    if 'response_bucket' in df.columns and 'ordered' in df.columns:
        prop_df = df[['response_bucket', 'ordered']].dropna()
        
        if len(prop_df) > 0:
            # Show sample rows from different buckets
            sample_size = min(15, len(prop_df))
            sample_df = prop_df.head(sample_size).copy()
            
            # Make it more readable
            display_sample = sample_df.copy()
            display_sample['Ordered'] = display_sample['ordered'].apply(lambda x: 'Yes' if x == 1 else 'No')
            display_sample = display_sample[['response_bucket', 'Ordered']]
            display_sample.columns = ['Response Time Bucket', 'Ordered?']
            
            st.dataframe(display_sample, use_container_width=True, hide_index=True)
            
            st.caption(f"""
            Showing {sample_size} of {len(prop_df):,} total rows used in the proportion tests.
            """)
            
            # Show how this data is used in comparisons
            st.markdown("#### How This Data is Used in Comparisons")
            st.markdown("""
            The z-test compares close rates between two buckets. For each bucket, we calculate:
            - **Number of leads** (total rows in that bucket)
            - **Number of orders** (rows where Ordered = Yes)
            - **Close rate** = Orders ÷ Leads
            """)
            
            # Show a quick example calculation
            if len(prop_df) > 0:
                example_buckets = prop_df['response_bucket'].unique()[:2]
                if len(example_buckets) >= 2:
                    bucket1, bucket2 = example_buckets[0], example_buckets[1]
                    b1_data = prop_df[prop_df['response_bucket'] == bucket1]
                    b2_data = prop_df[prop_df['response_bucket'] == bucket2]
                    
                    if len(b1_data) > 0 and len(b2_data) > 0:
                        b1_orders = b1_data['ordered'].sum()
                        b1_leads = len(b1_data)
                        b1_rate = b1_orders / b1_leads if b1_leads > 0 else 0
                        
                        b2_orders = b2_data['ordered'].sum()
                        b2_leads = len(b2_data)
                        b2_rate = b2_orders / b2_leads if b2_leads > 0 else 0
                        
                        example_calc = pd.DataFrame({
                            'Bucket': [str(bucket1), str(bucket2)],
                            'Total Leads': [b1_leads, b2_leads],
                            'Orders': [b1_orders, b2_orders],
                            'Close Rate': [f"{b1_rate*100:.1f}%", f"{b2_rate*100:.1f}%"],
                            'Calculation': [
                                f"{b1_orders} ÷ {b1_leads}",
                                f"{b2_orders} ÷ {b2_leads}"
                            ]
                        })
                        
                        st.dataframe(example_calc, use_container_width=True, hide_index=True)
                        st.caption("Example calculation for two buckets. The z-test compares these rates.")
    
    # Create human-friendly comparison table
    comparisons = []
    for result in stat_results['pairwise']:
        if hasattr(result, 'details') and result.details:
            p_val = result.p_value
            p_exp = get_p_value_explanation(p_val)
            diff = result.details.get('difference', 0)
            diff_pp = diff * 100
            
            # Make the comparison name more readable
            comp_name = result.test_name.replace('Z-Test for Proportions: ', '')
            
            # Determine practical significance
            if abs(diff_pp) >= 5:
                impact = "🔥 Large difference"
            elif abs(diff_pp) >= 2:
                impact = "📊 Moderate difference"
            else:
                impact = "📉 Small difference"
            
            comparisons.append({
                'Matchup': comp_name,
                'Difference': f"{diff_pp:+.1f} percentage points",
                'What This Means': f"{'First' if diff > 0 else 'Second'} bucket closes {abs(diff_pp):.1f}pp {'more' if diff > 0 else 'fewer'} deals",
                'Is It Real?': f"{p_exp['emoji']} {p_exp['verdict']}",
                'Chance of Luck': p_exp['luck_chance'],
                'Impact': impact
            })
    
    if comparisons:
        st.dataframe(pd.DataFrame(comparisons), use_container_width=True, hide_index=True)
        
        st.markdown("""
        **How to read this table:**
        - **Difference**: How many more percentage points the first bucket closes vs the second
        - **Is It Real?**: Whether we can trust this difference isn't just random chance
        - **Chance of Luck**: Lower % = more confident the difference is real
        """)
    
    # Highlight extreme comparison with more context
    extreme = stat_results.get('extreme_comparison')
    if extreme:
        p_exp = get_p_value_explanation(extreme.p_value)
        diff = extreme.details.get('difference', 0) * 100 if hasattr(extreme, 'details') and extreme.details else 0
        
        if extreme.is_significant:
            st.success(f"""
            **✅ The Big Finding:**
            
            {extreme.interpretation}
            
            **In practical terms:** If you get 1,000 leads per month, improving from the slowest 
            to fastest response could mean approximately **{abs(int(diff * 10))} additional sales per month**.
            
            **Confidence level:** {p_exp['confidence']} — {p_exp['trust_level'].lower()}.
            """)
        else:
            st.info(f"""
            **ℹ️ No conclusive difference found.**
            
            {extreme.interpretation}
            
            The difference we see ({diff:+.1f}pp) could be random variation. 
            Either there's no real difference, or we need more data to detect it.
            """)
        
        # =========================================================================
        # WORKED EXAMPLE - Show the actual calculation with user's numbers
        # =========================================================================
        st.markdown("---")
        st.markdown("### Let's Verify This With Your Data")
        
        st.markdown("""
        The z-test compares two proportions and asks: *"Is this difference too large to be random chance?"*
        
        Here's the step-by-step calculation for the key comparison:
        """)
        
        # Extract the details
        if hasattr(extreme, 'details') and extreme.details:
            b1 = extreme.details.get('bucket1', {})
            b2 = extreme.details.get('bucket2', {})
            
            fast_name = b1.get('name', 'Fast')
            slow_name = b2.get('name', 'Slow')
            fast_sales = b1.get('n_orders', 0)
            fast_leads = b1.get('n_leads', 1)
            slow_sales = b2.get('n_orders', 0)
            slow_leads = b2.get('n_leads', 1)
            fast_rate = b1.get('close_rate', 0)
            slow_rate = b2.get('close_rate', 0)
            
            # Step 1: Show the raw numbers
            st.markdown(f"#### Step 1: Your actual numbers")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                **{fast_name}**
                - Sales: **{fast_sales:,}**
                - Leads: **{fast_leads:,}**
                - Close rate: {fast_sales:,} ÷ {fast_leads:,} = **{fast_rate*100:.1f}%**
                """)
            
            with col2:
                st.markdown(f"""
                **{slow_name}**
                - Sales: **{slow_sales:,}**
                - Leads: **{slow_leads:,}**
                - Close rate: {slow_sales:,} ÷ {slow_leads:,} = **{slow_rate*100:.1f}%**
                """)
            
            # Step 2: The difference
            diff_pp = (fast_rate - slow_rate) * 100
            st.markdown(f"#### Step 2: Calculate the difference")
            st.markdown(f"""
            **Difference:** {fast_rate*100:.1f}% - {slow_rate*100:.1f}% = **{diff_pp:+.1f} percentage points**
            """)
            
            # Step 3: The test statistic with full calculation
            st.markdown(f"#### Step 3: Is this difference statistically significant?")
            
            z_stat = extreme.statistic if hasattr(extreme, 'statistic') else 0
            p_val = extreme.p_value if hasattr(extreme, 'p_value') else 1
            
            # Calculate intermediate values for display
            import numpy as np
            total_sales = fast_sales + slow_sales
            total_leads = fast_leads + slow_leads
            pooled_proportion = total_sales / total_leads if total_leads > 0 else 0
            standard_error = np.sqrt(pooled_proportion * (1 - pooled_proportion) * (1/fast_leads + 1/slow_leads)) if total_leads > 0 else 0
            
            st.markdown(f"""
            We calculate a **z-score** of **{z_stat:.2f}**, which tells us how many "standard deviations" 
            this difference is from what we'd expect by chance.
            
            - **Z-score:** {z_stat:.2f}
            - **P-value:** {p_val:.4f}
            
            **Translation:** {p_exp['plain_english']}
            """)
            
            # Show the full calculation
            st.markdown("---")
            st.markdown("#### Complete Z-Test Calculation")
            st.markdown("""
            Here's the complete step-by-step calculation of the z-test statistic:
            """)
            
            st.markdown(f"""
            **Step 3a: Calculate the pooled proportion**
            
            When testing if two proportions are different, we first calculate what the overall 
            proportion would be if we combined both groups:
            
            **Formula:** Pooled proportion (p̂) = (Total sales in both groups) ÷ (Total leads in both groups)
            
            **Calculation:**
            - Total sales = {fast_sales:,} + {slow_sales:,} = **{total_sales:,}**
            - Total leads = {fast_leads:,} + {slow_leads:,} = **{total_leads:,}**
            - p̂ = {total_sales:,} ÷ {total_leads:,} = **{pooled_proportion:.4f}** ({pooled_proportion*100:.2f}%)
            """)
            
            st.markdown(f"""
            **Step 3b: Calculate the standard error**
            
            The standard error tells us how much variation we'd expect in the difference 
            between proportions due to random sampling:
            
            **Formula:** SE = √[p̂(1-p̂)(1/n₁ + 1/n₂)]
            
            **Step-by-step calculation:**
            - p̂ = {pooled_proportion:.4f}
            - (1-p̂) = 1 - {pooled_proportion:.4f} = **{1-pooled_proportion:.4f}**
            - 1/n₁ = 1/{fast_leads:,} = **{1/fast_leads:.6f}**
            - 1/n₂ = 1/{slow_leads:,} = **{1/slow_leads:.6f}**
            - (1/n₁ + 1/n₂) = {1/fast_leads:.6f} + {1/slow_leads:.6f} = **{1/fast_leads + 1/slow_leads:.6f}**
            - p̂(1-p̂) = {pooled_proportion:.4f} × {1-pooled_proportion:.4f} = **{pooled_proportion * (1-pooled_proportion):.6f}**
            - p̂(1-p̂)(1/n₁ + 1/n₂) = {pooled_proportion * (1-pooled_proportion):.6f} × {1/fast_leads + 1/slow_leads:.6f} = **{pooled_proportion * (1-pooled_proportion) * (1/fast_leads + 1/slow_leads):.6f}**
            - SE = √[{pooled_proportion * (1-pooled_proportion) * (1/fast_leads + 1/slow_leads):.6f}] = **{standard_error:.4f}**
            """)
            
            st.markdown(f"""
            **Step 3c: Calculate the z-statistic**
            
            The z-statistic measures how many standard errors apart the two proportions are:
            
            **Formula:** z = (p₁ - p₂) ÷ SE
            
            **Step-by-step calculation:**
            - p₁ (fast bucket) = {fast_rate:.4f} ({fast_rate*100:.2f}%)
            - p₂ (slow bucket) = {slow_rate:.4f} ({slow_rate*100:.2f}%)
            - Difference = p₁ - p₂ = {fast_rate:.4f} - {slow_rate:.4f} = **{fast_rate - slow_rate:.4f}**
            - SE = {standard_error:.4f}
            - z = {fast_rate - slow_rate:.4f} ÷ {standard_error:.4f} = **{z_stat:.2f}**
            """)
            
            st.markdown(f"""
            **Step 3d: Interpret the z-statistic**
            
            - A z-score of **{z_stat:.2f}** means the difference is **{abs(z_stat):.2f} standard errors** away from zero
            - We then look up this z-score in a standard normal distribution to get the p-value
            - P-value = **{p_val:.4f}** ({p_val*100:.2f}% chance this difference is due to random chance)
            """)
            
            # Create a summary table
            calc_summary = pd.DataFrame({
                'Component': [
                    'Fast bucket proportion (p₁)',
                    'Slow bucket proportion (p₂)',
                    'Difference (p₁ - p₂)',
                    'Pooled proportion (p̂)',
                    'Standard error (SE)',
                    'Z-statistic',
                    'P-value'
                ],
                'Value': [
                    f"{fast_rate:.4f} ({fast_rate*100:.2f}%)",
                    f"{slow_rate:.4f} ({slow_rate*100:.2f}%)",
                    f"{fast_rate - slow_rate:.4f} ({diff_pp:.2f} percentage points)",
                    f"{pooled_proportion:.4f} ({pooled_proportion*100:.2f}%)",
                    f"{standard_error:.4f}",
                    f"{z_stat:.2f}",
                    f"{p_val:.4f}"
                ],
                'Calculation': [
                    f"{fast_sales:,} ÷ {fast_leads:,}",
                    f"{slow_sales:,} ÷ {slow_leads:,}",
                    f"{fast_rate:.4f} - {slow_rate:.4f}",
                    f"{total_sales:,} ÷ {total_leads:,}",
                    f"√[p̂(1-p̂)(1/n₁ + 1/n₂)]",
                    f"(p₁ - p₂) ÷ SE",
                    "From standard normal distribution"
                ]
            })
            
            st.dataframe(calc_summary, use_container_width=True, hide_index=True)
            
            # Step 4: Conclusion
            st.markdown(f"#### Step 4: The Verdict")
            
            if extreme.is_significant:
                st.success(f"""
                ✅ **The {diff_pp:.1f} percentage point difference is statistically significant.**
                
                With {fast_leads + slow_leads:,} leads in these two buckets, we can be 
                **{p_exp['confidence']} confident** this pattern is real, not random luck.
                """)
            else:
                st.warning(f"""
                ⚠️ **We cannot conclude the difference is statistically significant.**
                
                The {diff_pp:+.1f} percentage point difference could be due to random variation.
                """)
            
            # =========================================================================
            # VERIFICATION PANEL - Z-Test
            # =========================================================================
            render_z_test_verification(extreme)


def render_regression_step(df: pd.DataFrame, regression_result, show_math: bool) -> None:
    """Render the regression step with plain-English explanation."""
    # Bridge sentence to maintain narrative flow
    st.info(f"💡 **Why this step:** {get_step_bridge('regression')}")
    
    st.markdown("### Step 5: Is It Really Response Time, or Something Else?")
    
    # Plain English explanation with a concrete example
    st.markdown("""
    **Here's a tricky question:**
    We found that fast responders close more deals. But is that *because* they responded fast, 
    or is something else going on?
    
    **Consider this scenario:**
    """)
    
    st.info("""
    🤔 **The Sneaky Problem (Confounding)**
    
    Imagine referral leads are special:
    - They close at 25% (high-quality, warm leads)
    - Salespeople prioritize them → they get 10-minute responses
    
    Meanwhile, website leads:
    - They close at 8% (just browsing, less interested)
    - They come in at night → they get 60-minute responses
    
    **The misleading pattern:** Fast responses (10 min) → High close rates (25%)
    
    **The truth:** It's not the speed that matters — referrals would close at 25% 
    even if you took an hour to respond. And website leads would still only close 
    at 8% even if you responded instantly.
    
    **Regression helps us figure out:** Does speed *actually* matter, or are we 
    being fooled by lead source differences?
    """)
    
    st.markdown("**What we did:** We used regression to mathematically isolate the effect of speed from lead source.")
    
    # Check if we can actually control for lead source (check if it varies)
    if 'lead_source' in df.columns:
        lead_source_counts = df['lead_source'].value_counts()
        n_unique_sources = len(lead_source_counts)
        max_source_prop = lead_source_counts.max() / len(df) if len(df) > 0 else 0
        
        # If only one source, or one source dominates (>95%), we can't effectively control
        if n_unique_sources == 1:
            st.markdown("---")
            render_no_controls_explanation(
                available_controls=[],
                missing_controls=['lead_source'],
                reason="controls_dont_vary"
            )
        elif max_source_prop > 0.95:
            st.markdown("---")
            render_no_controls_explanation(
                available_controls=['lead_source'],
                missing_controls=['lead_source'],
                reason="insufficient_variation"
            )
    elif 'lead_source' not in df.columns:
        st.markdown("---")
        render_no_controls_explanation(
            available_controls=[],
            missing_controls=['lead_source'],
            reason="no_controls_available"
        )
    
    # Show result prominently
    if regression_result.is_response_time_significant:
        st.success("""
        **✅ The association persists after controlling for lead source.**
        
        Even when we compare leads from the *same source* (e.g., only looking at website leads, 
        or only looking at referrals), faster responses still predict more sales.
        
        **Important caveat:** This only controls for *one* potential confounder (lead source). 
        The association could still be explained by other unmeasured confounders such as:
        - Lead quality signals that drive prioritization (salespeople may prioritize leads that 
          appear more likely to convert, creating a spurious correlation)
        - Salesperson skill differences (better salespeople may both respond faster and close more)
        - Time-of-day or day-of-week effects
        
        **This analysis cannot prove causation.** This observational analysis has limitations 
        in establishing that faster responses *cause* higher conversion rates.
        """)
    else:
        st.warning("""
        **⚠️ Plot twist: The effect disappears when we account for lead source.**
        
        The relationship we saw earlier was likely a mirage. Different lead types have 
        different close rates AND different response times, creating an illusion that 
        speed matters.
        
        **Translation:** Focus on lead quality, not just response speed.
        """)
    
    # =========================================================================
    # SAMPLE INPUT DATA - Show actual data rows used in regression
    # =========================================================================
    st.markdown("---")
    st.markdown("## 📊 Sample Input Data")
    
    st.markdown("""
    Below are sample rows from your actual data that were used in the regression model.
    This shows the raw values before they're transformed into the regression formula.
    """)
    
    # Filter to rows that would be used in regression (have all required columns)
    regression_cols = ['ordered', 'response_bucket', 'lead_source']
    
    # Check if columns exist
    missing_cols = [col for col in regression_cols if col not in df.columns]
    if missing_cols:
        st.warning(f"⚠️ Cannot show sample data: Missing columns {missing_cols}")
    else:
        regression_df = df[regression_cols].dropna()
        
        if len(regression_df) == 0:
            st.warning("⚠️ No data available after filtering. All rows have missing values in required columns.")
        else:
            # Show sample rows
            sample_size = min(10, len(regression_df))
            sample_df = regression_df.head(sample_size).copy()
            
            # Make it more readable
            display_sample = sample_df.copy()
            display_sample['Ordered'] = display_sample['ordered'].apply(lambda x: 'Yes' if x == 1 else 'No')
            display_sample = display_sample[['response_bucket', 'lead_source', 'Ordered']]
            display_sample.columns = ['Response Time Bucket', 'Lead Source', 'Ordered?']
            
            st.dataframe(display_sample, use_container_width=True, hide_index=True)
            
            st.caption(f"""
            Showing {sample_size} of {len(regression_df):,} total rows used in the regression model.
            """)
            
            # Show data breakdown
            st.markdown("#### Data Breakdown by Category")
            
            breakdown_col1, breakdown_col2 = st.columns(2)
            
            with breakdown_col1:
                st.markdown("**By Response Time Bucket:**")
                try:
                    bucket_counts = regression_df.groupby('response_bucket').agg({
                        'ordered': ['count', 'sum', 'mean']
                    }).round(3)
                    bucket_counts.columns = ['Total Leads', 'Orders', 'Close Rate']
                    bucket_counts['Close Rate'] = bucket_counts['Close Rate'].apply(lambda x: f"{x*100:.1f}%")
                    st.dataframe(bucket_counts, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating bucket breakdown: {e}")
            
            with breakdown_col2:
                st.markdown("**By Lead Source:**")
                try:
                    source_counts = regression_df.groupby('lead_source').agg({
                        'ordered': ['count', 'sum', 'mean']
                    }).round(3)
                    source_counts.columns = ['Total Leads', 'Orders', 'Close Rate']
                    source_counts['Close Rate'] = source_counts['Close Rate'].apply(lambda x: f"{x*100:.1f}%")
                    st.dataframe(source_counts, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating source breakdown: {e}")
    
    # Show odds ratios with intuitive explanation
    st.markdown("---")
    st.markdown("### 🎯 How Much Does Speed Help?")
    
    st.markdown("""
    Below we show a **"success multiplier"** for each response speed compared to the slowest bucket (60+ min).
    
    **How to read this:**
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **Multiplier = 1.0**
        
        Same odds as slow responders.
        No advantage.
        """)
    with col2:
        st.markdown("""
        **Multiplier = 1.5**
        
        50% better odds of closing.
        If slow gets 10 sales, you get 15.
        """)
    with col3:
        st.markdown("""
        **Multiplier = 2.0**
        
        Twice the odds of closing.
        If slow gets 10 sales, you get 20.
        """)
    
    # Create human-friendly odds ratio table
    if not regression_result.odds_ratios.empty:
        or_df = regression_result.odds_ratios.copy()
        
        # Create friendly table
        friendly_rows = []
        for _, row in or_df.iterrows():
            or_val = row['odds_ratio']
            ci_low = row['ci_lower']
            ci_high = row['ci_upper']
            p_val = row.get('p_value', 0.01)
            
            p_exp = get_p_value_explanation(p_val)
            
            # Calculate implied multiplier in plain terms
            if or_val >= 1:
                multiplier_text = f"{or_val:.1f}x more likely to close"
                benefit = f"+{(or_val-1)*100:.0f}% better odds"
            else:
                multiplier_text = f"{or_val:.1f}x (lower odds)"
                benefit = f"{(or_val-1)*100:.0f}% lower odds"
            
            # Concrete example
            baseline_sales = 10
            your_sales = int(round(or_val * baseline_sales))
            
            friendly_rows.append({
                'Response Speed': row['bucket'],
                'Your Advantage': multiplier_text,
                'Compared to Slow': f"If slow gets {baseline_sales} sales, you get ~{your_sales}",
                'Confidence': f"{p_exp['emoji']} {p_exp['verdict']}",
                'Range': f"Could be {ci_low:.1f}x to {ci_high:.1f}x"
            })
        
        # Add reference row (the reference category doesn't appear in odds_ratios)
        reference_bucket = regression_result.reference_bucket or '60+ min'
        friendly_rows.append({
            'Response Speed': f'{reference_bucket} (baseline)',
            'Your Advantage': '1.0x (this is our comparison point)',
            'Compared to Slow': 'This is the baseline',
            'Confidence': '—',
            'Range': '—'
        })
        
        st.dataframe(pd.DataFrame(friendly_rows), use_container_width=True, hide_index=True)
    
    # Forest plot with explanation
    if not regression_result.odds_ratios.empty:
        st.markdown("#### 📊 Visual: Success Multiplier by Response Speed")
        
        st.markdown("""
        The chart below shows each response speed's advantage over slow responders (60+ min).
        - **Dots** = Our best estimate of the multiplier
        - **Lines** = The range of uncertainty (where the true value probably falls)
        - **Dashed line at 1** = "No advantage" — anything to the RIGHT is better than slow responders
        """)
        
        fig = create_forest_plot(regression_result.odds_ratios)
        st.plotly_chart(fig, use_container_width=True)
    
    # =========================================================================
    # DUMMY ENCODING EXPLANATION - Show how categorical variables are encoded
    # =========================================================================
    st.markdown("---")
    st.markdown("### 🔢 How Categorical Variables Are Encoded (Dummy Variables)")
    
    st.markdown("""
    Regression models need numbers, not text. So we convert categorical variables (like 
    "0-15 min" or "Website") into **dummy variables** (0s and 1s).
    
    **How it works:**
    - We pick one category as the **reference** (baseline)
    - For each other category, we create a new variable that's 1 if the row belongs to 
      that category, 0 otherwise
    - The reference category is represented by all dummy variables being 0
    """)
    
    if hasattr(regression_result, 'reference_bucket') and regression_result.reference_bucket:
        ref_bucket = regression_result.reference_bucket
        
        # Get unique buckets
        if 'response_bucket' in df.columns:
            buckets = sorted(df['response_bucket'].dropna().unique(), key=str)
            
            st.markdown(f"""
            **For Response Time Bucket:**
            
            - **Reference category:** `{ref_bucket}` (all dummy variables = 0)
            """)
            
            # Show encoding example
            encoding_rows = []
            for bucket in buckets:
                if str(bucket) != ref_bucket:
                    # Create a row showing the encoding
                    encoding = {}
                    encoding['Response Time'] = str(bucket)
                    encoding['Dummy Variable Name'] = f"C({ref_bucket})[T.{bucket}]"
                    encoding['Value When This Bucket'] = '1'
                    encoding['Value When Other Buckets'] = '0'
                    encoding['Interpretation'] = f"Compared to {ref_bucket}"
                    encoding_rows.append(encoding)
            
            if encoding_rows:
                encoding_df = pd.DataFrame(encoding_rows)
                st.dataframe(encoding_df, use_container_width=True, hide_index=True)
            
            # Show example with actual data
            st.markdown("#### Example: How a Sample Lead is Encoded")
            
            if len(regression_df) > 0:
                example_row = regression_df.iloc[0]
                example_bucket = example_row['response_bucket']
                example_source = example_row['lead_source']
                
                st.markdown(f"""
                **Example lead:**
                - Response time: **{example_bucket}**
                - Lead source: **{example_source}**
                - Ordered: **{'Yes' if example_row['ordered'] == 1 else 'No'}**
                """)
                
                # Show encoding
                encoding_example = []
                for bucket in buckets:
                    if str(bucket) != ref_bucket:
                        is_this_bucket = (str(example_bucket) == str(bucket))
                        encoding_example.append({
                            'Variable': f"Response Bucket = {bucket}",
                            'Dummy Value': '1' if is_this_bucket else '0',
                            'Meaning': f"{'This lead' if is_this_bucket else 'Not this lead'}"
                        })
                
                if encoding_example:
                    encoding_example_df = pd.DataFrame(encoding_example)
                    st.dataframe(encoding_example_df, use_container_width=True, hide_index=True)
    
    # =========================================================================
    # WORKED EXAMPLE - Show actual calculation with sample data
    # =========================================================================
    st.markdown("---")
    st.markdown("### 🧮 Worked Example: Calculating Predictions")
    
    st.markdown("""
    Let's see how the regression formula is applied to actual leads from your data.
    We'll calculate the predicted probability of ordering for a few example leads.
    """)
    
    if (hasattr(regression_result, 'coefficients') and not regression_result.coefficients.empty and
        len(regression_df) > 0):
        
        import numpy as np
        
        # Get the intercept (if available)
        intercept = 0
        # Check if 'variable' column exists (coefficients stored as column)
        if 'variable' in regression_result.coefficients.columns:
            intercept_row = regression_result.coefficients[regression_result.coefficients['variable'] == 'Intercept']
            if not intercept_row.empty:
                intercept = intercept_row.iloc[0].get('coefficient', 0)
        elif 'Intercept' in regression_result.coefficients.index:
            intercept = regression_result.coefficients.loc['Intercept'].get('coefficient', 0)
        elif hasattr(regression_result, 'model_object'):
            try:
                intercept = regression_result.model_object.params.get('Intercept', 0)
            except:
                pass
        
        # Pick 2-3 example leads
        n_examples = min(3, len(regression_df))
        example_leads = regression_df.head(n_examples)
        
        for idx, (lead_idx, lead) in enumerate(example_leads.iterrows(), 1):
            st.markdown(f"#### Example {idx}: Lead with {lead['response_bucket']} response, {lead['lead_source']} source")
            
            # Get the coefficients for this lead's categories
            bucket_coef = 0
            source_coef = 0
            
            # Find bucket coefficient
            # Statsmodels uses format: "C(response_bucket)[T.0-15 min]"
            bucket_str = str(lead['response_bucket'])
            import re
            
            # Check if this is the reference category (no coefficient, all dummies = 0)
            if regression_result.reference_bucket and str(bucket_str) == str(regression_result.reference_bucket):
                bucket_coef = 0  # Reference category
            else:
                # Look for coefficient matching this bucket
                # Check if 'variable' column exists
                if 'variable' in regression_result.coefficients.columns:
                    # Iterate through rows
                    for _, row in regression_result.coefficients.iterrows():
                        coef_name = str(row.get('variable', ''))
                        # Check if it's a response_bucket coefficient
                        if 'response_bucket' in coef_name:
                            # Extract bucket name from pattern [T.bucket_name]
                            match = re.search(r'\[T\.(.+?)\]', coef_name)
                            if match:
                                coef_bucket = match.group(1)
                                if str(coef_bucket) == str(bucket_str):
                                    bucket_coef = row.get('coefficient', 0)
                                    break
                else:
                    # Use index-based access
                    for coef_idx in regression_result.coefficients.index:
                        coef_name = str(coef_idx)
                        if 'response_bucket' in coef_name:
                            match = re.search(r'\[T\.(.+?)\]', coef_name)
                            if match:
                                coef_bucket = match.group(1)
                                if str(coef_bucket) == str(bucket_str):
                                    bucket_coef = regression_result.coefficients.loc[coef_idx].get('coefficient', 0)
                                    break
            
            # Find source coefficient
            source_str = str(lead['lead_source'])
            # Look for coefficient matching this source
            if 'variable' in regression_result.coefficients.columns:
                # Iterate through rows
                for _, row in regression_result.coefficients.iterrows():
                    coef_name = str(row.get('variable', ''))
                    # Check if it's a lead_source coefficient
                    if 'lead_source' in coef_name:
                        # Extract source name from pattern [T.source_name]
                        match = re.search(r'\[T\.(.+?)\]', coef_name)
                        if match:
                            coef_source = match.group(1)
                            if str(coef_source) == str(source_str):
                                source_coef = row.get('coefficient', 0)
                                break
            else:
                # Use index-based access
                for coef_idx in regression_result.coefficients.index:
                    coef_name = str(coef_idx)
                    if 'lead_source' in coef_name:
                        match = re.search(r'\[T\.(.+?)\]', coef_name)
                        if match:
                            coef_source = match.group(1)
                            if str(coef_source) == str(source_str):
                                source_coef = regression_result.coefficients.loc[coef_idx].get('coefficient', 0)
                                break
            
            # Calculate log-odds
            log_odds = intercept + bucket_coef + source_coef
            
            # Calculate probability
            probability = 1 / (1 + np.exp(-log_odds))
            
            # Show the calculation with detailed math
            st.markdown("**Step 1: Plug into the regression formula**")
            st.markdown(f"""
            **Formula:** log-odds = Intercept + β₁×[Bucket] + β₂×[Source]
            
            **Your values:**
            - Intercept = {intercept:.4f}
            - β₁ (for {lead['response_bucket']} bucket) = {bucket_coef:.4f}
            - β₂ (for {lead['lead_source']} source) = {source_coef:.4f}
            
            **Calculation:**
            - log-odds = {intercept:.4f} + {bucket_coef:.4f}×1 + {source_coef:.4f}×1
            - log-odds = {intercept:.4f} + {bucket_coef:.4f} + {source_coef:.4f}
            - log-odds = **{log_odds:.4f}**
            """)
            
            st.markdown("**Step 2: Convert log-odds to probability**")
            st.markdown(f"""
            **Formula:** probability = 1 ÷ (1 + e^(-log-odds))
            
            **Step-by-step calculation:**
            - log-odds = {log_odds:.4f}
            - -log-odds = -{log_odds:.4f} = **{-log_odds:.4f}**
            - e^(-log-odds) = e^({-log_odds:.4f}) = **{np.exp(-log_odds):.4f}**
            - 1 + e^(-log-odds) = 1 + {np.exp(-log_odds):.4f} = **{1 + np.exp(-log_odds):.4f}**
            - probability = 1 ÷ {1 + np.exp(-log_odds):.4f} = **{probability:.4f}**
            - probability = **{probability*100:.1f}%**
            """)
            
            # Show actual outcome
            actual_outcome = "Yes" if lead['ordered'] == 1 else "No"
            st.info(f"""
            **Actual outcome for this lead:** {actual_outcome}
            
            **Predicted probability:** {probability*100:.1f}%
            
            **Difference:** The model predicted a {probability*100:.1f}% chance of ordering, 
            and the lead {'ordered' if lead['ordered'] == 1 else 'did not order'}.
            """)
            
            if idx < n_examples:
                st.markdown("---")
    
    # =========================================================================
    # COMPLETE REGRESSION DATA - Show input data and coefficients
    # =========================================================================
    st.markdown("---")
    st.markdown("### 📊 Complete Regression Data and Coefficients")
    
    st.markdown("""
    Below are the complete data tables and coefficients used in the logistic regression model.
    You can verify every number used in the calculation.
    """)
    
    # 1. Input Data Summary
    st.markdown("#### 1. Input Data Summary")
    st.markdown("""
    This shows how many observations were used in the regression model, broken down by 
    response time bucket and lead source.
    """)
    
    if hasattr(regression_result, 'n_observations'):
        st.markdown(f"""
        **Total observations in model:** {regression_result.n_observations:,}
        
        **Model formula:** `{regression_result.formula}`
        
        **Reference category:** {regression_result.reference_bucket or 'Slowest bucket'}
        """)
    
    # 2. Coefficients Table
    st.markdown("---")
    st.markdown("#### 2. Regression Coefficients (Log-Odds)")
    st.markdown("""
    These are the raw coefficients from the logistic regression model. They represent 
    the change in **log-odds** of ordering compared to the reference category.
    
    **How to interpret:**
    - Positive coefficient = higher log-odds = more likely to order
    - Negative coefficient = lower log-odds = less likely to order
    - Zero = same odds as reference category
    """)
    
    if hasattr(regression_result, 'coefficients') and not regression_result.coefficients.empty:
        coef_df = regression_result.coefficients.copy()
        
        # Make it more readable
        display_coef = coef_df.copy()
        if 'coefficient' in display_coef.columns:
            display_coef['Log-Odds Coefficient'] = display_coef['coefficient'].apply(lambda x: f"{x:.4f}")
        if 'std_err' in display_coef.columns:
            display_coef['Standard Error'] = display_coef['std_err'].apply(lambda x: f"{x:.4f}")
        if 'p_value' in display_coef.columns:
            display_coef['P-Value'] = display_coef['p_value'].apply(lambda x: f"{x:.4f}")
        if 'z_value' in display_coef.columns:
            display_coef['Z-Value'] = display_coef['z_value'].apply(lambda x: f"{x:.2f}")
        
        # Show the table
        st.dataframe(display_coef, use_container_width=True, hide_index=True)
        
        st.caption("""
        **Note:** These coefficients are in log-odds units. To convert to odds ratios, 
        we use: **Odds Ratio = e^coefficient**. The odds ratios are shown in the table above.
        """)
    
    # 3. Conversion from Coefficients to Odds Ratios
    st.markdown("---")
    st.markdown("#### 3. Conversion: Coefficients → Odds Ratios")
    st.markdown("""
    The odds ratios shown earlier are calculated from these coefficients using the formula:
    **Odds Ratio = e^coefficient**
    
    This table shows the conversion for each variable:
    """)
    
    if (hasattr(regression_result, 'coefficients') and not regression_result.coefficients.empty and
        hasattr(regression_result, 'odds_ratios') and not regression_result.odds_ratios.empty):
        
        import numpy as np
        
        # Merge coefficients and odds ratios
        conversion_rows = []
        for idx, row in regression_result.coefficients.iterrows():
            # Get variable name from 'variable' column if it exists, otherwise use index
            if 'variable' in regression_result.coefficients.columns:
                var_name = str(row.get('variable', idx))
            else:
                var_name = idx if isinstance(idx, str) else str(idx)
            coef = row.get('coefficient', 0)
            or_val = None
            
            # Try to find matching odds ratio
            if not regression_result.odds_ratios.empty:
                # Try direct index match first (most reliable)
                try:
                    if idx in regression_result.odds_ratios.index:
                        or_val = regression_result.odds_ratios.loc[idx].get('odds_ratio', np.exp(coef))
                    elif var_name in regression_result.odds_ratios.index:
                        or_val = regression_result.odds_ratios.loc[var_name].get('odds_ratio', np.exp(coef))
                    else:
                        # Try string matching as fallback
                        or_index_str = regression_result.odds_ratios.index.astype(str)
                        matches = [i for i, idx_str in enumerate(or_index_str) if var_name.lower() in idx_str.lower()]
                        if matches:
                            or_val = regression_result.odds_ratios.iloc[matches[0]].get('odds_ratio', np.exp(coef))
                        else:
                            or_val = np.exp(coef)
                except (KeyError, TypeError):
                    # If matching fails, calculate from coefficient
                    or_val = np.exp(coef)
            else:
                or_val = np.exp(coef)
            
            conversion_rows.append({
                'Variable': var_name,
                'Coefficient (β)': f"{coef:.4f}",
                'Calculation': f"e^({coef:.4f})",
                'Odds Ratio': f"{or_val:.4f}",
                'Interpretation': (
                    f"{or_val:.2f}x {'higher' if or_val > 1 else 'lower'} odds than reference"
                    if or_val != 1.0 else "Same odds as reference"
                )
            })
        
        if conversion_rows:
            conversion_df = pd.DataFrame(conversion_rows)
            st.dataframe(conversion_df, use_container_width=True, hide_index=True)
    
    # 4. Model Summary Statistics
    st.markdown("---")
    st.markdown("#### 4. Model Summary Statistics")
    
    summary_stats = []
    if hasattr(regression_result, 'n_observations'):
        summary_stats.append(['Number of Observations', f"{regression_result.n_observations:,}"])
    if hasattr(regression_result, 'pseudo_r_squared'):
        summary_stats.append(['Pseudo R²', f"{regression_result.pseudo_r_squared:.4f} ({regression_result.pseudo_r_squared*100:.2f}%)"])
    if hasattr(regression_result, 'reference_bucket'):
        summary_stats.append(['Reference Category', regression_result.reference_bucket or 'Slowest bucket'])
    
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats, columns=['Statistic', 'Value'])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Technical details with regression explainer
    render_regression_explainer()
    
    # =========================================================================
    # VERIFICATION PANEL - Regression
    # =========================================================================
    render_regression_verification(regression_result)
    
    with st.expander("🔢 Technical Details (Logistic Regression)", expanded=False):
        st.markdown(f"""
        **Model Summary:**
        
        | Metric | Value | What it means |
        |:-------|:------|:--------------|
        | Pseudo R² | {regression_result.pseudo_r_squared:.3f} ({regression_result.pseudo_r_squared*100:.1f}%) | How much of the outcome variation we can explain |
        
        **About that Pseudo R² ({regression_result.pseudo_r_squared*100:.1f}%):**
        
        This might seem low, but that's normal! Many factors affect whether a lead closes:
        - The lead's actual need for your product
        - Their budget and timing
        - Competitive alternatives
        - Random chance
        
        Response time and lead source explain about {regression_result.pseudo_r_squared*100:.1f}% — 
        the rest is things we can't measure. {'That the effect is still significant means speed matters despite all these other factors.' if regression_result.is_response_time_significant else ''}
        """)
        
        if show_math:
            st.markdown("---")
            st.markdown("#### The Formula")
            st.latex(r"\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 \cdot \text{ResponseTime} + \beta_2 \cdot \text{LeadSource}")
            
            st.markdown("""
            **Breaking down each symbol:**
            
            | Symbol | Name | What it means |
            |:-------|:-----|:--------------|
            | p | Probability | Chance of closing the deal (0 to 1) |
            | p/(1-p) | Odds | Another way to express probability (e.g., "3 to 1 odds") |
            | log(...) | Log-odds | Mathematical transformation that makes the model work |
            | β₀ | Intercept | Baseline log-odds for the reference group |
            | β₁ | Response time coefficient | How much response speed shifts the log-odds |
            | β₂ | Lead source coefficient | How much lead source shifts the log-odds |
            
            **The key insight:** The **odds ratio** is e^β (e raised to the coefficient power).
            So if β₁ = 0.8 for fast responders, the odds ratio is e^0.8 ≈ 2.2x.
            """)


def render_weekly_trends_step(weekly_analysis, df: pd.DataFrame) -> None:
    """
    Render the week-over-week trends analysis step.
    
    WHY THIS MATTERS:
    -----------------
    Shows how metrics are changing over time.
    Helps identify if response times are improving or declining.
    
    PARAMETERS:
    -----------
    weekly_analysis : WeeklyAnalysis
        The weekly trends analysis result
    df : pd.DataFrame
        The full preprocessed DataFrame (needed for deep dive analysis)
    """
    # Bridge sentence to maintain narrative flow
    st.info(f"💡 **Why this step:** {get_step_bridge('weekly_trends')}")
    
    st.markdown("### 📅 Week-over-Week Trends")
    
    st.markdown("""
    **Why this matters:**
    Are things getting better or worse over time? This view helps you spot trends 
    and see if any changes you've made are working.
    """)
    
    # =========================================================================
    # SECTION 1: Summary Metrics with explanation
    # =========================================================================
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Weeks Analyzed",
            f"{weekly_analysis.weeks_analyzed}"
        )
        st.caption("How many weeks of data we have")
    
    # Show trend direction for close rate with explanation
    if 'close_rate' in weekly_analysis.trends:
        cr_trend = weekly_analysis.trends['close_rate']
        direction = cr_trend.get('direction', 'unknown')
        trend_info = {
            'improving': ('📈', 'Getting better!', 'green'),
            'declining': ('📉', 'Getting worse', 'red'),
            'stable': ('➡️', 'Staying steady', 'gray')
        }.get(direction, ('❓', 'Unknown', 'gray'))
        
        with col2:
            st.metric(
                "Close Rate Trend",
                f"{trend_info[0]} {direction.title()}"
            )
            st.caption(trend_info[1])
    
    # Show trend direction for response time with explanation
    if 'response_time' in weekly_analysis.trends:
        rt_trend = weekly_analysis.trends['response_time']
        direction = rt_trend.get('direction', 'unknown')
        # Note: for response time, "improving" means getting FASTER (lower)
        trend_info = {
            'improving': ('⚡', 'Getting faster!', 'green'),
            'declining': ('🐢', 'Getting slower', 'red'),
            'stable': ('➡️', 'Staying steady', 'gray')
        }.get(direction, ('❓', 'Unknown', 'gray'))
        
        with col3:
            st.metric(
                "Response Time Trend",
                f"{trend_info[0]} {direction.title()}"
            )
            st.caption(trend_info[1])
    
    # =========================================================================
    # SECTION 2: Percentage Points Explainer
    # =========================================================================
    render_percentage_points_explainer()
    
    # =========================================================================
    # SECTION 3: Weekly Data Table with human-friendly headers
    # =========================================================================
    st.markdown("#### 📊 Weekly Performance Data")
    
    st.markdown("""
    Each row is one week. Watch for patterns across multiple weeks — 
    single-week changes can be random noise, but consistent trends are meaningful.
    """)
    
    # Format the weekly stats for display with better column names
    display_df = format_weekly_stats_for_display(weekly_analysis.weekly_stats)
    
    # Rename columns to be more understandable
    column_renames = {
        'Week': 'Week Of',
        'Leads': 'Total Leads',
        'Orders': 'Sales Made',
        'Close Rate': 'Success Rate',
        'WoW Change': 'Change from Last Week',
        'Response Time': 'Avg Response Time'
    }
    display_df = display_df.rename(columns={k: v for k, v in column_renames.items() if k in display_df.columns})
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    # =========================================================================
    # SECTION 4: Week-over-Week Comparison with explanations
    # =========================================================================
    if weekly_analysis.weeks_analyzed >= 2:
        st.markdown("---")
        st.markdown("#### 📆 This Week vs Last Week")
        
        comparison = get_wow_comparison(weekly_analysis.weekly_stats)
        
        if 'error' not in comparison:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                change = comparison['changes']['close_rate_pp']
                wow = format_wow_change(
                    comparison['this_week']['close_rate'],
                    comparison['last_week']['close_rate']
                )
                st.metric(
                    "Success Rate",
                    f"{comparison['this_week']['close_rate']*100:.1f}%",
                    f"{wow['emoji']} {abs(change):.1f}pp {'up' if change > 0 else 'down'}" if change != 0 else "No change"
                )
                st.caption(f"Was {comparison['last_week']['close_rate']*100:.1f}% last week")
            
            with col2:
                orders_diff = comparison['changes']['n_orders_diff']
                st.metric(
                    "Sales This Week",
                    f"{comparison['this_week']['n_orders']:,}",
                    f"{orders_diff:+,} from last week"
                )
                st.caption(f"Was {comparison['last_week']['n_orders']:,} last week")
            
            with col3:
                change_pct = comparison['changes']['n_leads_pct']
                st.metric(
                    "Leads This Week",
                    f"{comparison['this_week']['n_leads']:,}",
                    f"{change_pct:+.1f}% from last week"
                )
                st.caption(f"Was {comparison['last_week']['n_leads']:,} last week")
            
            with col4:
                if comparison['this_week'].get('median_response_mins'):
                    rt_change = comparison['changes'].get('response_time_diff', 0)
                    emoji = '⚡' if rt_change < 0 else '🐢' if rt_change > 0 else '➡️'
                    st.metric(
                        "Response Time",
                        f"{comparison['this_week']['median_response_mins']:.0f} min",
                        f"{emoji} {abs(rt_change):.0f} min {'faster' if rt_change < 0 else 'slower'}" if rt_change != 0 else "No change",
                        delta_color="inverse"  # Lower is better
                    )
                    st.caption("Lower is better")
    
    # =========================================================================
    # SECTION 5: Insights with context
    # =========================================================================
    st.markdown("---")
    st.markdown("#### 💡 Key Insights")
    
    if weekly_analysis.insights:
        for insight in weekly_analysis.insights:
            st.info(insight)
    else:
        st.info("Keep collecting data — meaningful trends usually emerge after 4+ weeks.")
    
    # =========================================================================
    # SECTION 5.5: Connection to Main Finding - Real-time Validation
    # =========================================================================
    st.markdown("---")
    st.markdown("#### 🔗 Does This Support Our Main Finding?")
    
    st.markdown("""
    If faster responses truly cause higher conversion, we should see this relationship 
    in the week-over-week data: **weeks with faster responses should have higher close rates.**
    """)
    
    # Check if we have the trend data to make this assessment
    if 'close_rate' in weekly_analysis.trends and 'response_time' in weekly_analysis.trends:
        cr_trend = weekly_analysis.trends['close_rate']
        rt_trend = weekly_analysis.trends['response_time']
        
        cr_direction = cr_trend.get('direction', 'unknown')
        rt_direction = rt_trend.get('direction', 'unknown')
        
        # Check if trends are consistent with our main finding
        # (faster response = improving, better close rate = improving)
        if rt_direction == 'improving' and cr_direction == 'improving':
            st.success(f"""
            **✅ Real-time Validation: Trends Support the Finding**
            
            Over the past {weekly_analysis.weeks_analyzed} weeks:
            - Response times are **getting faster** ({rt_direction}) ⚡
            - Close rates are **improving** ({cr_direction}) 📈
            
            This is exactly what we'd expect if faster responses drive higher conversion.
            The week-over-week data is **consistent** with our statistical finding.
            """)
        elif rt_direction == 'declining' and cr_direction == 'declining':
            st.warning(f"""
            **⚠️ Concerning Pattern: Both Metrics Declining**
            
            Over the past {weekly_analysis.weeks_analyzed} weeks:
            - Response times are **getting slower** ({rt_direction}) 🐢
            - Close rates are **declining** ({cr_direction}) 📉
            
            This pattern is **consistent** with the hypothesis that response time affects 
            conversion — but it's moving in the wrong direction. Action may be needed.
            """)
        elif rt_direction == 'improving' and cr_direction == 'declining':
            st.info(f"""
            **ℹ️ Mixed Signals: Faster Responses, Lower Conversion**
            
            Over the past {weekly_analysis.weeks_analyzed} weeks:
            - Response times are **getting faster** ({rt_direction}) ⚡
            - Close rates are **declining** ({cr_direction}) 📉
            
            This is unexpected if speed is the primary driver of conversion. 
            Other factors may be at play (lead quality, seasonality, etc.).
            """)
        elif rt_direction == 'declining' and cr_direction == 'improving':
            st.info(f"""
            **ℹ️ Mixed Signals: Slower Responses, Higher Conversion**
            
            Over the past {weekly_analysis.weeks_analyzed} weeks:
            - Response times are **getting slower** ({rt_direction}) 🐢
            - Close rates are **improving** ({cr_direction}) 📈
            
            This suggests other factors may be contributing to conversion success,
            independent of response time. Lead quality or other improvements may explain this.
            """)
        else:
            st.info(f"""
            **ℹ️ Trends are Stable**
            
            Over the past {weekly_analysis.weeks_analyzed} weeks, metrics have been relatively stable.
            Continue monitoring as more data accumulates.
            """)
    else:
        st.info("""
        Trend data not available. Continue collecting data to see how weekly 
        changes in response time correlate with changes in close rate.
        """)
    
    # =========================================================================
    # SECTION 6: Interpretation Help
    # =========================================================================
    with st.expander("📖 How to Read This Analysis"):
        st.markdown("""
        ### Understanding the Trends
        
        **What the Trend Indicators Mean:**
        
        | Indicator | For Close Rate | For Response Time |
        |:----------|:---------------|:------------------|
        | 📈 Improving | Closing more deals | Responding faster |
        | 📉 Declining | Closing fewer deals | Responding slower |
        | ➡️ Stable | About the same | About the same |
        
        **What to Look For:**
        
        1. **Consistent patterns** — A 3+ week trend is more reliable than single-week spikes
        2. **Correlation** — When response time improves, does close rate follow?
        3. **Seasonality** — Some weeks may naturally be better/worse (holidays, etc.)
        
        **Don't Overreact To:**
        
        - Single-week changes (could be random noise)
        - Small fluctuations (±1-2 percentage points week-to-week is normal)
        - Weeks with very few leads (less reliable data)
        
        **Good Signs:**
        - Close rate trending up 📈
        - Response time trending down ⚡
        - Both improving together
        
        **Warning Signs:**
        - Close rate trending down 📉
        - Response time getting slower 🐢
        - Diverging trends (responding faster but closing less — might indicate other issues)
        """)
    
    # =========================================================================
    # SECTION 7: Week Deep Dive - Full Analysis for a Specific Week
    # =========================================================================
    st.markdown("---")
    render_week_deep_dive_section(weekly_analysis, df)
    
    # =========================================================================
    # SECTION 8: Week Comparison - Compare Two Specific Weeks
    # =========================================================================
    st.markdown("---")
    render_week_comparison_section(df)


def render_week_deep_dive_section(weekly_analysis, df: pd.DataFrame) -> None:
    """
    Render the week deep-dive section with full statistical analysis.
    
    WHY THIS SECTION:
    -----------------
    Users want to "zoom in" on specific weeks to see:
    - Full statistical tests for that week
    - Comparison to the overall dataset
    - Educational narrative explaining the week's story
    
    PARAMETERS:
    -----------
    weekly_analysis : WeeklyAnalysis
        The weekly trends analysis result
    df : pd.DataFrame
        The full preprocessed DataFrame
    """
    st.markdown("### 🔎 Deep Dive: Analyze a Specific Week")
    
    # Get available weeks
    available_weeks = get_available_weeks(df)
    
    if len(available_weeks) < 1:
        st.info("No weeks available for deep dive analysis.")
        return
    
    # Educational intro
    with st.expander("📖 Why analyze individual weeks?"):
        render_week_analysis_educational_intro()
    
    # Week selector
    week_options = {
        f"{w['week_label']} - {w['week_end']} ({w['n_leads']:,} leads, {w['close_rate']*100:.1f}% close rate)": w['week_start']
        for w in available_weeks
    }
    
    # Default to most recent week
    default_index = len(available_weeks) - 1
    
    selected_week_label = st.selectbox(
        "Select a week to analyze:",
        options=list(week_options.keys()),
        index=default_index,
        key="week_deep_dive_selector"
    )
    
    selected_week_start = week_options[selected_week_label]
    
    # Run analysis button (to avoid running on every interaction)
    if st.button("🔬 Run Full Analysis for This Week", type="primary", key="run_week_analysis"):
        st.session_state['week_deep_dive_result'] = run_weekly_statistical_analysis(
            df, selected_week_start
        )
        st.session_state['week_deep_dive_selected'] = selected_week_start
    
    # Display results if available
    if ('week_deep_dive_result' in st.session_state and 
        st.session_state.get('week_deep_dive_selected') == selected_week_start):
        
        week_result = st.session_state['week_deep_dive_result']
        overall_close_rate = df['ordered'].mean()
        
        # Show warnings first
        if week_result.warnings:
            for warning in week_result.warnings:
                st.warning(warning)
        
        # Summary metrics
        st.markdown("#### 📊 Week Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Leads", f"{week_result.n_leads:,}")
        with col2:
            st.metric("Orders", f"{week_result.n_orders:,}")
        with col3:
            diff = (week_result.close_rate - overall_close_rate) * 100
            st.metric(
                "Close Rate", 
                f"{week_result.close_rate*100:.1f}%",
                f"{diff:+.1f}pp vs overall"
            )
        with col4:
            if week_result.chi_square_result:
                sig_text = "✅ Significant" if week_result.chi_square_result.is_significant else "⚪ Not Significant"
            else:
                sig_text = "❌ Insufficient Data"
            st.metric("Response Time Effect", sig_text)
        
        # Narrative story
        st.markdown("---")
        story = generate_week_analysis_story(week_result, overall_close_rate)
        st.markdown(story)
        
        # Detailed analysis tabs
        st.markdown("---")
        st.markdown("#### 📈 Detailed Statistical Analysis")
        
        week_tabs = st.tabs(["Close Rates", "Chi-Square Test", "Proportions Tests", "Regression"])
        
        # Tab 1: Close Rates by Bucket
        with week_tabs[0]:
            st.info(get_week_educational_context('close_rates'))
            
            if not week_result.close_rates_by_bucket.empty:
                fig = create_close_rate_chart(week_result.close_rates_by_bucket)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show data table
                with st.expander("View Data Table"):
                    display_df = week_result.close_rates_by_bucket[
                        ['bucket', 'n_leads', 'n_orders', 'close_rate']
                    ].copy()
                    display_df['close_rate'] = display_df['close_rate'].apply(lambda x: f"{x*100:.1f}%")
                    display_df.columns = ['Response Time', 'Leads', 'Orders', 'Close Rate']
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Show Your Work section
                st.markdown("---")
                render_weekly_close_rate_calculations(week_result)
            else:
                st.info("Not enough data to show close rates by bucket.")
        
        # Tab 2: Chi-Square Test
        with week_tabs[1]:
            st.info(get_week_educational_context('chi_square'))
            
            if week_result.chi_square_result:
                chi_sq = week_result.chi_square_result
                p_exp = get_p_value_explanation(chi_sq.p_value)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Chi-Square Statistic", f"{chi_sq.statistic:.2f}")
                with col2:
                    st.metric("P-Value", f"{chi_sq.p_value:.4f}")
                
                if chi_sq.is_significant:
                    st.success(f"**{p_exp['emoji']} {p_exp['verdict']}**\n\n{p_exp['plain_english']}")
                else:
                    st.warning(f"**{p_exp['emoji']} {p_exp['verdict']}**\n\n{p_exp['plain_english']}")
                
                st.markdown(chi_sq.interpretation)
                
                # Show Your Work section
                st.markdown("---")
                worked_example = generate_weekly_chi_square_worked_example(week_result)
                render_weekly_chi_square_worked_example(worked_example)
            else:
                st.info("Chi-square test could not be run due to insufficient data this week.")
        
        # Tab 3: Proportions Tests
        with week_tabs[2]:
            st.info(get_week_educational_context('proportions'))
            
            if week_result.pairwise_results:
                # Create comparison table
                comparisons = []
                for result in week_result.pairwise_results:
                    if hasattr(result, 'details') and result.details:
                        p_val = result.p_value
                        p_exp = get_p_value_explanation(p_val)
                        diff = result.details.get('difference', 0) * 100
                        
                        comparisons.append({
                            'Comparison': result.test_name.replace('Z-Test for Proportions: ', ''),
                            'Difference (pp)': f"{diff:+.1f}",
                            'P-Value': f"{p_val:.4f}",
                            'Significant?': p_exp['emoji'] + " " + ("Yes" if result.is_significant else "No")
                        })
                
                if comparisons:
                    st.dataframe(pd.DataFrame(comparisons), use_container_width=True, hide_index=True)
                
                # Show Your Work section
                st.markdown("---")
                render_weekly_proportion_test_calculations(week_result)
            else:
                st.info("Pairwise comparisons could not be run due to insufficient data this week.")
        
        # Tab 4: Regression
        with week_tabs[3]:
            st.info(get_week_educational_context('regression'))
            
            if week_result.regression_result and not week_result.regression_result.odds_ratios.empty:
                reg = week_result.regression_result
                
                st.markdown(f"**Model Pseudo R²:** {reg.pseudo_r_squared:.3f}")
                
                if reg.is_response_time_significant:
                    st.success("Response time effect remains significant after controlling for lead source.")
                else:
                    st.warning("Response time effect is not significant after controlling for lead source.")
                
                # Odds ratios table
                st.markdown("**Odds Ratios by Response Time Bucket:**")
                or_df = reg.odds_ratios.copy()
                or_display = or_df[['bucket', 'odds_ratio', 'ci_lower', 'ci_upper', 'p_value']].copy()
                or_display.columns = ['Bucket', 'Odds Ratio', 'CI Lower', 'CI Upper', 'P-Value']
                or_display['Odds Ratio'] = or_display['Odds Ratio'].apply(lambda x: f"{x:.2f}x")
                or_display['P-Value'] = or_display['P-Value'].apply(lambda x: f"{x:.4f}")
                st.dataframe(or_display, use_container_width=True, hide_index=True)
            else:
                st.info("Logistic regression could not be run due to insufficient data this week.")


def render_week_comparison_section(df: pd.DataFrame) -> None:
    """
    Render the week comparison section for side-by-side analysis.
    
    WHY THIS SECTION:
    -----------------
    Users often want to compare two specific weeks to understand:
    - What improved or declined?
    - Did statistical significance change?
    - What story does the comparison tell?
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        The full preprocessed DataFrame
    """
    st.markdown("### ⚖️ Compare Two Weeks")
    
    st.markdown("""
    Select two weeks to compare side-by-side. This helps you understand 
    what changed between specific time periods.
    """)
    
    # Get available weeks
    available_weeks = get_available_weeks(df)
    
    if len(available_weeks) < 2:
        st.info("Need at least 2 weeks of data to compare weeks.")
        return
    
    # Create week options
    week_options = {
        f"{w['week_label']} - {w['week_end']} ({w['n_leads']:,} leads)": w['week_start']
        for w in available_weeks
    }
    week_labels = list(week_options.keys())
    
    # Two column layout for selectors
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Week 1 (Earlier)**")
        week1_label = st.selectbox(
            "Select first week:",
            options=week_labels,
            index=0,
            key="week_compare_1",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("**Week 2 (Later)**")
        # Default to the last week
        default_week2_index = min(len(week_labels) - 1, 1)
        week2_label = st.selectbox(
            "Select second week:",
            options=week_labels,
            index=default_week2_index,
            key="week_compare_2",
            label_visibility="collapsed"
        )
    
    week1_start = week_options[week1_label]
    week2_start = week_options[week2_label]
    
    # Check if same week selected
    if week1_start == week2_start:
        st.warning("Please select two different weeks to compare.")
        return
    
    # Run comparison button
    if st.button("📊 Compare These Weeks", type="primary", key="run_week_comparison"):
        with st.spinner("Running comparison analysis..."):
            comparison = compare_two_weeks(df, week1_start, week2_start)
            st.session_state['week_comparison_result'] = comparison
            st.session_state['week_comparison_weeks'] = (week1_start, week2_start)
    
    # Display results if available
    if ('week_comparison_result' in st.session_state and 
        st.session_state.get('week_comparison_weeks') == (week1_start, week2_start)):
        
        comparison = st.session_state['week_comparison_result']
        week1 = comparison.week1
        week2 = comparison.week2
        
        # Side-by-side metrics
        st.markdown("#### 📈 Key Metrics Comparison")
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.markdown(f"**{week1.week_label}**")
            st.metric("Leads", f"{week1.n_leads:,}")
            st.metric("Orders", f"{week1.n_orders:,}")
            st.metric("Close Rate", f"{week1.close_rate*100:.1f}%")
        
        with metric_col2:
            st.markdown(f"**{week2.week_label}**")
            lead_diff = week2.n_leads - week1.n_leads
            order_diff = week2.n_orders - week1.n_orders
            st.metric("Leads", f"{week2.n_leads:,}", f"{lead_diff:+,}")
            st.metric("Orders", f"{week2.n_orders:,}", f"{order_diff:+,}")
            st.metric("Close Rate", f"{week2.close_rate*100:.1f}%", 
                     f"{comparison.close_rate_change:+.1f}pp")
        
        with metric_col3:
            st.markdown("**Change**")
            lead_pct = (lead_diff / week1.n_leads * 100) if week1.n_leads > 0 else 0
            st.metric("Lead Volume", f"{lead_pct:+.1f}%")
            
            # Chi-square comparison
            week1_sig = "✅" if (week1.chi_square_result and week1.chi_square_result.is_significant) else "⚪"
            week2_sig = "✅" if (week2.chi_square_result and week2.chi_square_result.is_significant) else "⚪"
            st.metric("Speed→Close Rate", f"{week1_sig} → {week2_sig}")
            
            if comparison.significance_changed:
                st.warning("Significance changed!")
        
        # Bucket comparison chart
        st.markdown("---")
        st.markdown("#### 📊 Close Rates by Response Time Bucket")
        
        if not comparison.bucket_comparison.empty:
            # Create comparison chart
            bc = comparison.bucket_comparison
            
            # Prepare data for display
            display_data = []
            for _, row in bc.iterrows():
                bucket = row['bucket']
                rate1_col = [c for c in bc.columns if 'close_rate' in c and week1.week_label in c]
                rate2_col = [c for c in bc.columns if 'close_rate' in c and week2.week_label in c]
                
                rate1 = row[rate1_col[0]] * 100 if rate1_col and not pd.isna(row[rate1_col[0]]) else None
                rate2 = row[rate2_col[0]] * 100 if rate2_col and not pd.isna(row[rate2_col[0]]) else None
                change = row.get('change_pp', None)
                
                display_data.append({
                    'Response Time': bucket,
                    f'{week1.week_label} Rate': f"{rate1:.1f}%" if rate1 is not None else "N/A",
                    f'{week2.week_label} Rate': f"{rate2:.1f}%" if rate2 is not None else "N/A",
                    'Change': f"{change:+.1f}pp" if change is not None and not pd.isna(change) else "N/A"
                })
            
            st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)
        
        # Narrative comparison story
        st.markdown("---")
        st.markdown("#### 📖 The Comparison Story")
        
        comparison_story = generate_week_comparison_story(comparison, df['ordered'].mean())
        st.markdown(comparison_story)
        
        # Detailed comparison expander
        with st.expander("📊 View Detailed Test Results for Both Weeks"):
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.markdown(f"##### {week1.week_label}")
                if week1.chi_square_result:
                    st.markdown(f"**Chi-Square:** {week1.chi_square_result.statistic:.2f} (p={week1.chi_square_result.p_value:.4f})")
                    st.markdown(f"**Significant:** {'Yes' if week1.chi_square_result.is_significant else 'No'}")
                else:
                    st.markdown("*Chi-square test not available*")
                
                if week1.regression_result:
                    st.markdown(f"**Regression R²:** {week1.regression_result.pseudo_r_squared:.3f}")
                    st.markdown(f"**Speed Significant:** {'Yes' if week1.regression_result.is_response_time_significant else 'No'}")
            
            with detail_col2:
                st.markdown(f"##### {week2.week_label}")
                if week2.chi_square_result:
                    st.markdown(f"**Chi-Square:** {week2.chi_square_result.statistic:.2f} (p={week2.chi_square_result.p_value:.4f})")
                    st.markdown(f"**Significant:** {'Yes' if week2.chi_square_result.is_significant else 'No'}")
                else:
                    st.markdown("*Chi-square test not available*")
                
                if week2.regression_result:
                    st.markdown(f"**Regression R²:** {week2.regression_result.pseudo_r_squared:.3f}")
                    st.markdown(f"**Speed Significant:** {'Yes' if week2.regression_result.is_response_time_significant else 'No'}")


def render_recommendations(stat_results, regression_result, close_rates=None) -> None:
    """
    Render actionable recommendations based on the analysis.
    
    Handles multiple scenarios:
    - Both significant (strong evidence)
    - Chi-square only (confounding suspected)
    - Neither significant (no effect)
    - Reverse effects (slower is better)
    - Exceptional effects (very large differences)
    """
    st.subheader("💡 Recommendations")
    
    chi_sq = stat_results['chi_square']
    
    # Detect reverse effect if we have close_rates data
    is_reverse = False
    if close_rates is not None and len(close_rates) >= 2:
        fastest_rate = close_rates.iloc[0]['close_rate']
        slowest_rate = close_rates.iloc[-1]['close_rate']
        is_reverse = fastest_rate < slowest_rate
    
    # Detect exceptional effect
    p_value = chi_sq.p_value if hasattr(chi_sq, 'p_value') else None
    is_exceptional = (p_value is not None and p_value < 0.0001) or False
    
    # Handle reverse effect scenario
    if is_reverse and chi_sq.is_significant:
        st.warning("""
        ### ⚠️ Action: Investigate Counterintuitive Finding Before Acting
        
        **The finding is surprising:** Slower responses are associated with higher conversion rates.
        This contradicts conventional wisdom and requires careful investigation.
        
        **Recommended actions (in priority order):**
        1. **Verify data quality:**
           - Check for data collection errors or systematic biases
           - Verify that response time measurements are accurate
           - Examine if any filters or selection mechanisms might create this pattern
        
        2. **Investigate potential explanations:**
           - **Selection bias:** Are high-quality leads getting slower but more thoughtful responses?
           - **Time-of-day effects:** Do slower responses correlate with business hours when decision-makers are available?
           - **Quality vs. speed trade-off:** Do more thorough responses (requiring more time) lead to better outcomes?
           - **Confounders:** Are there unmeasured variables explaining both response time and conversion?
        
        3. **Consider the limitations of observational data:**
           - This analysis cannot definitively establish causation
           - Unmeasured confounders may explain the relationship
           - Do not make major operational changes based solely on this observational analysis
        
        4. **Consider qualitative investigation:**
           - Interview salespeople to understand response prioritization
           - Analyze response content quality vs. speed
           - Review time-of-day and day-of-week patterns
        
        **Critical Warning:** Do not automatically slow down responses based on this finding. 
        The association may be spurious or explained by unmeasured factors. This observational 
        analysis has limitations and cannot definitively establish causation.
        """)
    
    # Handle exceptional positive effect
    elif chi_sq.is_significant and regression_result.is_response_time_significant and is_exceptional:
        st.success("""
        ### 🚀 Action: Consider the Strength of This Association
        
        **The finding is exceptionally strong:** Response time shows an unusually large association 
        with conversion rates, and this pattern persists after controlling for confounders.
        
        **Recommended actions (in priority order):**
        1. **Consider the strength of this association:**
           - The exceptional effect size suggests this relationship warrants careful consideration
           - However, this observational analysis cannot definitively establish causation
           - Unmeasured confounders may explain the relationship
           - Consider this evidence when making decisions, but acknowledge the limitations
        
        2. **If you decide to act on this evidence:**
           - This could represent a major optimization opportunity
           - Set aggressive response time SLAs (e.g., respond within 15 minutes)
           - Implement real-time alerts for leads waiting too long
           - Consider after-hours coverage for night/weekend leads
           - Allocate resources to response speed infrastructure
           - Monitor response time as a key performance metric
           - Acknowledge that the relationship may be due to confounding
        
        3. **Continue monitoring and analysis:**
           - Track whether response time improvements correlate with conversion changes
           - Investigate what factors might be driving the apparent effect
           - Focus optimization efforts on factors that genuinely drive conversion
        
        4. **Document the mechanism:**
           - If causation is confirmed, investigate WHY response speed matters so much
           - Understand the mechanism to optimize more effectively
           - Consider whether there are threshold effects or optimal response windows
        
        **Why this matters:** Effects of this magnitude are rare and could have transformative 
        impact if causally established. However, exceptional effects in observational data can 
        also indicate exceptional confounding. This observational analysis cannot definitively 
        distinguish between these possibilities.
        """)
    
    # Handle standard both-significant scenario
    elif chi_sq.is_significant and regression_result.is_response_time_significant:
        st.markdown("""
        ### Action: Consider Evidence When Making Decisions
        
        Based on this analysis, faster response times are associated with 
        higher close rates, even after controlling for lead source. However, 
        this is observational data and cannot prove causation.
        
        **Recommended actions (in priority order):**
        1. **Consider the limitations of observational data:**
           - This analysis cannot definitively establish causation
           - The association could be explained by unmeasured confounders
           - Consider this evidence when making decisions, but acknowledge the limitations
        
        2. If you decide to act on this evidence:
           - Set response time SLAs (e.g., respond within 15 minutes)
           - Implement alerts for leads waiting too long
           - Consider after-hours coverage for night/weekend leads
           - Monitor response time as a key performance metric
           - Continue monitoring to see if improvements correlate with conversion changes
        
        3. Continue investigating the relationship:
           - The observational association may be due to confounding
           - Monitor whether response time improvements correlate with conversion changes
           - Focus optimization efforts on factors that genuinely drive conversion
        
        **Limitations of observational data:** The observed association could be due to 
        selection mechanisms (salespeople prioritizing high-quality leads), unmeasured 
        confounders, or reverse causation. This analysis cannot definitively rule these out.
        
        **Estimated impact (if association is causal):** Based on the observational data, 
        improving response time from the slowest bucket to the fastest could potentially 
        increase close rates by the observed difference. However, this estimate assumes 
        causation, which this observational analysis cannot definitively establish.
        """)
    
    # Handle chi-square significant but regression not (confounding suspected)
    elif chi_sq.is_significant:
        from explanations.common import render_contradiction_explanation
        st.markdown("""
        ### Action: Investigate Confounding Before Acting
        
        The data shows a relationship between response time and close rate,
        but this relationship disappears after controlling for lead source.
        This strongly suggests confounding rather than a direct causal effect.
        """)
        
        # Show contradiction explanation if we have the data
        try:
            render_contradiction_explanation(
                {'p_value': chi_sq.p_value} if hasattr(chi_sq, 'p_value') else {},
                {'p_value': getattr(regression_result, 'p_value', None)} if hasattr(regression_result, 'p_value') else {},
                chi_sq.is_significant,
                False
            )
        except:
            pass
        
        st.markdown("""
        **Recommended actions:**
        1. **Do NOT invest in response time improvements** based on this analysis alone
        2. **Acknowledge the limitations** of observational data in establishing causation
        3. **Analyze response time by lead source separately** to understand the pattern
        4. **Investigate lead source differences:**
           - Why do different sources have different response times?
           - Why do different sources have different conversion rates?
           - What drives the prioritization of certain lead types?
        5. **Focus on lead source optimization** rather than response time if that's the true driver
        
        **Key insight:** The apparent response time effect is likely explained by lead source 
        differences. Optimize lead source quality or allocation rather than response speed.
        """)
    
    # Handle neither significant
    else:
        st.markdown("""
        ### Action: Focus on Other Factors
        
        Response time does not appear to significantly affect close rates 
        in this data. Other factors may be more important.
        
        **Recommended actions:**
        1. **Do not invest in response time improvements** based on this analysis
        2. **Analyze other factors:**
           - Lead quality and source differences
           - Sales representative skill and experience
           - Time-of-day and day-of-week effects
           - Lead engagement and interest level
           - Product-market fit and pricing
        3. **Collect more data if sample size is limited:**
           - Small samples may fail to detect real but small effects
           - Consider if you have sufficient statistical power
        4. **Consider if the response time range is too narrow:**
           - If all responses are already fast (e.g., all < 30 minutes), 
             there may be little variation to detect an effect
        5. **Focus optimization efforts elsewhere:**
           - Investigate factors with clearer evidence of impact
           - Don't optimize response time if the evidence doesn't support it
        
        **Important:** This analysis cannot prove that response time has *no* effect — 
        only that we cannot detect an effect with the current data. A very small effect 
        might exist but be undetectable with this sample size. However, any such effect 
        would likely be too small to justify major operational changes.
        """)


def render_export_options(df, close_rates, stat_results, regression_result) -> None:
    """Render export options for the analysis results."""
    st.subheader("📥 Export Results")
    
    col1, col2 = st.columns(2)
    
    # Calculate key metrics for exports
    fastest_bucket = close_rates.iloc[0]
    slowest_bucket = close_rates.iloc[-1]
    rate_diff = fastest_bucket['close_rate'] - slowest_bucket['close_rate']
    rate_multiplier = fastest_bucket['close_rate'] / slowest_bucket['close_rate'] if slowest_bucket['close_rate'] > 0 else 1
    
    chi_sq = stat_results['chi_square']
    is_significant = chi_sq.is_significant
    p_value = chi_sq.p_value
    
    with col1:
        # Export summary as CSV - simplified and clear
        export_df = close_rates[['bucket', 'n_leads', 'n_orders', 'close_rate']].copy()
        export_df['close_rate'] = export_df['close_rate'].apply(lambda x: f"{x*100:.1f}%")
        export_df.columns = ['Response Time', 'Leads', 'Sales', 'Success Rate']
        
        # Add notes column
        notes = []
        for i, row in export_df.iterrows():
            if i == 0:
                notes.append("⭐ Best performance")
            elif i == len(export_df) - 1:
                notes.append("⚠️ Needs improvement")
            else:
                notes.append("")
        export_df['Notes'] = notes
        
        csv = export_df.to_csv(index=False)
        st.download_button(
            "📊 Download Close Rates (CSV)",
            csv,
            "close_rates_by_response_time.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        # Create plain-English summary
        summary_text = generate_plain_english_summary(
            df, close_rates, stat_results, regression_result,
            fastest_bucket, slowest_bucket, rate_diff, rate_multiplier,
            is_significant, p_value
        )
        
        st.download_button(
            "📄 Download Summary (TXT)",
            summary_text,
            "analysis_summary.txt",
            "text/plain",
            use_container_width=True
        )
    
    # =========================================================================
    # VERIFICATION CSV EXPORT
    # =========================================================================
    st.markdown("---")
    st.markdown("### 🔍 Verification Export")
    st.markdown("""
    Download all raw data and calculations to verify our analysis independently.
    This CSV contains:
    - All leads with computed response times and bucket assignments
    - Bucket summaries with formulas
    - Statistical test results
    - Regression coefficients
    """)
    
    verification_csv = create_verification_csv(
        df,
        chi_sq_result=stat_results.get('chi_square'),
        regression_result=regression_result,
        descriptive_stats=close_rates
    )
    
    st.download_button(
        "📥 Download Verification Data (CSV)",
        verification_csv,
        "response_time_analysis_verification.csv",
        "text/csv",
        use_container_width=True,
        help="Complete data export for independent verification of all calculations"
    )


def generate_plain_english_summary(
    df, close_rates, stat_results, regression_result,
    fastest_bucket, slowest_bucket, rate_diff, rate_multiplier,
    is_significant, p_value
) -> str:
    """
    Generate a plain-English summary that non-technical users can understand.
    
    WHY THIS MATTERS:
    -----------------
    The downloaded summary should tell a complete story without jargon.
    Someone reading this should understand:
    1. What we found (the bottom line)
    2. What the numbers mean
    3. What they should do about it
    """
    
    # Calculate confidence level as a percentage
    confidence_pct = (1 - p_value) * 100
    if confidence_pct > 99.9:
        confidence_pct = 99.9  # Cap for readability
    
    # Determine the strength of evidence
    if p_value < 0.001:
        confidence_words = "extremely confident"
        doubt_words = "virtually no chance"
    elif p_value < 0.01:
        confidence_words = "very confident"
        doubt_words = "less than 1% chance"
    elif p_value < 0.05:
        confidence_words = "confident"
        doubt_words = "less than 5% chance"
    else:
        confidence_words = "not confident"
        doubt_words = f"about {p_value*100:.0f}% chance"
    
    # Build the summary
    summary = f"""
================================================================================
                    RESPONSE TIME ANALYSIS - EXECUTIVE SUMMARY
================================================================================

THE BOTTOM LINE
---------------
"""
    
    if is_significant and regression_result.is_response_time_significant:
        summary += f"""
✅ STRONG ASSOCIATION BETWEEN RESPONSE TIME AND CONVERSION

We analyzed {len(df):,} leads and found a statistically significant association 
between faster response times and higher conversion rates. The probability this 
pattern is due to random chance is less than {p_value*100:.4f}%.

However, this is observational data and cannot prove causation. The association 
could be explained by unmeasured confounders or selection mechanisms. When making 
decisions, consider this evidence while acknowledging the limitations of observational data.

"""
    elif is_significant:
        summary += f"""
⚠️ RESPONSE TIME APPEARS TO MATTER, BUT WE'RE NOT 100% SURE WHY

We found a relationship between response time and sales, but it might be 
partly explained by other factors (like which types of leads get faster 
responses). More investigation is recommended.

"""
    else:
        summary += f"""
ℹ️ NO CLEAR RELATIONSHIP FOUND

Based on this data, we cannot confidently say that responding faster leads 
to more sales. The differences we observed could be random variation.

"""
    
    summary += f"""
================================================================================
                              THE KEY NUMBERS
================================================================================

YOUR DATA AT A GLANCE
---------------------
• Total leads analyzed: {len(df):,}
• Total sales made: {df['ordered'].sum():,}
• Overall success rate: {df['ordered'].mean()*100:.1f}%

RESPONSE TIME BREAKDOWN
-----------------------
"""
    
    # Add response time breakdown in a clear format
    for _, row in close_rates.iterrows():
        bar_length = int(row['close_rate'] * 50)  # Scale for visual
        bar = "█" * bar_length + "░" * (15 - bar_length)
        indicator = ""
        if row['bucket'] == fastest_bucket['bucket']:
            indicator = " ← BEST"
        elif row['bucket'] == slowest_bucket['bucket']:
            indicator = " ← SLOWEST"
        
        summary += f"• {row['bucket']:12} {bar} {row['close_rate']*100:5.1f}% ({row['n_orders']:,} sales){indicator}\n"
    
    summary += f"""

THE COMPARISON
--------------
• Fastest responders ({fastest_bucket['bucket']}): {fastest_bucket['close_rate']*100:.1f}% success rate
• Slowest responders ({slowest_bucket['bucket']}): {slowest_bucket['close_rate']*100:.1f}% success rate
• Difference: {abs(rate_diff)*100:.1f} percentage points
• Fast responders convert {rate_multiplier:.1f}x more often

================================================================================
                         WHAT THE STATISTICS MEAN
================================================================================

IS THIS ASSOCIATION REAL OR RANDOM LUCK?
----------------------------------------
We used statistical tests to check if the association is statistically significant:

• Result: {"The association IS statistically significant" if is_significant else "We can't be sure the association is real"}
• Statistical significance: {confidence_pct:.1f}% confidence the association is not due to random chance
• Translation: There's {doubt_words} this association is just random variation

IMPORTANT: Statistical significance measures whether the association is real (not random), 
NOT whether it is causal. This analysis cannot prove that faster responses CAUSE higher 
conversion rates - only that there is a statistically significant association.

Think of it like flipping a coin. If you got 60 heads out of 100 flips, 
you'd wonder if the coin is unfair. Statistics helps us answer whether the pattern 
is likely real (not random) - and for your data, {"the association is statistically significant" if is_significant else "we can't rule out that it's just chance"}.

However, even a real (non-random) association does not prove causation.

"""
    
    if regression_result.is_response_time_significant:
        summary += f"""
DOES IT HOLD UP UNDER SCRUTINY?
-------------------------------
We also checked: "Is this association still present when we account for lead source?"
(Some lead sources might naturally have both faster responses AND higher 
close rates, which could create a misleading pattern.)

✅ YES - The association persists after accounting for lead source.
   However, this only controls for ONE potential confounder. The association could 
   still be explained by:
   - Lead quality signals that drive prioritization
   - Salesperson skill differences
   - Time-of-day or day-of-week effects
   - Other unmeasured confounders

This observational analysis cannot prove causation. The limitations of observational 
data mean we cannot definitively establish that faster responses cause higher conversion rates.

"""
    elif is_significant:
        summary += f"""
DOES IT HOLD UP UNDER SCRUTINY?
-------------------------------
We also checked: "Is this still true when we account for lead source?"

⚠️ THE PICTURE IS LESS CLEAR - The effect shrinks when we account for 
   lead source. Part of the relationship might be explained by different 
   types of leads getting different response times.

"""
    
    summary += f"""
================================================================================
                              TECHNICAL DETAILS
================================================================================

For those who want the statistical specifics:

Chi-Square Test (tests if there's any relationship):
• Statistic: {stat_results['chi_square'].statistic:.2f}
• P-Value: {stat_results['chi_square'].p_value:.4f}
• Interpretation: {"Statistically significant" if is_significant else "Not statistically significant"}

Logistic Regression (controls for lead source):
• Response time effect: {"Significant" if regression_result.is_response_time_significant else "Not significant"}
• Pseudo R²: {regression_result.pseudo_r_squared:.3f}
• Interpretation: Response time explains about {regression_result.pseudo_r_squared*100:.1f}% of the 
  variation in outcomes (after accounting for lead source)

================================================================================
                              IMPORTANT CAVEATS
================================================================================

1. CORRELATION ≠ CAUSATION
   This is observational data, not experimental. We found a statistically 
   significant association, but we CANNOT prove that faster responses CAUSE 
   more sales. The association could be due to:
   - Selection mechanisms (salespeople prioritizing high-quality leads)
   - Unmeasured confounders (lead quality, salesperson skill, timing effects)
   - Reverse causation (high-value leads receive faster responses)

2. WE CAN ONLY CONTROL FOR WHAT WE MEASURE
   We controlled for lead source, but there are likely other factors we didn't 
   account for that could explain the relationship. We cannot control for factors 
   we don't observe or measure.

3. STATISTICAL SIGNIFICANCE ≠ CAUSAL CERTAINTY
   A statistically significant association means the pattern is unlikely to be 
   random. It does NOT mean the relationship is causal. This observational analysis 
   has limitations in establishing causation.

4. LIMITATIONS OF OBSERVATIONAL DATA
   This analysis cannot definitively prove that speed causes conversion. The 
   association could be explained by unmeasured confounders, selection mechanisms, 
   or reverse causation. When making decisions about staffing or infrastructure 
   investments, consider this evidence while acknowledging these limitations.

5. RESULTS ARE SPECIFIC TO YOUR DATA
   What works in this time period and market might not apply elsewhere.

================================================================================
                         END OF ANALYSIS SUMMARY
================================================================================
"""
    
    return summary

