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
from analysis.advanced import run_all_advanced_analyses
from analysis.weekly_trends import (
    analyze_weekly_trends, 
    format_weekly_stats_for_display,
    get_wow_comparison
)
from config.settings import ANALYSIS_MODES
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
    render_mixed_effects_explainer,
    render_within_rep_explainer,
    render_confounding_explainer,
    get_step_bridge,
    generate_chi_square_worked_example,
    render_chi_square_worked_example,
    generate_proportion_test_worked_example
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
    
    # Get analysis mode
    analysis_mode = settings.get('analysis_mode', 'standard')
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
        
        # Advanced (if selected)
        if analysis_mode == 'advanced':
            advanced_results = run_all_advanced_analyses(df, alpha)
        else:
            advanced_results = None
    
    # =========================================================================
    # EXECUTIVE SUMMARY
    # =========================================================================
    render_executive_summary(close_rates, stat_results, regression_result, advanced_results)
    
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
    
    # Define steps based on analysis mode
    # Always include Weekly Trends if we have date data
    base_steps = ['Data Overview', 'Close Rates', 'Chi-Square', 'Proportions', 'Regression']
    if weekly_analysis:
        base_steps.append('📅 Weekly Trends')
    
    if analysis_mode == 'standard':
        step_names = base_steps
    else:
        step_names = base_steps + ['Mixed Effects', 'Within-Rep', 'Confounding']
    
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
        render_regression_step(regression_result, show_math)
    
    # Step 6: Weekly Trends (if available)
    current_tab = 5
    if weekly_analysis:
        with tabs[current_tab]:
            render_weekly_trends_step(weekly_analysis)
        current_tab += 1
    
    # Advanced steps
    if analysis_mode == 'advanced' and advanced_results:
        with tabs[current_tab]:
            render_mixed_effects_step(advanced_results.get('mixed_effects'), show_math)
        
        with tabs[current_tab + 1]:
            render_within_rep_step(advanced_results.get('within_rep'), df, show_math)
        
        with tabs[current_tab + 2]:
            render_confounding_step(advanced_results.get('confounding'), df)
    
    # =========================================================================
    # EXPORT OPTIONS
    # =========================================================================
    st.markdown("---")
    render_export_options(df, close_rates, stat_results, regression_result)


def render_executive_summary(
    close_rates: pd.DataFrame,
    stat_results: Dict[str, Any],
    regression_result,
    advanced_results: Optional[Dict[str, Any]]
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
    st.markdown("### The Central Question")
    
    st.markdown("""
    This analysis addresses a fundamental business question: **Does responding faster 
    to leads cause higher conversion rates?**
    
    The answer to this question has direct operational implications. If speed causes 
    success, investments in response time infrastructure will yield measurable returns. 
    If the correlation is spurious, such investments would be misdirected.
    """)
    
    st.markdown("---")
    st.markdown("### The Conclusion")
    
    if chi_sq.is_significant and regression_result.is_response_time_significant:
        st.success(f"""
        **The evidence demonstrates a statistically significant association between response speed and conversion.**
        
        Leads receiving responses within **{fastest_bucket['bucket']}** convert at 
        **{fastest_bucket['close_rate']*100:.1f}%** — a rate **{rate_multiplier:.1f}× higher** than 
        leads waiting **{slowest_bucket['bucket']}** ({slowest_bucket['close_rate']*100:.1f}%).
        
        This pattern is:
        - **Statistically significant** — the probability of observing this by chance is {p_exp['luck_chance']}
        - **Robust to observed confounders** — the association persists after controlling for lead source
        - **Practically meaningful** — a {rate_diff*100:.1f} percentage point difference translates 
          to substantial revenue impact at scale
        
        **Recommendation:** Before committing to major investments, run a randomized controlled experiment 
        (A/B test) to establish causation definitively. While the observational evidence is strong, 
        only experimental validation can confirm that faster responses *cause* higher conversion rates.
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
        
        **Recommendation:** Before making major investments, consider running a controlled 
        experiment (A/B test) to establish causation definitively.
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
        
        {"**Association persists after controlling for lead source.** We tested whether lead source differences could explain this pattern. The association between response speed and conversion remains after controlling for lead source. However, this only addresses *one* potential confounder. Unmeasured confounders (such as lead quality signals that drive prioritization, salesperson skill differences, or time-of-day effects) may still explain the relationship. Only a randomized experiment can definitively establish causation." if regression_result.is_response_time_significant else "**Caution warranted.** When we control for lead source, the association weakens. Some of the apparent relationship may be attributable to confounding rather than a true causal mechanism. A controlled experiment would provide more definitive evidence."}
        
        ### Causal Inference Limitations
        
        **This analysis cannot establish causation.** This is observational data, not experimental. The observed association could be explained by:
        
        - **Selection mechanisms:** Salespeople may prioritize leads that appear more likely to convert, creating a spurious correlation between speed and success
        - **Unmeasured confounders:** Lead quality signals, salesperson skill differences, or time-of-day effects that we cannot observe or control for
        - **Reverse causation:** Higher-value leads may receive faster responses, rather than speed causing higher conversion
        
        **What we can conclude:** There is a statistically significant association between response time and conversion rates that persists after controlling for lead source. This is consistent with a causal effect, but observational data alone cannot prove it.
        
        **Recommended next step:** Before making major staffing or infrastructure investments, conduct a randomized controlled experiment (A/B test) where leads are randomly assigned to response time conditions, independent of lead characteristics.
        """)
    
    # =========================================================================
    # ADVANCED ANALYSIS FINDINGS (if applicable)
    # =========================================================================
    if advanced_results:
        st.markdown("---")
        st.markdown("### Advanced Analysis Findings")
        
        within_rep = advanced_results.get('within_rep')
        mixed_effects = advanced_results.get('mixed_effects')
        
        # Within-rep findings
        if within_rep and hasattr(within_rep, 'statistics'):
            within_p = within_rep.statistics.get('p_value', 1)
            if within_p < 0.05:
                st.success("""
                **Within-Person Analysis: Causal Evidence Strengthened**
                
                The most rigorous test available — comparing each salesperson to themselves — 
                confirms the effect. When individual representatives respond quickly, they 
                convert at higher rates than when those same individuals respond slowly.
                
                This finding is particularly important because it controls for all fixed 
                characteristics of the salesperson. The effect cannot be attributed to 
                better salespeople simply being faster; speed itself appears to matter.
                """)
            else:
                st.warning("""
                **Within-Person Analysis: Results Inconclusive**
                
                When we compare each salesperson to themselves, the effect of response 
                time becomes non-significant. This suggests that between-person differences 
                may explain some of the observed correlation.
                """)
        
        # Mixed effects findings
        if mixed_effects and hasattr(mixed_effects, 'statistics'):
            icc = mixed_effects.statistics.get('icc', 0)
            if icc > 0.15:
                st.info(f"""
                **Representative-Level Variation: {icc*100:.1f}% of outcome variation is between salespeople**
                
                This substantial clustering means that controlling for salesperson identity 
                is important for accurate inference. The mixed effects model accounts for this.
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
    
    if chi_sq.is_significant and rate_diff > 0:
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
    
    # Data quality assessment
    st.markdown("---")
    render_sample_size_guidance(len(df), "conversion rate")
    
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
    
    # CI Explainer
    render_ci_explainer()
    
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
    # VISIBLE WORKED EXAMPLE - Show the calculation with actual data
    # =========================================================================
    st.markdown("---")
    
    # Generate and display the worked example using actual data
    worked_example = generate_chi_square_worked_example(df)
    render_chi_square_worked_example(worked_example)
    
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
            
            # Step 3: The test statistic
            st.markdown(f"#### Step 3: Is this difference statistically significant?")
            
            z_stat = extreme.statistic if hasattr(extreme, 'statistic') else 0
            p_val = extreme.p_value if hasattr(extreme, 'p_value') else 1
            
            st.markdown(f"""
            We calculate a **z-score** of **{z_stat:.2f}**, which tells us how many "standard deviations" 
            this difference is from what we'd expect by chance.
            
            - **Z-score:** {z_stat:.2f}
            - **P-value:** {p_val:.4f}
            
            **Translation:** {p_exp['plain_english']}
            """)
            
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


def render_regression_step(regression_result, show_math: bool) -> None:
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
        
        **This analysis cannot prove causation.** Only a randomized experiment can definitively 
        establish that faster responses *cause* higher conversion rates.
        """)
    else:
        st.warning("""
        **⚠️ Plot twist: The effect disappears when we account for lead source.**
        
        The relationship we saw earlier was likely a mirage. Different lead types have 
        different close rates AND different response times, creating an illusion that 
        speed matters.
        
        **Translation:** Focus on lead quality, not just response speed.
        """)
    
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
    
    # Technical details with regression explainer
    render_regression_explainer()
    
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


def render_mixed_effects_step(result, show_math: bool) -> None:
    """Render the mixed effects step with first-principles explanation."""
    # Bridge sentence to maintain narrative flow
    st.info(f"💡 **Why this step:** {get_step_bridge('mixed_effects')}")
    
    st.markdown("### Step 6: Accounting for Individual Salesperson Differences")
    
    st.markdown("""
    **The Problem We Must Address:**
    
    Until now, our analysis has treated all leads as independent observations. But this 
    ignores a crucial reality: leads are handled by individual salespeople, and these 
    individuals differ from one another in ways that may confound our results.
    
    Consider the possibility: perhaps skilled salespeople both respond quickly *and* 
    close at higher rates — not because speed causes success, but because skill 
    drives both behaviors. If true, the apparent effect of response time would be 
    at least partially illusory.
    
    **The Solution:**
    
    A mixed effects model separates the overall population-level effect of response time 
    from the individual-level variation between salespeople. This allows us to ask: 
    *After accounting for the fact that some reps are better than others, does response 
    time still predict success?*
    """)
    
    if result:
        # Show the key finding
        if hasattr(result, 'is_significant') and result.is_significant:
            st.success("""
            **Conclusion: The effect survives adjustment for representative-level differences.**
            
            Even after accounting for variation between individual salespeople, response time 
            remains a significant predictor of outcomes. This strengthens our confidence that 
            the effect is genuine rather than an artifact of confounding by salesperson ability.
            """)
        else:
            st.warning("""
            **Conclusion: Results are less certain after adjustment.**
            
            When we account for individual salesperson differences, the effect of response 
            time becomes weaker or non-significant. This suggests that some of the apparent 
            effect may be attributable to confounding — better salespeople may simply both 
            respond faster and close more deals.
            """)
        
        if result.statistics:
            stats = result.statistics
            icc = stats.get('icc', 0)
            
            st.markdown("---")
            st.markdown("#### Key Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("ICC (Intraclass Correlation)", f"{icc*100:.1f}%")
                st.caption("Fraction of variation attributable to rep-level differences")
            
            with col2:
                st.metric("Representatives Analyzed", stats.get('n_reps', 'N/A'))
                st.caption("Number of unique salespeople in the dataset")
            
            # Interpret the ICC
            if icc > 0.2:
                icc_interpretation = "substantial — salesperson identity explains a meaningful portion of outcome variation"
            elif icc > 0.1:
                icc_interpretation = "moderate — salesperson effects are present but not dominant"
            else:
                icc_interpretation = "modest — most variation exists within salespeople rather than between them"
            
            st.markdown(f"""
            **Interpreting the ICC:**
            
            An ICC of {icc*100:.1f}% means that {icc*100:.1f}% of the variation in outcomes is 
            attributable to differences between salespeople. This is **{icc_interpretation}**.
            
            {'A high ICC reinforces the importance of controlling for salesperson effects — without this adjustment, our estimates could be biased.' if icc > 0.15 else 'A lower ICC suggests that salesperson-level confounding may be less of a concern for this dataset.'}
            """)
    else:
        st.info("Mixed effects analysis requires salesperson identification data. Ensure the dataset includes a representative or salesperson identifier.")
    
    # Add the detailed explainer
    render_mixed_effects_explainer()


def render_within_rep_step(result, df: pd.DataFrame, show_math: bool) -> None:
    """Render the within-rep analysis step with first-principles explanation."""
    # Bridge sentence to maintain narrative flow
    st.info(f"💡 **Why this step:** {get_step_bridge('within_rep')}")
    
    st.markdown("### Step 7: The Within-Person Test")
    
    st.markdown("""
    **The Strongest Form of Observational Evidence:**
    
    We now apply the most powerful analytical technique available for observational data: 
    the within-person comparison. This approach addresses a limitation that all prior 
    analyses share — the possibility that unmeasured differences between people explain 
    our results.
    
    **The Logic:**
    
    Instead of comparing different salespeople to each other, we compare each salesperson 
    to *themselves*. We ask: when this specific individual responds quickly, do they 
    close at a higher rate than when this same individual responds slowly?
    
    This controls for *everything* about the person that doesn't change between leads:
    - Their skill level
    - Their experience and training
    - Their territory and client base
    - Their personal style and approach
    - Countless unmeasured factors
    
    The only variable that changes is response time. If outcomes still vary with response 
    time under these conditions, the evidence for a causal effect becomes substantially stronger.
    """)
    
    if result:
        # Show the key finding
        if hasattr(result, 'statistics') and result.statistics.get('p_value', 1) < 0.05:
            st.success("""
            **Conclusion: The within-person effect is statistically significant.**
            
            When individual salespeople respond quickly, they close at higher rates than when 
            those same individuals respond slowly. This is compelling evidence that response 
            speed genuinely influences outcomes — the effect cannot be explained away by 
            differences in salesperson ability.
            """)
        else:
            st.warning("""
            **Conclusion: The within-person effect is not statistically significant.**
            
            When we examine individual salespeople, the relationship between response time 
            and outcomes weakens. This suggests that between-person differences — rather 
            than response speed itself — may explain some of the observed correlation.
            """)
        
        if hasattr(result, 'interpretation'):
            st.markdown(f"**Detailed Finding:** {result.interpretation}")
    
    # Show rep scatter if we have data
    if 'sales_rep' in df.columns:
        st.markdown("---")
        st.markdown("#### Representative-Level Performance Distribution")
        
        st.markdown("""
        The chart below visualizes each salesperson's average response time against their 
        close rate. Each point represents one individual.
        
        **What to look for:**
        - A downward slope suggests that faster-responding reps close at higher rates
        - The strength of this relationship indicates potential confounding
        - Outliers may warrant individual investigation
        """)
        
        rep_perf = calculate_rep_performance(df)
        if not rep_perf.empty:
            fig = create_rep_scatter(rep_perf)
            st.plotly_chart(fig, use_container_width=True)
    
    # Add the detailed explainer
    render_within_rep_explainer()


def render_confounding_step(result, df: pd.DataFrame) -> None:
    """Render the confounding assessment step with first-principles explanation."""
    # Bridge sentence to maintain narrative flow
    st.info(f"💡 **Why this step:** {get_step_bridge('confounding')}")
    
    st.markdown("### Step 8: Comprehensive Confounding Assessment")
    
    st.markdown("""
    **The Central Challenge of Causal Inference:**
    
    Throughout this analysis, we have been building toward a causal claim: that responding 
    faster to leads *causes* higher conversion rates. But observational data — no matter 
    how carefully analyzed — cannot definitively prove causation. We can only strengthen 
    or weaken our confidence in a causal interpretation.
    
    This final step synthesizes our findings and assesses the overall threat of confounding.
    
    **The Question:**
    
    How confident can we be that the observed relationship between response time and 
    outcomes reflects a genuine causal mechanism, rather than spurious correlation 
    driven by unmeasured confounders?
    """)
    
    if result:
        st.markdown("---")
        st.markdown("#### Assessment Summary")
        
        if hasattr(result, 'interpretation'):
            st.markdown(result.interpretation)
        
        if hasattr(result, 'confounding_level'):
            level = result.confounding_level
            if level == 'low':
                st.success("""
                **Overall Assessment: Low Confounding Risk**
                
                The effect of response time persists across multiple analytical approaches — 
                simple comparison, regression adjustment, and within-person analysis. This 
                consistency across methods strengthens our confidence in a causal interpretation.
                
                While we cannot eliminate all possibility of confounding (only a randomized 
                experiment could do that), the evidence supports treating response time as 
                a genuine causal factor in conversion outcomes.
                """)
            elif level == 'moderate':
                st.warning("""
                **Overall Assessment: Moderate Confounding Risk**
                
                The results are mixed across analytical approaches. The effect of response 
                time weakens when we apply more stringent controls, suggesting that some 
                portion of the apparent effect may be attributable to confounding.
                
                We recommend cautious interpretation. Consider running a controlled experiment 
                (A/B test) to establish causation definitively before making major investments 
                in response time infrastructure.
                """)
            else:
                st.error("""
                **Overall Assessment: High Confounding Risk**
                
                The effect of response time diminishes substantially or disappears when we 
                control for potential confounders. This suggests that the observed correlation 
                may be largely or entirely spurious.
                
                We recommend against making causal claims based on this data. The relationship 
                appears to be driven by confounding variables rather than a genuine causal 
                mechanism.
                """)
    else:
        st.info("""
        **Synthesizing the Evidence:**
        
        To assess confounding, we examine whether the response time effect remains stable 
        across different analytical approaches:
        
        1. **Simple comparison**: What is the raw difference in outcomes?
        2. **Regression adjustment**: Does the effect persist after controlling for lead source?
        3. **Mixed effects**: Does the effect persist after accounting for salesperson differences?
        4. **Within-person**: Does the effect persist when comparing each person to themselves?
        
        Consistency across these approaches strengthens causal interpretation. 
        Instability raises concern about confounding.
        """)
    
    # Add the detailed explainer
    render_confounding_explainer()


def render_weekly_trends_step(weekly_analysis, close_rates: pd.DataFrame = None) -> None:
    """
    Render the week-over-week trends analysis step.
    
    WHY THIS MATTERS:
    -----------------
    Shows how metrics are changing over time.
    Helps identify if response times are improving or declining.
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


def render_recommendations(stat_results, regression_result, advanced_results) -> None:
    """Render actionable recommendations based on the analysis."""
    st.subheader("💡 Recommendations")
    
    chi_sq = stat_results['chi_square']
    
    if chi_sq.is_significant and regression_result.is_response_time_significant:
        st.markdown("""
        ### Action: Validate with Experiment Before Major Investment
        
        Based on this analysis, faster response times are associated with 
        higher close rates, even after controlling for lead source. However, 
        this is observational data and cannot prove causation.
        
        **Recommended actions (in priority order):**
        1. **Run a randomized A/B test** to establish causation before major investments:
           - Randomly assign leads to fast vs. slow response conditions
           - Ensure assignment is independent of lead characteristics
           - Measure conversion rates in each condition
           - This is the only way to definitively prove that speed causes success
        
        2. If experimental validation confirms the effect:
           - Set response time SLAs (e.g., respond within 15 minutes)
           - Implement alerts for leads waiting too long
           - Consider after-hours coverage for night/weekend leads
           - Monitor response time as a key performance metric
        
        3. If experimental validation does NOT confirm the effect:
           - The observational association was likely due to confounding
           - Focus optimization efforts on factors that genuinely drive conversion
        
        **Why experimental validation matters:** The observed association could be due to 
        selection mechanisms (salespeople prioritizing high-quality leads), unmeasured 
        confounders, or reverse causation. Only a randomized experiment can rule these out.
        
        **Estimated impact (if causation is confirmed):** Based on the observational data, 
        improving response time from the slowest bucket to the fastest could potentially 
        increase close rates by the observed difference. However, this estimate assumes 
        causation, which requires experimental validation.
        """)
    elif chi_sq.is_significant:
        st.markdown("""
        ### Action: Investigate Further Before Acting
        
        The data shows a relationship between response time and close rate,
        but it may be partially explained by lead source or other factors.
        
        **Recommended actions:**
        1. Run an A/B test to establish causation
        2. Analyze response time by lead source separately
        3. Investigate if certain lead types warrant faster response
        4. Consider the cost-benefit of faster response times
        """)
    else:
        st.markdown("""
        ### Action: Focus on Other Factors
        
        Response time does not appear to significantly affect close rates 
        in this data. Other factors may be more important.
        
        **Recommended actions:**
        1. Analyze other factors (lead quality, rep skill, timing)
        2. Collect more data if sample size is limited
        3. Consider if the response time range is too narrow to show an effect
        4. Focus optimization efforts elsewhere
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
could be explained by unmeasured confounders or selection mechanisms. Before 
making major investments, we recommend running a randomized controlled experiment 
to establish causation definitively.

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

This observational analysis cannot prove causation. Only a randomized controlled 
experiment can definitively establish that faster responses cause higher conversion rates.

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
   random. It does NOT mean the relationship is causal. Only a randomized 
   experiment can establish causation.

4. RECOMMENDED NEXT STEP: RANDOMIZED EXPERIMENT
   Before making major staffing or infrastructure investments, run a randomized 
   controlled experiment (A/B test) where leads are randomly assigned to response 
   time conditions, independent of lead characteristics. This is the only way to 
   definitively prove that speed causes conversion.

5. RESULTS ARE SPECIFIC TO YOUR DATA
   What works in this time period and market might not apply elsewhere.

================================================================================
                         END OF ANALYSIS SUMMARY
================================================================================
"""
    
    return summary

