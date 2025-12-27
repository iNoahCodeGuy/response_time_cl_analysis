# =============================================================================
# Common Explainers
# =============================================================================
# Shared utility functions for explanations (step bridges, sample size, etc.)
# =============================================================================

import streamlit as st
import pandas as pd
from typing import Dict, Any


# Bridge sentences to explain why each step follows from the last
STEP_BRIDGES = {
    'data_overview': (
        "Let's start by understanding what we're working with. Before drawing any conclusions, "
        "we need to know the shape and quality of our data."
    ),
    'close_rates': (
        "Now let's see if faster responses are associated with higher close rates. "
        "This is the core question we're here to answer."
    ),
    'chi_square': (
        "We saw a pattern in the data. But patterns can appear purely by random chance — "
        "like flipping heads 6 times in a row. Let's test if this pattern is statistically real..."
    ),
    'proportions': (
        "We've established that *something* is going on. Now let's find exactly *where* "
        "the biggest gaps are — which response time thresholds matter most?"
    ),
    'regression': (
        "Wait — what if this pattern is just because better leads happen to get faster responses? "
        "Let's control for lead source and see if the effect holds..."
    ),
    'weekly_trends': (
        "Statistical tests tell us about the overall pattern. But does performance hold steady "
        "week-over-week, or are there concerning trends?"
    ),
    'mixed_effects': (
        "Different salespeople have different skill levels. Let's account for this and see "
        "if the response time effect holds within individuals..."
    ),
    'within_rep': (
        "The most rigorous test: when the *same* salesperson responds fast vs. slow, "
        "do they close more deals? This controls for everything about the person."
    ),
    'confounding': (
        "Before we finalize our conclusions, let's systematically assess how confident we can be "
        "that response time is truly driving outcomes, not hidden confounders."
    )
}


def get_step_bridge(step_name: str) -> str:
    """
    Get the bridge sentence for a given analysis step.
    
    Bridge sentences explain WHY we're doing each step and how it connects
    to the previous step. This maintains narrative flow for non-technical users.
    """
    return STEP_BRIDGES.get(step_name, "")


def render_sample_size_guidance(n: int, metric: str = "close rate") -> None:
    """
    Explain the relationship between sample size and estimate reliability.
    """
    if n >= 1000:
        st.success(f"""
        **Data Quality Assessment: Excellent**
        
        With {n:,} observations, we have substantial statistical power. The {metric} estimates 
        presented here are precise, and the confidence intervals are narrow. 
        Decisions based on this data rest on solid empirical ground.
        """)
    elif n >= 500:
        st.info(f"""
        **Data Quality Assessment: Good**
        
        With {n:,} observations, our estimates of {metric} are reasonably precise. 
        The confidence intervals provide an honest representation of remaining uncertainty. 
        This is sufficient data for most decision-making purposes.
        """)
    elif n >= 100:
        st.warning(f"""
        **Data Quality Assessment: Moderate**
        
        With {n:,} observations, meaningful uncertainty remains. The confidence intervals 
        are wider than ideal, meaning the true {metric} could deviate meaningfully from 
        our point estimates. Consider these results directionally indicative rather than precise.
        """)
    else:
        st.error(f"""
        **Data Quality Assessment: Limited**
        
        With only {n:,} observations, the estimates carry substantial uncertainty. 
        Statistical fluctuations can easily overwhelm true effects at this sample size. 
        These results should be treated as preliminary — additional data collection is 
        strongly recommended before acting on these findings.
        """)


def render_bucket_sample_sizes(close_rates_df: pd.DataFrame) -> None:
    """
    Display sample sizes by bucket with assessment of statistical reliability.
    """
    st.markdown("#### Sample Size Assessment by Category")
    
    st.markdown("""
    The reliability of any estimate is fundamentally tied to the amount of data underlying it. 
    Below, we assess the data density in each response time bucket.
    """)
    
    rows = []
    for _, row in close_rates_df.iterrows():
        n = row['n_leads']
        if n >= 1000:
            reliability = "✅ Excellent"
            assessment = "High-confidence estimates; narrow uncertainty range"
        elif n >= 500:
            reliability = "✓ Good"
            assessment = "Reliable estimates; acceptable uncertainty"
        elif n >= 200:
            reliability = "⚠️ Moderate"
            assessment = "Meaningful uncertainty; interpret with caution"
        else:
            reliability = "❌ Limited"
            assessment = "High uncertainty; additional data recommended"
        
        rows.append({
            'Response Time': row['bucket'],
            'Observations': f"{n:,}",
            'Conversions': f"{row['n_orders']:,}",
            'Reliability': reliability,
            'Assessment': assessment
        })
    
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_key_finding(
    fast_rate: float,
    slow_rate: float,
    fast_bucket: str,
    slow_bucket: str,
    is_significant: bool,
    multiplier: float,
    p_value: float = None
) -> None:
    """
    Render the primary conclusion using methodical, first-principles logic.
    
    Handles multiple scenarios:
    - Exceptional effects (very large differences)
    - Reverse effects (slower performs better)
    - Modest but significant effects
    - Non-significant results
    """
    diff_pp = (fast_rate - slow_rate) * 100
    abs_diff_pp = abs(diff_pp)
    is_reverse = diff_pp < 0  # Slower is actually better
    is_exceptional = abs_diff_pp > 10 or (p_value is not None and p_value < 0.0001 and abs_diff_pp > 5)
    
    # Handle reverse effect (slower is better)
    if is_significant and is_reverse and abs_diff_pp > 2:
        st.warning(f"""
        ## The Central Finding: Counterintuitive Result
        
        **Slower responses are associated with higher conversion rates.**
        
        This is a surprising finding that contradicts conventional wisdom. Leads receiving 
        slower responses ({slow_bucket}) convert at {slow_rate*100:.1f}%, compared to 
        {fast_rate*100:.1f}% for faster responses ({fast_bucket}). This {abs_diff_pp:.1f} 
        percentage point difference is statistically significant.
        
        | Metric | Slow Response ({slow_bucket}) | Fast Response ({fast_bucket}) | Difference |
        |:-------|:------------------------------|:------------------------------|:-----------|
        | **Conversion Rate** | {slow_rate*100:.1f}% | {fast_rate*100:.1f}% | +{abs_diff_pp:.1f} pp |
        | **Per 100 Leads** | {int(slow_rate*100)} conversions | {int(fast_rate*100)} conversions | +{int(abs_diff_pp)} |
        | **Per 1,000 Leads** | {int(slow_rate*1000)} conversions | {int(fast_rate*1000)} conversions | +{int(abs_diff_pp*10)} |
        
        **Possible Explanations:**
        - **Selection bias:** Salespeople may prioritize high-quality leads with slower but more thoughtful responses
        - **Time-of-day effects:** Slower responses might correlate with business hours when decision-makers are available
        - **Quality over speed:** More thorough initial responses may lead to better outcomes despite the delay
        - **Data artifact:** Unmeasured confounders may be driving this pattern
        
        **Recommendation:** This result requires careful investigation before acting. This 
        observational analysis cannot definitively determine whether this is a genuine causal relationship or spurious correlation.
        """)
    # Handle exceptional positive effect (very large difference)
    elif is_significant and is_exceptional and not is_reverse:
        st.success(f"""
        ## The Central Finding: Exceptional Effect Detected
        
        **Response speed demonstrates an exceptionally strong association with conversion outcomes.**
        
        The evidence shows a dramatic relationship: leads that receive faster responses 
        convert at substantially higher rates. This difference is both statistically 
        significant and practically substantial.
        
        | Metric | Fast Response ({fast_bucket}) | Slow Response ({slow_bucket}) | Difference |
        |:-------|:------------------------------|:------------------------------|:-----------|
        | **Conversion Rate** | {fast_rate*100:.1f}% | {slow_rate*100:.1f}% | +{diff_pp:.1f} pp |
        | **Per 100 Leads** | {int(fast_rate*100)} conversions | {int(slow_rate*100)} conversions | +{int(diff_pp)} |
        | **Per 1,000 Leads** | {int(fast_rate*1000)} conversions | {int(slow_rate*1000)} conversions | +{int(diff_pp*10)} |
        
        **The Magnitude:** Fast responders convert at {multiplier:.1f}× the rate of slow responders.
        This is a **large effect** that would have meaningful business impact if causally established.
        
        {"**Statistical Certainty:** The probability this is due to chance is extremely low (< 0.01%), indicating exceptional confidence in this finding." if p_value is not None and p_value < 0.0001 else ""}
        
        **The Implication:** This is among the strongest observed effects for response time. 
        Every hour of delay represents substantial lost conversion potential. However, remember 
        that statistical significance does not prove causation — this observational analysis has 
        limitations in establishing causation.
        """)
    # Handle standard positive significant effect
    elif is_significant and diff_pp > 2:
        st.success(f"""
        ## The Central Finding
        
        **Response speed is associated with conversion outcomes.**
        
        The evidence supports a clear conclusion: leads that receive faster responses 
        convert at meaningfully higher rates. This pattern persists after controlling 
        for confounding variables and testing against chance.
        
        | Metric | Fast Response ({fast_bucket}) | Slow Response ({slow_bucket}) | Difference |
        |:-------|:------------------------------|:------------------------------|:-----------|
        | **Conversion Rate** | {fast_rate*100:.1f}% | {slow_rate*100:.1f}% | +{diff_pp:.1f} pp |
        | **Per 100 Leads** | {int(fast_rate*100)} conversions | {int(slow_rate*100)} conversions | +{int(diff_pp)} |
        | **Per 1,000 Leads** | {int(fast_rate*1000)} conversions | {int(slow_rate*1000)} conversions | +{int(diff_pp*10)} |
        
        **The Magnitude:** Fast responders convert at {multiplier:.1f}× the rate of slow responders.
        
        **The Implication:** Every hour of delay in response time represents lost conversion potential. 
        The data suggests that investments in response speed infrastructure would yield measurable returns 
        if this association is causal. However, this observational analysis cannot definitively establish causation.
        """)
    # Handle small but significant effect
    elif is_significant and diff_pp > 0.5:
        st.info(f"""
        ## The Central Finding
        
        **A modest but statistically reliable effect exists.**
        
        Fast responders ({fast_bucket}) convert at {fast_rate*100:.1f}%, compared to 
        {slow_rate*100:.1f}% for slow responders ({slow_bucket}). The difference of 
        {diff_pp:.1f} percentage points is statistically significant, indicating this 
        pattern is unlikely to be random.
        
        **Interpretation:** The effect is real, but its practical magnitude requires careful 
        consideration in the context of implementation costs. Small effects can compound 
        meaningfully at scale, but may not justify significant operational changes without 
        further investigation.
        """)
    # Handle tiny but significant effect (statistically significant but tiny)
    elif is_significant and diff_pp <= 0.5:
        st.info(f"""
        ## The Central Finding
        
        **A statistically significant but very small effect detected.**
        
        Fast responders ({fast_bucket}) convert at {fast_rate*100:.1f}%, compared to 
        {slow_rate*100:.1f}% for slow responders ({slow_bucket}). The difference of 
        {diff_pp:.2f} percentage points is statistically significant but practically minimal.
        
        **Interpretation:** While this difference is statistically detectable, it may not 
        be practically meaningful. The cost of improving response times may outweigh the 
        marginal benefit. Consider focusing optimization efforts on factors with larger impacts.
        """)
    # Handle non-significant result
    else:
        st.warning(f"""
        ## The Central Finding
        
        **No conclusive effect detected.**
        
        While the data shows an apparent difference of {diff_pp:+.1f} percentage points 
        between fast and slow responders, this difference is not statistically significant.
        
        | Fast Response ({fast_bucket}) | Slow Response ({slow_bucket}) | Difference |
        |:------------------------------|:------------------------------|:-----------|
        | {fast_rate*100:.1f}% | {slow_rate*100:.1f}% | {diff_pp:+.1f} pp |
        
        **Interpretation:** We cannot rule out that this observed difference arose from 
        random sampling variation. Either no true effect exists, or the effect is too small 
        to detect with the current sample size. Additional data collection may resolve this uncertainty.
        
        **Recommendation:** Do not invest in response time improvements based solely on this analysis. 
        Consider focusing optimization efforts on factors with clearer evidence of impact.
        """)


def detect_narrative_scenario(
    close_rates: pd.DataFrame,
    chi_sq_result: Any,
    regression_result: Any,
    p_value: float = None
) -> Dict[str, Any]:
    """
    Detect the narrative scenario type to adapt storytelling.
    
    This helps determine how to frame the story based on what the data actually shows.
    
    Returns a dictionary with scenario classification and metadata.
    """
    if len(close_rates) < 2:
        return {'scenario': 'insufficient_data', 'confidence': 'low'}
    
    fastest = close_rates.iloc[0]
    slowest = close_rates.iloc[-1]
    fast_rate = fastest['close_rate']
    slow_rate = slowest['close_rate']
    rate_diff = fast_rate - slow_rate
    abs_rate_diff = abs(rate_diff)
    
    # Detect patterns
    is_reverse = rate_diff < 0
    chi_sig = chi_sq_result.is_significant if hasattr(chi_sq_result, 'is_significant') else False
    reg_sig = regression_result.is_response_time_significant if hasattr(regression_result, 'is_response_time_significant') else False
    
    # Check for exceptional effects
    p_val = p_value if p_value is not None else (chi_sq_result.p_value if hasattr(chi_sq_result, 'p_value') else 1.0)
    is_exceptional = (p_val < 0.0001 and abs_rate_diff > 5) or abs_rate_diff > 10
    
    # Check for non-monotonic patterns
    pattern_info = detect_non_monotonic_pattern(close_rates)
    is_non_monotonic = pattern_info.get('is_non_monotonic', False)
    
    # Determine scenario
    if is_non_monotonic:
        scenario = 'non_monotonic'
        story_type = 'complex_relationship'
        tone = 'info'
        priority = 'investigate_mechanism'
    elif is_reverse and chi_sig:
        scenario = 'reverse_effect'
        story_type = 'surprising_finding'
        tone = 'warning'
        priority = 'investigate_confounding'
    elif chi_sig and reg_sig and is_exceptional:
        scenario = 'exceptional_positive'
        story_type = 'strong_signal'
        tone = 'success'
        priority = 'urgent_validation'
    elif chi_sig and reg_sig:
        scenario = 'standard_positive'
        story_type = 'clear_pattern'
        tone = 'success'
        priority = 'validate_with_experiment'
    elif chi_sig and not reg_sig:
        scenario = 'confounding_suspected'
        story_type = 'uncertain_causation'
        tone = 'warning'
        priority = 'control_for_confounders'
    elif not chi_sig:
        scenario = 'no_effect'
        story_type = 'null_result'
        tone = 'info'
        priority = 'focus_elsewhere'
    else:
        scenario = 'unclear'
        story_type = 'ambiguous'
        tone = 'info'
        priority = 'investigate_further'
    
    return {
        'scenario': scenario,
        'story_type': story_type,
        'tone': tone,
        'priority': priority,
        'is_reverse': is_reverse,
        'is_exceptional': is_exceptional,
        'is_non_monotonic': is_non_monotonic,
        'rate_diff': rate_diff,
        'abs_rate_diff': abs_rate_diff,
        'fast_rate': fast_rate,
        'slow_rate': slow_rate,
        'chi_sig': chi_sig,
        'reg_sig': reg_sig,
        'pattern_info': pattern_info if is_non_monotonic else None
    }


def format_wow_change(current: float, previous: float) -> Dict[str, Any]:
    """
    Format a week-over-week change with logical interpretation.
    """
    diff = current - previous
    diff_pp = diff * 100
    
    if abs(diff_pp) < 0.1:
        return {
            'change_pp': diff_pp,
            'direction': 'stable',
            'emoji': '➡️',
            'description': 'Effectively unchanged from prior period',
            'interpretation': 'No meaningful movement detected'
        }
    elif diff_pp > 0:
        magnitude = 'substantially' if abs(diff_pp) > 2 else 'modestly'
        return {
            'change_pp': diff_pp,
            'direction': 'increased',
            'emoji': '📈',
            'description': f'Increased by {diff_pp:.1f} percentage points',
            'interpretation': f'Performance has {magnitude} improved'
        }
    else:
        magnitude = 'substantially' if abs(diff_pp) > 2 else 'modestly'
        return {
            'change_pp': diff_pp,
            'direction': 'decreased', 
            'emoji': '📉',
            'description': f'Decreased by {abs(diff_pp):.1f} percentage points',
            'interpretation': f'Performance has {magnitude} declined'
        }


def render_error_template(error_type: str, error_details: str = "", context: str = "") -> None:
    """
    Render appropriate error message templates for various failure scenarios.
    
    Parameters:
    -----------
    error_type : str
        Type of error: 'insufficient_data', 'test_failure', 'computation_error', 
                      'zero_variance', 'missing_results', 'perfect_separation'
    error_details : str
        Additional details about the error
    context : str
        Context where the error occurred (e.g., "chi-square test", "regression analysis")
    """
    if error_type == 'insufficient_data':
        st.error(f"""
        ## ⚠️ Insufficient Data for Analysis
        
        **The Problem:** {context} cannot be performed because there is insufficient data.
        
        {error_details if error_details else "One or more groups have too few observations to reliably perform statistical tests."}
        
        **Why This Matters:**
        - Statistical tests require minimum sample sizes to produce reliable results
        - Small samples lead to unstable estimates and unreliable conclusions
        - Tests may fail or produce misleading results with insufficient data
        
        **What You Can Do:**
        1. **Collect more data** — Increase sample size in the affected groups
        2. **Combine categories** — Merge sparse categories to increase counts
        3. **Relax time periods** — Analyze over longer time windows to accumulate more data
        4. **Focus on available data** — Analyze only groups with sufficient observations
        
        **Minimum Requirements:**
        - Chi-square test: Each cell should have ≥ 5 expected observations
        - Z-test: Each group needs ≥ 30 observations for reliable results
        - Regression: Generally requires ≥ 10 observations per predictor variable
        """)
    
    elif error_type == 'test_failure':
        st.error(f"""
        ## ❌ Statistical Test Failed
        
        **The Problem:** {context} encountered an error and could not complete.
        
        {error_details if error_details else "The test may have failed due to data issues, computational problems, or statistical assumptions not being met."}
        
        **Common Causes:**
        - **Perfect separation:** All observations in a group have the same outcome (no variance)
        - **Extreme data:** Values that cause numerical instability
        - **Missing data:** Critical variables have too many missing values
        - **Assumption violations:** Data doesn't meet test requirements (e.g., expected frequencies too low)
        
        **What You Can Do:**
        1. Check data quality and completeness
        2. Examine for extreme outliers or anomalies
        3. Verify that all groups have sufficient variation
        4. Consider alternative statistical approaches if assumptions cannot be met
        
        **Next Steps:** Review the data preprocessing steps and ensure all requirements are met before retrying.
        """)
    
    elif error_type == 'computation_error':
        st.error(f"""
        ## 🔧 Computational Error
        
        **The Problem:** {context} encountered a computational error.
        
        {error_details if error_details else "A mathematical or algorithmic issue prevented the analysis from completing."}
        
        **Possible Causes:**
        - Numerical overflow or underflow in calculations
        - Matrix inversion failures (e.g., singular matrices in regression)
        - Convergence failures in iterative algorithms
        - Memory limitations with large datasets
        
        **What You Can Do:**
        1. Check for extreme values that might cause numerical instability
        2. Verify data types and formats are correct
        3. Try removing or transforming problematic variables
        4. Consider using alternative computational approaches
        
        **Technical Details:** This error typically indicates a numerical or algorithmic issue rather than a data interpretation problem.
        """)
    
    elif error_type == 'zero_variance':
        st.warning(f"""
        ## 🔍 Zero Variance Detected
        
        **The Finding:** {context} shows no variation in the data.
        
        {error_details if error_details else "All observations have the same value, making statistical comparisons impossible."}
        
        **What This Means:**
        - **Perfect uniformity:** Every observation has identical values
        - **No statistical variation:** Cannot distinguish between groups
        - **No relationship to test:** If there's no variation, there's no relationship to analyze
        
        **Common Scenarios:**
        - All leads convert (100% close rate) or none convert (0% close rate)
        - All leads have identical response times
        - All leads come from the same source
        
        **Interpretation:**
        This may indicate:
        1. **Data quality issue:** Check if data is correctly loaded
        2. **Too narrow scope:** Data may be filtered to a single category
        3. **Business reality:** The process may truly have no variation (rare but possible)
        
        **What You Can Do:**
        - Expand the analysis scope (time period, categories, etc.)
        - Check data collection and filtering steps
        - Verify that the variable of interest actually varies in your dataset
        """)
    
    elif error_type == 'perfect_separation':
        st.warning(f"""
        ## ⚠️ Perfect Separation Detected
        
        **The Problem:** {context} shows perfect separation in the data.
        
        {error_details if error_details else "The outcome variable is perfectly predicted by one or more predictors, making statistical models unreliable."}
        
        **What This Means:**
        Perfect separation occurs when a predictor (or combination) completely separates outcomes.
        For example:
        - All fast responses convert, all slow responses don't (or vice versa)
        - A specific lead source always converts, others never do
        
        **Why This Is Problematic:**
        - Regression models cannot converge or produce unreliable estimates
        - Confidence intervals become infinite
        - Statistical tests may fail or be unreliable
        
        **Possible Explanations:**
        1. **Too few observations:** Small sample sizes can create apparent perfect separation
        2. **True pattern:** A genuinely deterministic relationship (rare but possible)
        3. **Data artifact:** Data quality issue or selection bias
        
        **What You Can Do:**
        1. Collect more data to see if the pattern holds
        2. Examine if this is a true deterministic relationship or a data issue
        3. Consider qualitative analysis to understand the mechanism
        4. Use alternative analytical approaches that don't require maximum likelihood estimation
        """)
    
    elif error_type == 'missing_results':
        st.warning(f"""
        ## ⚠️ Missing Analysis Results
        
        **The Problem:** Results from {context} are not available.
        
        {error_details if error_details else "The analysis step did not produce the expected output."}
        
        **Possible Causes:**
        - Analysis step was skipped or not completed
        - Results were not properly saved or passed between functions
        - Error occurred during analysis that prevented result generation
        
        **What You Can Do:**
        1. Check if all prerequisite steps completed successfully
        2. Verify data meets all requirements for this analysis
        3. Review error messages from previous steps
        4. Try re-running the analysis with adjusted parameters
        
        **Note:** Some analyses may be optional or conditional. Check if this analysis is required for your use case.
        """)
    
    else:
        st.error(f"""
        ## ⚠️ Analysis Error
        
        An unexpected error occurred: {error_type}
        
        {error_details if error_details else "Please check the data and try again."}
        
        Context: {context if context else "Unknown"}
        """)


def render_contradiction_explanation(
    chi_square_result: Dict[str, Any],
    regression_result: Dict[str, Any],
    chi_sig: bool,
    reg_sig: bool
) -> None:
    """
    Explain contradictory results between chi-square and regression.
    
    Handles cases where:
    - Chi-square significant but regression not (common - confounding suspected)
    - Chi-square not significant but regression significant (rare - controls reveal pattern)
    """
    if chi_sig and not reg_sig:
        # This is already handled well, but enhance it
        st.warning(f"""
        ## 🔍 Contradictory Results: Chi-Square vs Regression
        
        **The Pattern:** Chi-square test indicates a relationship, but regression (controlling for confounders) suggests the effect may be explained by other factors.
        
        **What This Means:**
        - **Chi-square result:** Response time and conversion rate are associated (p = {chi_square_result.get('p_value', 'N/A'):.4f})
        - **Regression result:** After controlling for lead source, response time is no longer significant (p = {regression_result.get('p_value', 'N/A'):.4f})
        
        **Interpretation:**
        This pattern strongly suggests **confounding**. The apparent relationship between response time and conversion may be explained by:
        
        1. **Lead source differences:** Different lead sources may have:
           - Different response times (e.g., phone leads get faster responses)
           - Different conversion rates (e.g., referral leads convert better)
           - These two differences create a spurious correlation
        
        2. **Selection mechanisms:** Salespeople may:
           - Prioritize high-quality leads for faster responses
           - Respond slower to lower-quality leads
           - The "quality" effect creates the apparent response time effect
        
        3. **Time-based confounders:** Response time correlates with:
           - Time of day (affects both response speed and decision-maker availability)
           - Day of week (business hours vs weekends)
           - These temporal factors affect conversion independently of response speed
        
        **Recommendation:**
        Do not invest in response time improvements based on this analysis. The association appears to be driven by confounding rather than a causal effect. This observational analysis cannot definitively establish causation, and we cannot deliberately delay responses to test the relationship.
        """)
    
    elif not chi_sig and reg_sig:
        # Rare but important case
        st.info(f"""
        ## 🔍 Interesting Pattern: Controls Reveal Hidden Effect
        
        **The Pattern:** Chi-square test shows no overall relationship, but regression (with controls) reveals a significant effect.
        
        **What This Means:**
        - **Chi-square result:** No significant association between response time and conversion (p = {chi_square_result.get('p_value', 'N/A'):.4f})
        - **Regression result:** After controlling for lead source, response time becomes significant (p = {regression_result.get('p_value', 'N/A'):.4f})
        
        **Interpretation:**
        This pattern indicates that **controlling for confounders reveals a hidden relationship**. Here's what's happening:
        
        1. **Masking effect:** Lead source (or other confounders) creates noise that obscures the response time effect when uncontrolled
        
        2. **Opposing forces:** The confounder has an effect that works against the response time effect, canceling it out in the raw comparison
        
        3. **Stratified relationship:** The response time effect may exist only within certain lead source categories
        
        **Why This Is Rare:**
        More commonly, adding controls reduces significance (as confounders explain the effect). When controls *increase* significance, it suggests:
        - The confounder and response time have opposite effects
        - Within-stratum effects are stronger than overall effects
        - There may be interaction effects worth investigating
        
        **Recommendation:**
        This result requires careful interpretation. The regression suggests an effect, but the contradictory chi-square result indicates this effect may be conditional or context-dependent. Consider:
        1. Analyzing response time effects separately within each lead source
        2. Testing for interaction effects between response time and lead source
        3. Acknowledging the limitations of observational data in establishing causation
        """)
    
    else:
        # Both significant or both not significant - no contradiction
        pass


def detect_non_monotonic_pattern(close_rates_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect non-monotonic patterns in response time buckets.
    
    Returns information about:
    - U-shaped pattern: Middle buckets perform worse than extremes
    - Inverted-U pattern: Middle buckets perform better than extremes
    - Peak/valley positions
    
    Returns empty dict if pattern is monotonic.
    """
    if len(close_rates_df) < 3:
        return {}  # Need at least 3 buckets to detect non-monotonic patterns
    
    rates = close_rates_df['close_rate'].values
    buckets = close_rates_df['bucket'].values
    
    # Find the bucket with highest and lowest rates
    max_idx = int(rates.argmax())
    min_idx = int(rates.argmin())
    
    # Check if peak/valley is in the middle (not at extremes)
    n_buckets = len(rates)
    is_peak_in_middle = 0 < max_idx < n_buckets - 1
    is_valley_in_middle = 0 < min_idx < n_buckets - 1
    
    pattern_type = None
    if is_peak_in_middle:
        # Check if this is an inverted-U (middle is best)
        if rates[0] < rates[max_idx] and rates[-1] < rates[max_idx]:
            pattern_type = "inverted_u"
            peak_bucket = buckets[max_idx]
            peak_rate = rates[max_idx]
            edge_avg = (rates[0] + rates[-1]) / 2
            return {
                'is_non_monotonic': True,
                'pattern_type': pattern_type,
                'description': f"Inverted-U pattern: {peak_bucket} performs best",
                'peak_bucket': peak_bucket,
                'peak_rate': peak_rate,
                'peak_rate_pct': peak_rate * 100,
                'edge_avg_rate': edge_avg,
                'edge_avg_pct': edge_avg * 100,
                'difference_pp': (peak_rate - edge_avg) * 100
            }
    
    if is_valley_in_middle:
        # Check if this is a U-shape (middle is worst)
        if rates[0] > rates[min_idx] and rates[-1] > rates[min_idx]:
            pattern_type = "u_shaped"
            valley_bucket = buckets[min_idx]
            valley_rate = rates[min_idx]
            edge_avg = (rates[0] + rates[-1]) / 2
            return {
                'is_non_monotonic': True,
                'pattern_type': pattern_type,
                'description': f"U-shaped pattern: {valley_bucket} performs worst",
                'valley_bucket': valley_bucket,
                'valley_rate': valley_rate,
                'valley_rate_pct': valley_rate * 100,
                'edge_avg_rate': edge_avg,
                'edge_avg_pct': edge_avg * 100,
                'difference_pp': (edge_avg - valley_rate) * 100
            }
    
    # Check for other non-monotonic patterns (e.g., zigzag)
    # Simple check: count direction changes
    direction_changes = 0
    for i in range(1, len(rates) - 1):
        if (rates[i] > rates[i-1] and rates[i+1] < rates[i]) or \
           (rates[i] < rates[i-1] and rates[i+1] > rates[i]):
            direction_changes += 1
    
    if direction_changes >= 2:
        return {
            'is_non_monotonic': True,
            'pattern_type': 'complex',
            'description': 'Complex non-monotonic pattern detected',
            'direction_changes': direction_changes
        }
    
    return {'is_non_monotonic': False}


def render_non_monotonic_pattern_explanation(pattern_info: Dict[str, Any]) -> None:
    """
    Explain non-monotonic patterns in response time effects.
    
    This is important because it challenges the assumption that faster is always better.
    """
    if not pattern_info.get('is_non_monotonic'):
        return
    
    pattern_type = pattern_info.get('pattern_type')
    
    if pattern_type == 'inverted_u':
        st.info(f"""
        ## 🔍 Non-Monotonic Pattern Detected: Optimal Response Window
        
        **The Finding:** The relationship between response time and conversion is not linear. 
        Instead, there appears to be an **optimal response window** rather than "faster is always better."
        
        **The Pattern:**
        - **Best Performance:** {pattern_info['peak_bucket']} ({pattern_info['peak_rate_pct']:.1f}% close rate)
        - **Extreme Performance (Average):** Fastest and slowest buckets average {pattern_info['edge_avg_pct']:.1f}%
        - **Difference:** {pattern_info['difference_pp']:.1f} percentage points advantage for optimal window
        
        **What This Means:**
        Response speed matters, but there may be an optimal timing window. Responding too quickly 
        might be as suboptimal as responding too slowly. This suggests:
        
        1. **Quality over speed trade-off:** The optimal window may allow time for:
           - More thoughtful, personalized responses
           - Better lead qualification before responding
           - Coordinated responses from the right team member
        
        2. **Customer expectations:** Customers may expect and value responses within a specific 
           timeframe (e.g., 15-60 minutes) rather than immediate responses
        
        3. **Lead characteristics:** Different lead types may have different optimal response windows
        
        **Business Implications:**
        - Don't optimize for maximum speed alone
        - Focus on consistent response times within the optimal window
        - Consider that immediate responses may not always be best
        - Test different response time strategies to find your optimal window
        
        **Next Steps:**
        Investigate why this optimal window exists. Is it related to response quality, 
        customer expectations, lead characteristics, or time-of-day effects? This pattern 
        suggests a more nuanced response time strategy than "faster is always better."
        """)
    
    elif pattern_type == 'u_shaped':
        st.warning(f"""
        ## 🔍 Non-Monotonic Pattern Detected: Suboptimal Middle Ground
        
        **The Finding:** The relationship between response time and conversion shows a U-shaped pattern. 
        Both very fast and very slow responses perform better than medium-speed responses.
        
        **The Pattern:**
        - **Worst Performance:** {pattern_info['valley_bucket']} ({pattern_info['valley_rate_pct']:.1f}% close rate)
        - **Better Performance (Average):** Fastest and slowest buckets average {pattern_info['edge_avg_pct']:.1f}%
        - **Difference:** {pattern_info['difference_pp']:.1f} percentage point disadvantage for middle window
        
        **What This Means:**
        This pattern suggests that medium response times represent a "worst of both worlds" scenario:
        
        1. **Too fast to be thoughtful:** Not enough time for quality responses
        2. **Too slow to capture urgency:** Missing the initial engagement window
        3. **Selection bias:** Medium response times might indicate:
           - Leads that are lower priority (neither urgent nor high-value)
           - Salespeople who are unsure how to prioritize
           - Leads that fall through organizational cracks
        
        **Business Implications:**
        - Avoid medium-speed responses if possible
        - Either respond very quickly (capture urgency) or take time for quality (deep engagement)
        - Investigate why leads receive medium-speed responses
        - Consider whether medium-speed bucket represents process inefficiencies
        
        **Next Steps:**
        Examine what characterizes the medium-speed bucket. Is this a prioritization issue? 
        A resource allocation problem? Understanding why this bucket underperforms can reveal 
        operational improvements beyond just response time.
        """)
    
    elif pattern_type == 'complex':
        st.warning(f"""
        ## 🔍 Complex Non-Monotonic Pattern Detected
        
        **The Finding:** The relationship between response time and conversion shows multiple 
        direction changes, indicating a complex, non-linear pattern.
        
        **What This Means:**
        The effect of response time is not simple or monotonic. Different response time windows 
        appear to have different effects, suggesting:
        
        - **Multiple factors at play:** Response time effects may vary by:
          - Lead type or source
          - Time of day
          - Day of week
          - Salesperson characteristics
          - Lead quality indicators
        
        - **Threshold effects:** There may be specific time thresholds where effects change dramatically
        
        - **Interaction effects:** Response time may interact with other variables in complex ways
        
        **Business Implications:**
        A one-size-fits-all response time strategy is unlikely to be optimal. Consider:
        
        1. **Segmented strategies:** Different response time targets for different lead types
        2. **Time-of-day considerations:** Optimal response times may vary by when leads come in
        3. **Quality vs. speed trade-offs:** The optimal balance may differ across contexts
        
        **Next Steps:**
        Analyze response time effects within subgroups (by lead source, time of day, etc.) to 
        identify which factors drive this complex pattern. Regression analysis with interaction 
        terms may help uncover the underlying relationships.
        """)


def render_minimal_sample_warning(n_per_group: int, group_name: str = "group") -> None:
    """
    Warn when sample sizes are too small for reliable statistical interpretation.
    
    This is more aggressive than the general sample size guidance - specifically 
    for cases where we have so few observations that interpretation is unreliable.
    """
    if n_per_group >= 10:
        return  # No warning needed
    
    if n_per_group < 5:
        severity = "error"
        icon = "❌"
        urgency = "critical"
    else:
        severity = "warning"
        icon = "⚠️"
        urgency = "severe"
    
    message = f"""
    ## {icon} Minimal Sample Size Warning
    
    **The Problem:** This analysis includes {group_name} with only **{n_per_group} observations**. 
    This is insufficient for reliable statistical interpretation.
    
    **Why This Matters:**
    With fewer than 10 observations per group:
    - **Statistical tests are unreliable:** P-values may be meaningless
    - **Confidence intervals are extremely wide:** Estimates carry massive uncertainty
    - **Outliers have disproportionate influence:** A single observation can skew results dramatically
    - **Patterns may be spurious:** Random variation can easily create false patterns
    
    **What You Should Know:**
    
    1. **These results are preliminary at best:**
       - Any patterns observed could be due to random chance
       - Point estimates are highly unstable
       - Confidence intervals will be extremely wide
    
    2. **Statistical tests may fail or be misleading:**
       - Chi-square test requires ≥5 expected observations per cell
       - Z-tests require ≥30 observations for reliable results
       - Regression requires ≥10 observations per predictor variable
    
    3. **You cannot confidently act on these findings:**
       - Do not make operational changes based on this analysis
       - Do not draw strong conclusions about relationships
       - Treat these results as exploratory only
    
    **What You Can Do:**
    
    1. **Collect more data** — This is the only reliable solution
       - Aim for at least 30 observations per group for basic comparisons
       - Aim for at least 100 observations per group for reliable regression analysis
    
    2. **Combine groups** — If you have multiple small groups, merge them temporarily
       - Create broader categories to increase sample sizes
       - Trade-off granularity for statistical reliability
    
    3. **Expand time window** — Analyze over a longer period
       - Combine multiple weeks or months of data
       - Accept that you're measuring longer-term patterns
    
    4. **Focus on groups with adequate data** — Analyze only buckets with sufficient observations
       - Temporarily exclude very small groups
       - Acknowledge the limitation in your conclusions
    
    **Bottom Line:** 
    With only {n_per_group} observations, these results should be treated as **preliminary and unreliable**. 
    Do not make business decisions based on this analysis. Collect more data before drawing conclusions.
    """
    
    if severity == "error":
        st.error(message)
    else:
        st.warning(message)


def render_ci_significance_contradiction(
    point_estimate: float,
    ci_lower: float,
    ci_upper: float,
    p_value: float,
    test_name: str = "statistical test",
    null_value: float = 0.0
) -> None:
    """
    Explain contradictions between confidence intervals and significance tests.
    
    Handles cases where:
    - CI includes null but test is significant (rare, suggests issue)
    - CI excludes null but test is not significant (rare, suggests issue)
    - Wide CI suggests uncertainty despite significance
    """
    ci_includes_null = ci_lower <= null_value <= ci_upper
    test_significant = p_value < 0.05
    
    # Check if CI is extremely wide relative to point estimate
    ci_width = ci_upper - ci_lower
    is_very_wide = ci_width > abs(point_estimate) * 2  # CI is >2× the estimate
    
    if ci_includes_null and test_significant:
        # This is unusual - should warn
        st.warning(f"""
        ## ⚠️ Contradiction: Confidence Interval vs. Significance Test
        
        **The Inconsistency:** 
        - The {test_name} is statistically significant (p = {p_value:.4f} < 0.05)
        - BUT the confidence interval ({ci_lower:.2f} to {ci_upper:.2f}) includes {null_value}, 
          suggesting the true effect might be {null_value}
        
        **What This Means:**
        This contradiction is unusual and suggests one of the following:
        
        1. **Statistical issue:** There may be a problem with the test assumptions or calculations
        2. **Boundary case:** The effect is right at the edge of significance (p-value just below 0.05)
        3. **Small sample size:** With small samples, CIs and tests can behave inconsistently
        4. **Multiple testing:** If many tests were run, some may be false positives
        
        **Interpretation:**
        Treat this result with caution. While the test suggests an effect exists, the confidence 
        interval suggests substantial uncertainty. The true effect might be very small or even zero.
        
        **Recommendation:**
        - Do not strongly act on this finding
        - Collect more data to clarify the contradiction
        - Check test assumptions and verify calculations
        - Consider that this might be a borderline false positive
        """)
    
    elif not ci_includes_null and not test_significant:
        # Also unusual - should note
        st.info(f"""
        ## ℹ️ Interesting Pattern: Confidence Interval vs. Significance Test
        
        **The Inconsistency:** 
        - The {test_name} is NOT statistically significant (p = {p_value:.4f} ≥ 0.05)
        - BUT the confidence interval ({ci_lower:.2f} to {ci_upper:.2f}) excludes {null_value}, 
          suggesting a non-zero effect
        
        **What This Means:**
        This pattern suggests:
        
        1. **Borderline significance:** The effect may be real but the test lacks power
        2. **Conservative test:** The test may be using a stricter threshold than the CI
        3. **Effect exists but is small:** The effect is non-zero but small relative to variability
        
        **Interpretation:**
        The confidence interval suggests a real effect, but the test indicates we cannot confidently 
        rule out randomness. This is common with small sample sizes or when effects are modest.
        
        **Recommendation:**
        - This result is suggestive but not conclusive
        - More data would help resolve the uncertainty
        - The effect may be real but small
        - Consider the practical significance, not just statistical significance
        """)
    
    elif is_very_wide:
        # Wide CI regardless of significance
        st.warning(f"""
        ## ⚠️ Wide Confidence Interval: High Uncertainty
        
        **The Finding:** 
        The confidence interval is very wide ({ci_lower:.2f} to {ci_upper:.2f}, spanning 
        {ci_width:.2f} units), indicating substantial uncertainty in the estimate.
        
        **What This Means:**
        Even though the point estimate is {point_estimate:.2f}, the true value could vary widely. 
        This wide interval suggests:
        
        1. **Limited data:** Small sample sizes produce wide intervals
        2. **High variability:** The underlying process has substantial variability
        3. **Unreliable estimate:** The point estimate may not be representative
        
        **Interpretation:**
        {"The test is significant, but the wide CI means we're uncertain about the effect magnitude." if test_significant else "The wide CI and non-significant test both suggest we have limited information."}
        
        **Recommendation:**
        - Interpret results with caution
        - Collect more data to narrow the interval
        - Consider that the true effect might be very different from the point estimate
        - Make conservative decisions given the uncertainty
        """)


def render_interaction_effect_explanation(
    main_effect: Dict[str, Any],
    interaction_effect: Dict[str, Any],
    subgroup_results: Dict[str, Dict[str, Any]]
) -> None:
    """
    Explain interaction effects when response time effects vary by subgroup (e.g., lead source).
    
    This is important because it means "faster is better" may only apply in certain contexts.
    """
    st.info(f"""
    ## 🔍 Interaction Effect Detected: Response Time Effects Vary by Context
    
    **The Finding:** The effect of response time on conversion is not uniform. It varies 
    significantly depending on other factors (e.g., lead source, time of day, lead quality).
    
    **What This Means:**
    
    The relationship between response time and conversion is **conditional** — it depends on 
    the context. This means:
    
    1. **One size does NOT fit all:** A single response time strategy may not be optimal
    2. **Context matters:** The same response time may have different effects in different situations
    3. **Subgroup analysis required:** We need to understand effects within each subgroup
    
    **The Evidence:**
    
    | Subgroup | Response Time Effect | Significance | Interpretation |
    |:---------|:---------------------|:-------------|:---------------|
    """)
    
    # Build table of subgroup results
    for subgroup_name, results in subgroup_results.items():
        effect = results.get('effect_size', 'N/A')
        sig = "✅ Significant" if results.get('is_significant', False) else "❌ Not significant"
        interpretation = results.get('interpretation', 'Varies by context')
        
        st.markdown(f"""
        | {subgroup_name} | {effect} | {sig} | {interpretation} |
        """)
    
    st.markdown(f"""
    **Business Implications:**
    
    1. **Segmented Strategy Required:**
       - Different response time targets for different lead types
       - Prioritize fast response for subgroups where it matters most
       - Optimize for quality over speed where speed doesn't matter
    
    2. **Resource Allocation:**
       - Focus response time improvements on high-impact subgroups
       - Don't waste resources optimizing response time where it doesn't affect outcomes
    
    3. **Personalization:**
       - Tailor response strategies to lead characteristics
       - Consider that optimal response time may be context-dependent
    
    **Example Interpretation:**
    
    If response time strongly affects conversion for "website" leads but not for "referral" leads:
    - **Website leads:** Prioritize speed — they may be comparison shopping
    - **Referral leads:** Focus on quality — speed is less critical, relationship matters more
    
    **Next Steps:**
    
    1. **Analyze each subgroup separately** to understand context-specific effects
    2. **Develop segmented strategies** rather than a one-size-fits-all approach
    3. **Test hypotheses** about why effects differ across subgroups
    4. **Monitor performance** by subgroup to validate the interaction
    
    **Caution:** 
    Interaction effects can be complex. Make sure you have adequate sample sizes within each 
    subgroup before drawing strong conclusions. Small subgroups may produce unreliable estimates.
    """)


def detect_imbalance(close_rates_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect extreme imbalance in bucket distributions.
    
    Returns information about imbalance if detected.
    """
    if len(close_rates_df) < 2:
        return {'is_imbalanced': False}
    
    total_leads = close_rates_df['n_leads'].sum()
    proportions = close_rates_df['n_leads'] / total_leads
    
    # Check for extreme imbalance: one bucket has >70% of data
    max_prop = proportions.max()
    max_idx = proportions.idxmax()
    max_bucket = close_rates_df.loc[max_idx, 'bucket']
    max_count = close_rates_df.loc[max_idx, 'n_leads']
    
    # Check how many buckets have <5% of data
    small_buckets = (proportions < 0.05).sum()
    very_small_buckets = (proportions < 0.02).sum()
    
    # Calculate imbalance ratio (max bucket / average of others)
    other_avg = (total_leads - max_count) / (len(close_rates_df) - 1) if len(close_rates_df) > 1 else 0
    imbalance_ratio = max_count / other_avg if other_avg > 0 else float('inf')
    
    # Determine severity
    if max_prop > 0.80:
        severity = "severe"
    elif max_prop > 0.70:
        severity = "moderate"
    elif max_prop > 0.60:
        severity = "mild"
    else:
        severity = None
    
    is_imbalanced = max_prop > 0.60  # Flag if any bucket has >60% of data
    
    if not is_imbalanced:
        return {'is_imbalanced': False}
    
    return {
        'is_imbalanced': True,
        'severity': severity,
        'max_bucket': max_bucket,
        'max_proportion': max_prop,
        'max_count': int(max_count),
        'small_bucket_count': int(small_buckets),
        'very_small_bucket_count': int(very_small_buckets),
        'imbalance_ratio': imbalance_ratio,
        'total_buckets': len(close_rates_df)
    }


def render_imbalance_warning(imbalance_info: Dict[str, Any]) -> None:
    """
    Warn about extreme imbalance in bucket distributions and explain implications.
    """
    if not imbalance_info.get('is_imbalanced'):
        return
    
    severity = imbalance_info.get('severity', 'moderate')
    max_bucket = imbalance_info['max_bucket']
    max_prop = imbalance_info['max_proportion']
    max_count = imbalance_info['max_count']
    small_buckets = imbalance_info['small_bucket_count']
    imbalance_ratio = imbalance_info['imbalance_ratio']
    
    if severity == "severe":
        icon = "🔴"
        alert_type = "error"
    elif severity == "moderate":
        icon = "⚠️"
        alert_type = "warning"
    else:
        icon = "ℹ️"
        alert_type = "info"
    
    message = f"""
    ## {icon} Imbalanced Data Distribution Detected
    
    **The Issue:** The data is extremely imbalanced across response time buckets. 
    **{max_bucket}** contains **{max_prop*100:.1f}%** ({max_count:,} leads) of all observations, 
    while other buckets have much smaller samples.
    
    **Why This Matters:**
    
    Extreme imbalance creates several problems for statistical analysis:
    
    1. **Comparison Reliability:**
       - Comparisons between the dominant bucket and small buckets are unreliable
       - Small buckets have high uncertainty (wide confidence intervals)
       - Statistical tests may lack power to detect real differences
    
    2. **Skewed Estimates:**
       - Overall patterns are dominated by the largest bucket
       - Effects in small buckets may be hidden or exaggerated
       - The analysis may primarily reflect patterns in the dominant category
    
    3. **Statistical Test Validity:**
       - Chi-square tests require adequate expected frequencies in all cells
       - With extreme imbalance, some comparisons may not meet minimum requirements
       - Results may be misleading or statistically invalid
    
    **Specific Problems:**
    """
    
    if small_buckets > 0:
        message += f"\n- **{small_buckets} bucket(s) have <5% of data** — these have insufficient sample sizes for reliable comparison\n"
    
    if imbalance_ratio > 5:
        message += f"- **The largest bucket is {imbalance_ratio:.1f}× larger than average** — creating disproportionate influence\n"
    
    message += f"""
    **What This Means for Your Analysis:**
    
    1. **Results are dominated by {max_bucket}:**
       - Patterns you observe primarily reflect this bucket's characteristics
       - Other buckets may have insufficient data to draw meaningful conclusions
    
    2. **Comparisons may be unreliable:**
       - Differences between small buckets and the dominant bucket are hard to interpret
       - Small bucket estimates have very high uncertainty
    
    3. **Consider alternative approaches:**
       - Focus analysis on buckets with adequate sample sizes
       - Consider combining small buckets into broader categories
       - Acknowledge limitations when interpreting results
    
    **Recommended Actions:**
    
    1. **Combine small buckets:**
       - Merge buckets with <5% of data to create larger, more reliable groups
       - Trade granularity for statistical reliability
    
    2. **Focus analysis appropriately:**
       - Acknowledge that results primarily reflect patterns in {max_bucket}
       - Be cautious when interpreting effects in small buckets
       - Consider whether the imbalance reflects a real pattern or data collection issue
    
    3. **Collect more balanced data (if possible):**
       - If this imbalance is due to data collection, future data should be more balanced
       - If it reflects reality (most leads truly fall in one bucket), acknowledge this limitation
    
    4. **Interpret with caution:**
       - Small bucket results should be treated as preliminary
       - Focus conclusions on patterns supported by adequate sample sizes
       - Be transparent about the imbalance in your reporting
    
    **Bottom Line:**
    The extreme imbalance ({max_prop*100:.1f}% in one bucket) means your analysis primarily 
    reflects patterns in {max_bucket}. Results for smaller buckets should be interpreted 
    with caution due to limited sample sizes. Consider combining categories or focusing 
    on buckets with adequate data.
    """
    
    if alert_type == "error":
        st.error(message)
    elif alert_type == "warning":
        st.warning(message)
    else:
        st.info(message)


def render_no_controls_explanation(
    available_controls: list,
    missing_controls: list = None,
    reason: str = None
) -> None:
    """
    Explain when regression cannot control for confounders due to lack of variation.
    
    This happens when:
    - All leads are from the same source (can't control for lead source)
    - All data is from the same time period (can't control for temporal effects)
    - Other control variables don't vary in the dataset
    """
    if missing_controls is None:
        missing_controls = []
    
    # Common reasons for missing controls
    if reason is None:
        if len(available_controls) == 0:
            reason = "no_controls_available"
        elif len(missing_controls) > 0:
            reason = "controls_dont_vary"
        else:
            reason = "insufficient_variation"
    
    st.warning(f"""
    ## ⚠️ Limited Ability to Control for Confounders
    
    **The Issue:** The regression analysis cannot control for certain potential confounders 
    because they do not vary in your dataset.
    
    **What This Means:**
    
    Regression analysis works by comparing outcomes across different values of control variables. 
    If a variable doesn't vary (e.g., all leads are from the same source), we cannot assess 
    its effect or use it to control for confounding.
    
    **Your Situation:**
    """)
    
    if reason == "no_controls_available":
        st.markdown("""
        - **No control variables are available** in your dataset
        - The analysis cannot account for potential confounders
        - Results reflect the raw association without any controls
        
        **Implications:**
        - We cannot rule out that observed effects are due to confounding
        - Unmeasured variables may explain the relationship
        - Causal conclusions are particularly risky
        """)
    
    elif reason == "controls_dont_vary":
        missing_list = ", ".join(missing_controls) if missing_controls else "certain variables"
        st.markdown(f"""
        - **Cannot control for: {missing_list}** — these variables don't vary in your data
        - Available controls: {', '.join(available_controls) if available_controls else 'None'}
        
        **Why This Matters:**
        
        If {missing_list} {'does not' if len(missing_controls) == 1 else 'do not'} vary, we cannot:
        - Assess whether {missing_list} explains the response time effect
        - Control for confounding by {missing_list}
        - Determine if the effect persists after accounting for {missing_list}
        
        **Common Examples:**
        - **Single lead source:** All leads from "website" — can't test if source differences explain the effect
        - **Single time period:** All data from one month — can't control for temporal trends
        - **Single sales team:** All leads handled by one team — can't control for team differences
        """)
    
    elif reason == "insufficient_variation":
        st.markdown("""
        - **Control variables have insufficient variation** to provide meaningful controls
        - While variables exist, they don't vary enough to separate their effects
        
        **Example:** If 98% of leads are from "website" and 2% from "referral", 
        we cannot reliably control for lead source because the comparison groups are too imbalanced.
        """)
    
    st.markdown("""
    **Impact on Interpretation:**
    
    1. **Confounding Risk:**
       - The observed association may be explained by variables we cannot control for
       - Without variation, we cannot test alternative explanations
       - Causal conclusions are less defensible
    
    2. **Regression Limitations:**
       - Regression can only control for variables that vary in the data
       - If a confounder doesn't vary, regression cannot address it
       - The "controls" we include may not fully address confounding
    
    3. **What We Can Still Learn:**
       - We can still assess the raw association between response time and conversion
       - We can still test statistical significance
       - We just cannot rule out that the effect is due to unmeasured or invariant confounders
    
    **Recommended Actions:**
    
    1. **Acknowledge the limitation:**
       - Be transparent that certain confounders cannot be controlled
       - Explain why (variables don't vary)
       - Note that this limits causal inference
    
    2. **Alternative approaches:**
       - Use qualitative methods to understand the relationship
       - Look for natural experiments or quasi-experiments
       - Consider whether you can collect more varied data in the future
    
    3. **Be cautious with conclusions:**
       - Don't claim causation without experimental evidence
       - Acknowledge that unmeasured confounders may explain the effect
       - Focus on the association, not causation
    
    4. **Future data collection:**
       - If possible, collect data with more variation in control variables
       - This will allow better control for confounders in future analyses
    
    **Bottom Line:**
    Your analysis cannot control for certain confounders because they don't vary in your data. 
    This means we cannot rule out that the observed response time effect is actually due to 
    these uncontrolled variables. Be cautious when drawing causal conclusions, and consider 
    the analysis as describing an association rather than proving causation.
    """)


def render_extreme_rate_guidance(conversion_rate: float, n_observations: int) -> None:
    """
    Provide special guidance for very high or very low conversion rates.
    
    Statistical considerations change when conversion rates are extreme:
    - Very high rates (>90%): Different statistical properties
    - Very low rates (<5%): Different statistical properties
    - Perfect or near-perfect rates: May indicate data issues or special circumstances
    """
    is_very_high = conversion_rate > 0.90
    is_very_low = conversion_rate < 0.05
    is_extreme = is_very_high or is_very_low
    is_perfect = conversion_rate >= 0.999 or conversion_rate <= 0.001
    
    if not is_extreme:
        return  # No special guidance needed
    
    if is_perfect:
        icon = "🔍"
        severity = "warning"
    elif is_very_high:
        icon = "📈"
        severity = "info"
    else:
        icon = "📉"
        severity = "info"
    
    message = f"""
    ## {icon} Extreme Conversion Rate Detected
    
    **The Finding:** Your overall conversion rate is **{conversion_rate*100:.1f}%**, which is 
    {"extremely high" if is_very_high else "extremely low"}.
    
    **Why This Matters:**
    """
    
    if is_very_high:
        message += f"""
        Very high conversion rates ({conversion_rate*100:.1f}%) have important implications:
        
        1. **Limited Room for Improvement:**
           - With {conversion_rate*100:.1f}% already converting, there's little room to increase
           - Response time improvements may have diminishing returns
           - Focus may shift to other metrics (e.g., deal size, customer satisfaction)
        
        2. **Statistical Properties:**
           - High rates create different statistical dynamics
           - Confidence intervals may behave differently at extremes
           - Small absolute improvements represent large relative improvements
        
        3. **Practical Significance:**
           - Even small absolute differences (e.g., 2-3 percentage points) may be practically meaningful
           - Moving from {conversion_rate*100:.1f}% to {(conversion_rate + 0.02)*100:.1f}% means eliminating most failures
           - But the absolute number of additional conversions may be modest
        
        4. **Data Quality Consideration:**
           - Very high rates may indicate:
             * Excellent lead quality or sales process
             * Selective lead filtering (only high-quality leads included)
             * Measurement issues (e.g., conversion defined too broadly)
        """
    
    elif is_very_low:
        message += f"""
        Very low conversion rates ({conversion_rate*100:.1f}%) have important implications:
        
        1. **Large Improvement Potential:**
           - With only {conversion_rate*100:.1f}% converting, there's substantial room for improvement
           - Response time effects could have large absolute impacts
           - Small improvements represent large relative gains
        
        2. **Statistical Considerations:**
           - Low rates require larger sample sizes for reliable estimates
           - With {n_observations:,} observations and {conversion_rate*100:.1f}% rate, you have approximately {int(n_observations * conversion_rate)} conversions
           - Confidence intervals may be wider or asymmetric
           - Statistical tests may need special consideration
        
        3. **Power Requirements:**
           - Detecting small effects requires more data at low baseline rates
           - You may need more observations to reliably detect response time effects
           - Consider whether your sample size is adequate
        
        4. **Practical Interpretation:**
           - A 5 percentage point improvement (from {conversion_rate*100:.1f}% to {(conversion_rate + 0.05)*100:.1f}%) 
             would be a **{(0.05/conversion_rate)*100:.0f}% relative increase**
           - Even modest absolute improvements can be transformative
        
        5. **Data Quality Consideration:**
           - Very low rates may indicate:
             * Poor lead quality or targeting
             * Conversion defined too narrowly
             * Measurement or tracking issues
        """
    
    if is_perfect:
        message += """
        
        **⚠️ Near-Perfect Rate Warning:**
        
        A conversion rate this extreme (near 0% or 100%) is unusual and may indicate:
        
        1. **Data Issues:**
           - Measurement errors
           - Data filtering that removed variation
           - Incorrect conversion definitions
        
        2. **Business Reality:**
           - Extremely selective lead qualification
           - Highly specialized product/service
           - Perfect or near-perfect process (rare)
        
        3. **Statistical Validity:**
           - Near-perfect rates can cause statistical test failures
           - Chi-square and regression assumptions may be violated
           - Results should be interpreted with extreme caution
        """
    
    message += f"""
    
    **Statistical Recommendations:**
    
    1. **Sample Size Considerations:**
       {"With very high conversion rates, you need fewer observations to detect effects, but absolute improvements are small." if is_very_high else f"With very low conversion rates, you need more observations. With {n_observations:,} observations and {conversion_rate*100:.1f}% rate, you have ~{int(n_observations * conversion_rate)} conversions — ensure this is adequate for statistical tests."}
    
    2. **Interpretation Focus:**
       - {"Focus on practical significance over statistical significance — even small improvements may be meaningful when starting high." if is_very_high else "Focus on relative improvements — a 50% relative increase (from 2% to 3%) may be more meaningful than the absolute 1 percentage point."}
    
    3. **Alternative Metrics:**
       {"Consider other success metrics (deal size, customer lifetime value, satisfaction) where there may be more room for improvement." if is_very_high else "Consider whether conversion is the right metric, or if you should also track intermediate metrics (e.g., engagement, qualified leads)."}
    
    4. **Effect Size Interpretation:**
       - {"Small absolute differences may represent substantial relative improvements." if is_very_high else "Focus on relative improvements — doubling conversion from 2% to 4% is a 100% relative increase."}
    
    **Bottom Line:**
    {"Your high conversion rate means response time improvements have limited absolute impact but may still be valuable. Consider other success metrics and focus on maintaining excellence." if is_very_high else f"Your low conversion rate suggests substantial improvement potential. Response time effects could have meaningful impact, but ensure adequate sample sizes for reliable detection."}
    """
    
    if severity == "warning":
        st.warning(message)
    else:
        st.info(message)

