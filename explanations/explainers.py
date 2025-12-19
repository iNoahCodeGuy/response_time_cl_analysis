# =============================================================================
# Explainers Module
# =============================================================================
# This module translates statistical concepts into clear, logical explanations
# using first-principles reasoning and methodical step-by-step logic.
#
# EXPLANATION PHILOSOPHY:
# -----------------------
# 1. Start with the fundamental question being answered
# 2. Explain WHY we need this concept before HOW it works
# 3. Build understanding through logical cause-and-effect chains
# 4. Define every term at the moment it's introduced
# 5. Use concrete examples to ground abstract concepts
# 6. Never assume prior knowledge — explain everything from first principles
#
# =============================================================================

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple


# =============================================================================
# P-VALUE EXPLAINERS
# =============================================================================

def get_p_value_explanation(p_value: float) -> Dict[str, Any]:
    """
    Convert a p-value into a logical, first-principles explanation.
    
    The p-value answers a fundamental epistemological question:
    "Given what we observed, how confident can we be that the pattern is real?"
    """
    # Cap confidence at 99.99% for readability
    confidence_pct = min((1 - p_value) * 100, 99.99)
    
    if p_value < 0.0001:
        return {
            'luck_chance': '< 0.01%',
            'confidence': f'{confidence_pct:.1f}%',
            'verdict': 'The pattern is real',
            'emoji': '✅',
            'trust_level': 'Extremely high certainty',
            'plain_english': (
                f"The probability of observing these results by chance alone is vanishingly small — "
                f"less than {p_value*100:.4f}%. To put this in perspective: if we repeated this analysis "
                f"ten thousand times with random data, we would expect to see results this strong "
                f"less than once. This is not chance. This is a genuine pattern in reality."
            ),
            'first_principles': (
                "When the probability of chance producing our results falls below one in ten thousand, "
                "we have moved beyond reasonable doubt. The pattern exists."
            ),
            'color': 'green'
        }
    elif p_value < 0.001:
        return {
            'luck_chance': f'{p_value*100:.2f}%',
            'confidence': f'{confidence_pct:.1f}%',
            'verdict': 'The pattern is almost certainly real',
            'emoji': '✅',
            'trust_level': 'Very high certainty',
            'plain_english': (
                f"The probability of chance alone producing these results is approximately {p_value*100:.2f}% — "
                f"roughly 1 in {int(1/p_value):,}. When we observe something this unlikely to occur by accident, "
                f"the logical conclusion is that we are observing a real phenomenon, not statistical noise."
            ),
            'first_principles': (
                "At this level of improbability, the burden of proof shifts. "
                "It becomes unreasonable to attribute these results to chance."
            ),
            'color': 'green'
        }
    elif p_value < 0.01:
        return {
            'luck_chance': f'{p_value*100:.1f}%',
            'confidence': f'{confidence_pct:.1f}%',
            'verdict': 'The pattern is very likely real',
            'emoji': '✅',
            'trust_level': 'High certainty',
            'plain_english': (
                f"There is approximately a {p_value*100:.1f}% probability — about 1 in {int(1/p_value):,} — "
                f"that these results occurred by random chance. While not impossible, this is sufficiently "
                f"improbable that we can proceed with confidence that the pattern reflects reality."
            ),
            'first_principles': (
                "When something has less than a 1% chance of being random, "
                "we are justified in treating it as a real effect."
            ),
            'color': 'green'
        }
    elif p_value < 0.05:
        return {
            'luck_chance': f'{p_value*100:.1f}%',
            'confidence': f'{confidence_pct:.1f}%',
            'verdict': 'The pattern appears real',
            'emoji': '✅',
            'trust_level': 'Good certainty',
            'plain_english': (
                f"The probability of these results arising from chance is {p_value*100:.1f}%. "
                f"By conventional scientific standards, this crosses the threshold for statistical significance. "
                f"We can reasonably conclude that a real pattern exists, though some caution remains warranted."
            ),
            'first_principles': (
                "The 5% threshold exists because we accept a small margin of error. "
                "Results below this threshold are more likely real than not."
            ),
            'color': 'green'
        }
    elif p_value < 0.10:
        return {
            'luck_chance': f'{p_value*100:.1f}%',
            'confidence': f'{confidence_pct:.1f}%',
            'verdict': 'The pattern may be real',
            'emoji': '⚠️',
            'trust_level': 'Moderate certainty',
            'plain_english': (
                f"There is a {p_value*100:.1f}% probability that chance alone explains these results. "
                f"This falls in an uncertain zone — not random enough to dismiss, but not certain enough "
                f"to act upon with full confidence. The evidence is suggestive but not conclusive."
            ),
            'first_principles': (
                "When probability falls between 5% and 10%, we are in epistemic uncertainty. "
                "More data would help resolve whether this pattern is genuine."
            ),
            'color': 'orange'
        }
    else:
        return {
            'luck_chance': f'{p_value*100:.0f}%',
            'confidence': f'{confidence_pct:.1f}%',
            'verdict': 'The pattern may be random noise',
            'emoji': '❌',
            'trust_level': 'Insufficient evidence',
            'plain_english': (
                f"There is a {p_value*100:.0f}% probability that what we observed is simply random variation — "
                f"the natural fluctuation present in any sample of data. This is too high a probability "
                f"to conclude that a real pattern exists. The apparent differences may be illusory."
            ),
            'first_principles': (
                "When chance can readily explain our observations, we cannot claim to have discovered "
                "a genuine pattern. The signal, if it exists, is lost in the noise."
            ),
            'color': 'red'
        }


def render_p_value_explainer(p_value: float, context: str = "this difference") -> None:
    """
    Render a logical, step-by-step p-value explanation in Streamlit.
    """
    exp = get_p_value_explanation(p_value)
    
    st.markdown(f"""
    #### The Fundamental Question: Is {context} real or merely noise?
    
    Before we can act on any observation, we must first establish whether that observation 
    reflects a genuine pattern in reality, or merely the random variation inherent in any sample of data.
    
    | Question | Answer |
    |:---------|:-------|
    | **Probability this is random chance** | {exp['luck_chance']} |
    | **Our certainty level** | {exp['confidence']} |
    | **Conclusion** | {exp['emoji']} {exp['verdict']} |
    
    **The Logic:** {exp['plain_english']}
    
    **First Principles:** {exp.get('first_principles', '')}
    """)


# =============================================================================
# ODDS RATIO EXPLAINERS
# =============================================================================

def get_odds_ratio_explanation(odds_ratio: float, ci_lower: float, ci_upper: float, 
                                bucket: str, reference: str = "60+ min") -> Dict[str, Any]:
    """
    Convert an odds ratio into a logical, first-principles explanation.
    
    The odds ratio answers a fundamental comparative question:
    "How much does being in one group change your chances of success?"
    """
    # Determine if significant (CI doesn't include 1)
    is_significant = ci_lower > 1 or ci_upper < 1
    
    # Calculate multiplier for explanation
    if odds_ratio >= 1:
        multiplier = odds_ratio
        direction = "higher"
        comparison = "advantage"
    else:
        multiplier = 1 / odds_ratio
        direction = "lower"
        comparison = "disadvantage"
    
    # Create concrete example using baseline of 10%
    baseline_rate = 0.10
    if odds_ratio > 0:
        implied_rate = (odds_ratio * baseline_rate) / (1 - baseline_rate + odds_ratio * baseline_rate)
    else:
        implied_rate = baseline_rate
    
    extra_sales_per_100 = (implied_rate - baseline_rate) * 100
    
    return {
        'multiplier': multiplier,
        'direction': direction,
        'comparison': comparison,
        'is_significant': is_significant,
        'bucket': bucket,
        'reference': reference,
        'headline': f"{multiplier:.1f}× the odds of success" if multiplier != 1.0 else "Equal odds",
        'example': (
            f"Consider a concrete scenario: if the baseline group ({reference}) converts 10 out of every 100 leads, "
            f"then the faster group ({bucket}) would convert approximately {int(implied_rate * 100)} out of 100. "
            f"The difference — {abs(int(extra_sales_per_100))} {'additional' if extra_sales_per_100 > 0 else 'fewer'} conversions "
            f"per 100 leads — represents the practical impact of this odds ratio."
        ) if multiplier != 1.0 else "No measurable difference in conversion odds.",
        'confidence_note': (
            f"The uncertainty range ({ci_lower:.2f}× to {ci_upper:.2f}×) does not include 1.0, "
            f"which means we can be confident this advantage is real, not a statistical artifact."
            if is_significant else
            f"The uncertainty range ({ci_lower:.2f}× to {ci_upper:.2f}×) includes 1.0, "
            f"which means we cannot rule out that this apparent difference is simply random variation."
        ),
        'first_principles': (
            "The odds ratio compares the ratio of success to failure between two groups. "
            f"An odds ratio of {odds_ratio:.2f} means that for every failure in the {bucket} group, "
            f"there are {odds_ratio:.2f} times as many successes compared to the {reference} group."
        )
    }


def render_odds_ratio_table(odds_ratios_df: pd.DataFrame) -> None:
    """
    Render a logical, first-principles explanation of odds ratios.
    """
    st.markdown("### Quantifying the Effect: How Much Does Speed Matter?")
    
    st.markdown("""
    We now arrive at a critical question: not merely *whether* response speed matters, 
    but *how much* it matters. To answer this, we must understand the concept of the odds ratio.
    
    **The Logic of Odds Ratios:**
    
    An odds ratio is a comparison tool. It tells us: for every unit of success in our reference group 
    (slow responders), how many units of success do we see in our comparison group (faster responders)?
    
    - **Odds ratio = 1.0** → No difference between groups
    - **Odds ratio = 2.0** → The faster group has twice the odds of success
    - **Odds ratio = 0.5** → The faster group has half the odds of success
    
    The table below quantifies the advantage (or disadvantage) of each response speed tier.
    """)
    
    # Create methodical table
    rows = []
    for _, row in odds_ratios_df.iterrows():
        exp = get_odds_ratio_explanation(
            row['odds_ratio'], 
            row['ci_lower'], 
            row['ci_upper'],
            row['bucket']
        )
        
        p_exp = get_p_value_explanation(row.get('p_value', 0.01))
        
        rows.append({
            'Response Time': row['bucket'],
            'Odds Multiplier': exp['headline'],
            'Practical Meaning': f"~{int(row['odds_ratio'] * 10)} conversions per 100 leads vs 10 for slow responders" if row['odds_ratio'] >= 1 else exp['example'],
            'Certainty': f"{p_exp['emoji']} {p_exp['verdict']}"
        })
    
    # Add the reference row
    rows.append({
        'Response Time': '60+ min (reference point)',
        'Odds Multiplier': '1.0× (baseline)',
        'Practical Meaning': 'All comparisons are made against this group',
        'Certainty': '—'
    })
    
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Add logical summary
    st.markdown("""
    **The Practical Implication:**
    
    Consider the scale of your operation. If you process 1,000 leads monthly and currently 
    respond slowly to most of them, the potential gain from faster response is substantial. 
    An odds ratio of 2.0 does not guarantee twice the sales — the relationship is more nuanced — 
    but it does indicate a meaningful, measurable advantage that compounds over time.
    """)


# =============================================================================
# CONFIDENCE INTERVAL EXPLAINERS
# =============================================================================

def get_ci_explanation(value: float, ci_lower: float, ci_upper: float, 
                       metric_name: str = "close rate") -> Dict[str, Any]:
    """
    Explain a confidence interval using first-principles reasoning.
    
    The confidence interval addresses a fundamental limitation of sampling:
    we observe a sample, but we want to know about the population.
    """
    range_size = ci_upper - ci_lower
    
    if range_size < 0.02:
        precision = "high precision"
        data_quality = "substantial data"
    elif range_size < 0.05:
        precision = "reasonable precision"
        data_quality = "adequate data"
    else:
        precision = "lower precision"
        data_quality = "limited data"
    
    return {
        'estimate': value,
        'lower': ci_lower,
        'upper': ci_upper,
        'range_size': range_size,
        'precision': precision,
        'plain_english': (
            f"Our point estimate is **{value*100:.1f}%**. However, this is based on a sample, "
            f"not the complete population. The true underlying {metric_name} likely falls "
            f"between **{ci_lower*100:.1f}%** and **{ci_upper*100:.1f}%**. "
            f"This interval reflects {precision}, indicating {data_quality}."
        ),
        'first_principles': (
            f"Every measurement from a sample carries inherent uncertainty. The confidence interval "
            f"quantifies that uncertainty. A narrower interval means more certainty; a wider interval "
            f"means less. This interval spans {range_size*100:.1f} percentage points."
        )
    }


def render_ci_explainer() -> None:
    """
    Render a first-principles explanation of confidence intervals.
    """
    with st.expander("📏 Understanding the Uncertainty Ranges"):
        st.markdown("""
        ### The Epistemological Problem of Sampling
        
        When we measure a close rate, we face a fundamental limitation: we are observing a *sample* 
        of data, not the complete universe of all possible leads. The number we calculate — say, 13.0% — 
        is our best estimate, but it is not the *true* rate. The true rate is unknowable with certainty.
        
        **The confidence interval addresses this limitation.** It provides a range within which 
        the true value most likely falls.
        
        #### The Mechanics:
        
        When we report "13.0% (11.9% - 14.2%)":
        - **13.0%** is our point estimate — the single best guess based on the data
        - **11.9% - 14.2%** is the 95% confidence interval — the range of plausible true values
        - We are 95% confident the true rate falls somewhere within this range
        
        #### Why This Matters for Decision-Making:
        
        | Narrow Interval | Wide Interval |
        |:----------------|:--------------|
        | Example: 12.5% - 13.5% | Example: 8% - 18% |
        | Based on substantial data | Based on limited data |
        | High precision in our estimate | Significant uncertainty |
        | Decisions can be made confidently | Caution warranted; more data needed |
        
        #### The Principle:
        
        The width of a confidence interval is inversely related to sample size. 
        Double your data, and the interval shrinks by approximately 30%. 
        This is why collecting more data is not merely desirable — it is the only way 
        to increase the precision of our knowledge.
        """)


# =============================================================================
# PERCENTAGE POINTS EXPLAINER
# =============================================================================

def render_percentage_points_explainer() -> None:
    """
    Explain the distinction between percentage points and percent change.
    """
    with st.expander("📐 The Distinction: Percentage Points vs. Percent Change"):
        st.markdown("""
        ### A Critical Distinction in Quantitative Reasoning
        
        When discussing changes in rates or percentages, precision in language matters enormously. 
        There are two fundamentally different ways to describe a change, and conflating them 
        leads to significant misunderstanding.
        
        #### The Two Measures:
        
        Consider a close rate that moves from **10%** to **12%**.
        
        | Measure | Notation | Value | Meaning |
        |:--------|:---------|:------|:--------|
        | **Percentage Points** | pp | +2pp | The absolute change: 12 minus 10 equals 2 |
        | **Percent Change** | % | +20% | The relative change: 2 divided by 10 equals 0.20, or 20% |
        
        #### Why This Matters:
        
        "A 20% improvement" sounds dramatic. "A 2 percentage point improvement" sounds modest. 
        Yet they describe the identical change. This is not merely semantic — it affects 
        how decisions are perceived and made.
        
        **We use percentage points throughout this analysis** because they provide an unambiguous, 
        absolute measure of change. When we say "the difference is 6.8 percentage points," 
        you know exactly what that means: one rate is 6.8 units higher than the other on 
        the percentage scale.
        
        #### Reference:
        
        | Notation | Example | Translation |
        |:---------|:--------|:------------|
        | +2pp | 10% → 12% | Absolute increase of 2 on the percentage scale |
        | -1.5pp | 10% → 8.5% | Absolute decrease of 1.5 on the percentage scale |
        | +0.3pp | 10% → 10.3% | A small absolute increase |
        """)


# =============================================================================
# CHI-SQUARE WALKTHROUGH
# =============================================================================

def render_chi_square_walkthrough(
    observed: Dict[str, Tuple[int, int]],  # bucket -> (orders, non_orders)
    expected: Dict[str, Tuple[float, float]],  # bucket -> (expected_orders, expected_non_orders)
    chi_sq_stat: float,
    p_value: float
) -> None:
    """
    Walk through the chi-square calculation using methodical, first-principles reasoning.
    """
    st.markdown("### The Mathematical Foundation")
    
    with st.expander("🔍 Step-by-Step Derivation", expanded=False):
        
        # Step 1: The Fundamental Question
        st.markdown("""
        #### Step 1: Establishing the Null Hypothesis
        
        Before we can claim that response time affects outcomes, we must consider the alternative: 
        **what if response time has no effect whatsoever?**
        
        This is not a philosophical question — it is a mathematical one. If response time truly 
        had no causal relationship with sales, we would still expect to see *some* variation in 
        close rates across buckets, simply due to random chance. The question becomes: 
        is the variation we observe *greater* than what chance alone would produce?
        
        To answer this, we construct a model of what the data *would* look like under the 
        assumption of no effect, then measure how far reality deviates from that model.
        """)
        
        # Step 2: Observed vs Expected
        st.markdown("#### Step 2: Constructing the Expected Values")
        
        comparison_rows = []
        for bucket in observed:
            obs_orders = observed[bucket][0]
            exp_orders = expected.get(bucket, (0, 0))[0]
            diff = obs_orders - exp_orders
            
            comparison_rows.append({
                'Response Time': bucket,
                'Observed (Actual)': f"{obs_orders:,}",
                'Expected (Null Hypothesis)': f"{exp_orders:.0f}",
                'Deviation': f"{diff:+.0f}",
                'Interpretation': 'Exceeds expectation' if diff > 10 else 'Below expectation' if diff < -10 else 'Within expectation'
            })
        
        st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)
        
        st.markdown("""
        **The Logic of Expected Values:**
        
        If response time has no effect, then the overall close rate should apply uniformly 
        across all buckets. The "expected" column shows what each bucket's sales *would be* 
        if every bucket converted at the overall average rate.
        
        The deviations tell us where reality diverges from this null hypothesis.
        """)
        
        # Step 3: The Formula
        st.markdown("#### Step 3: The Chi-Square Statistic")
        
        st.latex(r"\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}")
        
        st.markdown("""
        **Decomposing the Formula:**
        
        | Component | Symbol | Purpose |
        |:----------|:-------|:--------|
        | Observed count | O | What actually happened |
        | Expected count | E | What the null hypothesis predicts |
        | Difference | (O - E) | How far reality deviates from prediction |
        | Squared difference | (O - E)² | Ensures all deviations contribute positively |
        | Normalized | (O - E)²/E | Accounts for the size of each cell |
        | Sum | Σ | Aggregates across all categories |
        
        **Why Square the Differences?**
        
        A deviation of +50 and a deviation of -50 are equally surprising — both represent 
        significant departures from expectation. Squaring ensures they contribute equally 
        to the statistic rather than canceling each other out.
        
        **Why Divide by Expected?**
        
        A deviation of 50 in a bucket expecting 500 is less remarkable than a deviation 
        of 50 in a bucket expecting 100. Division normalizes for scale.
        """)
        
        # Step 4: The Result
        st.markdown("#### Step 4: Interpreting the Result")
        
        p_exp = get_p_value_explanation(p_value)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("χ² Statistic", f"{chi_sq_stat:.2f}")
            st.caption("Measures total deviation from null hypothesis")
        
        with col2:
            st.metric("P-Value", p_exp['luck_chance'])
            st.caption(f"{p_exp['emoji']} {p_exp['verdict']}")
        
        magnitude = 'substantial' if chi_sq_stat > 20 else 'moderate' if chi_sq_stat > 10 else 'modest'
        
        st.markdown(f"""
        **The Conclusion:**
        
        A χ² statistic of **{chi_sq_stat:.2f}** represents a **{magnitude}** deviation from 
        what we would expect under the null hypothesis.
        
        To translate this into a probability: if the null hypothesis were true (response time 
        has no effect), the probability of observing a χ² value this large or larger is 
        **{p_exp['luck_chance']}**.
        
        {p_exp['plain_english']}
        """)


# =============================================================================
# LOGISTIC REGRESSION EXPLAINER
# =============================================================================

def render_regression_explainer() -> None:
    """
    Explain the logic of regression using first-principles reasoning.
    """
    with st.expander("🔬 The Logic of Controlling for Confounding Variables"):
        st.markdown("""
        ### The Problem of Confounding: Why Simple Comparisons Can Deceive
        
        Consider a scenario that illustrates a fundamental problem in causal reasoning:
        
        | Lead Source | Average Response Time | Close Rate |
        |:------------|:----------------------|:-----------|
        | Referrals | 10 minutes | 25% |
        | Website | 45 minutes | 8% |
        
        A naive analysis would conclude: "Fast responses correlate with high close rates. 
        Therefore, speed causes sales."
        
        **But this reasoning contains a logical flaw.**
        
        Referrals are warm leads — they have a high close rate *regardless* of response time, 
        because they already trust the referrer. They also receive fast responses because 
        salespeople recognize their value and prioritize them.
        
        The causal structure may actually be:
        
        ```
        Lead Source → Close Rate (directly, through lead quality)
        Lead Source → Response Time (through salesperson prioritization)
        
        Result: Fast responses and high close rates appear together,
                but neither causes the other. Both are effects of lead source.
        ```
        
        ### The Solution: Regression Analysis
        
        Regression addresses this by mathematically isolating the effect of each variable.
        
        The question becomes: **Within each lead source**, does response time still predict success?
        
        - Among referrals only: Do fast-responded referrals close better than slow-responded referrals?
        - Among website leads only: Do fast-responded website leads close better than slow-responded website leads?
        
        If speed matters *within* each lead source — after removing the confounding — then we have 
        evidence that speed genuinely affects outcomes. If the effect disappears after controlling 
        for lead source, then the apparent effect was illusory.
        
        ### The Fundamental Principle
        
        Correlation does not imply causation. But regression brings us closer to causation by 
        removing alternative explanations. The more confounders we control, the more confident 
        we can be that the remaining effect is genuine.
        
        **Important caveat:** We can only control for variables we measure. If there are unmeasured 
        confounders, bias may remain. This is why randomized experiments remain the gold standard — 
        but when experiments are impractical, regression is the most powerful observational tool available.
        """)


# =============================================================================
# SAMPLE SIZE / DATA QUALITY EXPLAINERS
# =============================================================================

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


# =============================================================================
# MAIN INSIGHT RENDERER
# =============================================================================

def render_key_finding(
    fast_rate: float,
    slow_rate: float,
    fast_bucket: str,
    slow_bucket: str,
    is_significant: bool,
    multiplier: float
) -> None:
    """
    Render the primary conclusion using methodical, first-principles logic.
    """
    diff_pp = (fast_rate - slow_rate) * 100
    
    if is_significant and diff_pp > 2:
        st.success(f"""
        ## The Central Finding
        
        **Response speed is causally linked to conversion outcomes.**
        
        The evidence supports a clear conclusion: leads that receive faster responses 
        convert at meaningfully higher rates. This is not merely correlation — we have 
        controlled for confounding variables and tested against chance.
        
        | Metric | Fast Response ({fast_bucket}) | Slow Response ({slow_bucket}) | Difference |
        |:-------|:------------------------------|:------------------------------|:-----------|
        | **Conversion Rate** | {fast_rate*100:.1f}% | {slow_rate*100:.1f}% | +{diff_pp:.1f} pp |
        | **Per 100 Leads** | {int(fast_rate*100)} conversions | {int(slow_rate*100)} conversions | +{int(diff_pp)} |
        | **Per 1,000 Leads** | {int(fast_rate*1000)} conversions | {int(slow_rate*1000)} conversions | +{int(diff_pp*10)} |
        
        **The Magnitude:** Fast responders convert at {multiplier:.1f}× the rate of slow responders.
        
        **The Implication:** Every hour of delay in response time represents lost conversion potential. 
        The data suggests that investments in response speed infrastructure would yield measurable returns.
        """)
    elif is_significant:
        st.info(f"""
        ## The Central Finding
        
        **A modest but statistically reliable effect exists.**
        
        Fast responders ({fast_bucket}) convert at {fast_rate*100:.1f}%, compared to 
        {slow_rate*100:.1f}% for slow responders ({slow_bucket}). The difference of 
        {diff_pp:.1f} percentage points, while not dramatic, is statistically significant.
        
        **Interpretation:** The effect is real, but its practical magnitude requires consideration 
        in the context of implementation costs. Small effects can still compound meaningfully at scale.
        """)
    else:
        st.warning(f"""
        ## The Central Finding
        
        **No conclusive effect detected.**
        
        While the data shows an apparent difference of {diff_pp:+.1f} percentage points 
        between fast and slow responders, this difference is not statistically significant.
        
        **Interpretation:** We cannot rule out that this observed difference arose from 
        random sampling variation. Either no true effect exists, or the effect is too small 
        to detect with the current sample size. Additional data collection may resolve this uncertainty.
        """)


# =============================================================================
# WEEKLY TRENDS HELPERS
# =============================================================================

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


# =============================================================================
# ADVANCED ANALYSIS EXPLAINERS
# =============================================================================

def render_mixed_effects_explainer() -> None:
    """
    Explain mixed effects models using first-principles logic.
    """
    with st.expander("🔬 Understanding the Mixed Effects Approach"):
        st.markdown("""
        ### The Problem of Nested Data
        
        Our data has a hierarchical structure: leads are handled by individual sales representatives. 
        This creates a statistical complication that standard analysis ignores.
        
        **Consider the issue:**
        
        Some representatives are inherently better performers — through skill, experience, or territory. 
        These same representatives may also respond faster, perhaps due to better work habits or 
        lighter lead loads. This creates confounding at the representative level.
        
        If we ignore this structure, we cannot distinguish between two possibilities:
        1. Fast responses cause better outcomes
        2. Good representatives both respond fast AND close more deals (but not because of speed)
        
        ### The Solution: Accounting for Representative Effects
        
        A mixed effects model separates the analysis into two components:
        
        **Fixed Effects:** The overall relationship between response time and outcomes — 
        this is what we want to estimate.
        
        **Random Effects:** The variation between individual representatives — 
        this is what we want to remove.
        
        By modeling representative-level variation explicitly, we can estimate the 
        *within-representative* effect of response time. If speed matters when comparing 
        each rep's fast leads to their slow leads, we have stronger evidence of a causal relationship.
        
        ### The ICC (Intraclass Correlation)
        
        The ICC tells us: what fraction of total outcome variation is attributable to 
        representative-level differences? A high ICC means representatives differ substantially, 
        making mixed effects modeling essential. A low ICC means representative effects are minimal.
        """)


def render_within_rep_explainer() -> None:
    """
    Explain within-representative analysis using first-principles logic.
    """
    with st.expander("🔬 The Power of Within-Person Comparisons"):
        st.markdown("""
        ### The Most Powerful Form of Observational Evidence
        
        Between-person comparisons are always vulnerable to confounding. Different people 
        differ in countless ways — some measured, most unmeasured. Any correlation between 
        exposure and outcome could be driven by these underlying differences.
        
        **Within-person comparisons address this limitation.**
        
        Instead of asking: "Do fast-responding reps close more than slow-responding reps?"
        
        We ask: "When the *same* rep responds fast, do they close more than when that 
        *same* rep responds slowly?"
        
        ### Why This Matters
        
        This approach controls for everything about the person that doesn't change:
        - Their skill level
        - Their territory
        - Their style and approach
        - Their client relationships
        - Unmeasured factors we can't even identify
        
        The only thing that varies is response time. If outcomes still correlate with 
        response time under these conditions, the evidence for a causal effect becomes 
        substantially stronger.
        
        ### The Principle
        
        Each person serves as their own control. This is the closest observational data 
        can come to mimicking an experimental design, where the same unit is observed 
        under different treatment conditions.
        """)


def render_confounding_explainer() -> None:
    """
    Explain confounding assessment using first-principles logic.
    """
    with st.expander("🔬 Assessing the Threat of Confounding"):
        st.markdown("""
        ### The Fundamental Challenge of Causal Inference
        
        When we observe that A correlates with B, three explanations are possible:
        
        1. **A causes B** — the relationship we hope to establish
        2. **B causes A** — reverse causation
        3. **C causes both A and B** — confounding
        
        In our context: fast responses correlate with higher close rates. But does 
        speed cause success? Or does some third factor — lead quality, salesperson skill, 
        time of day — drive both?
        
        ### Assessing Confounding
        
        We approach this systematically:
        
        **1. Identify Potential Confounders**
        
        What variables might be associated with *both* response time and close rate? 
        Common candidates include:
        - Lead source (referrals vs. cold leads)
        - Time of inquiry (business hours vs. after-hours)
        - Representative assignment
        - Lead demographics or firmographics
        
        **2. Test for Association**
        
        For each potential confounder, we assess: Is it associated with response time? 
        Is it associated with close rate? A variable must be associated with both to 
        be a true confounder.
        
        **3. Control and Compare**
        
        We estimate the response time effect with and without controlling for confounders. 
        If the effect remains stable, confounding is likely minimal. If the effect 
        changes dramatically, confounding is present.
        
        ### Interpretation
        
        Perfect elimination of confounding is impossible with observational data. 
        However, if the effect persists across multiple control strategies — regression 
        adjustment, within-rep comparisons, mixed effects — our confidence in a causal 
        interpretation grows substantially.
        """)

