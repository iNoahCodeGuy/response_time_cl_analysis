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
    "Given what we observed, how confident can we be that the association is not due to random chance?"
    
    IMPORTANT: This measures statistical significance of an association, not causal certainty.
    A significant p-value means the association is unlikely to be random, but does NOT prove causation.
    """
    # Cap confidence at 99.9% for readability (99.99 rounds to 100.0% with .1f formatting)
    # Handle edge case where p_value might be 0 or negative
    if p_value <= 0:
        p_value = 0.0001  # Use a very small value to avoid display issues
    
    confidence_pct = min((1 - p_value) * 100, 99.9)
    
    if p_value < 0.0001:
        return {
            'luck_chance': '< 0.01%',
            'confidence': f'{confidence_pct:.1f}%',
            'verdict': 'The pattern is real',
            'emoji': '✅',
            'trust_level': 'Extremely high certainty',
            'plain_english': (
                f"The probability of observing this association by chance alone is vanishingly small — "
                f"less than {p_value*100:.4f}%. To put this in perspective: if we repeated this analysis "
                f"ten thousand times with random data, we would expect to see results this strong "
                f"less than once. This is not random chance. However, this does not prove causation — "
                f"the association could still be explained by unmeasured confounders or selection mechanisms."
            ),
            'first_principles': (
                "When the probability of chance producing our results falls below one in ten thousand, "
                "we can be confident the association is statistically significant. However, statistical "
                "significance measures whether the pattern is real (not random), not whether it is causal."
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
                f"The probability of chance alone producing this association is approximately {p_value*100:.2f}% — "
                f"roughly 1 in {int(1/p_value):,}. When we observe something this unlikely to occur by accident, "
                f"we can be confident we are observing a real association, not statistical noise. "
                f"However, this does not establish causation — the association could still be explained by confounders."
            ),
            'first_principles': (
                "At this level of improbability, we can reject the null hypothesis of no association. "
                "However, this only tells us the association is statistically significant, not that it is causal."
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
                f"that this association occurred by random chance. While not impossible, this is sufficiently "
                f"improbable that we can proceed with confidence that the association is statistically significant. "
                f"However, this does not prove causation."
            ),
            'first_principles': (
                "When something has less than a 1% chance of being random, "
                "we are justified in treating it as a statistically significant association. "
                "This is different from establishing causation, which requires experimental evidence."
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
                f"The probability of this association arising from chance is {p_value*100:.1f}%. "
                f"By conventional scientific standards, this crosses the threshold for statistical significance. "
                f"We can reasonably conclude that a statistically significant association exists. "
                f"However, this does not prove causation — observational data cannot establish causal relationships."
            ),
            'first_principles': (
                "The 5% threshold exists because we accept a small margin of error. "
                "Results below this threshold indicate a statistically significant association. "
                "Establishing causation requires experimental evidence beyond statistical significance."
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
# STEP BRIDGE SENTENCES
# =============================================================================

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


# =============================================================================
# CHI-SQUARE WORKED EXAMPLE GENERATOR
# =============================================================================

def generate_chi_square_worked_example(df: pd.DataFrame) -> dict:
    """
    Generate a complete worked example of the chi-square calculation using actual data.
    
    This function takes the user's actual data and returns all the components needed
    to show a step-by-step walkthrough of the chi-square calculation:
    - Observed counts per bucket
    - Expected counts per bucket (what we'd see if speed didn't matter)
    - Contribution to chi-square per bucket
    - Total chi-square and threshold
    
    Returns:
        dict with keys:
        - 'buckets': list of dicts with per-bucket calculations
        - 'total_chi_square': float
        - 'overall_rate': float
        - 'threshold': float (critical value at alpha=0.05)
        - 'times_threshold': float (how many times over threshold)
        - 'p_value': float
        - 'degrees_of_freedom': int
    """
    import numpy as np
    from scipy import stats
    
    # Calculate overall rate
    overall_rate = df['ordered'].mean()
    total_leads = len(df)
    total_orders = df['ordered'].sum()
    
    # Get bucket-level data
    bucket_data = df.groupby('response_bucket').agg({
        'ordered': ['sum', 'count']
    }).reset_index()
    bucket_data.columns = ['bucket', 'observed_orders', 'total_leads']
    bucket_data['observed_no_orders'] = bucket_data['total_leads'] - bucket_data['observed_orders']
    
    # Calculate expected values (what we'd expect if speed didn't matter)
    bucket_data['expected_orders'] = bucket_data['total_leads'] * overall_rate
    bucket_data['expected_no_orders'] = bucket_data['total_leads'] * (1 - overall_rate)
    
    # Calculate deviation and contribution to chi-square
    bucket_data['difference'] = bucket_data['observed_orders'] - bucket_data['expected_orders']
    bucket_data['contribution_orders'] = (
        (bucket_data['observed_orders'] - bucket_data['expected_orders']) ** 2 
        / bucket_data['expected_orders']
    )
    bucket_data['contribution_no_orders'] = (
        (bucket_data['observed_no_orders'] - bucket_data['expected_no_orders']) ** 2 
        / bucket_data['expected_no_orders']
    )
    bucket_data['total_contribution'] = bucket_data['contribution_orders'] + bucket_data['contribution_no_orders']
    
    # Build the bucket list
    buckets = []
    for _, row in bucket_data.iterrows():
        buckets.append({
            'name': row['bucket'],
            'observed_orders': int(row['observed_orders']),
            'observed_no_orders': int(row['observed_no_orders']),
            'total_leads': int(row['total_leads']),
            'expected_orders': float(row['expected_orders']),
            'expected_no_orders': float(row['expected_no_orders']),
            'difference': float(row['difference']),
            'contribution': float(row['total_contribution']),
            'interpretation': (
                'More sales than expected ↑' if row['difference'] > 10
                else 'Fewer sales than expected ↓' if row['difference'] < -10
                else 'Close to expected'
            )
        })
    
    # Calculate total chi-square
    total_chi_square = bucket_data['total_contribution'].sum()
    
    # Degrees of freedom = (rows - 1) * (cols - 1) = (n_buckets - 1) * (2 - 1) = n_buckets - 1
    degrees_of_freedom = len(bucket_data) - 1
    
    # Critical value at alpha = 0.05
    threshold = stats.chi2.ppf(0.95, degrees_of_freedom)
    
    # P-value
    p_value = 1 - stats.chi2.cdf(total_chi_square, degrees_of_freedom)
    
    return {
        'buckets': buckets,
        'total_chi_square': total_chi_square,
        'overall_rate': overall_rate,
        'total_leads': total_leads,
        'total_orders': int(total_orders),
        'threshold': threshold,
        'times_threshold': total_chi_square / threshold if threshold > 0 else 0,
        'p_value': p_value,
        'degrees_of_freedom': degrees_of_freedom
    }


def render_chi_square_worked_example(worked_example: dict) -> None:
    """
    Render a visible, step-by-step chi-square calculation using the user's actual data.
    
    This is NOT hidden in an expander - it's meant to be the primary explanation
    that shows users exactly how their numbers produce the statistical result.
    """
    st.markdown("### Let's Verify This With Your Data")
    
    st.markdown("""
    The chi-square test asks: *"Is the pattern in our data too strong to be random chance?"*
    
    Let's walk through the calculation using your actual numbers.
    """)
    
    # Step 1: What actually happened
    st.markdown("#### Step 1: What actually happened in your data?")
    
    fastest = worked_example['buckets'][0]
    slowest = worked_example['buckets'][-1]
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        **Your fastest responders ({fastest['name']})**
        - Leads: **{fastest['total_leads']:,}**
        - Closed: **{fastest['observed_orders']:,}**
        - Close rate: **{fastest['observed_orders']:,} ÷ {fastest['total_leads']:,} = {fastest['observed_orders']/fastest['total_leads']*100:.1f}%**
        """)
    
    with col2:
        st.markdown(f"""
        **Your slowest responders ({slowest['name']})**
        - Leads: **{slowest['total_leads']:,}**
        - Closed: **{slowest['observed_orders']:,}**
        - Close rate: **{slowest['observed_orders']:,} ÷ {slowest['total_leads']:,} = {slowest['observed_orders']/slowest['total_leads']*100:.1f}%**
        """)
    
    # Step 2: What would we expect if speed didn't matter?
    st.markdown("#### Step 2: What would we expect if speed didn't matter?")
    
    st.markdown(f"""
    If response time had **no effect**, every bucket would close at the overall average rate:
    
    **Overall close rate:** {worked_example['total_orders']:,} ÷ {worked_example['total_leads']:,} = **{worked_example['overall_rate']*100:.1f}%**
    
    So if speed didn't matter:
    - {fastest['name']} should have: {fastest['total_leads']:,} × {worked_example['overall_rate']*100:.1f}% = **{fastest['expected_orders']:.0f} sales** (not {fastest['observed_orders']:,})
    - {slowest['name']} should have: {slowest['total_leads']:,} × {worked_example['overall_rate']*100:.1f}% = **{slowest['expected_orders']:.0f} sales** (not {slowest['observed_orders']:,})
    """)
    
    # Step 3: Show the full calculation table with actual math
    st.markdown("#### Step 3: Measure the 'surprise' in each bucket")
    
    st.markdown("""
    For each bucket, we calculate: How far is reality from what we'd expect by chance?
    
    The formula is: **(Observed - Expected)² ÷ Expected**
    
    Let's calculate this for each bucket using your actual numbers:
    """)
    
    import pandas as pd
    calc_rows = []
    for bucket in worked_example['buckets']:
        # Calculate the actual formula step by step
        obs = bucket['observed_orders']
        exp = bucket['expected_orders']
        diff = obs - exp
        diff_squared = diff ** 2
        contribution = diff_squared / exp if exp > 0 else 0
        
        calc_rows.append({
            'Response Time': bucket['name'],
            'Observed (O)': f"{obs:,}",
            'Expected (E)': f"{exp:.0f}",
            'Difference (O-E)': f"{diff:+.0f}",
            'Formula: (O-E)²': f"({diff:+.0f})² = {diff_squared:.0f}",
            'Formula: (O-E)²/E': f"{diff_squared:.0f} ÷ {exp:.0f}",
            'Surprise Score': f"{contribution:.2f}",
            '': bucket['interpretation']
        })
    
    st.dataframe(pd.DataFrame(calc_rows), use_container_width=True, hide_index=True)
    
    # Show the actual formula calculation for one example bucket
    if worked_example['buckets']:
        example_bucket = worked_example['buckets'][0]
        obs_ex = example_bucket['observed_orders']
        exp_ex = example_bucket['expected_orders']
        diff_ex = obs_ex - exp_ex
        diff_sq_ex = diff_ex ** 2
        contrib_ex = diff_sq_ex / exp_ex if exp_ex > 0 else 0
        
        st.markdown(f"""
        **Example calculation for {example_bucket['name']}:**
        
        Step 1: Difference = Observed - Expected = {obs_ex:,} - {exp_ex:.0f} = **{diff_ex:+.0f}**
        
        Step 2: Square the difference = ({diff_ex:+.0f})² = **{diff_sq_ex:.0f}**
        
        Step 3: Divide by expected = {diff_sq_ex:.0f} ÷ {exp_ex:.0f} = **{contrib_ex:.2f}**
        
        This bucket contributes **{contrib_ex:.2f}** to the total chi-square statistic.
        """)
    
    # Step 4: The verdict
    st.markdown("#### Step 4: Add up the total surprise")
    
    # Show the sum
    contribution_parts = " + ".join([f"{b['contribution']:.2f}" for b in worked_example['buckets']])
    
    st.markdown(f"""
    **Total surprise score (χ²):** {contribution_parts} = **{worked_example['total_chi_square']:.2f}**
    """)
    
    # Compare to threshold
    p_exp = get_p_value_explanation(worked_example['p_value'])
    
    if worked_example['total_chi_square'] > worked_example['threshold']:
        st.success(f"""
        **The Verdict:**
        
        - Your surprise score: **{worked_example['total_chi_square']:.2f}**
        - Threshold for "too surprising to be random" (at 95% confidence): **{worked_example['threshold']:.2f}**
        - Your score is **{worked_example['times_threshold']:.1f}× higher** than the threshold
        
        ✅ **Conclusion: This pattern is almost certainly real, not random luck.**
        
        The probability of seeing a pattern this strong by pure chance is {p_exp['luck_chance']}.
        """)
    else:
        st.warning(f"""
        **The Verdict:**
        
        - Your surprise score: **{worked_example['total_chi_square']:.2f}**
        - Threshold for "too surprising to be random" (at 95% confidence): **{worked_example['threshold']:.2f}**
        
        ⚠️ **Conclusion: We cannot rule out that this pattern is random variation.**
        
        The data does not provide strong enough evidence to conclude that response time matters.
        """)


def generate_proportion_test_worked_example(
    fast_bucket: str,
    slow_bucket: str, 
    fast_sales: int,
    fast_leads: int,
    slow_sales: int,
    slow_leads: int,
    z_stat: float,
    p_value: float
) -> dict:
    """
    Generate a worked example for the z-test comparing two proportions.
    
    This shows the user exactly how we calculated whether one group
    is significantly better than another.
    """
    import numpy as np
    
    fast_rate = fast_sales / fast_leads if fast_leads > 0 else 0
    slow_rate = slow_sales / slow_leads if slow_leads > 0 else 0
    
    # Calculate pooled proportion (used in z-test)
    pooled = (fast_sales + slow_sales) / (fast_leads + slow_leads)
    
    # Calculate standard error
    se = np.sqrt(pooled * (1 - pooled) * (1/fast_leads + 1/slow_leads))
    
    # Calculate the difference
    diff = fast_rate - slow_rate
    diff_pp = diff * 100
    
    return {
        'fast_bucket': fast_bucket,
        'slow_bucket': slow_bucket,
        'fast_sales': fast_sales,
        'fast_leads': fast_leads,
        'slow_sales': slow_sales,
        'slow_leads': slow_leads,
        'fast_rate': fast_rate,
        'slow_rate': slow_rate,
        'diff': diff,
        'diff_pp': diff_pp,
        'pooled_rate': pooled,
        'standard_error': se,
        'z_stat': z_stat,
        'p_value': p_value,
        'is_significant': p_value < 0.05
    }


# =============================================================================
# CHI-SQUARE WALKTHROUGH (LEGACY - kept for compatibility)
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


# =============================================================================
# WEEKLY DEEP DIVE NARRATIVE GENERATORS
# =============================================================================
# These functions create educational storytelling for week-by-week analysis

def generate_week_analysis_story(
    week_result,
    overall_close_rate: float = None
) -> str:
    """
    Generate an educational narrative story for a single week's analysis.
    
    WHY THIS FUNCTION:
    ------------------
    Non-technical users need the analysis presented as a coherent story,
    not just a collection of statistics. This function weaves the data
    into a narrative that educates while informing.
    
    STRUCTURE:
    ----------
    1. Sets the scene (what happened this week)
    2. Presents the key finding (did speed matter?)
    3. Explains the statistical test (what did we check?)
    4. Interprets the result (what does it mean?)
    5. Provides actionable insight (what should you do?)
    
    PARAMETERS:
    -----------
    week_result : WeeklyDeepDiveResult
        The analysis results for the week
    overall_close_rate : float, optional
        The overall close rate for comparison
        
    RETURNS:
    --------
    str
        A markdown-formatted narrative story
    """
    story_parts = []
    
    # =======================================================================
    # CHAPTER 1: Setting the Scene
    # =======================================================================
    story_parts.append(f"## 📖 The Story of Week {week_result.week_label}\n")
    
    story_parts.append(
        f"During the week of **{week_result.week_label}**, your team received "
        f"**{week_result.n_leads:,} leads** and closed **{week_result.n_orders:,} sales**, "
        f"achieving a **{week_result.close_rate*100:.1f}% close rate**."
    )
    
    # Compare to overall if available
    if overall_close_rate is not None:
        diff = (week_result.close_rate - overall_close_rate) * 100
        if abs(diff) < 0.5:
            story_parts.append(
                f"This was right in line with your overall average ({overall_close_rate*100:.1f}%)."
            )
        elif diff > 0:
            story_parts.append(
                f"This was **{diff:.1f} percentage points above** your overall average "
                f"({overall_close_rate*100:.1f}%) — a strong week!"
            )
        else:
            story_parts.append(
                f"This was **{abs(diff):.1f} percentage points below** your overall average "
                f"({overall_close_rate*100:.1f}%) — room for improvement."
            )
    
    story_parts.append("\n")
    
    # =======================================================================
    # CHAPTER 2: Data Quality Check
    # =======================================================================
    if not week_result.has_sufficient_data:
        story_parts.append(
            "### ⚠️ A Note About This Week's Data\n\n"
            f"With only {week_result.n_leads} leads, the statistical tests for this week "
            "should be interpreted with caution. Smaller samples mean more uncertainty. "
            "Think of it like this: if you flip a coin 10 times and get 7 heads, that could easily "
            "be luck. But if you flip it 1,000 times and get 700 heads, you'd be confident "
            "something is unusual. The same principle applies here.\n\n"
        )
    
    # =======================================================================
    # CHAPTER 3: The Response Time Question
    # =======================================================================
    story_parts.append("### 🎯 Did Response Speed Matter This Week?\n\n")
    
    if week_result.chi_square_result is not None:
        chi_sq = week_result.chi_square_result
        p_exp = get_p_value_explanation(chi_sq.p_value)
        
        if chi_sq.is_significant:
            story_parts.append(
                f"**Yes — response speed had a statistically significant relationship with close rate.**\n\n"
                f"We ran a chi-square test (a standard method for detecting relationships between "
                f"categorical variables) and found that response time bucket and close rate are "
                f"**not independent** this week.\n\n"
                f"- **Chi-square statistic**: {chi_sq.statistic:.2f}\n"
                f"- **P-value**: {chi_sq.p_value:.4f}\n"
                f"- **What this means**: {p_exp['plain_english']}\n\n"
            )
        else:
            story_parts.append(
                f"**Not conclusively — the relationship was not statistically significant this week.**\n\n"
                f"We ran a chi-square test and found that we cannot confidently say response time "
                f"affected close rates during this specific week.\n\n"
                f"- **Chi-square statistic**: {chi_sq.statistic:.2f}\n"
                f"- **P-value**: {chi_sq.p_value:.4f}\n"
                f"- **What this means**: {p_exp['plain_english']}\n\n"
                f"This doesn't mean speed doesn't matter — it means this week's data alone doesn't "
                f"provide enough evidence to conclude it matters. With more data (or across all weeks), "
                f"the pattern may become clearer.\n\n"
            )
    else:
        story_parts.append(
            "We couldn't run the statistical test this week due to insufficient data. "
            "More leads are needed to draw reliable conclusions.\n\n"
        )
    
    # =======================================================================
    # CHAPTER 4: The Bucket-by-Bucket View
    # =======================================================================
    if not week_result.close_rates_by_bucket.empty:
        story_parts.append("### 📊 Performance by Response Speed\n\n")
        
        # Find fastest and slowest buckets
        cr_df = week_result.close_rates_by_bucket
        if len(cr_df) >= 2:
            fastest = cr_df.iloc[0]
            slowest = cr_df.iloc[-1]
            
            diff_pp = (fastest['close_rate'] - slowest['close_rate']) * 100
            
            story_parts.append(
                f"Looking at specific response time buckets:\n\n"
                f"- **Fastest ({fastest['bucket']})**: {fastest['close_rate']*100:.1f}% close rate "
                f"({int(fastest['n_orders'])} orders from {int(fastest['n_leads'])} leads)\n"
                f"- **Slowest ({slowest['bucket']})**: {slowest['close_rate']*100:.1f}% close rate "
                f"({int(slowest['n_orders'])} orders from {int(slowest['n_leads'])} leads)\n\n"
            )
            
            if diff_pp > 2:
                story_parts.append(
                    f"Fast responders closed at a **{diff_pp:.1f} percentage point higher rate** "
                    f"than slow responders this week.\n\n"
                )
            elif diff_pp < -2:
                story_parts.append(
                    f"Interestingly, slow responders actually had a **{abs(diff_pp):.1f} percentage point "
                    f"higher rate** this week — though this may be due to sample size variations.\n\n"
                )
            else:
                story_parts.append(
                    f"The difference between fast and slow responders was minimal "
                    f"({diff_pp:+.1f} percentage points) this week.\n\n"
                )
    
    # =======================================================================
    # CHAPTER 5: Warnings and Caveats
    # =======================================================================
    if week_result.warnings:
        story_parts.append("### 📝 Things to Keep in Mind\n\n")
        for warning in week_result.warnings:
            story_parts.append(f"- {warning}\n")
        story_parts.append("\n")
    
    # =======================================================================
    # CHAPTER 6: The Bottom Line
    # =======================================================================
    story_parts.append("### 💡 The Bottom Line for This Week\n\n")
    
    if week_result.chi_square_result and week_result.chi_square_result.is_significant:
        story_parts.append(
            f"The data from {week_result.week_label} supports the idea that response speed matters. "
            f"Faster responses were associated with better outcomes. While one week alone doesn't "
            f"prove causation, this aligns with the broader pattern in your data."
        )
    elif week_result.has_sufficient_data:
        story_parts.append(
            f"The data from {week_result.week_label} doesn't show a statistically significant "
            f"relationship between response speed and close rate. This week may have had "
            f"other factors dominating (lead quality, rep availability, etc.). Look at the "
            f"overall trends to get the full picture."
        )
    else:
        story_parts.append(
            f"This week had limited data, so we can't draw strong conclusions. "
            f"The overall analysis across all weeks provides more reliable insights."
        )
    
    return "\n".join(story_parts)


def generate_week_comparison_story(
    comparison_result,
    overall_close_rate: float = None
) -> str:
    """
    Generate an educational narrative comparing two weeks.
    
    WHY THIS FUNCTION:
    ------------------
    Comparing weeks helps users understand what's changing over time.
    This function tells the story of how performance evolved between
    two specific weeks.
    
    PARAMETERS:
    -----------
    comparison_result : WeekComparisonResult
        The comparison results from compare_two_weeks()
    overall_close_rate : float, optional
        The overall close rate for context
        
    RETURNS:
    --------
    str
        A markdown-formatted comparison narrative
    """
    week1 = comparison_result.week1
    week2 = comparison_result.week2
    change = comparison_result.close_rate_change
    
    story_parts = []
    
    # =======================================================================
    # TITLE
    # =======================================================================
    story_parts.append(
        f"## 📊 Comparing {week1.week_label} vs {week2.week_label}\n\n"
    )
    
    # =======================================================================
    # THE BIG PICTURE
    # =======================================================================
    story_parts.append("### The Big Picture\n\n")
    
    # Volume story
    lead_diff = week2.n_leads - week1.n_leads
    lead_pct = (lead_diff / week1.n_leads * 100) if week1.n_leads > 0 else 0
    
    if abs(lead_pct) < 10:
        volume_story = f"Lead volume stayed relatively stable ({week1.n_leads:,} → {week2.n_leads:,})."
    elif lead_diff > 0:
        volume_story = f"You received **{lead_pct:.0f}% more leads** ({week1.n_leads:,} → {week2.n_leads:,})."
    else:
        volume_story = f"Lead volume dropped by **{abs(lead_pct):.0f}%** ({week1.n_leads:,} → {week2.n_leads:,})."
    
    story_parts.append(volume_story + "\n\n")
    
    # Close rate story
    if abs(change) < 1:
        story_parts.append(
            f"Close rate was essentially flat: **{week1.close_rate*100:.1f}%** → **{week2.close_rate*100:.1f}%** "
            f"(a change of just {change:+.1f} percentage points).\n\n"
        )
    elif change > 0:
        story_parts.append(
            f"Close rate **improved significantly**: **{week1.close_rate*100:.1f}%** → **{week2.close_rate*100:.1f}%** "
            f"(up {change:.1f} percentage points). 🎉\n\n"
        )
    else:
        story_parts.append(
            f"Close rate **declined**: **{week1.close_rate*100:.1f}%** → **{week2.close_rate*100:.1f}%** "
            f"(down {abs(change):.1f} percentage points). ⚠️\n\n"
        )
    
    # =======================================================================
    # WHAT CHANGED IN THE DATA
    # =======================================================================
    story_parts.append("### What Changed?\n\n")
    
    # Look at bucket comparison if available
    if not comparison_result.bucket_comparison.empty:
        bc = comparison_result.bucket_comparison
        
        # Find which bucket changed the most
        if 'change_pp' in bc.columns:
            bc_sorted = bc.dropna(subset=['change_pp']).copy()
            if len(bc_sorted) > 0:
                most_improved = bc_sorted.loc[bc_sorted['change_pp'].idxmax()]
                most_declined = bc_sorted.loc[bc_sorted['change_pp'].idxmin()]
                
                if most_improved['change_pp'] > 2:
                    story_parts.append(
                        f"- **Biggest improvement**: The **{most_improved['bucket']}** bucket "
                        f"improved by {most_improved['change_pp']:.1f} percentage points\n"
                    )
                
                if most_declined['change_pp'] < -2:
                    story_parts.append(
                        f"- **Biggest decline**: The **{most_declined['bucket']}** bucket "
                        f"dropped by {abs(most_declined['change_pp']):.1f} percentage points\n"
                    )
    
    story_parts.append("\n")
    
    # =======================================================================
    # STATISTICAL SIGNIFICANCE STORY
    # =======================================================================
    story_parts.append("### What the Statistics Tell Us\n\n")
    
    week1_sig = week1.chi_square_result.is_significant if week1.chi_square_result else None
    week2_sig = week2.chi_square_result.is_significant if week2.chi_square_result else None
    
    if comparison_result.significance_changed:
        if week1_sig and not week2_sig:
            story_parts.append(
                f"**Interesting shift**: In {week1.week_label}, response time had a statistically "
                f"significant relationship with close rate. In {week2.week_label}, this relationship "
                f"was no longer significant.\n\n"
                f"**What does this mean?** It could indicate:\n"
                f"- Other factors (like lead quality) became more important\n"
                f"- The sample size was smaller, making patterns harder to detect\n"
                f"- Random variation masked the underlying relationship\n\n"
            )
        elif not week1_sig and week2_sig:
            story_parts.append(
                f"**Interesting shift**: In {week1.week_label}, we couldn't detect a significant "
                f"relationship between response time and close rate. In {week2.week_label}, "
                f"the relationship became statistically significant.\n\n"
                f"**What does this mean?** The impact of response speed on conversions became "
                f"more pronounced — possibly due to lead mix changes or team behavior changes.\n\n"
            )
    else:
        if week1_sig:
            story_parts.append(
                f"Response time showed a **consistent significant relationship** with close rate "
                f"in both weeks. The pattern is stable and likely reflects a real effect.\n\n"
            )
        elif week1_sig is False:
            story_parts.append(
                f"In both weeks, response time did **not** show a statistically significant "
                f"relationship with close rate. Either the effect is small, or other factors "
                f"are dominating outcomes.\n\n"
            )
    
    # =======================================================================
    # ACTIONABLE INSIGHTS
    # =======================================================================
    story_parts.append("### 💡 What This Means for You\n\n")
    
    if change > 2:
        story_parts.append(
            f"The improvement from {week1.week_label} to {week2.week_label} is meaningful. "
            f"Look at what changed: Did response times improve? Did lead quality change? "
            f"Understanding what drove this improvement can help you replicate it.\n"
        )
    elif change < -2:
        story_parts.append(
            f"The decline from {week1.week_label} to {week2.week_label} warrants attention. "
            f"Investigate what changed: Were response times slower? Were there different "
            f"lead sources? Understanding the root cause can help you course-correct.\n"
        )
    else:
        story_parts.append(
            f"Performance was relatively stable between these weeks. Continue monitoring — "
            f"consistent performance is good, but look for opportunities to improve.\n"
        )
    
    return "\n".join(story_parts)


def get_week_educational_context(test_name: str) -> str:
    """
    Get educational context explaining why a specific test matters for weekly analysis.
    
    WHY THIS FUNCTION:
    ------------------
    When users drill into a specific week, they may wonder why we're running
    these tests on a single week's data. This provides context.
    
    PARAMETERS:
    -----------
    test_name : str
        One of: 'chi_square', 'proportions', 'regression', 'close_rates'
        
    RETURNS:
    --------
    str
        Educational context for the test in weekly analysis
    """
    contexts = {
        'chi_square': """
**Why run a chi-square test on one week?**

The chi-square test tells us whether response time and close rate are related *during this specific week*.

Think of it this way: Your overall data might show a strong pattern, but individual weeks can vary. 
Some weeks might have the pattern, others might not. By testing each week, you can see:

- Was this a "typical" week where speed mattered?
- Or was this an "unusual" week where other factors dominated?

This helps you understand the consistency of the response time effect over time.
""",
        'proportions': """
**Why compare buckets within a single week?**

Pairwise comparisons show you exactly which response time groups differed from each other this week.

The chi-square test tells you *if* speed mattered. These comparisons tell you *where* it mattered:
- Did fast responders (0-15 min) beat medium responders (15-30 min)?
- Was the biggest gap between fast and slow, or somewhere in between?

This granular view helps you set appropriate response time targets.
""",
        'regression': """
**Why run regression on weekly data?**

Logistic regression helps us understand if the response time effect holds after controlling for lead source.

Different weeks might have different lead mixes. One week might have more referrals; another might 
have more website leads. By controlling for lead source, we're asking: "Among similar lead types, 
did response speed matter this week?"

This gives you a cleaner picture of the true response time effect for this specific period.
""",
        'close_rates': """
**Why look at close rates by bucket for each week?**

Close rates by response time bucket show the raw performance numbers for this specific week.

Before diving into statistical tests, it's important to see the actual data:
- How many leads fell into each response time bucket?
- What were the conversion rates?
- Were there any unusual patterns?

This is the foundation that all the other tests are built upon.
"""
    }
    
    return contexts.get(test_name, "")


def render_week_analysis_educational_intro() -> None:
    """
    Render an educational introduction to weekly deep-dive analysis.
    
    This explains to users why analyzing individual weeks matters
    and how to interpret the results.
    """
    st.markdown("""
    ### 🔍 Understanding Weekly Deep-Dive Analysis
    
    **Why analyze individual weeks?**
    
    Your overall data tells one story, but that story is made up of many chapters — 
    each week being one chapter. By examining individual weeks, you can:
    
    1. **Spot anomalies**: Was there a week that performed unusually well or poorly?
    2. **Test consistency**: Does response time matter every week, or only sometimes?
    3. **Connect to events**: Did a marketing campaign, staffing change, or holiday affect results?
    4. **Track improvement**: Are recent weeks showing better patterns than older ones?
    
    **A word of caution**: Individual weeks have smaller sample sizes than your full dataset. 
    This means more statistical uncertainty. A non-significant result in one week doesn't mean 
    speed doesn't matter — it might just mean that week didn't have enough data to detect the effect.
    
    Think of each week as a "mini experiment." Some will clearly show the pattern, others won't. 
    The overall analysis tells you the true story; weekly analysis tells you how consistent that story is.
    """)


def generate_weekly_chi_square_worked_example(week_result) -> dict:
    """
    Generate a worked example of chi-square calculation for a specific week.
    
    Shows step-by-step calculation using the week's actual data values.
    
    PARAMETERS:
    -----------
    week_result : WeeklyDeepDiveResult
        The week analysis results
        
    RETURNS:
    --------
    dict
        Worked example data with actual calculations
    """
    if not week_result.chi_square_result:
        return None
    
    chi_sq = week_result.chi_square_result
    overall_rate = week_result.close_rate
    total_leads = week_result.n_leads
    total_orders = week_result.n_orders
    
    # Try to get worked example data from chi-square details
    buckets_data = []
    threshold = 0
    times_threshold = 0
    
    if hasattr(chi_sq, 'details') and chi_sq.details:
        worked_ex = chi_sq.details.get('worked_example', {})
        if worked_ex and 'buckets' in worked_ex:
            # Use the worked example data from the chi-square test
            for bucket_info in worked_ex['buckets']:
                buckets_data.append({
                    'name': bucket_info['name'],
                    'total_leads': bucket_info['total_leads'],
                    'observed_orders': bucket_info['observed_orders'],
                    'expected_orders': bucket_info['expected_orders'],
                    'difference': bucket_info['difference'],
                    'contribution': bucket_info['contribution'],
                    'close_rate': bucket_info['observed_orders'] / bucket_info['total_leads'] if bucket_info['total_leads'] > 0 else 0
                })
            threshold = worked_ex.get('threshold', 0)
            times_threshold = worked_ex.get('times_threshold', 0)
        else:
            # Fallback: calculate from close_rates_by_bucket
            cr_df = week_result.close_rates_by_bucket
            for _, row in cr_df.iterrows():
                bucket_name = row['bucket']
                n_leads = int(row['n_leads'])
                n_orders = int(row['n_orders'])
                close_rate = row['close_rate']
                expected_orders = n_leads * overall_rate
                diff = n_orders - expected_orders
                contribution = (diff ** 2) / expected_orders if expected_orders > 0 else 0
                
                buckets_data.append({
                    'name': bucket_name,
                    'total_leads': n_leads,
                    'observed_orders': n_orders,
                    'expected_orders': expected_orders,
                    'difference': diff,
                    'contribution': contribution,
                    'close_rate': close_rate
                })
    
    # Calculate threshold if not available
    if threshold == 0:
        from scipy import stats
        threshold = stats.chi2.ppf(0.95, chi_sq.degrees_of_freedom) if chi_sq.degrees_of_freedom > 0 else 0
        times_threshold = chi_sq.statistic / threshold if threshold > 0 else 0
    
    return {
        'week_label': week_result.week_label,
        'buckets': buckets_data,
        'total_chi_square': chi_sq.statistic,
        'overall_rate': overall_rate,
        'total_leads': total_leads,
        'total_orders': total_orders,
        'threshold': threshold,
        'times_threshold': times_threshold,
        'p_value': chi_sq.p_value,
        'degrees_of_freedom': chi_sq.degrees_of_freedom
    }


def render_weekly_chi_square_worked_example(worked_example: dict) -> None:
    """
    Render step-by-step chi-square calculation for a specific week using actual data.
    
    PARAMETERS:
    -----------
    worked_example : dict
        Generated by generate_weekly_chi_square_worked_example()
    """
    if not worked_example:
        st.info("Chi-square worked example not available for this week.")
        return
    
    st.markdown("### 🔢 Show Your Work: Chi-Square Calculation")
    
    st.markdown(f"""
    Let's see exactly how we calculated the chi-square statistic for **{worked_example['week_label']}** 
    using your actual data values.
    """)
    
    # Step 1: The actual data
    st.markdown("#### Step 1: Your Actual Data This Week")
    
    st.markdown(f"""
    First, here's what actually happened during the week of **{worked_example['week_label']}**:
    - **Total leads**: {worked_example['total_leads']:,}
    - **Total orders**: {worked_example['total_orders']:,}
    - **Overall close rate**: {worked_example['total_orders']:,} ÷ {worked_example['total_leads']:,} = **{worked_example['overall_rate']*100:.1f}%**
    """)
    
    # Show bucket breakdown
    import pandas as pd
    bucket_rows = []
    for bucket in worked_example['buckets']:
        bucket_rows.append({
            'Response Time': bucket['name'],
            'Leads': f"{bucket['total_leads']:,}",
            'Orders': f"{bucket['observed_orders']:,}",
            'Close Rate': f"{bucket['observed_orders']:,} ÷ {bucket['total_leads']:,} = {bucket['close_rate']*100:.1f}%"
        })
    
    st.dataframe(pd.DataFrame(bucket_rows), use_container_width=True, hide_index=True)
    
    # Step 2: What we'd expect
    st.markdown("#### Step 2: What Would We Expect If Speed Didn't Matter?")
    
    st.markdown(f"""
    If response time had **no effect**, every bucket would close at the overall average rate of **{worked_example['overall_rate']*100:.1f}%**.
    
    So we'd expect:
    """)
    
    expected_rows = []
    for bucket in worked_example['buckets']:
        expected_rows.append({
            'Response Time': bucket['name'],
            'Calculation': f"{bucket['total_leads']:,} × {worked_example['overall_rate']*100:.1f}%",
            'Expected Orders': f"{bucket['expected_orders']:.1f}",
            'Actual Orders': f"{bucket['observed_orders']:,}",
            'Difference': f"{bucket['difference']:+.1f}"
        })
    
    st.dataframe(pd.DataFrame(expected_rows), use_container_width=True, hide_index=True)
    
    # Step 3: Calculate surprise score
    st.markdown("#### Step 3: Calculate the 'Surprise Score' for Each Bucket")
    
    st.markdown("""
    The chi-square formula measures how "surprised" we should be by each bucket's deviation:
    
    **Surprise Score = (Observed - Expected)² ÷ Expected**
    """)
    
    surprise_rows = []
    for bucket in worked_example['buckets']:
        diff_squared = bucket['difference'] ** 2
        surprise_rows.append({
            'Response Time': bucket['name'],
            'Difference': f"{bucket['difference']:+.1f}",
            'Difference²': f"{diff_squared:.1f}",
            'Expected': f"{bucket['expected_orders']:.1f}",
            'Surprise Score': f"{diff_squared:.1f} ÷ {bucket['expected_orders']:.1f} = {bucket['contribution']:.2f}"
        })
    
    st.dataframe(pd.DataFrame(surprise_rows), use_container_width=True, hide_index=True)
    
    # Step 4: Total chi-square
    st.markdown("#### Step 4: Add Up All the Surprise Scores")
    
    contribution_parts = " + ".join([f"{b['contribution']:.2f}" for b in worked_example['buckets']])
    
    st.markdown(f"""
    **Total Chi-Square Statistic (χ²):** {contribution_parts} = **{worked_example['total_chi_square']:.2f}**
    """)
    
    # Step 5: Interpretation
    st.markdown("#### Step 5: What Does This Number Mean?")
    
    p_exp = get_p_value_explanation(worked_example['p_value'])
    
    if worked_example['total_chi_square'] > worked_example.get('threshold', 0):
        st.success(f"""
        **The Verdict:**
        
        - Your chi-square statistic: **{worked_example['total_chi_square']:.2f}**
        - Critical value (at 95% confidence): **{worked_example.get('threshold', 0):.2f}**
        - P-value: **{worked_example['p_value']:.4f}**
        
        ✅ **{p_exp['verdict']}**
        
        {p_exp['plain_english']}
        """)
    else:
        st.warning(f"""
        **The Verdict:**
        
        - Your chi-square statistic: **{worked_example['total_chi_square']:.2f}**
        - P-value: **{worked_example['p_value']:.4f}**
        
        ⚠️ **{p_exp['verdict']}**
        
        {p_exp['plain_english']}
        """)


def render_weekly_close_rate_calculations(week_result) -> None:
    """
    Render step-by-step close rate calculations by bucket for a specific week.
    
    Shows the actual arithmetic: orders ÷ leads = close rate
    """
    if week_result.close_rates_by_bucket.empty:
        st.info("Close rate calculations not available for this week.")
        return
    
    st.markdown("### 🔢 Show Your Work: Close Rate Calculations")
    
    st.markdown(f"""
    Let's see how we calculated the close rate for each response time bucket during **{week_result.week_label}**.
    The formula is simple: **Close Rate = Orders ÷ Leads**
    """)
    
    import pandas as pd
    calc_rows = []
    
    for _, row in week_result.close_rates_by_bucket.iterrows():
        bucket = row['bucket']
        n_leads = int(row['n_leads'])
        n_orders = int(row['n_orders'])
        close_rate = row['close_rate']
        
        calc_rows.append({
            'Response Time': bucket,
            'Leads': f"{n_leads:,}",
            'Orders': f"{n_orders:,}",
            'Calculation': f"{n_orders:,} ÷ {n_leads:,}",
            'Close Rate': f"{close_rate*100:.1f}%",
            'Meaning': f"{n_orders} out of {n_leads} leads resulted in a sale"
        })
    
    st.dataframe(pd.DataFrame(calc_rows), use_container_width=True, hide_index=True)
    
    # Show comparison
    if len(week_result.close_rates_by_bucket) >= 2:
        fastest = week_result.close_rates_by_bucket.iloc[0]
        slowest = week_result.close_rates_by_bucket.iloc[-1]
        
        fastest_rate = fastest['close_rate']
        slowest_rate = slowest['close_rate']
        diff_pp = (fastest_rate - slowest_rate) * 100
        
        st.markdown("#### Comparison")
        
        st.markdown(f"""
        **Fastest ({fastest['bucket']}):** {int(fastest['n_orders']):,} ÷ {int(fastest['n_leads']):,} = {fastest_rate*100:.1f}%
        
        **Slowest ({slowest['bucket']}):** {int(slowest['n_orders']):,} ÷ {int(slowest['n_leads']):,} = {slowest_rate*100:.1f}%
        
        **Difference:** {fastest_rate*100:.1f}% - {slowest_rate*100:.1f}% = **{diff_pp:+.1f} percentage points**
        """)


def render_weekly_proportion_test_calculations(week_result) -> None:
    """
    Render step-by-step z-test calculations for pairwise comparisons in a specific week.
    """
    if not week_result.pairwise_results:
        st.info("Proportion test calculations not available for this week.")
        return
    
    st.markdown("### 🔢 Show Your Work: Proportion Test Calculations")
    
    st.markdown("""
    When comparing two buckets, we use a z-test to see if their close rates are significantly different.
    The formula accounts for sample size and calculates a z-score.
    """)
    
    # Show the first significant comparison (or the first one if none are significant)
    key_comparison = None
    for result in week_result.pairwise_results:
        if result.is_significant:
            key_comparison = result
            break
    
    if not key_comparison and week_result.pairwise_results:
        key_comparison = week_result.pairwise_results[0]
    
    if key_comparison and hasattr(key_comparison, 'details') and key_comparison.details:
        details = key_comparison.details
        b1 = details.get('bucket1', {})
        b2 = details.get('bucket2', {})
        
        bucket1_name = b1.get('name', 'Bucket 1')
        bucket2_name = b2.get('name', 'Bucket 2')
        n1 = b1.get('n_leads', 0)
        x1 = b1.get('n_orders', 0)
        p1 = b1.get('close_rate', 0)
        n2 = b2.get('n_leads', 0)
        x2 = b2.get('n_orders', 0)
        p2 = b2.get('close_rate', 0)
        
        # Calculate pooled proportion
        pooled = (x1 + x2) / (n1 + n2) if (n1 + n2) > 0 else 0
        
        st.markdown(f"#### Comparing {bucket1_name} vs {bucket2_name}")
        
        st.markdown(f"""
        **Step 1: Calculate each bucket's close rate**
        
        {bucket1_name}:
        - Orders: **{x1:,}**
        - Leads: **{n1:,}**
        - Close rate: {x1:,} ÷ {n1:,} = **{p1*100:.1f}%**
        
        {bucket2_name}:
        - Orders: **{x2:,}**
        - Leads: **{n2:,}**
        - Close rate: {x2:,} ÷ {n2:,} = **{p2*100:.1f}%**
        
        **Step 2: Calculate the difference**
        
        Difference: {p1*100:.1f}% - {p2*100:.1f}% = **{(p1-p2)*100:+.1f} percentage points**
        """)
        
        # Calculate standard error (simplified explanation)
        import numpy as np
        se = np.sqrt(pooled * (1 - pooled) * (1/n1 + 1/n2))
        z_score = key_comparison.statistic
        
        st.markdown(f"""
        **Step 3: Calculate the pooled proportion and standard error**
        
        Pooled proportion (overall rate across both buckets):
        ({x1:,} + {x2:,}) ÷ ({n1:,} + {n2:,}) = {pooled*100:.1f}%
        
        Standard error: √[{pooled*100:.1f}% × (1 - {pooled*100:.1f}%) × (1/{n1:,} + 1/{n2:,})] = **{se:.4f}**
        
        **Step 4: Calculate z-score**
        
        Z-score = Difference ÷ Standard Error = {(p1-p2):.4f} ÷ {se:.4f} = **{z_score:.2f}**
        """)
        
        # Interpretation
        p_exp = get_p_value_explanation(key_comparison.p_value)
        
        st.markdown(f"""
        **Step 5: Interpret the result**
        
        - Z-score: **{z_score:.2f}**
        - P-value: **{key_comparison.p_value:.4f}**
        - **{p_exp['verdict']}**
        
        {p_exp['plain_english']}
        """)

