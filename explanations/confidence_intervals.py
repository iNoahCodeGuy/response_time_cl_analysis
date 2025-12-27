# =============================================================================
# Confidence Interval Explainers
# =============================================================================
# This module provides explanations for confidence intervals and uncertainty.
# =============================================================================

import streamlit as st
from typing import Dict, Any


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
