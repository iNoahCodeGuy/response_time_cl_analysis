# =============================================================================
# Odds Ratio Explainers
# =============================================================================
# This module provides explanations for odds ratios and effect sizes.
# =============================================================================

import streamlit as st
import pandas as pd
from typing import Dict, Any
# Import at function level to avoid circular imports


def get_effect_size_category(odds_ratio: float) -> str:
    """
    Categorize odds ratio into effect size categories.
    
    Categories based on standard conventions:
    - Negligible: OR very close to 1.0 (0.95 - 1.05)
    - Small: OR 1.05 - 1.5 or 0.67 - 0.95
    - Medium: OR 1.5 - 3.0 or 0.33 - 0.67
    - Large: OR 3.0 - 5.0 or 0.20 - 0.33
    - Exceptional: OR > 5.0 or < 0.20
    """
    abs_or = abs(odds_ratio) if odds_ratio >= 1 else 1 / odds_ratio
    
    if abs_or < 1.05:
        return "negligible"
    elif abs_or < 1.5:
        return "small"
    elif abs_or < 3.0:
        return "medium"
    elif abs_or < 5.0:
        return "large"
    else:
        return "exceptional"


def get_odds_ratio_explanation(odds_ratio: float, ci_lower: float, ci_upper: float, 
                                bucket: str, reference: str = "60+ min") -> Dict[str, Any]:
    """
    Convert an odds ratio into a logical, first-principles explanation with nuanced effect sizes.
    
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
    
    # Categorize effect size
    effect_size = get_effect_size_category(odds_ratio)
    
    # Create concrete example using baseline of 10%
    baseline_rate = 0.10
    if odds_ratio > 0:
        implied_rate = (odds_ratio * baseline_rate) / (1 - baseline_rate + odds_ratio * baseline_rate)
    else:
        implied_rate = baseline_rate
    
    extra_sales_per_100 = (implied_rate - baseline_rate) * 100
    
    # Effect size interpretations
    effect_size_descriptions = {
        "negligible": (
            "This represents a **negligible effect** — the difference is so small as to be "
            "practically meaningless, even if statistically significant."
        ),
        "small": (
            "This represents a **small but potentially meaningful effect**. While modest, "
            "small effects can compound meaningfully at scale, especially in high-volume operations."
        ),
        "medium": (
            "This represents a **moderate effect** — substantial enough to warrant attention "
            "and potentially justify operational changes if causally established."
        ),
        "large": (
            "This represents a **large effect** — strong enough to be considered highly impactful. "
            "If this relationship is causal, it would represent a major driver of conversion outcomes."
        ),
        "exceptional": (
            "This represents an **exceptional effect** — unusually strong and potentially transformative. "
            "Effects of this magnitude are rare and warrant careful investigation to understand the mechanism."
        )
    }
    
    # Create headline with effect size emphasis
    if effect_size == "exceptional":
        headline = f"🚀 {multiplier:.1f}× the odds of success (EXCEPTIONAL)"
    elif effect_size == "large":
        headline = f"⭐ {multiplier:.1f}× the odds of success (LARGE)"
    elif effect_size == "medium":
        headline = f"{multiplier:.1f}× the odds of success (MODERATE)"
    elif effect_size == "small":
        headline = f"{multiplier:.1f}× the odds of success (SMALL)"
    else:
        headline = "Equal odds (negligible difference)" if multiplier == 1.0 else f"{multiplier:.2f}× the odds (NEGLIGIBLE)"
    
    return {
        'multiplier': multiplier,
        'direction': direction,
        'comparison': comparison,
        'is_significant': is_significant,
        'effect_size': effect_size,
        'effect_size_description': effect_size_descriptions.get(effect_size, ""),
        'bucket': bucket,
        'reference': reference,
        'headline': headline,
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
        
        # Import here to avoid circular dependency
        from .p_value import get_p_value_explanation
        p_exp = get_p_value_explanation(row.get('p_value', 0.01))
        
        # Enhanced practical meaning with effect size context
        practical_meaning = exp['example']
        if exp['effect_size'] in ['large', 'exceptional']:
            practical_meaning += f" [{exp['effect_size'].upper()} EFFECT]"
        
        rows.append({
            'Response Time': row['bucket'],
            'Odds Multiplier': exp['headline'],
            'Effect Size': exp['effect_size'].title(),
            'Practical Meaning': practical_meaning,
            'Certainty': f"{p_exp['emoji']} {p_exp['verdict']}"
        })
    
    # Add the reference row
    rows.append({
        'Response Time': '60+ min (reference point)',
        'Odds Multiplier': '1.0× (baseline)',
        'Effect Size': 'Baseline',
        'Practical Meaning': 'All comparisons are made against this group',
        'Certainty': '—'
    })
    
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Add logical summary with effect size interpretation
    st.markdown("""
    **Understanding Effect Sizes:**
    
    - **Negligible (< 1.05×):** Practically meaningless, even if statistically significant
    - **Small (1.05-1.5×):** Modest but potentially meaningful at scale
    - **Medium (1.5-3.0×):** Substantial enough to warrant attention
    - **Large (3.0-5.0×):** Highly impactful if causal
    - **Exceptional (> 5.0×):** Unusually strong, warrants careful investigation
    
    **The Practical Implication:**
    
    Consider the scale of your operation. If you process 1,000 leads monthly and currently 
    respond slowly to most of them, the potential gain from faster response varies by effect size:
    - **Small effects** may compound meaningfully but require careful cost-benefit analysis
    - **Medium effects** typically justify operational changes if causally established
    - **Large/exceptional effects** represent major opportunities but require experimental validation
    
    Remember: Odds ratios quantify association, not causation. Statistical significance tells us 
    the pattern is real, but only experimental evidence can prove that response speed *causes* 
    higher conversion rates.
    """)
