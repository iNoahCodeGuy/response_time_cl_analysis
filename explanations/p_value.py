# =============================================================================
# P-Value Explainers
# =============================================================================
# This module provides explanations for p-values and statistical significance.
#
# WHY THIS MODULE EXISTS:
# -----------------------
# P-values are fundamental to understanding statistical results, but they're
# often misunderstood. This module translates p-values into clear, actionable
# interpretations.
# =============================================================================

import streamlit as st
from typing import Dict, Any


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
