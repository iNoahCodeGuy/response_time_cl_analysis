# =============================================================================
# Explanation Templates Module
# =============================================================================
# This module provides structured explanations for statistical concepts using
# first-principles reasoning and methodical logical progression.
#
# EXPLANATION PHILOSOPHY:
# -----------------------
# 1. Begin with the fundamental question being addressed
# 2. Explain WHY this matters before HOW it works
# 3. Build understanding through logical cause-and-effect chains
# 4. Define every term precisely when introduced
# 5. Provide concrete examples to ground abstract concepts
# 6. Never assume prior knowledge
#
# =============================================================================


# =============================================================================
# EXPLANATION TEMPLATES
# =============================================================================

EXPLANATION_TEMPLATES = {
    
    # =========================================================================
    # STEP 1: DESCRIPTIVE STATISTICS
    # =========================================================================
    'descriptive_stats': {
        'title': "Establishing the Foundation",
        'what_it_does': """
Before any statistical inference can proceed, we must first understand the 
characteristics of our data. This foundational step answers:

- What is the total volume of observations available for analysis?
- How are these observations distributed across response time categories?
- What is the baseline conversion rate we are seeking to explain?
- Are there any data quality issues that could compromise our conclusions?
""",
        'why_it_matters': """
Statistical conclusions are only as reliable as the data underlying them. 
A thorough understanding of the data's structure allows us to:

1. Assess whether sample sizes are sufficient for reliable inference
2. Identify potential biases or anomalies that require attention
3. Establish the baseline against which effects will be measured
4. Set realistic expectations for the precision of our estimates
""",
        'first_principles': """
The principle here is simple but essential: before we can determine whether 
one variable affects another, we must first understand each variable in isolation. 
What does the distribution of response times look like? What is the overall 
conversion rate? Only with this foundation can meaningful comparison proceed.
""",
        'key_questions': [
            "Is the sample size sufficient for the precision we require?",
            "Are observations reasonably distributed across categories?",
            "Are there outliers or anomalies that warrant investigation?",
            "Is the baseline conversion rate within expected ranges?"
        ]
    },
    
    # =========================================================================
    # STEP 2: CHI-SQUARE TEST
    # =========================================================================
    'chi_square': {
        'title': "Testing for Association",
        'what_it_does': """
The chi-square test addresses a fundamental question: Is there *any* systematic 
relationship between response time and conversion outcomes?

The test proceeds by:
1. Calculating what the data would look like if response time had no effect
2. Measuring how far the actual data deviates from this null expectation
3. Determining the probability of observing such deviation by chance alone
""",
        'why_it_matters': """
Before we can claim that response time affects outcomes, we must first 
establish that a relationship exists at all. This is the most basic 
question — and answering it incorrectly leads to either:

- **False positives:** Claiming an effect that is merely noise
- **False negatives:** Missing a genuine effect hidden in the data

The chi-square test provides a principled framework for distinguishing 
signal from noise.
""",
        'how_to_interpret': {
            'significant': """
**The association is statistically significant.**

The observed differences in conversion rates across response time categories 
are substantially larger than what we would expect from random sampling variation 
alone. We can conclude with confidence that a relationship exists between 
response time and outcomes.

**Important caveat:** This establishes *association*, not *causation*. The 
relationship may be genuine, or it may be explained by confounding variables 
that correlate with both response time and outcomes.
""",
            'not_significant': """
**The association is not statistically significant.**

The observed differences between categories fall within the range that could 
plausibly arise from random sampling variation. We cannot conclude with 
confidence that a relationship exists.

**Important caveat:** Absence of evidence is not evidence of absence. A true 
effect may exist but be too small to detect with the current sample size. 
Additional data could resolve this uncertainty.
"""
        },
        'first_principles': """
The logic of the chi-square test is elegant in its simplicity. We ask: 
if response time truly had no effect on outcomes, what would the data 
look like? The answer: all response time categories would have 
approximately the same conversion rate — the overall average.

We then measure how far reality deviates from this null expectation. 
Large deviations are unlikely under the null hypothesis, so observing 
them provides evidence against it.
""",
        'caveats': [
            "Establishes association, not causation",
            "Does not quantify the strength or direction of the effect",
            "Does not identify which specific categories differ",
            "Does not control for potential confounding variables"
        ]
    },
    
    # =========================================================================
    # STEP 3: PROPORTION Z-TEST
    # =========================================================================
    'z_test': {
        'title': "Pairwise Comparison of Categories",
        'what_it_does': """
Having established that *some* association exists, we now identify precisely 
*which* categories differ from one another. The z-test for proportions 
compares two specific groups directly.

The test asks: Is the observed difference between these two conversion rates 
larger than what random sampling variation would produce?
""",
        'why_it_matters': """
Knowing that a general association exists is insufficient for action. 
We need to know the specific comparisons that matter:

- Is there a meaningful difference between the fastest and slowest responders?
- Where exactly in the response time spectrum does the benefit appear?
- Are adjacent categories distinguishable, or do differences only emerge at extremes?

These answers inform where to focus operational improvements.
""",
        'how_to_interpret': {
            'significant': """
**The difference is statistically significant.**

The observed gap between these two categories exceeds what we would 
expect from random sampling variation. We can conclude with confidence 
that these categories have genuinely different underlying conversion rates.

This provides specific, actionable intelligence: moving from the slower 
category to the faster category would, all else equal, yield improved outcomes.
""",
            'not_significant': """
**The difference is not statistically significant.**

The observed gap between these categories falls within the range that could 
plausibly result from random sampling variation. We cannot confidently 
distinguish them as genuinely different.

This may indicate that the true difference is negligible, or that our 
sample size is insufficient to detect a modest effect.
"""
        },
        'first_principles': """
The logic proceeds as follows: if both groups were drawn from populations 
with identical conversion rates, what range of differences would we 
expect to observe simply due to sampling variability?

We standardize the observed difference by this expected variability. 
When the standardized value (the z-score) exceeds a critical threshold, 
we conclude the difference is unlikely to be mere sampling noise.
""",
        'caveats': [
            "Multiple comparisons inflate the false positive rate — appropriate adjustment is essential",
            "Statistical significance is distinct from practical significance — a 'significant' difference may still be too small to matter operationally",
            "Pairwise tests do not address confounding — a significant difference may still be spurious"
        ]
    },
    
    # =========================================================================
    # STEP 4: LOGISTIC REGRESSION
    # =========================================================================
    'logistic_regression': {
        'title': "Isolating the Effect: Controlling for Confounding",
        'what_it_does': """
Logistic regression addresses a fundamental limitation of simple comparisons: 
confounding. It estimates the effect of response time *while mathematically 
holding other factors constant*.

The model simultaneously considers multiple predictors — response time 
and lead source, for example — and estimates the independent contribution 
of each. This allows us to ask: among leads from the same source, does 
response time still predict outcomes?
""",
        'why_it_matters': """
Simple correlations can deceive. Consider the possibility that:

1. Certain lead sources (e.g., referrals) have inherently higher conversion rates
2. These same sources receive faster responses due to prioritization
3. Fast responses and high conversion appear correlated — but the true driver is lead source

Without controlling for this confounding, we would incorrectly attribute 
to speed what is actually an artifact of lead quality.

Regression provides the analytical machinery to disentangle these effects.
""",
        'how_to_interpret': {
            'significant': """
**The effect of response time remains significant after adjustment.**

Even after accounting for differences in lead source, response time 
continues to predict conversion outcomes. This substantially strengthens 
the case for a genuine causal relationship.

The odds ratios quantify the magnitude of this effect:
- OR = 1.5: The odds of conversion are 50% higher
- OR = 2.0: The odds of conversion are doubled
- The further from 1.0, the larger the effect
""",
            'not_significant': """
**The effect of response time is not significant after adjustment.**

When we control for lead source, the apparent relationship between 
response time and conversion weakens or disappears. This suggests 
that the observed correlation was partially or entirely confounded — 
a statistical artifact rather than a genuine causal relationship.

This is valuable information: it redirects attention from response 
time optimization to other, more promising interventions.
"""
        },
        'first_principles': """
The logic is fundamentally comparative. Instead of asking "do fast 
responders close more than slow responders?" (which conflates many 
variables), we ask:

"Among referral leads only, do fast responders close more?"
"Among website leads only, do fast responders close more?"

Regression answers these questions simultaneously, producing an 
overall estimate that is adjusted for the distribution of lead sources.
""",
        'caveats': [
            "We can only control for measured confounders — unmeasured variables may still bias results",
            "The model assumes specific functional forms that may not perfectly match reality",
            "Even controlled observational analysis cannot definitively establish causation"
        ]
    },
    
    # =========================================================================
    # STEP 5: MIXED EFFECTS MODEL (ADVANCED)
    # =========================================================================
    'mixed_effects': {
        'title': "Accounting for Hierarchical Structure",
        'what_it_does': """
Our data has an inherent hierarchical structure: leads are nested within 
salespeople. Some representatives may be systematically better performers — 
and these same individuals may also respond more quickly.

The mixed effects model explicitly accounts for this structure by:
1. Estimating the between-representative variation (how much do reps differ?)
2. Isolating the within-representative effect of response time
3. Producing estimates that are not confounded by representative-level differences
""",
        'why_it_matters': """
Consider the confounding mechanism: skilled salespeople may both 
respond quickly *and* close at higher rates — not because speed 
causes success, but because skill drives both behaviors.

Standard regression conflates these effects. Mixed effects modeling 
provides a principled solution by separating:

- **Between-person variation:** Differences in baseline performance across reps
- **Within-person effects:** The impact of response time *for a given rep*

If speed only matters between reps (fast reps are better), the implication 
is that hiring matters more than process. If speed matters within reps 
(each rep performs better when responding fast), the implication is that 
process optimization will yield returns.
""",
        'how_to_interpret': {
            'significant': """
**The within-person effect of response time is significant.**

Even after accounting for systematic differences between representatives, 
faster responses predict better outcomes. This provides stronger evidence 
for a genuine causal mechanism: speed itself appears to matter, not merely 
the traits of people who happen to be fast.
""",
            'not_significant': """
**The effect weakens after accounting for representative differences.**

The apparent effect of response time is partially or wholly explained by 
between-representative confounding. Faster responders may simply be better 
performers overall, with speed being incidental to their success.
"""
        },
        'first_principles': """
The key insight is that between-person comparisons are vulnerable to 
unmeasured confounding, while within-person comparisons are not.

By modeling each representative's baseline performance separately, 
we effectively compare each person to themselves across different 
response time conditions. This is the closest observational data 
can come to mimicking an experimental design.
""",
        'caveats': [
            "Requires sufficient observations per representative for reliable estimation",
            "Assumes representative effects are randomly distributed in the population",
            "Linear approximations may not perfectly capture binary outcome dynamics"
        ]
    },
    
    # =========================================================================
    # STEP 6: WITHIN-REP ANALYSIS (ADVANCED)
    # =========================================================================
    'within_rep': {
        'title': "The Within-Person Test: Strongest Observational Evidence",
        'what_it_does': """
This analysis applies the most powerful technique available for observational 
causal inference: the within-person comparison.

For each salesperson, we compare their outcomes when responding quickly 
versus when responding slowly. Every characteristic of the person that 
does not vary between leads — their skill, experience, territory, style — 
is automatically controlled.

The only variable that changes is response time. Any remaining association 
with outcomes is therefore less likely to be confounded.
""",
        'why_it_matters': """
Between-person comparisons are always vulnerable to unmeasured confounding. 
Different people differ in countless ways, most of which we cannot measure 
or control. Any correlation between exposure and outcome could be driven 
by these underlying differences.

Within-person comparisons solve this problem elegantly. Each person serves 
as their own control. Whatever unmeasured factors make one rep different 
from another are held constant when we compare that rep's fast leads to 
their slow leads.

This is the closest observational data can come to mimicking a randomized 
experiment, where the same unit is observed under different conditions.
""",
        'how_to_interpret': {
            'significant': """
**The within-person effect is statistically significant.**

When individual salespeople respond quickly, they achieve better outcomes 
than when those same individuals respond slowly. This cannot be explained 
by between-person confounding — the comparison holds person-level factors 
constant.

This is compelling evidence that response speed genuinely influences outcomes. 
The mechanism appears to operate at the level of the individual interaction, 
not merely as a correlate of salesperson quality.
""",
            'not_significant': """
**The within-person effect is not statistically significant.**

When we compare each salesperson to themselves, the relationship between 
response time and outcomes weakens or disappears. This suggests that the 
observed correlation may be driven by between-person differences rather 
than a true causal effect of speed.

The implication: who responds may matter more than how quickly they respond.
"""
        },
        'first_principles': """
The fundamental insight is that causation operates at the level of 
individual events, not at the level of average comparisons. If speed 
causes success, then speed should predict success *for the same person* 
across different occasions — not merely in aggregate comparisons that 
conflate person-level and event-level variation.
""",
        'caveats': [
            "Requires each representative to have both fast and slow responses",
            "Sample size is effectively the number of reps with variation, not total observations",
            "Assumes that within-person variation in response time is essentially random"
        ]
    },
    
    # =========================================================================
    # CONFIDENCE INTERVALS
    # =========================================================================
    'confidence_interval': {
        'title': "Quantifying Uncertainty: The Confidence Interval",
        'what_it_does': """
A confidence interval addresses a fundamental limitation of sampling: 
we observe a sample, but we want to know about the population.

When we report "Close rate: 10% (95% CI: 8% - 12%)", we are saying:
- Our point estimate is 10%
- We are 95% confident the true population rate falls between 8% and 12%
- This range reflects the uncertainty inherent in estimating from a sample
""",
        'why_it_matters': """
Point estimates without uncertainty measures are incomplete — and 
potentially misleading. A close rate of 10% could come from:

- 1,000 leads with 100 conversions (narrow interval: ~8.3% to ~11.9%)
- 100 leads with 10 conversions (wide interval: ~5.5% to ~17.4%)

The same point estimate carries vastly different implications depending 
on the precision with which it is estimated. Confidence intervals make 
this precision explicit.

For comparing groups, confidence intervals are particularly valuable:
- Non-overlapping intervals suggest reliably different populations
- Overlapping intervals indicate uncertainty about which group is truly higher
""",
        'first_principles': """
The philosophical foundation is epistemic humility. We cannot know 
the true population parameter with certainty from sample data alone. 
The confidence interval is a principled expression of this limitation.

A 95% confidence interval has a specific frequentist interpretation: 
if we repeated our sampling procedure many times, 95% of the resulting 
intervals would contain the true population value.
"""
    },
    
    # =========================================================================
    # P-VALUES
    # =========================================================================
    'p_value': {
        'title': "The P-Value: Quantifying Evidence Against the Null Hypothesis",
        'what_it_does': """
The p-value addresses a specific question: If the null hypothesis were true 
(i.e., if there were no real effect), what is the probability of observing 
results as extreme as, or more extreme than, what we actually observed?

A small p-value indicates that our observed results would be very unlikely 
under the null hypothesis. This provides evidence — though not proof — 
against the null hypothesis.
""",
        'why_it_matters': """
P-values provide a principled framework for distinguishing signal from noise.

Any sample of data will show some variation. Two groups will almost never 
have exactly identical outcomes, even if they are drawn from identical 
populations. The question is whether observed differences are large enough 
to be meaningful, or small enough to be explained by random sampling variation.

The p-value quantifies this: it tells us how surprised we should be to see 
these results if there were truly no underlying difference.
""",
        'how_to_interpret': {
            'small': """
**p < 0.05: Conventionally significant**

The observed results would be unlikely (less than 5% probable) if the 
null hypothesis were true. By convention, this threshold justifies 
rejecting the null hypothesis and concluding that a real effect exists.

Important: This is not the probability that the null is true. It is 
the probability of the data given the null — a subtle but crucial distinction.
""",
            'large': """
**p ≥ 0.05: Not conventionally significant**

The observed results are plausibly explained by random sampling variation. 
We cannot confidently reject the null hypothesis.

Important: This is not evidence that no effect exists. It is evidence 
that we cannot reliably detect an effect with the current data. A true 
effect may exist but be too small to distinguish from noise.
"""
        },
        'first_principles': """
The logic of hypothesis testing is indirect. We do not prove our hypothesis; 
we attempt to disprove the null. The p-value quantifies how strong that 
attempted disproof is.

A p-value of 0.01 means: "If the null were true, there is only a 1% chance 
we would see data this extreme. Since we did see such data, either we 
witnessed a 1-in-100 event, or the null is false. The latter is more parsimonious."
""",
        'caveats': [
            "P-values indicate statistical significance, not practical importance — small effects can be 'significant' with large samples",
            "Non-significant results do not prove absence of effect — they indicate insufficient evidence",
            "Multiple testing inflates false positive rates — appropriate corrections are essential",
            "The 0.05 threshold is conventional, not sacred — context should inform interpretation"
        ]
    }
}


def get_explanation(
    analysis_type: str,
    is_significant: bool = None,
    p_value: float = None,
    **kwargs
) -> dict:
    """
    Get the appropriate explanation for an analysis type.
    
    PARAMETERS:
    -----------
    analysis_type : str
        One of the keys in EXPLANATION_TEMPLATES
    is_significant : bool, optional
        Whether the result was significant
    p_value : float, optional
        The p-value from the test
    **kwargs
        Additional context for the explanation
        
    RETURNS:
    --------
    dict
        Explanation with all components, customized for the result
        
    EXAMPLE:
    --------
    >>> exp = get_explanation('chi_square', is_significant=True, p_value=0.03)
    >>> print(exp['interpretation'])
    """
    if analysis_type not in EXPLANATION_TEMPLATES:
        return {
            'title': analysis_type,
            'content': f"No explanation available for {analysis_type}"
        }
    
    template = EXPLANATION_TEMPLATES[analysis_type].copy()
    
    # Select the appropriate interpretation based on significance
    if 'how_to_interpret' in template and is_significant is not None:
        if is_significant:
            template['interpretation'] = template['how_to_interpret'].get(
                'significant', template['how_to_interpret'].get('small', '')
            )
        else:
            template['interpretation'] = template['how_to_interpret'].get(
                'not_significant', template['how_to_interpret'].get('large', '')
            )
    
    # Add p-value context if provided
    if p_value is not None:
        template['p_value_context'] = f"The p-value is {p_value:.4f}. "
        if p_value < 0.001:
            template['p_value_context'] += "This is highly significant - the result would be extremely unlikely by chance alone."
        elif p_value < 0.01:
            template['p_value_context'] += "This is very significant - strong evidence against the null hypothesis."
        elif p_value < 0.05:
            template['p_value_context'] += "This is significant at the standard threshold."
        elif p_value < 0.10:
            template['p_value_context'] += "This is marginally significant - suggestive but not conclusive."
        else:
            template['p_value_context'] += "This is not significant - we cannot rule out chance as an explanation."
    
    return template

