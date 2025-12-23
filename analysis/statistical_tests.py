# =============================================================================
# Statistical Tests Module
# =============================================================================
# This module performs statistical hypothesis tests on the lead data.
#
# WHY THIS MODULE EXISTS:
# -----------------------
# Descriptive stats show us patterns, but we need statistical tests to know
# if those patterns are "real" or just random noise.
#
# Key questions we answer:
# 1. Is there ANY relationship between response time and close rate?
# 2. Is bucket A statistically different from bucket B?
# 3. How confident can we be in these differences?
#
# MAIN FUNCTIONS:
# ---------------
# - run_chi_square_test(): Test overall association
# - run_proportion_z_test(): Compare two specific buckets
# - run_pairwise_comparisons(): Compare all bucket pairs
# =============================================================================

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass


@dataclass
class TestResult:
    """
    Container for statistical test results.
    
    WHY THIS CLASS EXISTS:
    ----------------------
    Standardizes how we return test results.
    Makes it easy to display results in the UI.
    """
    test_name: str
    statistic: float
    p_value: float
    degrees_of_freedom: int
    is_significant: bool
    alpha: float
    interpretation: str
    details: Dict[str, Any]


def run_chi_square_test(
    df: pd.DataFrame,
    alpha: float = 0.05
) -> TestResult:
    """
    Run a chi-square test of independence.
    
    WHY THIS TEST:
    --------------
    The chi-square test answers: "Is there ANY association between
    response time bucket and close rate?"
    
    If significant: Response time and close rate are related.
    If not significant: We can't conclude they're related.
    
    HOW IT WORKS (Plain English):
    -----------------------------
    1. We create a table: rows = response buckets, columns = ordered (yes/no)
    2. We calculate what we'd EXPECT if there was no relationship
    3. We compare what we OBSERVED to what we EXPECTED
    4. Large differences → relationship exists → significant result
    
    THE MATH:
    ---------
    χ² = Σ (Observed - Expected)² / Expected
    
    We then compare this to a chi-square distribution to get the p-value.
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        Preprocessed DataFrame with response_bucket and ordered columns
    alpha : float
        Significance level (default 0.05)
        
    RETURNS:
    --------
    TestResult
        Contains statistic, p-value, significance, and interpretation
        
    EXAMPLE:
    --------
    >>> result = run_chi_square_test(df)
    >>> print(f"Chi-square: {result.statistic:.2f}")
    >>> print(f"P-value: {result.p_value:.4f}")
    >>> print(f"Significant? {result.is_significant}")
    """
    # Create contingency table
    # Rows: response buckets
    # Columns: ordered (0/1)
    contingency = pd.crosstab(
        df['response_bucket'],
        df['ordered']
    )
    
    # Run the chi-square test
    # Returns: chi2 statistic, p-value, degrees of freedom, expected frequencies
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    
    # Determine significance
    is_significant = p_value < alpha
    
    # Generate interpretation
    if is_significant:
        interpretation = (
            f"The chi-square test is SIGNIFICANT (p = {p_value:.4f} < {alpha}). "
            f"This means response time bucket and close rate are NOT independent - "
            f"there IS a relationship between how fast you respond and whether "
            f"the customer orders. The differences in close rates across buckets "
            f"are unlikely to be due to random chance alone."
        )
    else:
        interpretation = (
            f"The chi-square test is NOT significant (p = {p_value:.4f} > {alpha}). "
            f"This means we CANNOT conclude that response time affects close rate. "
            f"The differences we see across buckets could reasonably be due to "
            f"random variation. Note: This doesn't prove there's NO effect - "
            f"we may just need more data."
        )
    
    # Create expected frequencies DataFrame for easier access
    expected_df = pd.DataFrame(
        expected,
        index=contingency.index,
        columns=contingency.columns
    )
    
    # Calculate contributions per bucket for transparency
    # (Observed - Expected)² / Expected for each cell
    contributions = ((contingency - expected_df) ** 2 / expected_df)
    
    # Calculate overall close rate
    overall_rate = df['ordered'].mean()
    
    # Calculate the critical value (threshold) at alpha level
    threshold = stats.chi2.ppf(1 - alpha, dof)
    
    # Build detailed worked example for each bucket
    worked_example_buckets = []
    for bucket in contingency.index:
        obs_orders = contingency.loc[bucket, 1] if 1 in contingency.columns else 0
        obs_no_orders = contingency.loc[bucket, 0] if 0 in contingency.columns else 0
        exp_orders = expected_df.loc[bucket, 1] if 1 in expected_df.columns else 0
        exp_no_orders = expected_df.loc[bucket, 0] if 0 in expected_df.columns else 0
        
        total_leads = obs_orders + obs_no_orders
        diff = obs_orders - exp_orders
        
        # Contribution from orders cell
        contrib_orders = contributions.loc[bucket, 1] if 1 in contributions.columns else 0
        contrib_no_orders = contributions.loc[bucket, 0] if 0 in contributions.columns else 0
        total_contrib = contrib_orders + contrib_no_orders
        
        worked_example_buckets.append({
            'name': bucket,
            'observed_orders': int(obs_orders),
            'observed_no_orders': int(obs_no_orders),
            'total_leads': int(total_leads),
            'expected_orders': float(exp_orders),
            'expected_no_orders': float(exp_no_orders),
            'difference': float(diff),
            'contribution': float(total_contrib),
            'interpretation': (
                'More sales than expected ↑' if diff > 10
                else 'Fewer sales than expected ↓' if diff < -10
                else 'Close to expected'
            )
        })
    
    # Package details with enhanced worked example
    details = {
        'contingency_table': contingency.to_dict(),
        'expected_frequencies': expected_df.to_dict(),
        'contributions': contributions.to_dict(),
        'n_observations': len(df),
        'n_buckets': len(contingency),
        'worked_example': {
            'buckets': worked_example_buckets,
            'overall_rate': overall_rate,
            'total_leads': len(df),
            'total_orders': int(df['ordered'].sum()),
            'threshold': threshold,
            'times_threshold': chi2 / threshold if threshold > 0 else 0
        }
    }
    
    return TestResult(
        test_name="Chi-Square Test of Independence",
        statistic=chi2,
        p_value=p_value,
        degrees_of_freedom=dof,
        is_significant=is_significant,
        alpha=alpha,
        interpretation=interpretation,
        details=details
    )


def run_proportion_z_test(
    df: pd.DataFrame,
    bucket1: str,
    bucket2: str,
    alpha: float = 0.05
) -> TestResult:
    """
    Compare close rates between two specific response buckets.
    
    WHY THIS TEST:
    --------------
    The chi-square test tells us IF there's a relationship.
    This test tells us if specific buckets are different.
    
    For example: "Is the close rate for 0-15 min SIGNIFICANTLY higher
    than for 60+ min?"
    
    HOW IT WORKS (Plain English):
    -----------------------------
    1. Calculate close rate for each bucket (p1 and p2)
    2. Calculate the pooled proportion (overall rate)
    3. Calculate standard error
    4. Create z-score: how many standard errors apart are p1 and p2?
    5. Compare to normal distribution for p-value
    
    THE MATH:
    ---------
    z = (p1 - p2) / √[p̂(1-p̂)(1/n1 + 1/n2)]
    
    where p̂ is the pooled proportion
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        Preprocessed DataFrame
    bucket1 : str
        First bucket to compare (e.g., '0-15 min')
    bucket2 : str
        Second bucket to compare (e.g., '60+ min')
    alpha : float
        Significance level
        
    RETURNS:
    --------
    TestResult
        Test results with interpretation
        
    EXAMPLE:
    --------
    >>> result = run_proportion_z_test(df, '0-15 min', '60+ min')
    >>> print(f"Difference significant? {result.is_significant}")
    """
    # Filter to each bucket
    group1 = df[df['response_bucket'] == bucket1]
    group2 = df[df['response_bucket'] == bucket2]
    
    # Check we have data
    if len(group1) == 0 or len(group2) == 0:
        return TestResult(
            test_name=f"Z-Test: {bucket1} vs {bucket2}",
            statistic=np.nan,
            p_value=np.nan,
            degrees_of_freedom=0,
            is_significant=False,
            alpha=alpha,
            interpretation="Cannot perform test: one or both buckets have no data.",
            details={}
        )
    
    # Calculate proportions
    n1 = len(group1)
    n2 = len(group2)
    x1 = group1['ordered'].sum()  # Number of orders in group 1
    x2 = group2['ordered'].sum()  # Number of orders in group 2
    p1 = x1 / n1  # Close rate for group 1
    p2 = x2 / n2  # Close rate for group 2
    
    # Use statsmodels proportions_ztest
    from statsmodels.stats.proportion import proportions_ztest
    
    count = np.array([x1, x2])
    nobs = np.array([n1, n2])
    
    z_stat, p_value = proportions_ztest(count, nobs, alternative='two-sided')
    
    # Determine significance
    is_significant = p_value < alpha
    
    # Calculate the difference and relative difference
    diff = p1 - p2
    relative_diff = (p1 / p2 - 1) * 100 if p2 > 0 else float('inf')
    
    # Generate interpretation
    if is_significant:
        if diff > 0:
            interpretation = (
                f"The difference is SIGNIFICANT (p = {p_value:.4f}). "
                f"Leads in '{bucket1}' have a {p1*100:.1f}% close rate vs "
                f"{p2*100:.1f}% for '{bucket2}'. "
                f"That's {abs(diff)*100:.1f} percentage points higher "
                f"({abs(relative_diff):.0f}% relative increase). "
                f"This difference is unlikely to be due to chance."
            )
        else:
            interpretation = (
                f"The difference is SIGNIFICANT (p = {p_value:.4f}). "
                f"Leads in '{bucket1}' have a {p1*100:.1f}% close rate vs "
                f"{p2*100:.1f}% for '{bucket2}'. "
                f"That's {abs(diff)*100:.1f} percentage points lower. "
                f"This difference is unlikely to be due to chance."
            )
    else:
        interpretation = (
            f"The difference is NOT significant (p = {p_value:.4f}). "
            f"While '{bucket1}' has a {p1*100:.1f}% close rate vs "
            f"{p2*100:.1f}% for '{bucket2}', this {abs(diff)*100:.1f} "
            f"percentage point difference could be due to random variation."
        )
    
    details = {
        'bucket1': {
            'name': bucket1,
            'n_leads': n1,
            'n_orders': int(x1),
            'close_rate': p1
        },
        'bucket2': {
            'name': bucket2,
            'n_leads': n2,
            'n_orders': int(x2),
            'close_rate': p2
        },
        'difference': diff,
        'relative_difference_pct': relative_diff
    }
    
    return TestResult(
        test_name=f"Z-Test for Proportions: {bucket1} vs {bucket2}",
        statistic=z_stat,
        p_value=p_value,
        degrees_of_freedom=0,  # Z-test doesn't have dof in the usual sense
        is_significant=is_significant,
        alpha=alpha,
        interpretation=interpretation,
        details=details
    )


def run_pairwise_comparisons(
    df: pd.DataFrame,
    alpha: float = 0.05,
    adjustment: str = 'bonferroni'
) -> List[TestResult]:
    """
    Compare all pairs of response buckets.
    
    WHY THIS TEST:
    --------------
    We often want to know which specific buckets differ from each other.
    This runs all pairwise comparisons with multiple testing correction.
    
    MULTIPLE TESTING PROBLEM:
    -------------------------
    If we run many tests at α=0.05, we'll get false positives.
    With 4 buckets = 6 pairwise comparisons, we'd expect ~0.3 false positives.
    
    We adjust using Bonferroni correction: divide α by number of comparisons.
    This is conservative but protects against false positives.
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        Preprocessed DataFrame
    alpha : float
        Overall significance level (will be adjusted)
    adjustment : str
        Multiple testing adjustment method ('bonferroni' or 'none')
        
    RETURNS:
    --------
    List[TestResult]
        Results for each pairwise comparison
    """
    # Get unique buckets
    buckets = df['response_bucket'].dropna().unique()
    buckets = sorted(buckets, key=lambda x: str(x))  # Sort for consistent order
    
    # Generate all pairs
    pairs = []
    for i, b1 in enumerate(buckets):
        for b2 in buckets[i+1:]:
            pairs.append((b1, b2))
    
    n_comparisons = len(pairs)
    
    # Adjust alpha for multiple comparisons
    if adjustment == 'bonferroni':
        adjusted_alpha = alpha / n_comparisons
    else:
        adjusted_alpha = alpha
    
    # Run each comparison
    results = []
    for bucket1, bucket2 in pairs:
        result = run_proportion_z_test(df, str(bucket1), str(bucket2), adjusted_alpha)
        
        # Update interpretation to mention adjustment
        if adjustment == 'bonferroni':
            result.interpretation += (
                f" (Using Bonferroni correction: α = {adjusted_alpha:.4f} "
                f"to account for {n_comparisons} comparisons)"
            )
        
        results.append(result)
    
    return results


def run_all_statistical_tests(
    df: pd.DataFrame,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Run all statistical tests and package results.
    
    WHY THIS FUNCTION:
    ------------------
    Main entry point for standard statistical analysis.
    Runs chi-square and pairwise comparisons, packages results.
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        Preprocessed DataFrame
    alpha : float
        Significance level
        
    RETURNS:
    --------
    Dict[str, Any]
        All test results organized by test type
    """
    results = {}
    
    # Chi-square test
    results['chi_square'] = run_chi_square_test(df, alpha)
    
    # Pairwise comparisons
    results['pairwise'] = run_pairwise_comparisons(df, alpha)
    
    # Find the most extreme comparison (fastest vs slowest)
    buckets = sorted(df['response_bucket'].dropna().unique(), key=lambda x: str(x))
    if len(buckets) >= 2:
        results['extreme_comparison'] = run_proportion_z_test(
            df, str(buckets[0]), str(buckets[-1]), alpha
        )
    
    # Summary
    n_significant = sum(1 for r in results['pairwise'] if r.is_significant)
    
    results['summary'] = {
        'overall_significant': results['chi_square'].is_significant,
        'n_pairwise_significant': n_significant,
        'n_pairwise_total': len(results['pairwise']),
        'alpha': alpha
    }
    
    return results


def calculate_effect_size(
    df: pd.DataFrame,
    bucket1: str,
    bucket2: str
) -> Dict[str, float]:
    """
    Calculate effect size measures for the difference between buckets.
    
    WHY EFFECT SIZE MATTERS:
    ------------------------
    P-values tell us IF there's an effect, not HOW BIG it is.
    With large samples, tiny differences become "significant".
    
    Effect size tells us: "Is this difference practically meaningful?"
    
    MEASURES WE CALCULATE:
    ----------------------
    1. Absolute difference: p1 - p2
    2. Relative difference: (p1 - p2) / p2
    3. Odds ratio: [p1/(1-p1)] / [p2/(1-p2)]
    4. Cohen's h: Arcsine difference (standard effect size for proportions)
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        Preprocessed DataFrame
    bucket1, bucket2 : str
        Buckets to compare
        
    RETURNS:
    --------
    Dict[str, float]
        Effect size measures
    """
    group1 = df[df['response_bucket'] == bucket1]
    group2 = df[df['response_bucket'] == bucket2]
    
    p1 = group1['ordered'].mean()
    p2 = group2['ordered'].mean()
    
    # Absolute difference
    abs_diff = p1 - p2
    
    # Relative difference
    rel_diff = (p1 - p2) / p2 if p2 > 0 else float('inf')
    
    # Odds ratio
    odds1 = p1 / (1 - p1) if p1 < 1 else float('inf')
    odds2 = p2 / (1 - p2) if p2 < 1 else float('inf')
    odds_ratio = odds1 / odds2 if odds2 > 0 else float('inf')
    
    # Cohen's h (arcsine transformation)
    # h = 2 * (arcsin(√p1) - arcsin(√p2))
    h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
    
    # Interpretation of Cohen's h
    if abs(h) < 0.2:
        h_interpretation = "small"
    elif abs(h) < 0.5:
        h_interpretation = "medium"
    else:
        h_interpretation = "large"
    
    return {
        'absolute_difference': abs_diff,
        'relative_difference': rel_diff,
        'odds_ratio': odds_ratio,
        'cohens_h': h,
        'cohens_h_interpretation': h_interpretation,
        'p1': p1,
        'p2': p2
    }

