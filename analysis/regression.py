# =============================================================================
# Regression Module
# =============================================================================
# This module performs logistic regression analysis on lead data.
#
# WHY THIS MODULE EXISTS:
# -----------------------
# Simple comparisons don't account for confounding variables.
# For example, if website leads are both:
#   - Slower to respond to (night traffic)
#   - Lower close rate (browsing, not buying)
# Then slow response and low close rate are correlated, but NOT causally.
#
# Logistic regression lets us estimate the effect of response time
# WHILE CONTROLLING for lead source (and other confounders).
#
# MAIN FUNCTIONS:
# ---------------
# - run_logistic_regression(): Main regression with controls
# - get_odds_ratios(): Extract and format odds ratios
# - interpret_regression(): Plain-English interpretation
# =============================================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import logit
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class RegressionResult:
    """
    Container for regression results.
    
    WHY THIS CLASS EXISTS:
    ----------------------
    Packages all regression outputs in a clean, accessible format.
    Makes it easy to display in the UI and generate explanations.
    """
    model_name: str
    formula: str
    n_observations: int
    pseudo_r_squared: float
    coefficients: pd.DataFrame
    odds_ratios: pd.DataFrame
    p_values: Dict[str, float]
    is_response_time_significant: bool
    interpretation: str
    model_object: Any  # The actual statsmodels result


def prepare_regression_data(
    df: pd.DataFrame,
    reference_bucket: Optional[str] = None
) -> pd.DataFrame:
    """
    Prepare data for logistic regression.
    
    WHY THIS FUNCTION:
    ------------------
    Regression requires:
    1. Categorical variables to be dummy-encoded
    2. A reference category for each variable
    3. No missing values in model variables
    
    HOW IT WORKS:
    -------------
    1. Remove rows with missing values
    2. Create dummy variables for response bucket
    3. Create dummy variables for lead source
    4. Return clean DataFrame ready for regression
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        Preprocessed DataFrame
    reference_bucket : str, optional
        Which bucket to use as reference (default: slowest)
        
    RETURNS:
    --------
    pd.DataFrame
        Regression-ready DataFrame with dummy variables
    """
    # Start with copy
    result = df.copy()
    
    # Remove missing values
    cols_needed = ['ordered', 'response_bucket', 'lead_source']
    result = result.dropna(subset=cols_needed)
    
    # Ensure ordered is 0/1
    result['ordered'] = result['ordered'].astype(int)
    
    # Set reference category for response bucket
    # By default, use the slowest bucket (last one) as reference
    if reference_bucket is None:
        buckets_sorted = sorted(result['response_bucket'].dropna().unique(), key=str)
        reference_bucket = str(buckets_sorted[-1]) if buckets_sorted else None
    
    # Convert to categorical with proper reference
    result['response_bucket'] = pd.Categorical(
        result['response_bucket'].astype(str),
        categories=sorted(result['response_bucket'].astype(str).unique(), key=str)
    )
    
    return result


def run_logistic_regression(
    df: pd.DataFrame,
    include_lead_source: bool = True,
    include_sales_rep: bool = False,
    alpha: float = 0.05
) -> RegressionResult:
    """
    Run logistic regression predicting order probability.
    
    WHY THIS REGRESSION:
    --------------------
    We want to know: "After controlling for lead source, does response
    time still predict close rate?"
    
    If yes → response time has an independent effect
    If no → the relationship was spurious (explained by lead source)
    
    HOW IT WORKS (Plain English):
    -----------------------------
    Logistic regression models the LOG-ODDS of ordering.
    
    log(p / (1-p)) = β₀ + β₁×[bucket_1] + β₂×[bucket_2] + ... + β×[source]
    
    Each coefficient tells us how much that variable changes the log-odds
    compared to the reference category.
    
    INTERPRETING ODDS RATIOS:
    -------------------------
    We convert coefficients to odds ratios: OR = exp(β)
    
    OR = 1.0: No effect compared to reference
    OR = 1.5: 50% higher odds of ordering
    OR = 0.5: 50% lower odds of ordering
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        Preprocessed DataFrame
    include_lead_source : bool
        Whether to control for lead source
    include_sales_rep : bool
        Whether to control for sales rep (use with caution - many categories)
    alpha : float
        Significance level
        
    RETURNS:
    --------
    RegressionResult
        Complete regression results with interpretation
        
    EXAMPLE:
    --------
    >>> result = run_logistic_regression(df, include_lead_source=True)
    >>> print(f"Response time significant? {result.is_response_time_significant}")
    >>> print(result.interpretation)
    """
    # Prepare data
    reg_data = prepare_regression_data(df)
    
    # Build formula
    formula_parts = ['ordered ~ C(response_bucket)']
    
    if include_lead_source and 'lead_source' in reg_data.columns:
        formula_parts.append('C(lead_source)')
    
    if include_sales_rep and 'sales_rep' in reg_data.columns:
        # Note: Including sales rep as fixed effect can cause issues
        # if there are many reps or small samples per rep
        n_reps = reg_data['sales_rep'].nunique()
        if n_reps <= 20:  # Only include if reasonable number
            formula_parts.append('C(sales_rep)')
    
    formula = ' + '.join(formula_parts)
    
    # Fit the model
    try:
        model = logit(formula, data=reg_data).fit(disp=0)  # disp=0 suppresses output
    except Exception as e:
        # Return error result
        return RegressionResult(
            model_name="Logistic Regression (FAILED)",
            formula=formula,
            n_observations=len(reg_data),
            pseudo_r_squared=0,
            coefficients=pd.DataFrame(),
            odds_ratios=pd.DataFrame(),
            p_values={},
            is_response_time_significant=False,
            interpretation=f"Model failed to converge: {str(e)}",
            model_object=None
        )
    
    # Extract coefficients
    coef_df = pd.DataFrame({
        'variable': model.params.index,
        'coefficient': model.params.values,
        'std_error': model.bse.values,
        'z_stat': model.tvalues.values,
        'p_value': model.pvalues.values,
        'ci_lower': model.conf_int()[0].values,
        'ci_upper': model.conf_int()[1].values
    })
    
    # Calculate odds ratios
    coef_df['odds_ratio'] = np.exp(coef_df['coefficient'])
    coef_df['or_ci_lower'] = np.exp(coef_df['ci_lower'])
    coef_df['or_ci_upper'] = np.exp(coef_df['ci_upper'])
    
    # Check if response time is significant
    # Look for any response bucket coefficient that's significant
    bucket_rows = coef_df[coef_df['variable'].str.contains('response_bucket')]
    is_significant = (bucket_rows['p_value'] < alpha).any()
    
    # Get p-values as dictionary
    p_values = dict(zip(model.params.index, model.pvalues))
    
    # Create odds ratios DataFrame for response buckets only
    bucket_or = bucket_rows[['variable', 'odds_ratio', 'or_ci_lower', 'or_ci_upper', 'p_value']].copy()
    bucket_or['bucket'] = bucket_or['variable'].str.extract(r'\[T\.(.+?)\]')[0]
    bucket_or = bucket_or[['bucket', 'odds_ratio', 'or_ci_lower', 'or_ci_upper', 'p_value']]
    bucket_or.columns = ['bucket', 'odds_ratio', 'ci_lower', 'ci_upper', 'p_value']
    
    # Generate interpretation
    interpretation = generate_regression_interpretation(
        model, bucket_or, include_lead_source, alpha
    )
    
    return RegressionResult(
        model_name="Logistic Regression",
        formula=formula,
        n_observations=int(model.nobs),
        pseudo_r_squared=model.prsquared,
        coefficients=coef_df,
        odds_ratios=bucket_or,
        p_values=p_values,
        is_response_time_significant=is_significant,
        interpretation=interpretation,
        model_object=model
    )


def generate_regression_interpretation(
    model,
    bucket_or: pd.DataFrame,
    controlled_for_source: bool,
    alpha: float
) -> str:
    """
    Generate a plain-English interpretation of regression results.
    
    WHY THIS FUNCTION:
    ------------------
    Statistical output is hard to read. This converts it to
    business-friendly language.
    """
    interpretation = []
    
    # Overall significance
    bucket_rows = [c for c in model.params.index if 'response_bucket' in c]
    any_significant = any(model.pvalues[c] < alpha for c in bucket_rows)
    
    if any_significant:
        interpretation.append(
            "**Response time IS statistically significant** after controlling for "
            + ("lead source" if controlled_for_source else "other factors") + "."
        )
    else:
        interpretation.append(
            "**Response time is NOT statistically significant** after controlling for "
            + ("lead source" if controlled_for_source else "other factors") + ". "
            "The apparent relationship in the raw data may be explained by confounding."
        )
    
    # Interpret specific buckets
    interpretation.append("\n\n**Odds Ratios (compared to slowest bucket):**")
    
    for _, row in bucket_or.iterrows():
        or_val = row['odds_ratio']
        bucket = row['bucket']
        p_val = row['p_value']
        
        if or_val > 1:
            pct_change = (or_val - 1) * 100
            direction = "higher"
        else:
            pct_change = (1 - or_val) * 100
            direction = "lower"
        
        sig_text = "✓" if p_val < alpha else ""
        
        interpretation.append(
            f"- **{bucket}**: {or_val:.2f}x odds of ordering "
            f"({pct_change:.0f}% {direction}) {sig_text}"
        )
    
    # Model fit
    interpretation.append(
        f"\n\nModel Pseudo R²: {model.prsquared:.3f} "
        "(higher = better fit, but low values are normal for binary outcomes)"
    )
    
    return "\n".join(interpretation)


def compare_models(
    df: pd.DataFrame,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Compare models with and without lead source control.
    
    WHY THIS COMPARISON:
    --------------------
    Shows whether the response time effect changes when we control
    for lead source. If it shrinks substantially, lead source was
    confounding the relationship.
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        Preprocessed DataFrame
    alpha : float
        Significance level
        
    RETURNS:
    --------
    Dict[str, Any]
        Comparison of models with interpretation
    """
    # Model 1: Response bucket only
    model_simple = run_logistic_regression(df, include_lead_source=False)
    
    # Model 2: Response bucket + lead source
    model_controlled = run_logistic_regression(df, include_lead_source=True)
    
    # Compare the fastest bucket's odds ratio
    if len(model_simple.odds_ratios) > 0 and len(model_controlled.odds_ratios) > 0:
        or_simple = model_simple.odds_ratios.iloc[0]['odds_ratio']
        or_controlled = model_controlled.odds_ratios.iloc[0]['odds_ratio']
        
        change = (or_controlled - or_simple) / or_simple * 100
        
        interpretation = (
            f"The odds ratio for the fastest bucket changed from {or_simple:.2f} "
            f"(without controls) to {or_controlled:.2f} (with lead source control). "
        )
        
        if abs(change) > 20:
            interpretation += (
                f"This {abs(change):.0f}% change suggests lead source was "
                f"{'inflating' if change < 0 else 'masking'} the response time effect."
            )
        else:
            interpretation += (
                f"This {abs(change):.0f}% change is modest - lead source explains "
                f"only a small portion of the relationship."
            )
    else:
        interpretation = "Could not compare models - missing data."
    
    return {
        'model_simple': model_simple,
        'model_controlled': model_controlled,
        'interpretation': interpretation
    }


def get_model_summary_table(result: RegressionResult) -> pd.DataFrame:
    """
    Create a summary table of regression results for display.
    
    WHY THIS TABLE:
    ---------------
    Provides a clean, formatted table that can be displayed in Streamlit.
    Focus on the key information users need.
    """
    if result.odds_ratios.empty:
        return pd.DataFrame()
    
    summary = result.odds_ratios.copy()
    
    # Format columns
    summary['OR (95% CI)'] = summary.apply(
        lambda r: f"{r['odds_ratio']:.2f} ({r['ci_lower']:.2f}-{r['ci_upper']:.2f})",
        axis=1
    )
    summary['p-value'] = summary['p_value'].apply(lambda p: f"{p:.4f}")
    summary['Significant'] = summary['p_value'].astype(float) < 0.05
    summary['Significant'] = summary['Significant'].map({True: '✓', False: ''})
    
    # Return formatted columns
    return summary[['bucket', 'OR (95% CI)', 'p-value', 'Significant']]

