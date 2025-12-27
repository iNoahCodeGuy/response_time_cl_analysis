# =============================================================================
# Verification Panels Module
# =============================================================================
# This module provides inline verification panels for all statistical tests.
# Each panel shows step-by-step calculations with actual data values so users
# can verify the math independently.
#
# WHY THIS MODULE EXISTS:
# -----------------------
# Users need confidence that calculations are correct. These panels show:
# - Exact input values used
# - Step-by-step formula calculations with actual numbers
# - Intermediate results
# - Final computed values
# - Excel/Python formulas for independent verification
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, Optional, List, Tuple
from analysis.statistical_tests import TestResult
from analysis.regression import RegressionResult
from explanations.formulas import get_formula


def render_chi_square_verification(
    df: pd.DataFrame,
    chi_sq_result: TestResult
) -> None:
    """
    Render verification panel for chi-square test showing step-by-step calculation.
    
    Shows:
    - Contingency table with row/column totals
    - Expected frequency calculation
    - Chi-square contribution per cell
    - Degrees of freedom calculation
    - Excel formulas for verification
    """
    with st.expander("🔍 Verify Chi-Square Calculation", expanded=False):
        st.markdown("### Input Values")
        
        # Get contingency table from details
        if hasattr(chi_sq_result, 'details') and chi_sq_result.details:
            details = chi_sq_result.details
            contingency_dict = details.get('contingency_table', {})
            expected_dict = details.get('expected_frequencies', {})
            
            # Convert to DataFrame for display
            contingency_df = pd.DataFrame(contingency_dict)
            expected_df = pd.DataFrame(expected_dict)
            
            # Calculate totals
            contingency_df.loc['Total'] = contingency_df.sum()
            contingency_df['Total'] = contingency_df.sum(axis=1)
            
            st.markdown("#### Observed Contingency Table")
            st.dataframe(contingency_df, use_container_width=True)
            
            st.markdown("#### Expected Frequencies (if no relationship)")
            expected_df.loc['Total'] = expected_df.sum()
            expected_df['Total'] = expected_df.sum(axis=1)
            st.dataframe(expected_df, use_container_width=True)
            
            # Show calculation for one example cell
            if len(contingency_df) > 1:
                first_bucket = contingency_df.index[0]
                if first_bucket != 'Total':
                    obs_orders = contingency_df.loc[first_bucket, 1] if 1 in contingency_df.columns else 0
                    exp_orders = expected_df.loc[first_bucket, 1] if 1 in expected_df.columns else 0
                    
                    st.markdown("### Step-by-Step Calculation")
                    st.markdown(f"""
                    **Example: {first_bucket} bucket**
                    
                    **Step 1: Observed vs Expected**
                    - Observed orders: **{obs_orders:,}**
                    - Expected orders: **{exp_orders:.0f}**
                    - Difference: {obs_orders:,} - {exp_orders:.0f} = **{obs_orders - exp_orders:+.0f}**
                    
                    **Step 2: Chi-Square Contribution**
                    - Formula: (Observed - Expected)² ÷ Expected
                    - Calculation: ({obs_orders:,} - {exp_orders:.0f})² ÷ {exp_orders:.0f}
                    - = ({obs_orders - exp_orders:+.0f})² ÷ {exp_orders:.0f}
                    - = **{(obs_orders - exp_orders)**2 / exp_orders if exp_orders > 0 else 0:.2f}**
                    """)
            
            # Show degrees of freedom
            st.markdown("### Degrees of Freedom")
            n_buckets = len(contingency_df) - 1  # Exclude Total row
            n_cols = len(contingency_df.columns) - 1  # Exclude Total column
            dof = (n_buckets - 1) * (n_cols - 1)
            
            st.markdown(f"""
            **Formula:** (Number of rows - 1) × (Number of columns - 1)
            
            **Calculation:** ({n_buckets} - 1) × ({n_cols} - 1) = **{dof}**
            
            This matches the reported degrees of freedom: **{chi_sq_result.degrees_of_freedom}**
            """)
            
            # Show total chi-square
            contributions = details.get('contributions', {})
            if contributions:
                st.markdown("### Total Chi-Square Statistic")
                total_chi_sq = chi_sq_result.statistic
                st.markdown(f"""
                **Sum of all contributions:** **{total_chi_sq:.2f}**
                
                **P-value:** {chi_sq_result.p_value:.4f}
                """)
            
            # Excel formulas
            st.markdown("### Verify Independently in Excel")
            st.markdown("""
            **To verify in Excel:**
            
            1. Create a pivot table with response_bucket as rows, ordered as columns
            2. For each cell, calculate: `=(Observed-Expected)^2/Expected`
            3. Sum all cells to get chi-square statistic
            4. Use `=CHISQ.DIST.RT(ChiSquare, DegreesOfFreedom)` for p-value
            """)


def render_z_test_verification(
    z_test_result: TestResult
) -> None:
    """
    Render verification panel for z-test showing pooled proportion and SE calculation.
    
    Shows:
    - p1, p2 calculations (orders/leads for each bucket)
    - Pooled proportion calculation
    - Standard error calculation
    - Final z-score
    - P-value lookup explanation
    """
    with st.expander("🔍 Verify Z-Test Calculation", expanded=False):
        if not hasattr(z_test_result, 'details') or not z_test_result.details:
            st.info("Detailed calculation data not available.")
            return
        
        details = z_test_result.details
        b1 = details.get('bucket1', {})
        b2 = details.get('bucket2', {})
        
        bucket1_name = b1.get('name', 'Bucket 1')
        bucket2_name = b2.get('name', 'Bucket 2')
        n1 = b1.get('n_leads', 0)
        n2 = b2.get('n_leads', 0)
        x1 = b1.get('n_orders', 0)
        x2 = b2.get('n_orders', 0)
        p1 = b1.get('close_rate', 0)
        p2 = b2.get('close_rate', 0)
        
        st.markdown("### Input Values")
        st.markdown(f"""
        **{bucket1_name}:**
        - Leads: **{n1:,}**
        - Orders: **{x1:,}**
        - Close rate: p₁ = {x1:,} ÷ {n1:,} = **{p1:.4f}** ({p1*100:.2f}%)
        
        **{bucket2_name}:**
        - Leads: **{n2:,}**
        - Orders: **{x2:,}**
        - Close rate: p₂ = {x2:,} ÷ {n2:,} = **{p2:.4f}** ({p2*100:.2f}%)
        """)
        
        st.markdown("### Step-by-Step Calculation")
        
        # Pooled proportion
        total_orders = x1 + x2
        total_leads = n1 + n2
        pooled_p = total_orders / total_leads if total_leads > 0 else 0
        
        st.markdown(f"""
        **Step 1: Calculate Pooled Proportion**
        
        When testing if two proportions are different, we first calculate what the overall 
        proportion would be if we combined both groups:
        
        **Formula:** p̂ = (x₁ + x₂) ÷ (n₁ + n₂)
        
        **Calculation:**
        - Total orders = {x1:,} + {x2:,} = **{total_orders:,}**
        - Total leads = {n1:,} + {n2:,} = **{total_leads:,}**
        - Pooled proportion = {total_orders:,} ÷ {total_leads:,} = **{pooled_p:.4f}** ({pooled_p*100:.2f}%)
        """)
        
        # Standard error
        se = np.sqrt(pooled_p * (1 - pooled_p) * (1/n1 + 1/n2)) if total_leads > 0 else 0
        
        st.markdown(f"""
        **Step 2: Calculate Standard Error**
        
        **Formula:** SE = √[p̂(1-p̂)(1/n₁ + 1/n₂)]
        
        **Calculation:**
        - SE = √[{pooled_p:.4f} × (1 - {pooled_p:.4f}) × (1/{n1:,} + 1/{n2:,})]
        - SE = √[{pooled_p:.4f} × {1-pooled_p:.4f} × {1/n1 + 1/n2:.6f}]
        - SE = √[{pooled_p * (1-pooled_p) * (1/n1 + 1/n2):.6f}]
        - SE = **{se:.6f}**
        """)
        
        # Z-statistic
        z_stat = z_test_result.statistic
        diff = p1 - p2
        
        st.markdown(f"""
        **Step 3: Calculate Z-Statistic**
        
        **Formula:** z = (p₁ - p₂) ÷ SE
        
        **Calculation:**
        - Difference = {p1:.4f} - {p2:.4f} = **{diff:.4f}**
        - z = {diff:.4f} ÷ {se:.6f} = **{z_stat:.2f}**
        """)
        
        # P-value
        p_val = z_test_result.p_value
        
        st.markdown(f"""
        **Step 4: P-Value**
        
        The p-value is the probability of observing a z-score this extreme (or more) 
        if there's no real difference between the groups.
        
        **P-value:** **{p_val:.4f}**
        
        **Interpretation:** {'Significant' if p_val < 0.05 else 'Not significant'} 
        (p {'<' if p_val < 0.05 else '>'} 0.05)
        """)
        
        # Excel formula
        st.markdown("### Verify Independently in Excel")
        st.markdown(f"""
        **Excel formula for z-test:**
        
        ```
        =({x1}/{n1} - {x2}/{n2}) / SQRT(({total_orders}/{total_leads}) * (1 - {total_orders}/{total_leads}) * (1/{n1} + 1/{n2}))
        ```
        
        **P-value formula:**
        ```
        =2*(1-NORM.S.DIST(ABS({z_stat:.2f}), TRUE))
        ```
        """)


def render_regression_verification(
    regression_result: RegressionResult
) -> None:
    """
    Render verification panel for logistic regression showing coefficient to odds ratio conversion.
    
    Shows:
    - Raw coefficient value (beta)
    - How odds ratio is calculated: exp(beta)
    - Confidence interval calculation: exp(beta +/- 1.96 * SE)
    - Reference category explanation
    """
    with st.expander("🔍 Verify Regression Calculation", expanded=False):
        if regression_result.odds_ratios.empty:
            st.info("No odds ratios available for verification.")
            return
        
        st.markdown("### Model Formula")
        st.markdown(f"""
        **Formula:** {regression_result.formula}
        
        **Observations:** {regression_result.n_observations:,}
        **Pseudo R²:** {regression_result.pseudo_r_squared:.4f}
        """)
        
        st.markdown("### Reference Category")
        if regression_result.reference_bucket:
            st.markdown(f"""
            **Reference bucket:** {regression_result.reference_bucket}
            
            All other buckets are compared to this reference. The reference bucket has an 
            odds ratio of 1.0 (baseline).
            """)
        
        st.markdown("### Coefficient to Odds Ratio Conversion")
        st.markdown("""
        In logistic regression, coefficients are on the log-odds scale. We convert them 
        to odds ratios using the exponential function.
        """)
        
        # Show conversion for each bucket
        for _, row in regression_result.odds_ratios.iterrows():
            bucket = row['bucket']
            or_val = row['odds_ratio']
            ci_lower = row['ci_lower']
            ci_upper = row['ci_upper']
            p_val = row['p_value']
            
            # Find corresponding coefficient
            coef_row = regression_result.coefficients[
                regression_result.coefficients['variable'].str.contains(bucket, na=False)
            ]
            
            if not coef_row.empty:
                beta = coef_row.iloc[0]['coefficient']
                se = coef_row.iloc[0]['std_error']
                ci_lower_coef = coef_row.iloc[0]['ci_lower']
                ci_upper_coef = coef_row.iloc[0]['ci_upper']
                
                st.markdown(f"""
                **{bucket} (compared to {regression_result.reference_bucket or 'reference'}):**
                
                **Step 1: Coefficient (log-odds scale)**
                - β = **{beta:.4f}**
                - Standard error = **{se:.4f}**
                
                **Step 2: Convert to Odds Ratio**
                - Formula: OR = exp(β)
                - Calculation: OR = exp({beta:.4f}) = **{or_val:.2f}**
                
                **Step 3: Confidence Interval**
                - Lower bound: exp(β - 1.96 × SE) = exp({beta:.4f} - 1.96 × {se:.4f})
                - = exp({ci_lower_coef:.4f}) = **{ci_lower:.2f}**
                - Upper bound: exp(β + 1.96 × SE) = exp({beta:.4f} + 1.96 × {se:.4f})
                - = exp({ci_upper_coef:.4f}) = **{ci_upper:.2f}**
                
                **P-value:** {p_val:.4f} {'✓ Significant' if p_val < 0.05 else 'Not significant'}
                """)
        
        # Excel formulas
        st.markdown("### Verify Independently")
        st.markdown("""
        **In Python (statsmodels):**
        ```python
        from statsmodels.formula.api import logit
        model = logit('ordered ~ C(response_bucket) + C(lead_source)', data=df).fit()
        print(model.summary())
        ```
        
        **To verify odds ratios:**
        - OR = exp(coefficient)
        - CI = exp(coefficient ± 1.96 × std_error)
        """)


def render_effect_size_verification(
    effect_size: Dict[str, float],
    bucket1_name: str,
    bucket2_name: str
) -> None:
    """
    Render verification panel for effect size showing Cohen's h and odds ratio calculations.
    
    Shows:
    - Absolute difference: p1 - p2
    - Relative difference: (p1 - p2) / p2
    - Odds ratio: (p1/(1-p1)) / (p2/(1-p2))
    - Cohen's h: 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))
    """
    with st.expander("🔍 Verify Effect Size Calculation", expanded=False):
        p1 = effect_size.get('p1', 0)
        p2 = effect_size.get('p2', 0)
        
        st.markdown("### Input Values")
        st.markdown(f"""
        - **{bucket1_name} close rate:** p₁ = **{p1:.4f}** ({p1*100:.2f}%)
        - **{bucket2_name} close rate:** p₂ = **{p2:.4f}** ({p2*100:.2f}%)
        """)
        
        st.markdown("### Step-by-Step Calculations")
        
        # Absolute difference
        abs_diff = effect_size.get('absolute_difference', 0)
        st.markdown(f"""
        **1. Absolute Difference**
        - Formula: p₁ - p₂
        - Calculation: {p1:.4f} - {p2:.4f} = **{abs_diff:.4f}** ({abs_diff*100:.2f} percentage points)
        """)
        
        # Relative difference
        rel_diff = effect_size.get('relative_difference', 0)
        st.markdown(f"""
        **2. Relative Difference**
        - Formula: (p₁ - p₂) ÷ p₂
        - Calculation: ({p1:.4f} - {p2:.4f}) ÷ {p2:.4f} = **{rel_diff:.4f}** ({rel_diff*100:.0f}% relative change)
        """)
        
        # Odds ratio
        odds_ratio = effect_size.get('odds_ratio', 0)
        odds1 = p1 / (1 - p1) if p1 < 1 else float('inf')
        odds2 = p2 / (1 - p2) if p2 < 1 else float('inf')
        
        st.markdown(f"""
        **3. Odds Ratio**
        - Formula: OR = [p₁/(1-p₁)] ÷ [p₂/(1-p₂)]
        - Step 1: Odds for {bucket1_name} = {p1:.4f} ÷ (1 - {p1:.4f}) = {p1:.4f} ÷ {1-p1:.4f} = **{odds1:.4f}**
        - Step 2: Odds for {bucket2_name} = {p2:.4f} ÷ (1 - {p2:.4f}) = {p2:.4f} ÷ {1-p2:.4f} = **{odds2:.4f}**
        - Step 3: Odds ratio = {odds1:.4f} ÷ {odds2:.4f} = **{odds_ratio:.2f}**
        """)
        
        # Cohen's h
        cohens_h = effect_size.get('cohens_h', 0)
        h_interpretation = effect_size.get('cohens_h_interpretation', 'unknown')
        
        sqrt_p1 = np.sqrt(p1)
        sqrt_p2 = np.sqrt(p2)
        arcsin_p1 = np.arcsin(sqrt_p1)
        arcsin_p2 = np.arcsin(sqrt_p2)
        
        st.markdown(f"""
        **4. Cohen's h (Effect Size)**
        - Formula: h = 2 × [arcsin(√p₁) - arcsin(√p₂)]
        - Step 1: √p₁ = √{p1:.4f} = **{sqrt_p1:.4f}**
        - Step 2: arcsin(√p₁) = arcsin({sqrt_p1:.4f}) = **{arcsin_p1:.4f}** radians
        - Step 3: √p₂ = √{p2:.4f} = **{sqrt_p2:.4f}**
        - Step 4: arcsin(√p₂) = arcsin({sqrt_p2:.4f}) = **{arcsin_p2:.4f}** radians
        - Step 5: h = 2 × ({arcsin_p1:.4f} - {arcsin_p2:.4f}) = 2 × {arcsin_p1 - arcsin_p2:.4f} = **{cohens_h:.4f}**
        - **Interpretation:** {h_interpretation} effect
        """)
        
        # Excel formulas
        st.markdown("### Verify Independently")
        st.markdown(f"""
        **Excel formulas:**
        
        - Absolute difference: `={p1:.4f}-{p2:.4f}`
        - Relative difference: `=({p1:.4f}-{p2:.4f})/{p2:.4f}`
        - Odds ratio: `=({p1:.4f}/(1-{p1:.4f}))/({p2:.4f}/(1-{p2:.4f}))`
        - Cohen's h: `=2*(ASIN(SQRT({p1:.4f}))-ASIN(SQRT({p2:.4f})))`
        """)


def render_ci_verification(
    n: int,
    p: float,
    ci_lower: float,
    ci_upper: float,
    confidence_level: float = 0.95
) -> None:
    """
    Render verification panel for Wilson score confidence interval.
    
    Shows:
    - p-hat, n, z values
    - Numerator calculation
    - Denominator calculation
    - Final bounds
    """
    with st.expander("🔍 Verify Confidence Interval Calculation", expanded=False):
        z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
        
        st.markdown("### Input Values")
        st.markdown(f"""
        - **Sample size:** n = **{n:,}**
        - **Sample proportion:** p̂ = **{p:.4f}** ({p*100:.2f}%)
        - **Confidence level:** {confidence_level*100:.0f}%
        - **Z-score:** z = **{z:.4f}** (for {confidence_level*100:.0f}% confidence)
        """)
        
        st.markdown("### Wilson Score Interval Formula")
        st.markdown("""
        The Wilson score interval is more accurate than normal approximation for proportions.
        
        **Formula:**
        ```
        CI = [p̂ + z²/(2n) ± z√(p̂(1-p̂)/n + z²/(4n²))] / [1 + z²/n]
        ```
        """)
        
        st.markdown("### Step-by-Step Calculation")
        
        # Denominator
        denominator = 1 + z**2 / n
        
        st.markdown(f"""
        **Step 1: Calculate Denominator**
        - Denominator = 1 + z²/n
        - = 1 + {z:.4f}²/{n:,}
        - = 1 + {z**2:.4f}/{n:,}
        - = 1 + {z**2/n:.6f}
        - = **{denominator:.6f}**
        """)
        
        # Center
        center = (p + z**2 / (2 * n)) / denominator
        
        st.markdown(f"""
        **Step 2: Calculate Center**
        - Center = [p̂ + z²/(2n)] / denominator
        - = [{p:.4f} + {z:.4f}²/(2 × {n:,})] / {denominator:.6f}
        - = [{p:.4f} + {z**2:.4f}/(2 × {n:,})] / {denominator:.6f}
        - = [{p:.4f} + {z**2/(2*n):.6f}] / {denominator:.6f}
        - = {p + z**2/(2*n):.6f} / {denominator:.6f}
        - = **{center:.6f}**
        """)
        
        # Spread
        spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
        
        st.markdown(f"""
        **Step 3: Calculate Spread**
        - Spread = z × √[(p̂(1-p̂) + z²/(4n))/n] / denominator
        - = {z:.4f} × √[({p:.4f} × {1-p:.4f} + {z:.4f}²/(4 × {n:,}))/{n:,}] / {denominator:.6f}
        - = {z:.4f} × √[({p*(1-p):.6f} + {z**2/(4*n):.6f})/{n:,}] / {denominator:.6f}
        - = {z:.4f} × √[{p*(1-p) + z**2/(4*n):.6f}/{n:,}] / {denominator:.6f}
        - = {z:.4f} × √[{p*(1-p)/n + z**2/(4*n**2):.8f}] / {denominator:.6f}
        - = {z:.4f} × {np.sqrt((p*(1-p) + z**2/(4*n))/n):.6f} / {denominator:.6f}
        - = **{spread:.6f}**
        """)
        
        # Final bounds
        st.markdown(f"""
        **Step 4: Calculate Bounds**
        - Lower bound = center - spread = {center:.6f} - {spread:.6f} = **{ci_lower:.4f}** ({ci_lower*100:.2f}%)
        - Upper bound = center + spread = {center:.6f} + {spread:.6f} = **{ci_upper:.4f}** ({ci_upper*100:.2f}%)
        """)
        
        # Excel formula
        st.markdown("### Verify Independently")
        st.markdown(f"""
        **Excel formula (Wilson score interval):**
        
        Lower bound:
        ```
        =(({p:.4f}+{z:.4f}^2/(2*{n}))-{z:.4f}*SQRT(({p:.4f}*(1-{p:.4f})+{z:.4f}^2/(4*{n}))/{n}))/(1+{z:.4f}^2/{n})
        ```
        
        Upper bound:
        ```
        =(({p:.4f}+{z:.4f}^2/(2*{n}))+{z:.4f}*SQRT(({p:.4f}*(1-{p:.4f})+{z:.4f}^2/(4*{n}))/{n}))/(1+{z:.4f}^2/{n})
        ```
        """)


def render_bucketing_verification(
    df: pd.DataFrame,
    bucket_boundaries: List[float] = None,
    bucket_labels: List[str] = None
) -> None:
    """
    Render verification panel for response time bucketing showing sample leads.
    
    Shows:
    - Bucket boundaries
    - Sample of 5-10 leads with their calculated response times
    - Which bucket each was assigned to and why
    """
    with st.expander("🔍 Verify Response Time Bucketing", expanded=False):
        from config.settings import DEFAULT_BUCKETS, DEFAULT_BUCKET_LABELS
        
        if bucket_boundaries is None:
            bucket_boundaries = DEFAULT_BUCKETS
        if bucket_labels is None:
            bucket_labels = DEFAULT_BUCKET_LABELS
        
        st.markdown("### Bucket Boundaries")
        st.markdown("""
        Each lead is assigned to a bucket based on its response time (in minutes):
        """)
        
        boundary_table = []
        for i in range(len(bucket_labels)):
            lower = bucket_boundaries[i]
            upper = bucket_boundaries[i+1]
            label = bucket_labels[i]
            
            if upper == float('inf'):
                boundary_str = f"{lower:.0f}+ minutes"
            else:
                boundary_str = f"{lower:.0f} to {upper:.0f} minutes"
            
            boundary_table.append({
                'Bucket': label,
                'Boundaries': boundary_str,
                'Rule': f"{lower:.0f} ≤ response_time < {upper:.0f}" if upper != float('inf') else f"response_time ≥ {lower:.0f}"
            })
        
        st.dataframe(pd.DataFrame(boundary_table), use_container_width=True, hide_index=True)
        
        st.markdown("### Sample Leads")
        st.markdown("""
        Here are sample leads showing how response times are calculated and assigned to buckets:
        """)
        
        # Get sample leads (non-null response times)
        sample_df = df[df['response_time_mins'].notna()].head(10).copy()
        
        if len(sample_df) > 0:
            sample_data = []
            for _, row in sample_df.iterrows():
                lead_id = row.get('lead_id', f"Lead {_}")
                lead_time = row.get('lead_time', 'N/A')
                response_time = row.get('first_response_time', 'N/A')
                response_mins = row.get('response_time_mins', 0)
                bucket = row.get('response_bucket', 'N/A')
                
                sample_data.append({
                    'Lead ID': str(lead_id),
                    'Lead Time': str(lead_time)[:19] if isinstance(lead_time, pd.Timestamp) else str(lead_time),
                    'Response Time': str(response_time)[:19] if isinstance(response_time, pd.Timestamp) else str(response_time),
                    'Response (minutes)': f"{response_mins:.1f}",
                    'Assigned Bucket': str(bucket),
                    'Calculation': f"({response_time} - {lead_time}) = {response_mins:.1f} min"
                })
            
            st.dataframe(pd.DataFrame(sample_data), use_container_width=True, hide_index=True)
            
            st.markdown("### Verification")
            st.markdown("""
            **To verify bucketing:**
            
            1. Calculate response time: `first_response_time - lead_time` (in minutes)
            2. Check which bucket boundaries the value falls between
            3. Confirm the assigned bucket matches the boundaries
            
            **Example:**
            - If response_time = 25.3 minutes
            - Boundaries: 0-15, 15-30, 30-60, 60+
            - 25.3 falls in the range 15-30, so bucket = "15-30 min"
            """)
        else:
            st.info("No sample leads available for display.")

