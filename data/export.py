# =============================================================================
# Data Export Module
# =============================================================================
# This module provides CSV export functionality for verification.
# Users can download all raw data and calculations to verify independently.
#
# WHY THIS MODULE EXISTS:
# -----------------------
# Users need to verify calculations independently. This module exports:
# - Raw data with all computed fields
# - Bucket summaries with formulas
# - Test results
# - Regression coefficients
# =============================================================================

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import io


def create_verification_csv(
    df: pd.DataFrame,
    chi_sq_result: Optional[Any] = None,
    regression_result: Optional[Any] = None,
    descriptive_stats: Optional[pd.DataFrame] = None
) -> str:
    """
    Create a CSV string containing all data and calculations for verification.
    
    The CSV contains multiple sections separated by headers:
    1. Raw Data - all leads with computed fields
    2. Bucket Summaries - per-bucket statistics
    3. Test Results - chi-square and other test statistics
    4. Regression Coefficients - if regression was run
    
    PARAMETERS:
    -----------
    df : pd.DataFrame
        Preprocessed DataFrame with all computed fields
    chi_sq_result : TestResult, optional
        Chi-square test result
    regression_result : RegressionResult, optional
        Logistic regression result
    descriptive_stats : pd.DataFrame, optional
        Descriptive statistics by bucket
        
    RETURNS:
    --------
    str
        CSV content as string (can be written to file or returned for download)
    """
    output = io.StringIO()
    
    # =========================================================================
    # SECTION 1: RAW DATA
    # =========================================================================
    output.write("=" * 80 + "\n")
    output.write("SECTION 1: RAW DATA\n")
    output.write("=" * 80 + "\n")
    output.write("\n")
    output.write("This section contains all leads with their computed response times and bucket assignments.\n")
    output.write("You can verify:\n")
    output.write("- Response time = first_response_time - lead_time (in minutes)\n")
    output.write("- Bucket assignment matches response time boundaries\n")
    output.write("- Close rate = sum(ordered) / count(leads) for each bucket\n")
    output.write("\n")
    
    # Select key columns for export
    export_cols = []
    for col in ['lead_id', 'lead_time', 'first_response_time', 'response_time_mins', 
                'response_bucket', 'ordered', 'lead_source', 'sales_rep']:
        if col in df.columns:
            export_cols.append(col)
    
    # Add any other columns that exist
    for col in df.columns:
        if col not in export_cols:
            export_cols.append(col)
    
    raw_data = df[export_cols].copy()
    raw_data.to_csv(output, index=False)
    output.write("\n\n")
    
    # =========================================================================
    # SECTION 2: BUCKET SUMMARIES
    # =========================================================================
    output.write("=" * 80 + "\n")
    output.write("SECTION 2: BUCKET SUMMARIES\n")
    output.write("=" * 80 + "\n")
    output.write("\n")
    output.write("This section shows summary statistics for each response time bucket.\n")
    output.write("You can verify:\n")
    output.write("- n_leads = count of leads in bucket\n")
    output.write("- n_orders = sum of 'ordered' column for bucket\n")
    output.write("- close_rate = n_orders / n_leads\n")
    output.write("- expected_orders = n_leads * overall_close_rate (for chi-square)\n")
    output.write("- chi_sq_contribution = (observed - expected)^2 / expected\n")
    output.write("\n")
    
    if descriptive_stats is not None:
        # Use provided descriptive stats
        bucket_summary = descriptive_stats.copy()
    else:
        # Calculate from raw data
        bucket_summary = df.groupby('response_bucket').agg({
            'ordered': ['sum', 'count']
        }).reset_index()
        bucket_summary.columns = ['bucket', 'n_orders', 'n_leads']
        bucket_summary['close_rate'] = bucket_summary['n_orders'] / bucket_summary['n_leads']
    
    # Add expected values for chi-square
    overall_rate = df['ordered'].mean()
    if 'n_leads' in bucket_summary.columns:
        bucket_summary['expected_orders'] = bucket_summary['n_leads'] * overall_rate
        bucket_summary['chi_sq_contribution'] = (
            (bucket_summary['n_orders'] - bucket_summary['expected_orders']) ** 2 
            / bucket_summary['expected_orders']
        )
    
    bucket_summary.to_csv(output, index=False)
    output.write("\n\n")
    
    # =========================================================================
    # SECTION 3: TEST RESULTS
    # =========================================================================
    output.write("=" * 80 + "\n")
    output.write("SECTION 3: STATISTICAL TEST RESULTS\n")
    output.write("=" * 80 + "\n")
    output.write("\n")
    
    if chi_sq_result:
        output.write("Chi-Square Test Results:\n")
        output.write(f"Test Name: {chi_sq_result.test_name}\n")
        output.write(f"Chi-Square Statistic: {chi_sq_result.statistic:.4f}\n")
        output.write(f"Degrees of Freedom: {chi_sq_result.degrees_of_freedom}\n")
        output.write(f"P-Value: {chi_sq_result.p_value:.4f}\n")
        output.write(f"Significant: {'Yes' if chi_sq_result.is_significant else 'No'}\n")
        output.write(f"Alpha Level: {chi_sq_result.alpha}\n")
        output.write("\n")
        
        # Contingency table if available
        if hasattr(chi_sq_result, 'details') and chi_sq_result.details:
            details = chi_sq_result.details
            if 'contingency_table' in details:
                output.write("Contingency Table (Observed):\n")
                contingency_df = pd.DataFrame(details['contingency_table'])
                contingency_df.to_csv(output)
                output.write("\n")
            
            if 'expected_frequencies' in details:
                output.write("Expected Frequencies:\n")
                expected_df = pd.DataFrame(details['expected_frequencies'])
                expected_df.to_csv(output)
                output.write("\n")
    
    output.write("\n")
    
    # =========================================================================
    # SECTION 4: REGRESSION COEFFICIENTS
    # =========================================================================
    if regression_result:
        output.write("=" * 80 + "\n")
        output.write("SECTION 4: LOGISTIC REGRESSION COEFFICIENTS\n")
        output.write("=" * 80 + "\n")
        output.write("\n")
        output.write(f"Model Formula: {regression_result.formula}\n")
        output.write(f"Observations: {regression_result.n_observations:,}\n")
        output.write(f"Pseudo R²: {regression_result.pseudo_r_squared:.4f}\n")
        output.write("\n")
        
        if not regression_result.coefficients.empty:
            output.write("All Coefficients:\n")
            regression_result.coefficients.to_csv(output, index=False)
            output.write("\n")
        
        if not regression_result.odds_ratios.empty:
            output.write("Odds Ratios (Response Buckets):\n")
            regression_result.odds_ratios.to_csv(output, index=False)
            output.write("\n")
    
    # =========================================================================
    # SECTION 5: VERIFICATION FORMULAS
    # =========================================================================
    output.write("=" * 80 + "\n")
    output.write("SECTION 5: EXCEL FORMULAS FOR VERIFICATION\n")
    output.write("=" * 80 + "\n")
    output.write("\n")
    output.write("Use these Excel formulas to verify calculations:\n")
    output.write("\n")
    output.write("Close Rate:\n")
    output.write("  =SUMIF(response_bucket, \"0-15 min\", ordered) / COUNTIF(response_bucket, \"0-15 min\")\n")
    output.write("\n")
    output.write("Chi-Square Contribution (for one bucket):\n")
    output.write("  =((ObservedOrders - ExpectedOrders)^2) / ExpectedOrders\n")
    output.write("\n")
    output.write("P-Value from Chi-Square:\n")
    if chi_sq_result:
        output.write(f"  =CHISQ.DIST.RT({chi_sq_result.statistic:.4f}, {chi_sq_result.degrees_of_freedom})\n")
    output.write("\n")
    output.write("Odds Ratio from Coefficient:\n")
    output.write("  =EXP(coefficient)\n")
    output.write("\n")
    
    return output.getvalue()


def create_verification_dataframes(
    df: pd.DataFrame,
    chi_sq_result: Optional[Any] = None,
    regression_result: Optional[Any] = None,
    descriptive_stats: Optional[pd.DataFrame] = None
) -> Dict[str, pd.DataFrame]:
    """
    Create separate DataFrames for each verification section.
    
    Useful if you want to export to multiple CSV files or Excel sheets.
    
    RETURNS:
    --------
    Dict[str, pd.DataFrame]
        Dictionary with keys: 'raw_data', 'bucket_summary', 'test_results', 
        'regression_coefficients', 'regression_odds_ratios'
    """
    result = {}
    
    # Raw data
    result['raw_data'] = df.copy()
    
    # Bucket summary
    if descriptive_stats is not None:
        result['bucket_summary'] = descriptive_stats.copy()
    else:
        bucket_summary = df.groupby('response_bucket').agg({
            'ordered': ['sum', 'count']
        }).reset_index()
        bucket_summary.columns = ['bucket', 'n_orders', 'n_leads']
        bucket_summary['close_rate'] = bucket_summary['n_orders'] / bucket_summary['n_leads']
        overall_rate = df['ordered'].mean()
        bucket_summary['expected_orders'] = bucket_summary['n_leads'] * overall_rate
        bucket_summary['chi_sq_contribution'] = (
            (bucket_summary['n_orders'] - bucket_summary['expected_orders']) ** 2 
            / bucket_summary['expected_orders']
        )
        result['bucket_summary'] = bucket_summary
    
    # Test results
    if chi_sq_result:
        test_results = pd.DataFrame({
            'test_name': [chi_sq_result.test_name],
            'statistic': [chi_sq_result.statistic],
            'degrees_of_freedom': [chi_sq_result.degrees_of_freedom],
            'p_value': [chi_sq_result.p_value],
            'is_significant': [chi_sq_result.is_significant],
            'alpha': [chi_sq_result.alpha]
        })
        result['test_results'] = test_results
    else:
        result['test_results'] = pd.DataFrame()
    
    # Regression
    if regression_result:
        result['regression_coefficients'] = regression_result.coefficients.copy()
        result['regression_odds_ratios'] = regression_result.odds_ratios.copy()
    else:
        result['regression_coefficients'] = pd.DataFrame()
        result['regression_odds_ratios'] = pd.DataFrame()
    
    return result

