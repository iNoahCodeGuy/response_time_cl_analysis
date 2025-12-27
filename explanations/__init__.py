# =============================================================================
# Explanations Package
# =============================================================================
# This package provides plain-English explanations for all statistical concepts.
# The goal is to make complex statistics accessible to non-technical users.
#
# DESIGN PHILOSOPHY:
# ------------------
# Every explanation should answer three questions:
# 1. WHAT are we calculating?
# 2. WHY does it matter for the business?
# 3. HOW should we interpret the result?
# =============================================================================

from .templates import EXPLANATION_TEMPLATES, get_explanation
from .formulas import LATEX_FORMULAS, get_formula

# Import from new split modules
from .p_value import get_p_value_explanation, render_p_value_explainer
from .odds_ratio import (
    get_odds_ratio_explanation, 
    render_odds_ratio_table,
    get_effect_size_category
)
from .confidence_intervals import (
    get_ci_explanation, 
    render_ci_explainer, 
    render_percentage_points_explainer
)
from .common import (
    get_step_bridge,
    render_sample_size_guidance,
    render_bucket_sample_sizes,
    render_key_finding,
    format_wow_change,
    render_error_template,
    render_contradiction_explanation,
    detect_non_monotonic_pattern,
    render_non_monotonic_pattern_explanation,
    render_minimal_sample_warning,
    render_ci_significance_contradiction,
    render_interaction_effect_explanation,
    detect_imbalance,
    render_imbalance_warning,
    render_no_controls_explanation,
    render_extreme_rate_guidance,
    detect_narrative_scenario,
    STEP_BRIDGES
)

# Import remaining functions from original explainers.py (to be migrated)
# These will be split into chi_square.py, regression.py, weekly.py, advanced.py in the future
from .explainers import (
    generate_chi_square_worked_example,
    render_chi_square_worked_example,
    generate_proportion_test_worked_example,
    render_chi_square_walkthrough,
    render_regression_explainer,
    render_mixed_effects_explainer,
    render_within_rep_explainer,
    render_confounding_explainer,
    generate_week_analysis_story,
    generate_week_comparison_story,
    get_week_educational_context,
    render_week_analysis_educational_intro,
    generate_weekly_chi_square_worked_example,
    render_weekly_chi_square_worked_example,
    render_weekly_close_rate_calculations,
    render_weekly_proportion_test_calculations
)

# Import verification panels
from .verification_panels import (
    render_chi_square_verification,
    render_z_test_verification,
    render_regression_verification,
    render_effect_size_verification,
    render_ci_verification,
    render_bucketing_verification
)

__all__ = [
    # Templates and formulas
    'EXPLANATION_TEMPLATES',
    'get_explanation',
    'LATEX_FORMULAS',
    'get_formula',
    # P-value explainers
    'get_p_value_explanation',
    'render_p_value_explainer',
    # Odds ratio explainers
    'get_odds_ratio_explanation',
    'render_odds_ratio_table',
    'get_effect_size_category',
    # Confidence interval explainers
    'get_ci_explanation',
    'render_ci_explainer',
    'render_percentage_points_explainer',
    # Common utilities
    'get_step_bridge',
    'render_sample_size_guidance',
    'render_bucket_sample_sizes',
    'render_key_finding',
    'format_wow_change',
    'render_error_template',
    'render_contradiction_explanation',
    'detect_non_monotonic_pattern',
    'render_non_monotonic_pattern_explanation',
    'render_minimal_sample_warning',
    'render_ci_significance_contradiction',
    'render_interaction_effect_explanation',
    'detect_imbalance',
    'render_imbalance_warning',
    'render_no_controls_explanation',
    'render_extreme_rate_guidance',
    'detect_narrative_scenario',
    'STEP_BRIDGES',
    # Chi-square (from explainers.py - to be migrated)
    'generate_chi_square_worked_example',
    'render_chi_square_worked_example',
    'generate_proportion_test_worked_example',
    'render_chi_square_walkthrough',
    # Regression (from explainers.py - to be migrated)
    'render_regression_explainer',
    # Advanced (from explainers.py - to be migrated)
    'render_mixed_effects_explainer',
    'render_within_rep_explainer',
    'render_confounding_explainer',
    # Weekly (from explainers.py - to be migrated)
    'generate_week_analysis_story',
    'generate_week_comparison_story',
    'get_week_educational_context',
    'render_week_analysis_educational_intro',
    'generate_weekly_chi_square_worked_example',
    'render_weekly_chi_square_worked_example',
    'render_weekly_close_rate_calculations',
    'render_weekly_proportion_test_calculations',
    # Verification panels
    'render_chi_square_verification',
    'render_z_test_verification',
    'render_regression_verification',
    'render_effect_size_verification',
    'render_ci_verification',
    'render_bucketing_verification',
]