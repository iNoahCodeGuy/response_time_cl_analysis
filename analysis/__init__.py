# =============================================================================
# Analysis Package
# =============================================================================
# This package contains all statistical analysis functions:
# - Preprocessing (creating response time buckets)
# - Descriptive statistics (counts, rates, summaries)
# - Statistical tests (chi-square, z-tests)
# - Regression models (logistic regression with controls)
# =============================================================================

from .preprocessing import calculate_response_time, create_response_buckets
from .descriptive import calculate_summary_stats, calculate_close_rates
from .statistical_tests import run_chi_square_test, run_proportion_z_test
from .regression import run_logistic_regression

