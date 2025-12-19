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

