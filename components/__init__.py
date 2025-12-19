# =============================================================================
# Components Package
# =============================================================================
# This package contains reusable Streamlit UI components:
# - File upload with drag-and-drop
# - Column mapping interface
# - Settings panel (analysis mode toggle)
# - Step-by-step display for explanations
# - Charts and visualizations
# - Results dashboard
# =============================================================================

from .upload import render_upload_section
from .mapping_ui import render_column_mapping
from .settings_panel import render_settings_panel
from .step_display import display_step, display_result_card
from .charts import (
    create_close_rate_chart,
    create_funnel_chart,
    create_heatmap,
    create_forest_plot,
    create_rep_scatter
)
from .results_dashboard import render_results_dashboard

