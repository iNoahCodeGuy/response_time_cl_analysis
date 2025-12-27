# =============================================================================
# Application Settings
# =============================================================================
# This file contains all configuration constants for the Response Time Analyzer.
# 
# WHY CENTRALIZE SETTINGS?
# ------------------------
# 1. Easy to change values without hunting through code
# 2. Junior developers can understand app behavior at a glance
# 3. Makes testing easier (can override settings)
#
# HOW TO USE:
# -----------
# from config.settings import DEFAULT_BUCKETS, APP_TITLE
# =============================================================================


# =============================================================================
# APP METADATA
# =============================================================================
# Basic information about the application displayed in the UI

APP_TITLE = "Lead Response Time Analyzer"
APP_ICON = "🚗"  # Displayed in browser tab
APP_DESCRIPTION = """
Analyze how quickly your sales team responds to leads and measure the impact 
on close rates. Upload your data, and we'll walk you through the statistics 
step by step.
"""

# =============================================================================
# RESPONSE TIME BUCKETS
# =============================================================================
# These define how we group response times for analysis.
# 
# WHY THESE BUCKETS?
# ------------------
# - 0-15 min: Industry "speed to lead" best practice
# - 15-30 min: Still responsive
# - 30-60 min: Delayed response
# - 60+ min: Slow response
#
# Users can customize these in the sidebar

DEFAULT_BUCKETS = [0, 15, 30, 60, float('inf')]
DEFAULT_BUCKET_LABELS = ['0-15 min', '15-30 min', '30-60 min', '60+ min']

# Alternative bucket configurations users might want
BUCKET_PRESETS = {
    'standard': {
        'boundaries': [0, 15, 30, 60, float('inf')],
        'labels': ['0-15 min', '15-30 min', '30-60 min', '60+ min'],
        'description': 'Standard industry buckets'
    },
    'aggressive': {
        'boundaries': [0, 5, 15, 30, float('inf')],
        'labels': ['0-5 min', '5-15 min', '15-30 min', '30+ min'],
        'description': 'For high-volume, fast-response teams'
    },
    'relaxed': {
        'boundaries': [0, 30, 60, 120, float('inf')],
        'labels': ['0-30 min', '30-60 min', '1-2 hrs', '2+ hrs'],
        'description': 'For complex B2B sales with longer cycles'
    }
}


# =============================================================================
# STATISTICAL SETTINGS
# =============================================================================
# Parameters for statistical tests

# Significance level (alpha) for hypothesis tests
# 0.05 means we accept a 5% chance of false positives
DEFAULT_ALPHA = 0.05

# Confidence level for confidence intervals
# 0.95 means 95% confidence intervals
DEFAULT_CONFIDENCE_LEVEL = 0.95

# Minimum sample size warnings
# Below these thresholds, we warn users about statistical power
MIN_SAMPLE_SIZE_WARNING = 30  # Per bucket for reliable estimates
MIN_SAMPLE_SIZE_ERROR = 10   # Below this, results are unreliable


# =============================================================================
# ANALYSIS MODES
# =============================================================================
# Define what tests are included in each analysis mode

ANALYSIS_MODES = {
    'standard': {
        'name': 'Standard Analysis',
        'description': 'Essential tests for most use cases',
        'tests': [
            'descriptive_stats',
            'chi_square',
            'z_test_proportions', 
            'logistic_regression'
        ],
        'icon': '📊'
    }
}


# =============================================================================
# FILE UPLOAD SETTINGS
# =============================================================================
# Constraints for file uploads

MAX_FILE_SIZE_MB = 50  # Maximum file size in megabytes
ALLOWED_FILE_TYPES = ['csv', 'xlsx', 'xls']
PREVIEW_ROWS = 5  # Number of rows to show in data preview


# =============================================================================
# COLUMN MAPPING DEFAULTS
# =============================================================================
# Expected column names (we'll try to auto-detect these)

EXPECTED_COLUMNS = {
    'lead_time': {
        'description': 'When the lead came in',
        'type': 'datetime',
        'required': True,
        'common_names': [
            'lead_time', 'lead_date', 'created_at', 'submission_time',
            'lead_created', 'date_created', 'timestamp', 'lead_timestamp',
            'received_time'  # GitHub repo compatibility
        ]
    },
    'response_time': {
        'description': 'When the first response was sent',
        'type': 'datetime', 
        'required': True,
        'common_names': [
            'response_time', 'first_response', 'first_message', 'contacted_at',
            'first_contact', 'response_date', 'first_reply', 'replied_at',
            'first_contact_time'  # GitHub repo compatibility
        ]
    },
    'lead_source': {
        'description': 'Where the lead came from',
        'type': 'categorical',
        'required': False,  # Optional - analysis works but without lead source controls
        'common_names': [
            'lead_source', 'source', 'channel', 'lead_channel', 
            'marketing_source', 'origin', 'lead_origin', 'traffic_source'
        ]
    },
    'sales_rep': {
        'description': 'Who handled the lead',
        'type': 'categorical',
        'required': False,  # Optional - analysis works but without rep-level controls
        'common_names': [
            'sales_rep', 'rep', 'salesperson', 'agent', 'owner',
            'assigned_to', 'rep_name', 'sales_agent', 'team_member'
        ]
    },
    'ordered': {
        'description': 'Whether the customer ordered',
        'type': 'boolean',
        'required': True,
        'common_names': [
            'ordered', 'sold', 'converted', 'closed', 'won',
            'is_sale', 'purchased', 'closed_won', 'sale', 'order'
        ]
    }
}

# Values that indicate a positive outcome (customer ordered)
POSITIVE_OUTCOME_VALUES = [
    True, 'true', 'True', 'TRUE', 
    1, '1', 1.0,
    'yes', 'Yes', 'YES', 'y', 'Y',
    'ordered', 'Ordered', 'ORDERED',
    'sold', 'Sold', 'SOLD',
    'won', 'Won', 'WON',
    'closed won', 'Closed Won', 'CLOSED WON',
    'converted', 'Converted', 'CONVERTED'
]


# =============================================================================
# UI STYLING
# =============================================================================
# Colors and styling for the application

# Color scheme for response time buckets (green = fast = good)
BUCKET_COLORS = {
    '0-15 min': '#2ECC71',   # Green - fastest response
    '15-30 min': '#F1C40F',  # Yellow - okay
    '30-60 min': '#E67E22',  # Orange - slow
    '60+ min': '#E74C3C'     # Red - too slow
}

# Chart theme
CHART_TEMPLATE = 'plotly_white'
CHART_COLOR_SEQUENCE = ['#2ECC71', '#F1C40F', '#E67E22', '#E74C3C']


# =============================================================================
# SAMPLE DATA SETTINGS
# =============================================================================
# Parameters for generating realistic sample data

SAMPLE_DATA_CONFIG = {
    'n_leads': 10000,  # Number of sample leads to generate
    
    # Lead sources with their base close rates
    'lead_sources': {
        'Dealer Referral': {'base_close_rate': 0.15, 'weight': 0.10},
        'Website Form': {'base_close_rate': 0.06, 'weight': 0.35},
        'Phone Call': {'base_close_rate': 0.12, 'weight': 0.20},
        'Walk-in': {'base_close_rate': 0.18, 'weight': 0.15},
        'Third Party Lead': {'base_close_rate': 0.04, 'weight': 0.20}
    },
    
    # Sales rep skill distribution (affects both speed and close rate)
    'n_reps': 12,
    'rep_skill_std': 0.3,  # Standard deviation of rep skill
    
    # Response time distribution parameters (log-normal)
    'response_time_median_mins': 25,
    'response_time_std_log': 1.2
}

