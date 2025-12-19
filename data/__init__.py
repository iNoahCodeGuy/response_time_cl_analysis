# =============================================================================
# Data Package
# =============================================================================
# This package handles all data-related operations:
# - Loading CSV/Excel files
# - Parsing datetime columns
# - Mapping user columns to our expected format
# - Generating sample data for demos
# - Analyzing how many weeks of data are uploaded (NEW!)
#
# MODULE GUIDE FOR JUNIOR DEVELOPERS:
# ------------------------------------
# 1. loader.py         - Reads CSV/Excel files into pandas DataFrames
# 2. datetime_parser.py - Converts string dates to Python datetime objects
# 3. column_mapper.py   - Maps user's column names to our expected names
# 4. sample_generator.py - Creates fake data for demos and testing
# 5. weeks_analyzer.py   - Checks if user has enough weeks of data
# =============================================================================

from .loader import load_file, validate_file
from .datetime_parser import parse_datetime_column, detect_datetime_format
from .column_mapper import ColumnMapper
from .sample_generator import generate_sample_data
from .weeks_analyzer import analyze_weeks_of_data

