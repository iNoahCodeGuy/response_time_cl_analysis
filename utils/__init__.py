# =============================================================================
# Utility Functions
# =============================================================================
# Shared utilities for the application
# =============================================================================

import sys
import os


def setup_project_path():
    """
    Add the project root directory to sys.path for imports.
    
    WHY THIS EXISTS:
    -----------------
    This Streamlit app uses relative imports across modules. To enable imports
    from the project root (like `from data.loader import ...`), we need to add
    the project root to sys.path.
    
    USAGE:
    ------
    Call this function at the top of any module that needs to import from
    the project root:
    
        from utils import setup_project_path
        setup_project_path()
        from data.loader import load_file
    
    NOTE:
    -----
    This is a temporary solution. A better approach would be to:
    1. Install the project as a package (setup.py or pyproject.toml)
    2. Always run the app from the project root with PYTHONPATH set
    3. Use absolute imports with proper package structure
    
    However, for a Streamlit app that may be run from different locations,
    this pattern ensures imports work correctly.
    """
    # Get the project root (parent of utils directory, or current dir if utils doesn't exist)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Add to sys.path if not already present
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    return project_root

