"""
Constants for api_pipeline.

Provides DATA_PATH and CONSTITUTION_PATH with sensible defaults.
Users can override by creating character/constants.py with custom paths.
"""

from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).resolve().parents[1]

# Try to import from character.constants for backwards compatibility
# Falls back to defaults based on repo structure
try:
    from character.constants import DATA_PATH
except ImportError:
    DATA_PATH = ROOT_DIR / "data"

try:
    from character.constants import CONSTITUTION_PATH
except ImportError:
    CONSTITUTION_PATH = ROOT_DIR / "constitutions"
