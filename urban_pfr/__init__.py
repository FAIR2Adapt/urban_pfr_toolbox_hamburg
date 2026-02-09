"""
urban-pfr: Urban Pluvial Flood Risk Assessment
Risk = Hazard × Exposure × Vulnerability (IPCC framework)
"""

__version__ = "0.1.0"
__license__ = ""

from .analyzer import PFRAnalyzer
from .indicators import topsis_with_shannon_entropy, compute_social_vulnerability
from .exposure import calculate_exposure_residents, calculate_exposure_wellbeing
from .hazard import calculate_hazard_mobility_accessibility, calculate_hazard_wellbeing
from .risk import calculate_risk, delaunay_smoothing
from .viz import create_risk_visualization, classify_risk_values
from .validation import (
    show_data_requirements,
    validate_input_data,
    generate_validation_report,
    preprocess_and_fix_data,
)