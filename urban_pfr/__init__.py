__version__ = "0.3.0"
__license__ = ""

from .analyzer import PFRAnalyzer
from .indicators import topsis_with_shannon_entropy, compute_social_vulnerability
from .exposure import calculate_exposure_residents, calculate_exposure_wellbeing
from .hazard import calculate_hazard_mobility_accessibility, calculate_hazard_wellbeing
from .risk import calculate_risk, delaunay_smoothing
from .thiessen import create_thiessen_polygons
try:
    from .rocrate_io import resolve_rocrate_inputs, create_output_rocrate
except ImportError:
    pass  # fdo-resolver not installed; FDO mode unavailable
from .viz import create_risk_visualization, classify_values
from .validation import (
    show_data_requirements,
    validate_input_data,
    generate_validation_report,
    preprocess_and_fix_data,
)
