"""Visualization contract (request/result models, view maps, classification, renderer policy); classification reuses the legacy head/tail algorithm so new and legacy maps classify identically; sidecars are plain JSON, not RO-Crate."""

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml


def _legacy_viz():
    """Lazy import: keeps light workers/--help from loading the full scientific stack."""
    from urban_pfr import viz
    return viz


def head_tail_breaks(values, n_iterations=3):
    """Delegates to the legacy algorithm (imported lazily)."""
    return _legacy_viz().head_tail_breaks(values, n_iterations=n_iterations)


#TODO All variables and views require a scientific review and refactors, and agreement is needed on the definition and use of all variables.
# TODO science validation for all variables and thresholds review


# Sidecar schema versioning policy
SCHEMA_VERSION = "1.4"

TIERS = ("private", "public")
RENDERERS = ("auto", "vector", "rasterized")
FRAMINGS = ("fit", "focus")
SOURCE_FORMATS = ("auto", "gpkg", "fgb")
CLASSIFICATION_MODES = ("head_tail", "precomputed")

# Semantic dataset roles (vocabulary aligned with the dashboard contract)
ROLE_PRIVATE_BUILDINGS = "private_building_risk"
ROLE_PRIVATE_STATS = "private_statistical_units"
ROLE_PUBLIC_RISK = "public_risk_healpix"
ROLE_PUBLIC_VULN = "public_vulnerability_healpix"

VIEW_VARIABLES = {
    "risk_ma": "PFRMA",
    "risk_wb": "PFRWB",
    "risk_ma_raw": "PFRMA",
    "risk_wb_raw": "PFRWB",
    "vulnerability": "SVPF",
    "hazard_ma": "HMA",
    "hazard_wb": "HWB",
    "exposure_ma": "R",
    "exposure_wb": "R_G",
}

# Public-only raw validation views: private risk is already raw, so they have no private role
_PUBLIC_ONLY_VIEWS = ("risk_ma_raw", "risk_wb_raw")

_PRIVATE_VIEW_ROLES = {view: ROLE_PRIVATE_BUILDINGS for view in VIEW_VARIABLES
                       if view not in _PUBLIC_ONLY_VIEWS}
_PUBLIC_VIEW_ROLES = {
    "risk_ma": ROLE_PUBLIC_RISK,
    "risk_wb": ROLE_PUBLIC_RISK,
    "risk_ma_raw": ROLE_PUBLIC_RISK,
    "risk_wb_raw": ROLE_PUBLIC_RISK,
    "hazard_ma": ROLE_PUBLIC_RISK,
    "hazard_wb": ROLE_PUBLIC_RISK,
    "vulnerability": ROLE_PUBLIC_VULN,
}

# Public display DEFAULT = SAVED smoothed PFR columns on HEALPix cells
# TODO for data-contract, visualization is read-only and never transforms scientific values
# Visualization should notes not calculate smoothing or HEALPix aggregation.

# The HEALPix cells use smoothed values with different HEALPix shapes
_PUBLIC_VIEW_VARIABLES = {"risk_ma": "PFRMA_smoothed", "risk_wb": "PFRWB_smoothed"}
_PUBLIC_BIVARIATE_AXES = {"risk_map_pfr": ("PFRMA_smoothed", "PFRWB_smoothed")}

VIEW_FAMILIES = {
    "risk_ma": "risk",
    "risk_wb": "risk",
    "risk_ma_raw": "risk",
    "risk_wb_raw": "risk",
    "vulnerability": "vulnerability",
    "hazard_ma": "hazard",
    "hazard_wb": "hazard",
    "exposure_ma": "exposure",
    "exposure_wb": "exposure",
}

# Data-geometry representation per view
SPATIAL_REPRESENTATION = {
    ROLE_PRIVATE_BUILDINGS: "building_footprint",
    ROLE_PRIVATE_STATS: "statistical_unit",
    ROLE_PUBLIC_RISK: "healpix",
    ROLE_PUBLIC_VULN: "healpix",
}

# TODO:  Map style and Palettes, and legend semantics  need to be review with CS3 to support both scientific paper and methodology paper.
HAZARD_COLORS = ["#FFFFFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"]
EXPOSURE_COLORS = ["#FFFFFF", "#C7E9C0", "#74C476", "#238B45", "#00441B"]

def _default_view_colors(view):
    """Legacy paper palettes are fetched lazily to avoid importing matplotlib at module load."""
    v = _legacy_viz()
    mapping = {
        "risk_ma": v.DEFAULT_PFRMA_COLORS,
        "risk_wb": v.DEFAULT_PFRWB_COLORS,
        "risk_ma_raw": v.DEFAULT_PFRMA_COLORS,
        "risk_wb_raw": v.DEFAULT_PFRWB_COLORS,
        "vulnerability": v.DEFAULT_SVPF_COLORS,
        "hazard_ma": HAZARD_COLORS,
        "hazard_wb": HAZARD_COLORS,
        "exposure_ma": EXPOSURE_COLORS,
        "exposure_wb": EXPOSURE_COLORS,
    }
    return mapping[view]

# Dashboard-named composite  views using the paper's Figs 4-6 layer semantics
BIVARIATE_VIEWS = {
    "risk_map_pfr": ("PFRMA", "PFRWB"),
    "vulnerability_drivers": ("Sensitivity", "CopingCapacity"),
}
VIEW_TITLES = {
    "risk_map_pfr": "Risk map (PFR) — Mobility & Accessibility x Well-being",
    "vulnerability_drivers": "Vulnerability drivers — Sensitivity x Coping capacity",
    "vulnerability": "Social vulnerability background (SVPF)",
}
_BIV_RAMP_X = {"risk_map_pfr": ["#FFF2CC", "#FFD966", "#FF9900"],
               "vulnerability_drivers": ["#D6E4F0", "#7FA8D0", "#3B6BA5"]}
_BIV_RAMP_Y = {"risk_map_pfr": ["#F1D4E5", "#D791C0", "#A84B94"],
               "vulnerability_drivers": ["#DDEEDD", "#8FBC8F", "#2E8B57"]}
BIV_AXIS_LABELS = {"risk_map_pfr": ("PFR$_{MA}$", "PFR$_{WB}$"),
                   "vulnerability_drivers": ("Sensitivity", "Low coping")}


def is_bivariate(view):
    return view in BIVARIATE_VIEWS


def bivariate_axes(view):
    return BIVARIATE_VIEWS[view]


def bivariate_colors(view):
    """3x3 multiplicative blend matrix [row=y][col=x] of the two axis ramps."""
    from matplotlib.colors import to_hex, to_rgb
    xs = [to_rgb(c) for c in _BIV_RAMP_X[view]]
    ys = [to_rgb(c) for c in _BIV_RAMP_Y[view]]
    return [[to_hex((x[0] * y[0], x[1] * y[1], x[2] * y[2])) for x in xs] for y in ys]


def compute_breaks2(values):
    """Two head/tail breaks -> 3 classes for one bivariate axis (NaN/inf ignored)."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise NoFiniteValuesError("no finite values for a bivariate axis")
    return [float(b) for b in head_tail_breaks(finite, n_iterations=2)]


def classify3(values, breaks):
    """3-class assignment per bivariate axis (legacy semantics: NaN -> 0)."""
    arr = np.asarray(values, dtype=float)
    cls = np.zeros(arr.shape[0], dtype=int)
    finite = np.isfinite(arr)
    if len(breaks) > 0:
        cls[finite] = np.digitize(arr[finite], np.asarray(breaks, dtype=float))
    return np.clip(cls, 0, 2)


FAMILY_LABELS = {
    "risk": ["No risk", "Low", "Medium", "High", "Very high"],
    "hazard": ["No hazard", "Low", "Medium", "High", "Very high"],
    "exposure": ["None", "Low", "Medium", "High", "Very high"],
    "vulnerability": ["Very low", "Low", "Medium", "High", "Very high"],
}

# Transparent-over-basemap style (dashboard-like): zero class nearly invisible, light thin edges
BASEMAP_STYLE = {"alpha_zero": 0.12, "alpha_data": 0.78, "edge_rgba": (0.3, 0.3, 0.3, 0.25), "edge_lw": 0.2}

# Basemap presets matching the dashboard's options
BASEMAP_PRESETS = {
    "positron": "CartoDB.Positron",     # Light — CARTO Positron
    "voyager": "CartoDB.Voyager",       # CARTO Voyager
    "esri-topo": "Esri.WorldTopoMap",   # Esri Topographic
    "satellite": "Esri.WorldImagery",   # Satellite (Esri)
    # Add the approved official URL template
}

# Display-only boundary outline styling and processing (see static._boundary_outline)
BOUNDARY_STYLE = {"close_buffer_m": 600.0, "simplify_m": 50.0, "color": "#4a4a4a", "linewidth": 1.2}


def resolve_basemap(name):
    """Map a friendly preset (positron/voyager/esri-topo/satellite) to a contextily provider key."""
    if not name:
        return None
    return BASEMAP_PRESETS.get(str(name).strip().lower(), str(name).strip())

# Centralized deterministic renderer thresholds 
RENDER_THRESHOLDS = {
    "max_vector_detail": 50000,
    "max_vector_fast": 150000,
    "min_edge_px": 4.0,
    "pdf_vector_cap": 20000,
    "marker_min_px": 2.0,     # below this median feature size, auto switches to visible markers
    "marker_size_pt2": 3.0,   # scatter marker area for marker_overview
}

DEFAULT_FIG_WIDTH_IN = 10.0
MAX_FIG_HEIGHT_IN = 20.0
MIN_FIG_HEIGHT_IN = 3.0
EXTENT_PADDING = 0.02


class VisualizationError(Exception):
    """Base error for the visualization subsystem."""


class ValidationError(VisualizationError):
    """Invalid request parameters or unsupported combinations."""


class EmptySelectionError(VisualizationError):
    """The bbox selects no features."""


class NoFiniteValuesError(VisualizationError):
    """No finite values remain after removing NaN/inf."""


class MissingPublicOutputError(VisualizationError):
    """A required public output file is absent (fail closed, no private fallback)."""


class InvalidBreaksFileError(ValidationError):
    """The precomputed breaks file is missing, unparseable, or has invalid content."""


class InvalidStyleFileError(ValidationError):
    """The style file is missing, unparseable, or has invalid content."""


@dataclass
class VisualizationRequest:
    run_dir: str
    tier: str
    view: str
    output: str
    bbox: Optional[List[float]] = None
    bbox_crs: Optional[str] = None
    framing: str = "fit"
    renderer: str = "auto"
    classification_mode: str = "head_tail"
    breaks_file: Optional[str] = None
    source_format: str = "auto"
    dpi: int = 300
    figsize: Optional[Tuple[float, float]] = None
    style_file: Optional[str] = None
    context_file: Optional[str] = None
    basemap: Optional[str] = None
    boundary_file: Optional[str] = None


@dataclass
class VisualizationResult:
    output_path: str
    provenance_path: str
    source_role: str
    source_filename: str
    tier: str
    view: str
    variables: List[str]
    source_crs: str
    displayed_extent: List[float]
    renderer_requested: str
    renderer_used: str
    renderer_reason: str
    feature_count: int
    classification_breaks: List[float] = field(default_factory=list)


def view_role(tier, view):
    """Map (tier, view) to a dataset role; unsupported combinations raise ValidationError."""
    if tier not in TIERS:
        raise ValidationError("unknown tier '%s'; expected one of %s" % (tier, list(TIERS)))
    all_views = sorted(set(VIEW_VARIABLES) | set(BIVARIATE_VIEWS))
    if view not in all_views:
        raise ValidationError("unknown view '%s'; expected one of %s" % (view, all_views))
    if view in _PUBLIC_ONLY_VIEWS and tier != "public":
        raise ValidationError(
            "view '%s' is a public raw-validation view; the private '%s' view is already raw" % (
                view, view.replace("_raw", "")))
    if view == "risk_map_pfr":
        return ROLE_PUBLIC_RISK if tier == "public" else ROLE_PRIVATE_BUILDINGS
    if view == "vulnerability_drivers":
        return ROLE_PUBLIC_VULN if tier == "public" else ROLE_PRIVATE_BUILDINGS
    if tier == "public":
        if view not in _PUBLIC_VIEW_ROLES:
            raise ValidationError(
                "view '%s' is unsupported for the public tier (exposure is building-level only)" % view)
        return _PUBLIC_VIEW_ROLES[view]
    return _PRIVATE_VIEW_ROLES[view]


def view_variable(view):
    if view not in VIEW_VARIABLES:
        raise ValidationError("unknown view '%s'" % view)
    return VIEW_VARIABLES[view]


def view_variables(tier, view):
    """Tier-aware stored-column resolution: public risk defaults read the SAVED smoothed columns, public *_raw read the saved raw columns; nothing is computed."""
    view_role(tier, view)
    if is_bivariate(view):
        if tier == "public" and view in _PUBLIC_BIVARIATE_AXES:
            return list(_PUBLIC_BIVARIATE_AXES[view])
        return list(BIVARIATE_VIEWS[view])
    if tier == "public" and view in _PUBLIC_VIEW_VARIABLES:
        return [_PUBLIC_VIEW_VARIABLES[view]]
    return [view_variable(view)]


def variable_state(variables):
    """'smoothed' when every drawn column is a *_smoothed store, else 'raw'."""
    return "smoothed" if all(str(v).endswith("_smoothed") for v in variables) else "raw"


def view_title(tier, view, variables):
    """Explicit figure title: tier, spatial representation, exact variable, raw/smoothed state."""
    rep = "building" if tier == "private" else "HEALPix"
    if view in _PUBLIC_ONLY_VIEWS:
        return "Public HEALPix risk — %s (raw validation)" % variables[0]
    if view in ("risk_ma", "risk_wb"):
        base = str(variables[0]).replace("_smoothed", "")
        return "%s %s risk — %s (%s)" % (tier.capitalize(), rep, base, variable_state(variables))
    named = VIEW_TITLES.get(view)
    if named:
        return "%s — %s %s (%s)" % (named, tier, rep, variable_state(variables))
    return "%s — %s %s (%s, %s)" % (view, tier, rep, ", ".join(variables), variable_state(variables))


def view_labels(view, style=None):
    """Labels for a view."""
    if style and view in style.get("view_labels", {}):
        return list(style["view_labels"][view])
    return list(FAMILY_LABELS[VIEW_FAMILIES[view]])


def view_colors(view, style=None):
    """Palette for a view; an optional style dict (from load_style_file) overrides per view."""
    if style and view in style.get("view_colors", {}):
        return list(style["view_colors"][view])
    return list(_default_view_colors(view))


def load_style_file(path):
    """Load/validate a JSON/YAML style file {"view_colors": {view: [5 colors]}, "view_labels": {view: [5 labels]}}; unknown top-level keys are ignored, unknown views are errors."""
    try:
        with open(path, "r") as f:
            text = f.read()
    except (IOError, OSError) as e:
        raise InvalidStyleFileError("cannot read style file '%s': %s" % (path, e))
    try:
        if str(path).endswith((".yaml", ".yml")):
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
    except Exception as e:
        raise InvalidStyleFileError("cannot parse style file '%s': %s" % (path, e))
    if not isinstance(data, dict):
        raise InvalidStyleFileError("style file must contain a mapping")
    style = {"view_colors": {}, "view_labels": {}}
    for key in ("view_colors", "view_labels"):
        section = data.get(key, {}) or {}
        if not isinstance(section, dict):
            raise InvalidStyleFileError("'%s' must be a mapping of view -> list of 5 strings" % key)
        for view, vals in section.items():
            if view not in VIEW_VARIABLES:
                raise InvalidStyleFileError("unknown view '%s' in '%s'; expected one of %s" % (
                    view, key, sorted(VIEW_VARIABLES)))
            if not (isinstance(vals, list) and len(vals) == 5
                    and all(isinstance(v, str) and v.strip() for v in vals)):
                raise InvalidStyleFileError(
                    "'%s' entry for view '%s' must be a list of exactly 5 non-empty strings" % (key, view))
            style[key][view] = [str(v) for v in vals]
    return style


def validate_bbox(bbox, bbox_crs):
    """Validate [xmin, ymin, xmax, ymax] + mandatory bbox_crs; returns bbox as floats."""
    if bbox is None:
        return None
    if bbox_crs is None or str(bbox_crs).strip() == "":
        raise ValidationError("bbox_crs is mandatory when bbox is supplied")
    try:
        vals = [float(v) for v in bbox]
    except (TypeError, ValueError):
        raise ValidationError("bbox must be four numbers [xmin, ymin, xmax, ymax]")
    if len(vals) != 4:
        raise ValidationError("bbox must have exactly four values, got %d" % len(vals))
    if not all(math.isfinite(v) for v in vals):
        raise ValidationError("bbox values must be finite")
    if vals[0] >= vals[2] or vals[1] >= vals[3]:
        raise ValidationError("bbox requires xmin < xmax and ymin < ymax: %s" % vals)
    return vals


def validate_request(req):
    """Validate the full request; raises ValidationError on any unsupported combination."""
    view_role(req.tier, req.view)
    if req.renderer not in RENDERERS:
        raise ValidationError("unknown renderer '%s'; expected %s" % (req.renderer, list(RENDERERS)))
    if req.framing not in FRAMINGS:
        raise ValidationError("unknown framing '%s'; expected %s" % (req.framing, list(FRAMINGS)))
    if req.source_format not in SOURCE_FORMATS:
        raise ValidationError("unknown source_format '%s'; expected %s" % (req.source_format, list(SOURCE_FORMATS)))
    if req.classification_mode not in CLASSIFICATION_MODES:
        raise ValidationError(
            "unknown classification_mode '%s'; expected %s" % (req.classification_mode, list(CLASSIFICATION_MODES)))
    req.bbox = validate_bbox(req.bbox, req.bbox_crs)
    if req.framing == "fit" and req.bbox is not None:
        raise ValidationError("framing='fit' requires bbox to be absent; use framing='focus' with bbox")
    if req.framing == "focus" and req.bbox is None:
        raise ValidationError("framing='focus' requires a bbox")
    if is_bivariate(req.view) and req.style_file:
        raise ValidationError("bivariate views do not support style files")
    if req.classification_mode == "precomputed" and not req.breaks_file:
        raise ValidationError("classification_mode='precomputed' requires breaks_file")
    if req.classification_mode == "head_tail" and req.breaks_file:
        raise ValidationError("breaks_file is only valid with classification_mode='precomputed' (no silent fallback)")
    if not (str(req.output).endswith(".png") or str(req.output).endswith(".pdf")):
        raise ValidationError("output must end in .png or .pdf, got '%s'" % req.output)
    if req.dpi <= 0:
        raise ValidationError("dpi must be positive")
    return req


def _parse_breaks_document(path):
    """Read + parse a JSON/YAML breaks document (safe_load only)."""
    try:
        with open(path, "r") as f:
            text = f.read()
    except (IOError, OSError) as e:
        raise InvalidBreaksFileError("cannot read breaks file '%s': %s" % (path, e))
    try:
        if str(path).endswith((".yaml", ".yml")):
            return yaml.safe_load(text)
        return json.loads(text)
    except Exception as e:
        raise InvalidBreaksFileError("cannot parse breaks file '%s': %s" % (path, e))


def _validate_break_values(vals_in, n_min=1, n_max=4):
    if not isinstance(vals_in, list) or not (n_min <= len(vals_in) <= n_max):
        raise InvalidBreaksFileError(
            "breaks must be a list of %d to %d numbers" % (n_min, n_max))
    try:
        vals = [float(v) for v in vals_in]
    except (TypeError, ValueError):
        raise InvalidBreaksFileError("breaks file entries must all be numbers")
    if not all(math.isfinite(v) for v in vals):
        raise InvalidBreaksFileError("breaks file entries must be finite")
    if any(lo >= hi for lo, hi in zip(vals, vals[1:])):
        raise InvalidBreaksFileError("breaks must be strictly ascending: %s" % vals)
    return vals


def load_breaks_file(path, expected_variable=None):
    """Load univariate precomputed breaks , plain lists stay accepted for backward compatibility."""
    data = _parse_breaks_document(path)
    if isinstance(data, dict) and "breaks" in data:
        declared = data.get("variable")
        if expected_variable is not None and declared is not None and str(declared) != str(expected_variable):
            raise InvalidBreaksFileError(
                "breaks file declares variable '%s' but the selected view draws '%s'" % (
                    declared, expected_variable))
        data = data["breaks"]
    if not isinstance(data, list) or not (1 <= len(data) <= 4):
        raise InvalidBreaksFileError("breaks file must contain a list of 1 to 4 numbers (5-class scheme)")
    return _validate_break_values(data)


def load_bivariate_breaks_file(path, expected_x, expected_y):
    """Versioned bivariate breaks {"variables": {"x", "y"}, "breaks": {"x": [b1, b2], "y": [b1, b2]}} — exactly 2 ascending breaks per axis; declared variables must match the view's axes."""
    data = _parse_breaks_document(path)
    if not (isinstance(data, dict) and isinstance(data.get("breaks"), dict)):
        raise InvalidBreaksFileError(
            "bivariate breaks file must be a mapping with 'breaks': {'x': [...], 'y': [...]}")
    declared = data.get("variables") or {}
    if not isinstance(declared, dict):
        raise InvalidBreaksFileError("'variables' must be a mapping with 'x' and 'y'")
    for axis, expected in (("x", expected_x), ("y", expected_y)):
        got = declared.get(axis)
        if got is not None and str(got) != str(expected):
            raise InvalidBreaksFileError(
                "bivariate breaks axis '%s' declares variable '%s' but the view draws '%s'" % (
                    axis, got, expected))
    out = {}
    for axis in ("x", "y"):
        if axis not in data["breaks"]:
            raise InvalidBreaksFileError("bivariate breaks file is missing axis '%s'" % axis)
        out[axis] = _validate_break_values(data["breaks"][axis], n_min=2, n_max=2)
    return out["x"], out["y"]


def compute_breaks(values, mode="head_tail", precomputed=None):
    """Returns (breaks, bounds, invalid_counts): NaN/inf excluded but counted"""
    arr = np.asarray(values, dtype=float)
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise NoFiniteValuesError(
            "no finite values remain after removing %d NaN and %d inf values" % (n_nan, n_inf))
    if mode == "precomputed":
        if precomputed is None:
            raise ValidationError("precomputed mode requires explicit breaks")
        breaks = [float(b) for b in precomputed]
    elif mode == "head_tail":
        breaks = [float(b) for b in head_tail_breaks(finite)]
    else:
        raise ValidationError("unknown classification mode '%s'" % mode)
    max_val = float(finite.max())
    # Upper bound must stay monotonic even when precomputed breaks exceed the data maximum
    upper = max([max_val] + list(breaks)) * 1.01
    bounds = [0.0] + list(breaks) + [upper]
    while len(bounds) < 6:
        bounds.insert(-1, bounds[-1])
    bounds = bounds[:6]
    return breaks, bounds, {"nan": n_nan, "inf": n_inf}


def classify(values, breaks, n_classes=5):
    """Assign classes from breaks (legacy semantics: NaN->0, v<=break1 -> class 0)."""
    arr = np.asarray(values, dtype=float)
    cls = np.zeros(arr.shape[0], dtype=int)
    finite = np.isfinite(arr)
    if len(breaks) > 0:
        cls[finite] = np.digitize(arr[finite], np.asarray(breaks, dtype=float))
    return np.clip(cls, 0, n_classes - 1)


def decide_renderer(renderer_requested, feature_count, median_feature_px, output_suffix,
                    thresholds=None):
    """Deterministic renderer decision; returns (renderer_used, reason)."""
    t = thresholds if thresholds is not None else RENDER_THRESHOLDS
    if renderer_requested == "rasterized":
        return "rasterized_export", "renderer explicitly requested as rasterized"
    if renderer_requested == "vector":
        warn = ""
        if output_suffix == ".pdf" and feature_count > t["pdf_vector_cap"]:
            warn = "; warning: %d features > pdf_vector_cap %d, expect a very large file" % (
                feature_count, t["pdf_vector_cap"])
        if median_feature_px >= t["min_edge_px"]:
            return "vector_detail", "vector requested; median feature %.1fpx >= %.1fpx edge threshold%s" % (
                median_feature_px, t["min_edge_px"], warn)
        return "vector_fast", "vector requested; median feature %.1fpx < %.1fpx, edges dropped%s" % (
            median_feature_px, t["min_edge_px"], warn)
    if median_feature_px < t["marker_min_px"]:
        return "marker_overview", "auto: median feature %.1fpx < %.1fpx — sub-pixel polygons drawn as class-colored markers" % (
            median_feature_px, t["marker_min_px"])
    if output_suffix == ".pdf" and feature_count > t["pdf_vector_cap"]:
        return "rasterized_export", "auto: %d features > pdf_vector_cap %d" % (
            feature_count, t["pdf_vector_cap"])
    if feature_count <= t["max_vector_detail"] and median_feature_px >= t["min_edge_px"]:
        return "vector_detail", "auto: %d features <= %d and median %.1fpx >= %.1fpx" % (
            feature_count, t["max_vector_detail"], median_feature_px, t["min_edge_px"])
    if feature_count <= t["max_vector_fast"]:
        return "vector_fast", "auto: %d features <= max_vector_fast %d" % (
            feature_count, t["max_vector_fast"])
    return "rasterized_export", "auto: %d features > max_vector_fast %d" % (
        feature_count, t["max_vector_fast"])


def derive_figsize(figsize, extent, is_geographic, mid_latitude):
    """figsize=None -> base width 10in, height from displayed-extent aspect, capped at 20in."""
    if figsize is not None:
        return (float(figsize[0]), float(figsize[1]))
    width = DEFAULT_FIG_WIDTH_IN
    ext_w = max(extent[2] - extent[0], 1e-12)
    ext_h = max(extent[3] - extent[1], 1e-12)
    ratio = ext_h / ext_w
    if is_geographic:
        ratio = ratio / max(math.cos(math.radians(mid_latitude)), 1e-6)
    height = min(MAX_FIG_HEIGHT_IN, max(MIN_FIG_HEIGHT_IN, width * ratio))
    return (width, height)
