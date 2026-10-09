# data.py
"""Dataset discovery and geospatial input handling for saved run outputs."""
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import pandas as pd

from . import spec

# Expected filenames per role and format; no recursive guessing
_PRIVATE_ROLES = {
    spec.ROLE_PRIVATE_BUILDINGS: {
        "gpkg": "private/buildings_with_risk.gpkg",
        "fgb": "private/buildings_with_risk.fgb",
    },
    spec.ROLE_PRIVATE_STATS: {
        "gpkg": "private/statistical_units_with_vulnerability.gpkg",
        "fgb": "private/statistical_units_with_vulnerability.fgb",
    },
}
_PUBLIC_ROLES = {
    spec.ROLE_PUBLIC_RISK: {
        "gpkg": "public/risk_healpix.gpkg",
        "fgb": "public/risk_healpix.fgb",
    },
    spec.ROLE_PUBLIC_VULN: {
        "gpkg": "public/vulnerability_healpix.gpkg",
        "fgb": "public/vulnerability_healpix.fgb",
    },
}


class OutputCatalog:
    """Resolves semantic dataset roles inside one run directory, strictly per tier."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        if not self.run_dir.is_dir():
            raise spec.ValidationError("run_dir does not exist or is not a directory: %s" % run_dir)

    def _table(self, tier):
        if tier == "public":
            return _PUBLIC_ROLES
        if tier == "private":
            return _PRIVATE_ROLES
        raise spec.ValidationError("unknown tier '%s'" % tier)

    def resolve(self, tier, role, source_format="auto"):
        """Return (path, format) for a role; public misses fail closed with public-only info."""
        table = self._table(tier)
        if role not in table:
            raise spec.ValidationError("role '%s' does not belong to tier '%s'" % (role, tier))
        rels = table[role]
        if source_format == "auto":
            order = ["gpkg", "fgb"]  # auto prefers GPKG for local static rendering
        else:
            order = [source_format]
        for fmt in order:
            candidate = self.run_dir / rels[fmt]
            if candidate.is_file():
                return candidate, fmt
        expected = " or ".join(rels[f] for f in order)
        if tier == "public":
            raise spec.MissingPublicOutputError(
                "public output missing for role '%s': expected %s under the run directory "
                "(no fallback to other tiers)" % (role, expected))
        raise spec.VisualizationError(
            "output missing for role '%s': expected %s under the run directory" % (role, expected))


def _is_driver_error(exc):
    """True for expected geospatial-IO failures (GDAL/pyogrio/fiona/OS), not programming errors."""
    mod = type(exc).__module__ or ""
    return mod.startswith(("pyogrio", "fiona")) or isinstance(exc, OSError)


def _sanitized_read(path, reader):
    """Run a read callable; driver failures re-raise typed with basename-only messages."""
    p = Path(path)
    if not p.is_file():
        raise spec.VisualizationError("source file '%s' is missing" % p.name)
    try:
        return reader()
    except spec.VisualizationError:
        raise
    except Exception as exc:
        if _is_driver_error(exc):
            raise spec.VisualizationError(
                "cannot read source '%s': corrupt, unreadable, or unsupported file" % p.name) from exc
        raise


def _read_gdf(path, columns=None, bbox=None):
    """Engine-agnostic read; feature-detects column pruning, falls back to full read."""
    kwargs = {}
    if bbox is not None:
        kwargs["bbox"] = tuple(bbox)

    def reader():
        if columns is not None:
            try:
                return gpd.read_file(str(path), columns=list(columns), **kwargs)
            except TypeError:
                pass
        return gpd.read_file(str(path), **kwargs)

    return _sanitized_read(path, reader)


def read_attribute_series(path, column):
    """Attribute-only full-run read of one column (no geometry when the engine supports it)."""

    def reader():
        try:
            return gpd.read_file(str(path), columns=[column], ignore_geometry=True)
        except TypeError:
            try:
                return gpd.read_file(str(path), ignore_geometry=True)
            except TypeError:
                return gpd.read_file(str(path))

    df = _sanitized_read(path, reader)
    if column not in df.columns:
        raise spec.VisualizationError("required field '%s' not found in %s" % (column, Path(path).name))
    series = df[column]
    if not pd.api.types.is_numeric_dtype(series):
        raise spec.VisualizationError("field '%s' in %s is not numeric" % (column, Path(path).name))
    return series


def layer_extent(path):
    """Layer extent from driver metadata where available; None -> caller falls back."""
    try:
        import pyogrio
        info = pyogrio.read_info(str(path))
        tb = info.get("total_bounds")
        if tb is not None:
            return [float(v) for v in tb]
    except Exception:
        pass
    try:
        import fiona
        with fiona.open(str(path)) as src:
            return [float(v) for v in src.bounds]
    except Exception:
        return None


def transform_bbox(bbox, bbox_crs, target_crs):
    """Transform [xmin, ymin, xmax, ymax] from bbox_crs to the source CRS."""
    from shapely.geometry import box as shapely_box
    gs = gpd.GeoSeries([shapely_box(bbox[0], bbox[1], bbox[2], bbox[3])], crs=bbox_crs)
    if target_crs is not None and str(gs.crs) != str(target_crs):
        gs = gs.to_crs(target_crs)
    tb = gs.total_bounds
    return [float(v) for v in tb]


def read_features(path, variables, bbox_source_crs=None):
    """Read geometry + required variable(s), driver-level bbox filter + exact two-phase check."""
    if isinstance(variables, str):
        variables = [variables]
    gdf = _read_gdf(path, columns=list(variables), bbox=bbox_source_crs)
    for variable in variables:
        if variable not in gdf.columns:
            raise spec.VisualizationError("required field '%s' not found in %s" % (variable, Path(path).name))
    if "geometry" not in gdf.columns or gdf.geometry is None:
        raise spec.VisualizationError("no geometry column in %s" % Path(path).name)
    if gdf.crs is None:
        raise spec.VisualizationError("no parseable CRS in %s" % Path(path).name)
    for variable in variables:
        if not pd.api.types.is_numeric_dtype(gdf[variable]):
            raise spec.VisualizationError("field '%s' in %s is not numeric" % (variable, Path(path).name))
    geom_types = set(t for t in gdf.geom_type.unique() if t is not None)
    if geom_types and not geom_types <= {"Polygon", "MultiPolygon"}:
        raise spec.VisualizationError(
            "only polygonal layers are supported, %s contains %s" % (
                Path(path).name, sorted(geom_types)))
    if bbox_source_crs is not None:
        if len(gdf) > 0:
            from shapely.geometry import box as shapely_box
            window = shapely_box(*bbox_source_crs)
            gdf = gdf[gdf.geometry.intersects(window)]  # exact check on the returned subset only
        if len(gdf) == 0:
            extent = layer_extent(path)
            raise spec.EmptySelectionError(
                "bbox %s (in source CRS) selects no features; source extent is %s. "
                "bbox order is [minx, miny, maxx, maxy] — for EPSG:4326 that means "
                "x=longitude first; a swapped [lat, lon] input is the most common cause." % (
                    [round(v, 2) for v in bbox_source_crs],
                    [round(v, 2) for v in extent] if extent else "unknown"))
    else:
        if len(gdf) == 0:
            raise spec.VisualizationError("layer %s is empty" % Path(path).name)
    gdf = gdf[~gdf.geometry.isna()]
    if "healpix_id" in gdf.columns and gdf["healpix_id"].duplicated().any():
        raise spec.VisualizationError("duplicate healpix_id values in %s" % Path(path).name)
    return gdf
