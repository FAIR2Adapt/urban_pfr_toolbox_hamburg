"""Static map rendering from saved urban_pfr outputs."""

import datetime
import functools
import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from . import data, spec

# Basemap cache and request limits; cache location overridable via URBAN_PFR_TILE_CACHE
TILE_CACHE_DIR = Path(
    os.environ.get(
        "URBAN_PFR_TILE_CACHE",
        str(Path.home() / ".cache" / "urban_pfr_tiles"),
    )
)
MAX_BASEMAP_TILES = 1500
BASEMAP_RETRIES = 3

# Files above this size get a sampled fingerprint (head + tail + size) instead of a full hash
_FULL_HASH_CAP_BYTES = 256 * 1024 * 1024


def _source_fingerprint(path):
    """(sha256_hexdigest, mode, size_bytes) of the source file."""
    import hashlib
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha256()
    with open(str(p), "rb") as f:
        if size <= _FULL_HASH_CAP_BYTES:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
            mode = "full"
        else:
            h.update(f.read(1 << 20))
            f.seek(-(1 << 20), 2)
            h.update(f.read(1 << 20))
            h.update(str(size).encode("ascii"))
            mode = "sampled"
    return h.hexdigest(), mode, size


def _package_version():
    try:
        from urban_pfr import __version__
        return __version__
    except Exception:
        return None


def _median_feature_px(gdf, extent, fig_width_in, dpi):
    """Approximate median feature size in output pixels from sampled geometry areas."""
    n = len(gdf)
    sample = gdf.geometry if n <= 5000 else gdf.geometry.iloc[:: max(1, n // 5000)]
    areas = np.asarray(sample.area, dtype=float)
    areas = areas[np.isfinite(areas) & (areas > 0)]
    if areas.size == 0:
        return 0.0
    ext_w = max(extent[2] - extent[0], 1e-12)
    px_per_unit = (fig_width_in * dpi) / ext_w
    return float(np.median(np.sqrt(areas)) * px_per_unit)


def _pad_extent(extent, fraction):
    w = max(extent[2] - extent[0], 1e-12)
    h = max(extent[3] - extent[1], 1e-12)
    return [extent[0] - w * fraction, extent[1] - h * fraction,
            extent[2] + w * fraction, extent[3] + h * fraction]


def _write_sidecar(path, payload):
    with open(str(path), "w") as f:
        json.dump(payload, f, indent=2)


def visualize_output(run_dir, tier, view, output, bbox=None, bbox_crs=None, framing="fit",
                     renderer="auto", classification_mode="head_tail", breaks_file=None,
                     source_format="auto", dpi=300, figsize=None, style_file=None,
                     context_file=None, basemap=None, boundary_file=None):
    """Render one static map from an existing run directory; returns VisualizationResult."""
    if basemap is True:
        basemap = "positron"
    basemap = spec.resolve_basemap(basemap) if basemap else None
    req = spec.VisualizationRequest(
        run_dir=str(run_dir), tier=tier, view=view, output=str(output), bbox=bbox,
        bbox_crs=bbox_crs, framing=framing, renderer=renderer,
        classification_mode=classification_mode, breaks_file=breaks_file,
        source_format=source_format, dpi=dpi, figsize=figsize,
        style_file=str(style_file) if style_file else None,
        context_file=str(context_file) if context_file else None,
        basemap=str(basemap) if basemap else None,
        boundary_file=str(boundary_file) if boundary_file else None)
    spec.validate_request(req)
    _guard_public_aux(req)
    style = spec.load_style_file(req.style_file) if req.style_file else None

    role = spec.view_role(req.tier, req.view)
    bivariate = spec.is_bivariate(req.view)
    variables = spec.view_variables(req.tier, req.view)
    catalog = data.OutputCatalog(req.run_dir)
    source_path, fmt_used = catalog.resolve(req.tier, role, req.source_format)

    precomputed = pre_xy = None
    if req.classification_mode == "precomputed":
        if bivariate:
            pre_xy = spec.load_bivariate_breaks_file(req.breaks_file, variables[0], variables[1])
        else:
            precomputed = spec.load_breaks_file(req.breaks_file, expected_variable=variables[0])

    if req.tier == "public":
        ids = data.read_attribute_series(source_path, "healpix_id")
        if ids.duplicated().any():
            raise spec.VisualizationError("duplicate healpix_id values in %s" % source_path.name)

    bbox_source = None
    if req.bbox is not None:
        source_crs_for_bbox = _source_crs(source_path)
        if source_crs_for_bbox is None:
            raise spec.VisualizationError(
                "cannot determine the source CRS of %s for bbox transformation" % source_path.name)
        bbox_source = data.transform_bbox(req.bbox, req.bbox_crs, source_crs_for_bbox)

    gdf = data.read_features(source_path, variables, bbox_source_crs=bbox_source)

    # Full-run classification scope: reuse the loaded columns when no bbox filtered them
    if bivariate:
        if pre_xy is not None:
            breaks_x, breaks_y = pre_xy
            both = np.concatenate([np.asarray(gdf[variables[0]], dtype=float),
                                   np.asarray(gdf[variables[1]], dtype=float)])
        else:
            if bbox_source is None:
                fx, fy = gdf[variables[0]], gdf[variables[1]]
            else:
                fx = data.read_attribute_series(source_path, variables[0])
                fy = data.read_attribute_series(source_path, variables[1])
            breaks_x = spec.compute_breaks2(fx.values)
            breaks_y = spec.compute_breaks2(fy.values)
            both = np.concatenate([np.asarray(fx, dtype=float), np.asarray(fy, dtype=float)])
        invalid_counts = {"nan": int(np.isnan(both).sum()), "inf": int(np.isinf(both).sum())}
        breaks = breaks_x + breaks_y
        bounds = None
    else:
        if bbox_source is None:
            full_values = gdf[variables[0]]
        else:
            full_values = data.read_attribute_series(source_path, variables[0])
        breaks, bounds, invalid_counts = spec.compute_breaks(
            full_values.values, mode=req.classification_mode, precomputed=precomputed)
    source_crs = str(gdf.crs)
    is_geographic = bool(getattr(gdf.crs, "is_geographic", False))
    feature_count = int(len(gdf))

    if req.framing == "focus":
        displayed_extent = _pad_extent(bbox_source, spec.EXTENT_PADDING)
    else:
        meta_extent = data.layer_extent(source_path)
        base = meta_extent if meta_extent is not None else [float(v) for v in gdf.total_bounds]
        displayed_extent = _pad_extent(base, spec.EXTENT_PADDING)

    mid_lat = (displayed_extent[1] + displayed_extent[3]) / 2.0
    fig_w, fig_h = spec.derive_figsize(req.figsize, displayed_extent, is_geographic, mid_lat)

    median_px = _median_feature_px(gdf, displayed_extent, fig_w, req.dpi)
    suffix = Path(req.output).suffix.lower()
    renderer_used, renderer_reason = spec.decide_renderer(
        req.renderer, feature_count, median_px, suffix)

    if bivariate:
        cx = spec.classify3(gdf[variables[0]].values, breaks_x)
        cy = spec.classify3(gdf[variables[1]].values, breaks_y)
        matrix = spec.bivariate_colors(req.view)
        base_cols = [matrix[cy[i]][cx[i]] for i in range(len(gdf))]
        if req.basemap:
            from matplotlib.colors import to_rgba
            a0, a1 = spec.BASEMAP_STYLE["alpha_zero"], spec.BASEMAP_STYLE["alpha_data"]
            facecolors = [to_rgba(c, a0 if (cx[i] == 0 and cy[i] == 0) else a1)
                          for i, c in enumerate(base_cols)]
        else:
            facecolors = base_cols
        colors = [c for row in matrix for c in row]
        labels = ["Low", "Medium", "High"]
    else:
        classes = spec.classify(gdf[variables[0]].values, breaks, n_classes=5)
        colors = spec.view_colors(req.view, style)
        labels = spec.view_labels(req.view, style)
        if req.basemap:
            # Dashboard-like transparency: the zero class stays nearly invisible over the tiles
            from matplotlib.colors import to_rgba
            facecolors = [to_rgba(colors[c], spec.BASEMAP_STYLE["alpha_zero"] if c == 0
                                  else spec.BASEMAP_STYLE["alpha_data"]) for c in classes]
        else:
            facecolors = [colors[c] for c in classes]

    context_gdf = None
    if req.context_file:
        context_gdf = _load_context(req.context_file, source_crs, displayed_extent)

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    fig = Figure(figsize=(fig_w, fig_h), dpi=req.dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)

    if context_gdf is not None:
        geom_types = set(context_gdf.geom_type.unique())
        if geom_types <= {"LineString", "MultiLineString"}:
            context_gdf.plot(ax=ax, color="#bbbbbb", linewidth=0.8, zorder=1)
        else:
            context_gdf.plot(ax=ax, facecolor="#d9d9d9", edgecolor="#bbbbbb", linewidth=0.3, zorder=1)

    if renderer_used == "marker_overview":
        # Sub-pixel polygons: draw class-colored centroid markers with a fixed visual size
        pts = gdf.geometry.centroid
        ax.scatter(pts.x.values, pts.y.values, s=spec.RENDER_THRESHOLDS["marker_size_pt2"],
                   c=facecolors, linewidths=0, zorder=2, rasterized=True)
    else:
        plot_kwargs = {"color": facecolors, "rasterized": renderer_used == "rasterized_export", "zorder": 2}
        if renderer_used == "vector_detail":
            if req.basemap:
                plot_kwargs["edgecolor"] = tuple(spec.BASEMAP_STYLE["edge_rgba"])
                plot_kwargs["linewidth"] = spec.BASEMAP_STYLE["edge_lw"]
            else:
                plot_kwargs["edgecolor"] = "black"
                plot_kwargs["linewidth"] = 0.3
        else:
            plot_kwargs["edgecolor"] = "none"
            plot_kwargs["linewidth"] = 0.0
        gdf.plot(ax=ax, **plot_kwargs)

    if is_geographic:
        ax.set_aspect(1.0 / max(math.cos(math.radians(mid_lat)), 1e-6))
    else:
        ax.set_aspect("equal")
    ax.set_xlim(displayed_extent[0], displayed_extent[2])
    ax.set_ylim(displayed_extent[1], displayed_extent[3])
    ax.set_axis_off()

    basemap_zoom = basemap_attempts = None
    if req.basemap:
        basemap_zoom, basemap_attempts = _add_basemap(
            ax, source_crs, req.basemap, displayed_extent, is_geographic, mid_lat,
            fig_width_px=fig_w * req.dpi)

    if req.boundary_file:
        _draw_boundary(ax, req.boundary_file, source_crs)

    if bivariate:
        _add_bivariate_legend(ax, spec.bivariate_colors(req.view), *spec.BIV_AXIS_LABELS[req.view])
    else:
        legend_handles = [Patch(facecolor=c, edgecolor="black", linewidth=0.4, label=l)
                          for c, l in zip(colors, labels)]
        if context_gdf is not None:
            legend_handles.append(Patch(facecolor="#d9d9d9", edgecolor="#bbbbbb", linewidth=0.4, label="Context"))
        ax.legend(handles=legend_handles, title=variables[0], loc="lower right",
                  fontsize=8, title_fontsize=9, framealpha=0.9)
    ax.set_title(spec.view_title(req.tier, req.view, variables), fontsize=12)
    if req.tier == "private":
        fig.text(0.5, 0.01, "PRIVATE — building-level, do not publish",
                 ha="center", fontsize=9, color="red")

    out_path = Path(req.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prov_path = Path(str(out_path) + ".provenance.json")

    src_sha256, src_sha_mode, src_size = _source_fingerprint(source_path)
    sidecar = {
        "schema_version": spec.SCHEMA_VERSION,
        "generator": "urban_pfr.visualization/%s" % (_package_version() or "unknown"),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "run_dir_name": Path(req.run_dir).resolve().name,
        "tier": req.tier,
        "view": req.view,
        "role": role,
        "spatial_representation": spec.SPATIAL_REPRESENTATION[role],
        "source_filename": source_path.name,
        "source_format": fmt_used,
        "source_crs": source_crs,
        "source_sha256": src_sha256,
        "source_sha256_mode": src_sha_mode,
        "source_size_bytes": src_size,
        "variables": variables,
        "variable_state": spec.variable_state(variables),
        "bbox": req.bbox,
        "bbox_crs": req.bbox_crs,
        "framing": req.framing,
        "displayed_extent": [float(v) for v in displayed_extent],
        "figsize_in": [fig_w, fig_h],
        "dpi": req.dpi,
        "classification": ({
            "mode": req.classification_mode,
            "scope": "full_run",
            "axes": {"x": {"variable": variables[0], "breaks": breaks_x},
                     "y": {"variable": variables[1], "breaks": breaks_y}},
            "palette_matrix": spec.bivariate_colors(req.view),
            "labels": labels,
        } if bivariate else {
            "mode": req.classification_mode,
            "scope": "full_run",
            "breaks": breaks,
            "bounds": bounds,
            "labels": labels,
            "palette": colors,
        }),
        "style_file": Path(req.style_file).name if req.style_file else None,
        "style_sha256": (_source_fingerprint(req.style_file)[0] if req.style_file else None),
        "breaks_file": Path(req.breaks_file).name if req.breaks_file else None,
        "context_file": Path(req.context_file).name if req.context_file else None,
        "basemap": req.basemap,
        "basemap_zoom": basemap_zoom,
        "basemap_attempts": basemap_attempts,
        "boundary_file": Path(req.boundary_file).name if req.boundary_file else None,
        "renderer": {"requested": req.renderer, "used": renderer_used, "reason": renderer_reason},
        "feature_count": feature_count,
        "invalid_value_counts": invalid_counts,
    }

    # Atomic publish: unique PID+UUID temps then os.replace — concurrent/crashed runs never share temps or leave partial finals
    token = "%s-%s" % (os.getpid(), uuid.uuid4().hex[:8])
    tmp_img = Path("%s.tmp-%s" % (out_path, token))
    tmp_prov = Path("%s.tmp-%s" % (prov_path, token))
    try:
        fig.savefig(str(tmp_img), dpi=req.dpi, bbox_inches="tight", format=suffix.lstrip("."))
        _write_sidecar(tmp_prov, sidecar)
        os.replace(str(tmp_img), str(out_path))
        os.replace(str(tmp_prov), str(prov_path))
    except Exception:
        for t in (tmp_img, tmp_prov):
            if t.exists():
                os.remove(str(t))
        # Pair invariant: no image may exist without its sidecar
        if out_path.exists() and not prov_path.exists():
            os.remove(str(out_path))
        raise

    return spec.VisualizationResult(
        output_path=str(out_path),
        provenance_path=str(prov_path),
        source_role=role,
        source_filename=source_path.name,
        tier=req.tier,
        view=req.view,
        variables=variables,
        source_crs=source_crs,
        displayed_extent=[float(v) for v in displayed_extent],
        renderer_requested=req.renderer,
        renderer_used=renderer_used,
        renderer_reason=renderer_reason,
        feature_count=feature_count,
        classification_breaks=list(breaks),
    )


def _guard_public_aux(req):
    """Reject public requests whose auxiliary files resolve (symlink-safe) under the run's private directory — before any file is opened or output created; errors name the argument role only, never the private filename."""
    if req.tier != "public":
        return
    private_dir = (Path(req.run_dir) / "private").resolve()
    for aux_role, p in (("boundary_file", req.boundary_file), ("context_file", req.context_file),
                        ("style_file", req.style_file), ("breaks_file", req.breaks_file)):
        if not p:
            continue
        rp = Path(p).resolve()
        if rp == private_dir or str(rp).startswith(str(private_dir) + os.sep):
            raise spec.ValidationError(
                "restricted auxiliary input for a public map: %s" % aux_role)


def _render_one_kwargs(kwargs):
    return visualize_output(**kwargs)


def _resolve_max_workers(max_workers):
    """Default is min(2, cpu) — each worker holds a FULL copy of the layer in memory."""
    if max_workers is None:
        return min(2, os.cpu_count() or 1)
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise spec.ValidationError("max_workers must be a positive integer or None, got %r" % (max_workers,))
    return max_workers


def render_many(requests, max_workers=None):
    """Render visualize_output kwargs dicts in parallel PROCESSES (results in input order, failures surface on collect); memory is roughly per worker (each loads its full layer) — full-city PRIVATE should use max_workers=1; default min(2, cpu)."""
    from concurrent.futures import ProcessPoolExecutor
    workers = _resolve_max_workers(max_workers)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_render_one_kwargs, dict(kw)) for kw in requests]
        return [f.result() for f in futures]


def _true_mid_latitude(extent, crs, is_geographic):
    """Geographic latitude of the extent centre (projected extents are reprojected first)."""
    if is_geographic:
        return (extent[1] + extent[3]) / 2.0
    import geopandas as gpd
    from shapely.geometry import Point
    centre = Point((extent[0] + extent[2]) / 2.0, (extent[1] + extent[3]) / 2.0)
    return float(gpd.GeoSeries([centre], crs=crs).to_crs("EPSG:4326").iloc[0].y)


def _basemap_zoom(extent, lat_deg, is_geographic, fig_width_px):
    """Tile zoom matched to the OUTPUT pixel width (contextily's default assumes ~96dpi and blurs)."""
    cos_lat = max(math.cos(math.radians(lat_deg)), 0.01)
    width_m = (extent[2] - extent[0]) * (111320.0 * cos_lat if is_geographic else 1.0)
    target_res = width_m / max(float(fig_width_px), 1.0)
    ground0 = 156543.03392 * cos_lat
    zoom = int(math.ceil(math.log(max(ground0 / max(target_res, 1e-9), 1.0), 2)))
    return max(1, min(zoom, 19))


def _estimate_tiles(extent, lat_deg, is_geographic, zoom):
    """Approximate tile count for the extent at a zoom (politeness cap input)."""
    cos_lat = max(math.cos(math.radians(lat_deg)), 0.01)
    to_m = 111320.0 if is_geographic else 1.0
    width_m = (extent[2] - extent[0]) * to_m * (cos_lat if is_geographic else 1.0)
    height_m = (extent[3] - extent[1]) * to_m
    tile_m = 40075016.686 * cos_lat / (2 ** zoom)
    return (width_m / tile_m + 1.0) * (height_m / tile_m + 1.0)


def prune_tile_cache(max_age_days=90, cache_dir=None):
    """Delete cached tiles older than the configured age.
    Returns the number of deleted files and bytes freed. This may be
    invoked automatically when URBAN_PFR_TILE_CACHE_MAX_AGE_DAYS is set.
    """
    root = Path(cache_dir) if cache_dir else TILE_CACHE_DIR
    if not root.is_dir():
        return 0, 0
    cutoff = time.time() - float(max_age_days) * 86400.0
    deleted = freed = 0
    for p in root.rglob("*"):
        try:
            if p.is_file() and p.stat().st_mtime < cutoff:
                size = p.stat().st_size
                p.unlink()
                deleted += 1
                freed += size
        except OSError:
            continue
    return deleted, freed


_AUTO_PRUNED = False


def _ensure_cache_dir(path=None):
    """Create the LOCAL tile-cache dir owner-only (0700) — tile names reveal which areas were viewed."""
    root = Path(path) if path else TILE_CACHE_DIR
    try:
        root.mkdir(parents=True, exist_ok=True)
        os.chmod(str(root), 0o700)
    except OSError:
        pass
    return root


def _auto_prune_if_configured(cache_dir=None):
    """Once per process: apply the opt-in cache TTL from URBAN_PFR_TILE_CACHE_MAX_AGE_DAYS (unset = keep everything)."""
    global _AUTO_PRUNED
    if _AUTO_PRUNED:
        return None
    _AUTO_PRUNED = True
    raw = os.environ.get("URBAN_PFR_TILE_CACHE_MAX_AGE_DAYS")
    if not raw:
        return None
    try:
        days = float(raw)
    except ValueError:
        return None
    if days <= 0:
        return None
    return prune_tile_cache(days, cache_dir=cache_dir)


def _default_tile_adder(ax, crs, provider_key, zoom):
    """Real tile fetch via contextily with a persistent LOCAL disk cache (injectable for tests)."""
    import contextily as ctx
    _ensure_cache_dir()
    _auto_prune_if_configured()
    try:
        ctx.set_cache_dir(str(TILE_CACHE_DIR))
    except Exception as e:
        # Silent-fallback would re-download every tile on every render (slow, rate-limit risk)
        import warnings
        warnings.warn("tile disk cache disabled (%s); tiles will be re-downloaded each render" % e)
    provider = ctx.providers
    for part in str(provider_key).split("."):
        provider = provider[part]
    ctx.add_basemap(ax, crs=crs, source=provider, zoom=zoom, attribution_size=6)


_TILE_ADDER = _default_tile_adder


def _add_basemap(ax, crs, provider_key, extent, is_geographic, mid_lat, fig_width_px):
    """Web-tile basemap under the data layer; returns (zoom_used, attempts). Opt-in network only."""
    lat_deg = _true_mid_latitude(extent, crs, is_geographic)
    zoom = _basemap_zoom(extent, lat_deg, is_geographic, fig_width_px)
    while zoom > 1 and _estimate_tiles(extent, lat_deg, is_geographic, zoom) > MAX_BASEMAP_TILES:
        zoom -= 1
    last_err = None
    for attempt in range(1, BASEMAP_RETRIES + 1):
        try:
            _TILE_ADDER(ax, crs, provider_key, zoom)
            return zoom, attempt
        except Exception as e:
            last_err = e
            if attempt < BASEMAP_RETRIES:
                time.sleep(attempt)
    raise spec.VisualizationError(
        "basemap '%s' failed after %d attempts: %s" % (provider_key, BASEMAP_RETRIES, last_err))


def _add_bivariate_legend(ax, matrix, xlabel, ylabel):
    """3x3 bivariate legend outside the map, bottom-right (paper Figs 4-6 style)."""
    from matplotlib.colors import to_rgba
    leg = ax.inset_axes([1.03, 0.02, 0.16, 0.16])
    grid = np.array([[to_rgba(matrix[r][c]) for c in range(3)] for r in range(3)])
    leg.imshow(grid, origin="lower", aspect="auto", interpolation="nearest")
    leg.set_xticks([0, 2])
    leg.set_xticklabels(["Low", "High"], fontsize=6)
    leg.set_yticks([0, 2])
    leg.set_yticklabels(["Low", "High"], fontsize=6)
    leg.set_xlabel(xlabel, fontsize=7, labelpad=2)
    leg.set_ylabel(ylabel, fontsize=7, labelpad=2)
    leg.tick_params(length=0)


@functools.lru_cache(maxsize=8)
def _boundary_outline(path_str, mtime, target_crs_str):
    """Cached display outline: dissolve service features, close water carve-outs, drop holes."""
    import geopandas as gpd
    bnd = gpd.read_file(path_str)
    if bnd.crs is None:
        raise spec.ValidationError("boundary file has no parseable CRS: %s" % Path(path_str).name)
    if str(bnd.crs) != target_crs_str:
        bnd = bnd.to_crs(target_crs_str)
    merged = bnd.geometry.union_all() if hasattr(bnd.geometry, "union_all") else bnd.geometry.unary_union
    if not gpd.GeoSeries([merged], crs=bnd.crs).crs.is_geographic:
        from shapely.geometry import MultiPolygon, Polygon
        style = spec.BOUNDARY_STYLE
        closed = merged.buffer(style["close_buffer_m"]).buffer(-style["close_buffer_m"]).simplify(style["simplify_m"])
        if closed.geom_type == "Polygon":
            merged = Polygon(closed.exterior)
        elif closed.geom_type == "MultiPolygon":
            merged = MultiPolygon([Polygon(g.exterior) for g in closed.geoms])
    return merged


def _draw_boundary(ax, path, target_crs):
    """Display-only administrative boundary outline on top (e.g. fetched from an Esri service)."""
    import geopandas as gpd
    p = Path(path)
    if not p.is_file():
        raise spec.ValidationError( "boundary_file does not exist: %s" % Path(path).name)
    outline = _boundary_outline(str(p.resolve()), p.stat().st_mtime, str(target_crs))
    style = spec.BOUNDARY_STYLE
    gpd.GeoSeries([outline], crs=target_crs).boundary.plot(
        ax=ax, color=style["color"], linewidth=style["linewidth"], linestyle=(0, (6, 3)), zorder=5)


def _load_context(path, target_crs, extent_in_target):
    """Display-only background layer: driver-bbox-clipped to the view extent, reprojected for drawing."""
    import geopandas as gpd
    p = Path(path)
    if not p.is_file():
        raise spec.ValidationError("context_file does not exist: %s" % path)
    ctx_crs = _source_crs(p)
    bbox = data.transform_bbox(extent_in_target, target_crs, ctx_crs) if ctx_crs is not None else None
    ctx = data._read_gdf(p, bbox=bbox)
    if len(ctx) == 0:
        return None
    if ctx.crs is None:
        raise spec.ValidationError("context file has no parseable CRS: %s" % p.name)
    if str(ctx.crs) != str(target_crs):
        ctx = ctx.to_crs(target_crs)
    return ctx


def _source_crs(path):
    """Read the source CRS from driver metadata (fallback: None, resolved after full read)."""
    try:
        import pyogrio
        crs = pyogrio.read_info(str(path)).get("crs")
        return crs
    except Exception:
        pass
    try:
        import fiona
        with fiona.open(str(path)) as src:
            return src.crs_wkt or src.crs
    except Exception:
        return None
