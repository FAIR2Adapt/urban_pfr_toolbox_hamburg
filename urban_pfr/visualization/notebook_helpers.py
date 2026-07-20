"""Shared helpers for reproducible visualization notebooks."""

from pathlib import Path

# Esri Living Atlas boundary source
_ESRI_BOUNDARY_ENDPOINT = (
    "https://services.arcgis.com/P3ePLMYs2RVChkJx/"
    "arcgis/rest/services/World_Administrative_Divisions/"
    "FeatureServer/0/query"
)
_ESRI_BOUNDARY_PARAMS = {
    "where": "NAME='Hamburg' AND COUNTRY='Germany'",
    "outFields": "NAME,COUNTRY",
    "f": "geojson",
    "outSR": "4326",
}


def ensure_hamburg_boundary(inputs_dir, endpoint=None, params=None):
    """Return the cached Esri Living Atlas Hamburg boundary; download once (timeout + raise_for_status) only when the cache is missing — a failed download never breaks offline rendering."""
    p = Path(inputs_dir) / "hamburg_boundary_esri.geojson"
    if p.exists():
        return p
    try:
        import requests
        response = requests.get(endpoint or _ESRI_BOUNDARY_ENDPOINT,
                                params=params or _ESRI_BOUNDARY_PARAMS, timeout=30)
        response.raise_for_status()
        p.write_bytes(response.content)
        print("downloaded Hamburg boundary from Esri Living Atlas ->", p)
        return p
    except Exception as e:
        raise
        # return None


def _expected_sidecar(run_dir, tier, view, kw):
    """Sidecar subset the CURRENT request would produce (drives the freshness comparison)."""
    from urban_pfr.visualization import data as vdata, spec, static as vstatic
    role = spec.view_role(tier, view)
    variables = spec.view_variables(tier, view)
    src, _fmt = vdata.OutputCatalog(run_dir).resolve(tier, role, kw.get("source_format", "auto"))
    sha, _mode, _size = vstatic._source_fingerprint(src)
    basemap = kw.get("basemap")
    if basemap is True:
        basemap = "positron"
    bbox = kw.get("bbox")
    expected = {
        "schema_version": spec.SCHEMA_VERSION,
        "tier": tier,
        "view": view,
        "role": role,
        "source_filename": src.name,
        "source_sha256": sha,
        "variables": list(variables),
        "variable_state": spec.variable_state(variables),
        "classification.mode": kw.get("classification_mode", "head_tail"),
        "framing": kw.get("framing", "fit"),
        "bbox": [float(v) for v in bbox] if bbox is not None else None,
        "bbox_crs": kw.get("bbox_crs"),
        "dpi": kw.get("dpi", 300),
        "renderer.requested": kw.get("renderer", "auto"),
        "basemap": spec.resolve_basemap(basemap) if basemap else None,
        "boundary_file": Path(str(kw["boundary_file"])).name if kw.get("boundary_file") else None,
        "context_file": Path(str(kw["context_file"])).name if kw.get("context_file") else None,
        "style_file": Path(str(kw["style_file"])).name if kw.get("style_file") else None,
    }
    if kw.get("style_file"):
        expected["style_sha256"] = vstatic._source_fingerprint(kw["style_file"])[0]
    if kw.get("figsize"):
        expected["figsize_in"] = [float(v) for v in kw["figsize"]]
    if kw.get("breaks_file") and not spec.is_bivariate(view):
        expected["classification.breaks"] = spec.load_breaks_file(kw["breaks_file"], variables[0])
    return expected


def _is_fresh(out_path, expected):
    """Fresh only when a valid sidecar matches EVERY expected field ('a.b' = nested lookup); missing, malformed, or incompatible sidecars are stale."""
    import json
    sc = Path(str(out_path) + ".provenance.json")
    if not sc.exists() or expected is None:
        return False
    try:
        with open(str(sc)) as f:
            payload = json.load(f)
    except Exception:
        return False
    for key, want in expected.items():
        node = payload
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return False
            node = node[part]
        if node != want:
            return False
    return True


def make_renderer(run_dir, figs_dir=None, force=False):
    """Return (render, figs_dir): render(tier, view, name, **kw) writes figs/<tier>/<name>, skipping only when the existing sidecar matches the CURRENT request (schema, source fingerprint, variables, classification, bbox, dpi, renderer, style, basemap/boundary/context); force=True always re-renders."""
    from urban_pfr.visualization import visualize_output
    run_dir = Path(run_dir)
    figs = Path(figs_dir) if figs_dir else run_dir / "figures"

    def render(tier, view, name, **kw):
        out = figs / tier / name
        if out.exists() and not force:
            try:
                expected = _expected_sidecar(run_dir, tier, view, kw)
            except Exception:
                expected = None  # let the real render surface the real error
            if _is_fresh(out, expected):
                print("exists, skipped:", "%s/%s" % (tier, name))
                return None
        try:
            r = visualize_output(run_dir, tier, view, out, renderer=kw.pop("renderer", "auto"), **kw)
            print("created:", "%s/%s" % (tier, name), "|", r.feature_count, "features |", r.renderer_used)
            return r
        except Exception as e:
            print(name, "SKIPPED (offline?):", e)
            return None

    return render, figs


def render_standard_views(render, basemap, boundary, private_figsize=None):
    """Dashboard-named layers (paper Figs 4-6) + single-variable risk views, both tiers."""
    for tier in ["public", "private"]:
        big = {"figsize": private_figsize} if (tier == "private" and private_figsize) else {}
        render(tier, "risk_map_pfr", "%s_risk_map_pfr_basemap.png" % tier,
               basemap=basemap, boundary_file=boundary, **big)
        render(tier, "vulnerability_drivers", "%s_vulnerability_drivers_basemap.png" % tier,
               basemap=basemap, boundary_file=boundary, **big)
        render(tier, "vulnerability", "%s_vulnerability_basemap.png" % tier,
               basemap=basemap, boundary_file=boundary, **big)
    render("public", "risk_wb", "public_risk_wb_basemap.png", basemap=basemap, boundary_file=boundary)
    for view in ["risk_ma", "risk_wb"]:
        big = {"figsize": private_figsize} if private_figsize else {}
        render("private", view, "private_%s_basemap.png" % view,
               basemap=basemap, boundary_file=boundary, **big)
