"""Offline test for the urban_pfr.visualization subsystem."""


# next step test coverage:
# TODO(ci-docker): Run this test module inside the project Docker image.
# TODO(ci-remote): Add an opt-in remote smoke test using synthetic public
# outputs only. Require an explicit endpoint configuration and never transmit
# private data, local paths, or private bounding boxes.
# TODO(ci-concurrency): Add a simultaneous same-output rendering test.
#
# Local commands:
#   python tests/test_visualization.py
#   python -m pytest tests/test_visualization.py -q

import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

# Support direct execution from the repository without an editable install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import geopandas as gpd
from shapely.geometry import box

try:
    import pytest
except ImportError:
    pytest = None

from urban_pfr.viz import head_tail_breaks
from urban_pfr.visualization import VisualizationResult, spec
from urban_pfr.visualization.static import visualize_output


# ---- synthetic-data builders (deterministic; no production data) ----

def _synthetic_buildings(n=20, crs="EPSG:25832"):
    x0, y0 = 565000, 5935000
    polys = [box(x0 + (i % 5) * 40, y0 + (i // 5) * 40,
                 x0 + (i % 5) * 40 + 15, y0 + (i // 5) * 40 + 10) for i in range(n)]
    rng = np.random.RandomState(7)
    return gpd.GeoDataFrame({
        "PFRMA": rng.rand(n) * 10,
        "PFRWB": rng.rand(n) * 5,
        "SVPF": rng.rand(n),
        "Sensitivity": rng.rand(n),
        "CopingCapacity": rng.rand(n),
        "HMA": rng.rand(n),
        "HWB": rng.rand(n) * 3,
        "R": rng.rand(n) * 100,
        "R_G": rng.rand(n) * 30,
    }, geometry=polys, crs=crs)


def _make_run_dir(root, private=True, public=False, pfrwb_values=None, public_smoothed=True):
    run = Path(root)
    if private:
        b = _synthetic_buildings()
        if pfrwb_values is not None:
            b["PFRWB"] = pfrwb_values
        (run / "private").mkdir(parents=True, exist_ok=True)
        b.to_file(str(run / "private" / "buildings_with_risk.gpkg"), driver="GPKG")
    if public:
        cells = _synthetic_buildings(n=6)
        cells["healpix_id"] = np.arange(6, dtype="int64")
        if public_smoothed:
            # Public fixtures contain the saved smoothed fields expected from the pipeline.
            cells["PFRMA_smoothed"] = cells["PFRMA"].values * 0.5 + 0.05
            cells["PFRWB_smoothed"] = cells["PFRWB"].values * 0.25 + 0.02
        (run / "public").mkdir(parents=True, exist_ok=True)
        cells.to_file(str(run / "public" / "risk_healpix.gpkg"), driver="GPKG")
    return run


# ---- reusable local assertion helpers ----

def _assert_raises(exception_type, callable_, *args, **kwargs):
    try:
        callable_(*args, **kwargs)
    except exception_type as exc:
        return exc
    raise AssertionError("expected %s" % exception_type.__name__)


def _read_sidecar(result):
    return json.loads(Path(result.provenance_path).read_text())


# ---- specification and classification checks ----

def check_golden_head_tail_breaks():
    values = np.array([0, 0, np.nan, 1, 2, 3, 4, 5, 10, 20, 50], dtype=float)
    breaks, bounds, counts = spec.compute_breaks(values, mode="head_tail")
    assert breaks == [float(b) for b in head_tail_breaks(values)]
    assert counts["nan"] == 1 and counts["inf"] == 0
    assert len(bounds) == 6 and bounds[0] == 0.0


def check_small_dataset_classification_padding():
    breaks1, bounds1, _ = spec.compute_breaks([5.0])
    assert breaks1 == [] and len(bounds1) == 6
    assert list(spec.classify([0.0, 5.0], breaks1)) == [0, 0]
    breaks2, bounds2, _ = spec.compute_breaks([2.0, 8.0])
    assert breaks2 == [5.0] and len(bounds2) == 6
    assert list(spec.classify([0.0, 2.0, 8.0, np.nan], breaks2)) == [0, 0, 1, 0]


def check_no_finite_values_raises():
    _assert_raises(spec.NoFiniteValuesError, spec.compute_breaks, [np.nan, np.inf, -np.inf])


def check_bbox_validation_rules():
    _assert_raises(spec.ValidationError, spec.validate_bbox, [0, 0, 1, 1], None)
    _assert_raises(spec.ValidationError, spec.validate_bbox, [1, 0, 0, 1], "EPSG:25832")
    assert spec.validate_bbox([0, 0, 1, 1], "EPSG:25832") == [0.0, 0.0, 1.0, 1.0]


def check_tier_view_role_mappings():
    assert spec.view_role("private", "risk_wb") == spec.ROLE_PRIVATE_BUILDINGS
    assert spec.view_role("public", "risk_wb") == spec.ROLE_PUBLIC_RISK
    assert spec.view_role("public", "vulnerability") == spec.ROLE_PUBLIC_VULN
    assert spec.view_role("public", "risk_map_pfr") == spec.ROLE_PUBLIC_RISK
    assert spec.view_role("public", "vulnerability_drivers") == spec.ROLE_PUBLIC_VULN
    assert spec.view_variable("exposure_wb") == "R_G"
    _assert_raises(spec.ValidationError, spec.view_role, "public", "exposure_ma")


def check_view_labels_and_palettes():
    assert spec.view_labels("risk_ma")[0] == "No risk"
    assert spec.view_labels("hazard_wb")[0] == "No hazard"
    assert spec.view_labels("exposure_ma")[0] == "None"
    assert spec.view_labels("vulnerability")[0] == "Very low"
    assert len(spec.view_colors("risk_wb")) == 5


def check_risk_view_variable_mappings():
    assert spec.view_variables("private", "risk_ma") == ["PFRMA"]
    assert spec.view_variables("private", "risk_wb") == ["PFRWB"]
    assert spec.view_variables("public", "risk_ma") == ["PFRMA_smoothed"]
    assert spec.view_variables("public", "risk_wb") == ["PFRWB_smoothed"]
    assert spec.view_variables("public", "risk_ma_raw") == ["PFRMA"]
    assert spec.view_variables("public", "risk_wb_raw") == ["PFRWB"]
    assert spec.view_variables("public", "risk_map_pfr") == ["PFRMA_smoothed", "PFRWB_smoothed"]
    assert spec.view_variables("private", "risk_map_pfr") == ["PFRMA", "PFRWB"]
    assert spec.variable_state(["PFRMA_smoothed", "PFRWB_smoothed"]) == "smoothed"
    assert spec.variable_state(["PFRMA"]) == "raw"
    _assert_raises(spec.ValidationError, spec.view_role, "private", "risk_ma_raw")


def check_bivariate_specification():
    assert spec.is_bivariate("risk_map_pfr") and not spec.is_bivariate("risk_wb")
    assert spec.bivariate_axes("vulnerability_drivers") == ("Sensitivity", "CopingCapacity")
    matrix = spec.bivariate_colors("risk_map_pfr")
    assert len(matrix) == 3 and len(matrix[0]) == 3
    assert list(spec.classify3([0, 1, 5, 100], [1.0, 5.0])) == [0, 1, 2, 2]


def check_renderer_policy_is_deterministic():
    a = spec.decide_renderer("auto", 100, 10.0, ".png")
    assert a == spec.decide_renderer("auto", 100, 10.0, ".png")
    assert a[0] == "vector_detail"
    assert spec.decide_renderer("auto", 100000, 2.0, ".png")[0] == "vector_fast"
    assert spec.decide_renderer("auto", 200000, 2.0, ".png")[0] == "rasterized_export"
    assert spec.decide_renderer("auto", 10000, 1.0, ".png")[0] == "marker_overview"
    assert spec.decide_renderer("vector", 10000, 1.0, ".png")[0] == "vector_fast"
    assert spec.decide_renderer("auto", 30000, 10.0, ".pdf")[0] == "rasterized_export"
    assert spec.decide_renderer("rasterized", 10, 10.0, ".png")[0] == "rasterized_export"
    assert spec.decide_renderer("vector", 300000, 10.0, ".png")[0] == "vector_detail"


def check_bbox_axis_order_transform():
    from urban_pfr.visualization import data as data_mod
    # always_xy requires longitude before latitude.
    tb = data_mod.transform_bbox([9.9, 53.4, 10.3, 53.8], "EPSG:4326", "EPSG:25832")
    assert 400000 < tb[0] < 700000 and 5850000 < tb[1] < 6050000, tb


def check_empty_selection_reports_swap_hint():
    from urban_pfr.visualization import data as data_mod
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(tmp, private=True)
        src = run / "private" / "buildings_with_risk.gpkg"
        far = data_mod.transform_bbox([53.4, 9.9, 53.8, 10.3], "EPSG:4326", "EPSG:25832")
        exc = _assert_raises(spec.EmptySelectionError,
                             data_mod.read_features, src, "PFRWB", bbox_source_crs=far)
        assert "[minx, miny, maxx, maxy]" in str(exc) and "lat, lon" in str(exc)


# ---- style and breaks-file checks ----

def check_style_file_loading():
    with tempfile.TemporaryDirectory() as tmp:
        sf = Path(tmp) / "style.json"
        sf.write_text('{"view_colors": {"risk_ma": ["#FFFFFF", "#FEE6CE", "#FDAE6B", "#E6550D", "#A63603"]},'
                      ' "view_labels": {"risk_ma": ["No risk", "L", "M", "H", "VH"]}}')
        style = spec.load_style_file(sf)
        assert spec.view_colors("risk_ma", style)[3] == "#E6550D"
        assert spec.view_labels("risk_ma", style)[4] == "VH"
        assert spec.view_colors("risk_wb", style) == spec.view_colors("risk_wb")
        bad_view = Path(tmp) / "bad1.json"
        bad_view.write_text('{"view_colors": {"nope": ["#1", "#2", "#3", "#4", "#5"]}}')
        _assert_raises(spec.InvalidStyleFileError, spec.load_style_file, bad_view)
        bad_len = Path(tmp) / "bad2.json"
        bad_len.write_text('{"view_colors": {"risk_ma": ["#FFFFFF"]}}')
        _assert_raises(spec.InvalidStyleFileError, spec.load_style_file, bad_len)


def check_breaks_file_validation():
    with tempfile.TemporaryDirectory() as tmp:
        good = Path(tmp) / "b.json"
        good.write_text("[1.0, 2.5, 7.0]")
        assert spec.load_breaks_file(good) == [1.0, 2.5, 7.0]
        versioned = Path(tmp) / "v.json"
        versioned.write_text('{"schema_version": "1.2", "variable": "PFRWB", "breaks": [1.0, 2.0]}')
        assert spec.load_breaks_file(versioned) == [1.0, 2.0]
        bad_order = Path(tmp) / "bad.json"
        bad_order.write_text("[3.0, 1.0]")
        _assert_raises(spec.InvalidBreaksFileError, spec.load_breaks_file, bad_order)
        bad_type = Path(tmp) / "bad2.yaml"
        bad_type.write_text("- 1.0\n- not_a_number\n")
        _assert_raises(spec.InvalidBreaksFileError, spec.load_breaks_file, bad_type)
        too_many = Path(tmp) / "toomany.json"
        too_many.write_text("[1, 2, 3, 4, 5]")
        _assert_raises(spec.InvalidBreaksFileError, spec.load_breaks_file, too_many)


def check_precomputed_breaks_variable_matching():
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(tmp, private=True, public=True)
        wrong = Path(tmp) / "wrong.json"
        wrong.write_text('{"schema_version": "1.0", "variable": "PFRWB", "breaks": [0.5, 1.0]}')
        exc = _assert_raises(spec.InvalidBreaksFileError, visualize_output,
                             run, "private", "risk_ma", Path(tmp) / "a.png",
                             classification_mode="precomputed", breaks_file=wrong)
        assert "PFRWB" in str(exc) and "PFRMA" in str(exc)
        raw_decl = Path(tmp) / "rawdecl.json"
        raw_decl.write_text('{"schema_version": "1.0", "variable": "PFRMA", "breaks": [0.5, 1.0]}')
        _assert_raises(spec.InvalidBreaksFileError, visualize_output,
                       run, "public", "risk_ma", Path(tmp) / "b.png",
                       classification_mode="precomputed", breaks_file=raw_decl)
        ok = Path(tmp) / "ok.json"
        ok.write_text('{"schema_version": "1.0", "variable": "PFRMA_smoothed", "breaks": [0.1, 0.2]}')
        r = visualize_output(run, "public", "risk_ma", Path(tmp) / "c.png",
                             classification_mode="precomputed", breaks_file=ok)
        assert r.classification_breaks == [0.1, 0.2]
        plain = Path(tmp) / "plain.json"
        plain.write_text("[0.1, 0.2]")
        r2 = visualize_output(run, "private", "risk_ma", Path(tmp) / "d.png",
                              classification_mode="precomputed", breaks_file=plain)
        assert r2.classification_breaks == [0.1, 0.2]


def check_bivariate_precomputed_breaks():
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(tmp, private=True)
        bad = Path(tmp) / "bad.json"
        bad.write_text(json.dumps({"schema_version": "1.0",
                                   "variables": {"x": "PFRMA", "y": "PFRMA"},
                                   "breaks": {"x": [0.1, 0.2], "y": [0.1, 0.2]}}))
        _assert_raises(spec.InvalidBreaksFileError, visualize_output,
                       run, "private", "risk_map_pfr", Path(tmp) / "e.png",
                       classification_mode="precomputed", breaks_file=bad)
        good = Path(tmp) / "good.json"
        good.write_text(json.dumps({"schema_version": "1.0",
                                    "variables": {"x": "PFRMA", "y": "PFRWB"},
                                    "breaks": {"x": [0.1, 0.2], "y": [0.1, 0.2]}}))
        r = visualize_output(run, "private", "risk_map_pfr", Path(tmp) / "f.png",
                             classification_mode="precomputed", breaks_file=good)
        sc = _read_sidecar(r)
        assert sc["classification"]["mode"] == "precomputed"
        assert sc["classification"]["axes"]["x"]["breaks"] == [0.1, 0.2]


# ---- rendering checks ----

def check_classification_equal_across_bboxes():
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(tmp, private=True)
        full = gpd.read_file(str(Path(run) / "private" / "buildings_with_risk.gpkg"))
        tb = full.total_bounds
        w, h = tb[2] - tb[0], tb[3] - tb[1]
        bbox_a = [tb[0], tb[1], tb[0] + w * 0.6, tb[1] + h * 0.6]
        bbox_b = [tb[0] + w * 0.4, tb[1] + h * 0.4, tb[2], tb[3]]
        r_a = visualize_output(run, "private", "risk_wb", Path(tmp) / "a.png",
                               bbox=bbox_a, bbox_crs="EPSG:25832", framing="focus")
        r_b = visualize_output(run, "private", "risk_wb", Path(tmp) / "b.png",
                               bbox=bbox_b, bbox_crs="EPSG:25832", framing="focus")
        assert r_a.classification_breaks == r_b.classification_breaks
        assert r_a.classification_breaks == [float(x) for x in head_tail_breaks(full["PFRWB"].values)]
        assert r_a.feature_count != len(full) or r_b.feature_count != len(full)


def check_pdf_render_produces_output():
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(tmp, private=True)
        r = visualize_output(run, "private", "risk_wb", Path(tmp) / "fig.pdf")
        assert Path(r.output_path).exists() and Path(r.output_path).stat().st_size > 0
        assert Path(r.provenance_path).exists()


def check_styled_precomputed_render_records_provenance():
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(tmp, private=True)
        sf = Path(tmp) / "style.json"
        sf.write_text('{"view_colors": {"risk_wb": ["#FFFFFF", "#F2D8E7", "#DE9FC6", "#B05A9C", "#6E205F"]}}')
        bf = Path(tmp) / "breaks.json"
        bf.write_text("[1.0, 2.0, 3.0]")
        ctx = Path(run) / "private" / "buildings_with_risk.gpkg"
        r = visualize_output(run, "private", "risk_wb", Path(tmp) / "combo.png",
                             classification_mode="precomputed", breaks_file=bf,
                             style_file=sf, context_file=ctx)
        assert r.classification_breaks == [1.0, 2.0, 3.0]
        sc = _read_sidecar(r)
        assert sc["style_file"] == "style.json" and sc["context_file"] == "buildings_with_risk.gpkg"
        assert sc["style_sha256"]
        assert sc["classification"]["palette"][4] == "#6E205F"
        assert sc["classification"]["bounds"] == sorted(sc["classification"]["bounds"])


def check_bivariate_rendering():
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(tmp, private=True)
        r = visualize_output(run, "private", "risk_map_pfr", Path(tmp) / "biv.png")
        assert r.variables == ["PFRMA", "PFRWB"] and Path(r.output_path).exists()
        sc = _read_sidecar(r)
        assert sc["classification"]["axes"]["x"]["variable"] == "PFRMA"
        assert len(sc["classification"]["palette_matrix"]) == 3
        _assert_raises(spec.ValidationError, visualize_output,
                       run, "private", "risk_map_pfr", Path(tmp) / "x.png",
                       style_file=Path(tmp) / "style.json")


def check_basemap_retry_with_injected_fetcher():
    from urban_pfr.visualization import static as static_mod
    calls = {"n": 0}

    def flaky_adder(ax, crs, provider_key, zoom):
        calls["n"] += 1
        if calls["n"] < 2:
            raise ConnectionError("transient tile failure")

    original_adder = static_mod._TILE_ADDER
    original_sleep = static_mod.time.sleep
    static_mod._TILE_ADDER = flaky_adder
    static_mod.time.sleep = lambda _seconds: None  # retries must not wait in real time
    try:
        with tempfile.TemporaryDirectory() as tmp:
            run = _make_run_dir(tmp, private=True)
            r = visualize_output(run, "private", "risk_wb", Path(tmp) / "bm.png", basemap="positron")
            sc = _read_sidecar(r)
            assert sc["basemap"] == "CartoDB.Positron"
            assert sc["basemap_attempts"] == 2 and calls["n"] == 2
            assert isinstance(sc["basemap_zoom"], int)
    finally:
        static_mod._TILE_ADDER = original_adder
        static_mod.time.sleep = original_sleep


def check_basemap_zoom_respects_tile_cap():
    from urban_pfr.visualization import static as static_mod
    assert spec.resolve_basemap("positron") == "CartoDB.Positron"
    assert spec.resolve_basemap("SATELLITE") == "Esri.WorldImagery"
    assert spec.resolve_basemap(None) is None
    extent = [0.0, 0.0, 40000.0, 35000.0]
    zoom = static_mod._basemap_zoom(extent, 53.5, False, 12000)
    capped = zoom
    while capped > 1 and static_mod._estimate_tiles(extent, 53.5, False, capped) > static_mod.MAX_BASEMAP_TILES:
        capped -= 1
    assert capped <= zoom
    assert static_mod._estimate_tiles(extent, 53.5, False, capped) <= static_mod.MAX_BASEMAP_TILES


def check_boundary_overlay_renders():
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(tmp, private=True)
        bnd = gpd.GeoDataFrame(geometry=[box(565000, 5935000, 565200, 5935160).buffer(50)],
                               crs="EPSG:25832")
        bf = Path(tmp) / "boundary.geojson"
        bnd.to_file(str(bf), driver="GeoJSON")
        r = visualize_output(run, "private", "risk_wb", Path(tmp) / "b.png", boundary_file=bf)
        assert _read_sidecar(r)["boundary_file"] == "boundary.geojson"
        assert Path(r.output_path).exists()


# ---- tiered raw/smoothed risk contract checks ----

def check_public_risk_uses_smoothed_healpix():
    with tempfile.TemporaryDirectory() as tmp:
        # No private file exists at all: public rendering cannot depend on it
        run = _make_run_dir(tmp, private=False, public=True)
        r = visualize_output(run, "public", "risk_ma", Path(tmp) / "pm.png")
        sc = _read_sidecar(r)
        assert sc["variables"] == ["PFRMA_smoothed"] and sc["variable_state"] == "smoothed"
        assert sc["spatial_representation"] == "healpix" and sc["role"] == spec.ROLE_PUBLIC_RISK
        assert visualize_output(run, "public", "risk_wb", Path(tmp) / "pw.png").variables == ["PFRWB_smoothed"]


def check_public_raw_risk_uses_raw_healpix():
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(tmp, private=False, public=True)
        r = visualize_output(run, "public", "risk_ma_raw", Path(tmp) / "pr.png")
        sc = _read_sidecar(r)
        assert sc["variables"] == ["PFRMA"] and sc["variable_state"] == "raw"
        assert sc["spatial_representation"] == "healpix"
        assert visualize_output(run, "public", "risk_wb_raw", Path(tmp) / "pz.png").variables == ["PFRWB"]


def check_private_risk_uses_building_footprints():
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(tmp, private=True)
        r = visualize_output(run, "private", "risk_ma", Path(tmp) / "m.png")
        sc = _read_sidecar(r)
        assert sc["variables"] == ["PFRMA"] and sc["variable_state"] == "raw"
        assert sc["spatial_representation"] == "building_footprint"
        assert visualize_output(run, "private", "risk_wb", Path(tmp) / "w.png").variables == ["PFRWB"]


def check_raw_and_smoothed_breaks_are_independent():
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(tmp, private=False, public=True)
        smoothed = visualize_output(run, "public", "risk_ma", Path(tmp) / "s.png")
        raw = visualize_output(run, "public", "risk_ma_raw", Path(tmp) / "r.png")
        assert smoothed.classification_breaks != raw.classification_breaks


def check_bivariate_variables_per_tier():
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(tmp, private=True, public=True)
        pub = visualize_output(run, "public", "risk_map_pfr", Path(tmp) / "pb.png")
        assert pub.variables == ["PFRMA_smoothed", "PFRWB_smoothed"]
        prv = visualize_output(run, "private", "risk_map_pfr", Path(tmp) / "vb.png")
        assert prv.variables == ["PFRMA", "PFRWB"]


def check_missing_smoothed_variable_does_not_fallback():
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(tmp, private=False, public=True, public_smoothed=False)
        out = Path(tmp) / "x.png"
        exc = _assert_raises(spec.VisualizationError, visualize_output, run, "public", "risk_ma", out)
        assert "PFRMA_smoothed" in str(exc)
        assert not out.exists()
        assert not Path(str(out) + ".provenance.json").exists()
        # The public fixture intentionally contains no private output.
        assert visualize_output(run, "public", "risk_ma_raw", Path(tmp) / "r.png").variables == ["PFRMA"]


# ---- public/private isolation checks ----

def check_public_fails_closed_without_public_outputs():
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(tmp, private=True, public=False)
        out = Path(tmp) / "fig.png"
        exc = _assert_raises(spec.MissingPublicOutputError, visualize_output, run, "public", "risk_wb", out)
        assert "private" not in str(exc)
        assert not out.exists()


def check_no_output_files_on_failure():
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(tmp, private=True)
        out1 = Path(tmp) / "empty.png"
        _assert_raises(spec.EmptySelectionError, visualize_output, run, "private", "risk_wb", out1,
                       bbox=[0, 0, 1, 1], bbox_crs="EPSG:25832", framing="focus")
        assert not out1.exists()
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(tmp, private=True, pfrwb_values=np.full(20, np.nan))
        out2 = Path(tmp) / "nan.png"
        _assert_raises(spec.NoFiniteValuesError, visualize_output, run, "private", "risk_wb", out2)
        assert not out2.exists()


def check_public_auxiliary_file_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(tmp, private=True, public=True)
        private_file = Path(run) / "private" / "buildings_with_risk.gpkg"
        for param in ("boundary_file", "context_file", "style_file", "breaks_file"):
            out = Path(tmp) / ("g_%s.png" % param)
            kwargs = {param: private_file}
            if param == "breaks_file":
                kwargs["classification_mode"] = "precomputed"
            exc = _assert_raises(spec.ValidationError, visualize_output,
                                 run, "public", "risk_wb", out, **kwargs)
            assert param in str(exc), "error must name the rejected argument role"
            assert not out.exists()
            assert not Path(str(out) + ".provenance.json").exists()
            assert not list(Path(tmp).glob("*.tmp-*")), "no temp files may be created before rejection"


def check_public_error_does_not_disclose_private_source():
    # Contract: a public-facing rejection names the argument role but never the private
    # filename, the private directory, or any absolute path.
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(tmp, private=True, public=True)
        private_file = Path(run) / "private" / "buildings_with_risk.gpkg"
        exc = _assert_raises(spec.ValidationError, visualize_output,
                             run, "public", "risk_wb", Path(tmp) / "g.png",
                             boundary_file=private_file)
        message = str(exc)
        assert "boundary_file" in message
        assert str(Path(run).resolve()) not in message, "absolute run path disclosed"
        assert "/private/" not in message and "\\private\\" not in message
        assert "buildings_with_risk" not in message, (
            "public rejection message disclosed the private source filename")


def check_symlinked_private_auxiliary_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(tmp, private=True, public=True)
        private_file = Path(run) / "private" / "buildings_with_risk.gpkg"
        link = Path(tmp) / "innocent.gpkg"
        os.symlink(private_file, link)
        _assert_raises(spec.ValidationError, visualize_output,
                       run, "public", "risk_wb", Path(tmp) / "sym.png", context_file=link)


def check_public_sidecar_has_no_private_leak():
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(tmp, private=True, public=True)
        result = visualize_output(run, "public", "risk_wb", Path(tmp) / "pub.png")
        assert Path(result.output_path).exists()
        sidecar_text = Path(result.provenance_path).read_text()
        assert "private/" not in sidecar_text
        assert "buildings_with_risk" not in sidecar_text
        assert str(Path(tmp).resolve()) not in sidecar_text, "absolute path leaked into sidecar"


# ---- provenance checks ----

def check_sidecar_source_fingerprint():
    import hashlib
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(tmp, private=True)
        result = visualize_output(run, "private", "risk_wb", Path(tmp) / "fp.png")
        sc = _read_sidecar(result)
        assert sc["schema_version"] == spec.SCHEMA_VERSION
        src = run / "private" / "buildings_with_risk.gpkg"
        assert sc["source_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
        assert sc["source_sha256_mode"] == "full"
        assert sc["source_size_bytes"] == src.stat().st_size


# ---- temporary-file and atomic-output checks (sequential; see TODO(ci-concurrency)) ----

def check_temporary_paths_are_unique_per_invocation():
    from urban_pfr.visualization import static as static_mod
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(tmp, private=True)
        out = Path(tmp) / "unique.png"
        observed = []
        original = static_mod._write_sidecar

        def recording_failure(path, payload):
            observed.append(str(path))
            raise OSError("stop after recording the temp path")

        static_mod._write_sidecar = recording_failure
        try:
            for _ in range(2):
                _assert_raises(OSError, visualize_output, run, "private", "risk_wb", out)
        finally:
            static_mod._write_sidecar = original
        assert len(observed) == 2 and observed[0] != observed[1]
        pid = str(os.getpid())
        assert all(".tmp-" in p and pid in p for p in observed)


def check_failed_render_cleans_own_temp_files():
    from urban_pfr.visualization import static as static_mod
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(tmp, private=True)
        out = Path(tmp) / "atomic.png"
        original = static_mod._write_sidecar

        def failing_writer(path, payload):
            raise OSError("simulated sidecar write failure")

        static_mod._write_sidecar = failing_writer
        try:
            _assert_raises(OSError, visualize_output, run, "private", "risk_wb", out)
        finally:
            static_mod._write_sidecar = original
        assert not out.exists(), "a final image must never exist without its sidecar"
        assert not Path(str(out) + ".provenance.json").exists()
        assert list(Path(tmp).glob("atomic.png*.tmp-*")) == []


def check_failed_render_preserves_foreign_temp_files():
    from urban_pfr.visualization import static as static_mod
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(tmp, private=True)
        out = Path(tmp) / "atomic.png"
        foreign = Path(str(out) + ".tmp-999-deadbeef")
        foreign.write_text("temp file owned by another invocation")
        original = static_mod._write_sidecar

        def failing_writer(path, payload):
            raise OSError("simulated sidecar write failure")

        static_mod._write_sidecar = failing_writer
        try:
            _assert_raises(OSError, visualize_output, run, "private", "risk_wb", out)
        finally:
            static_mod._write_sidecar = original
        assert foreign.exists(), "one invocation must not delete another's temp files"


def check_output_pair_is_atomic():
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(tmp, private=True)
        out = Path(tmp) / "pair.png"
        r = visualize_output(run, "private", "risk_wb", out)
        assert Path(r.output_path).exists() and Path(r.provenance_path).exists()
        assert list(Path(tmp).glob("pair.png*.tmp-*")) == []


def check_render_many_worker_bounds():
    from urban_pfr.visualization import static as static_mod
    assert static_mod._resolve_max_workers(None) <= 2
    assert static_mod._resolve_max_workers(1) == 1
    assert static_mod._resolve_max_workers(5) == 5
    for bad in (0, -3, True, False, 2.5, "2"):
        _assert_raises(spec.ValidationError, static_mod._resolve_max_workers, bad)


# ---- freshness checks (notebook helper cache) ----

def check_identical_request_skips_rerender():
    from urban_pfr.visualization import notebook_helpers as nh
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(tmp, private=True)
        render, _figs = nh.make_renderer(run)
        assert render("private", "risk_wb", "f.png") is not None
        assert render("private", "risk_wb", "f.png") is None


def check_request_change_invalidates_cached_figure():
    from urban_pfr.visualization import notebook_helpers as nh
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(tmp, private=True, public=True)
        render, _figs = nh.make_renderer(run)
        assert render("private", "risk_wb", "f.png") is not None
        assert render("private", "risk_wb", "f.png", dpi=150) is not None, "dpi change must re-render"
        assert render("public", "risk_ma", "same.png") is not None
        assert render("public", "risk_ma", "same.png") is None
        assert render("public", "risk_ma_raw", "same.png") is not None, \
            "raw/smoothed variable change must re-render"


def check_source_change_invalidates_cached_figure():
    from urban_pfr.visualization import notebook_helpers as nh
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(tmp, private=True)
        render, _figs = nh.make_renderer(run)
        assert render("private", "risk_wb", "f.png") is not None
        src = Path(run) / "private" / "buildings_with_risk.gpkg"
        b = gpd.read_file(str(src))
        b["PFRWB"] = b["PFRWB"].values * 2.0
        b.to_file(str(src), driver="GPKG")
        assert render("private", "risk_wb", "f.png") is not None, \
            "source fingerprint change must re-render"


def check_corrupt_sidecar_invalidates_cached_figure():
    from urban_pfr.visualization import notebook_helpers as nh
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(tmp, private=True)
        render, figs = nh.make_renderer(run)
        assert render("private", "risk_wb", "f.png") is not None
        (figs / "private" / "f.png.provenance.json").write_text("not json at all")
        assert render("private", "risk_wb", "f.png") is not None, \
            "malformed sidecar must be treated as stale"


# ---- input-error and API-behavior checks ----

def check_corrupt_source_raises_sanitized_error():
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(tmp, private=True)
        target = Path(run) / "private" / "buildings_with_risk.gpkg"
        target.write_bytes(b"this is not a geopackage")
        out = Path(tmp) / "x.png"
        exc = _assert_raises(spec.VisualizationError, visualize_output, run, "private", "risk_ma", out)
        assert "buildings_with_risk.gpkg" in str(exc)
        assert str(Path(tmp).resolve()) not in str(exc), "absolute path leaked into the message"
        assert exc.__cause__ is not None, "original driver exception must be chained"
        assert not out.exists()


def check_successful_render_is_silent_and_returns_result():
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(tmp, private=True)
        stdout_buffer = io.StringIO()
        with contextlib.redirect_stdout(stdout_buffer):
            result = visualize_output(run, "private", "risk_wb", Path(tmp) / "quiet.png")
        assert isinstance(result, VisualizationResult)
        assert stdout_buffer.getvalue() == "", "library API must not print"
        # stderr is not asserted: third-party libraries may emit warnings there


def check_invalid_input_raises_typed_not_systemexit():
    with tempfile.TemporaryDirectory() as tmp:
        run = _make_run_dir(tmp, private=True)
        try:
            visualize_output(run, "private", "not_a_view", Path(tmp) / "x.png")
            raise AssertionError("expected ValidationError")
        except SystemExit:
            raise AssertionError("library API must never raise SystemExit")
        except spec.ValidationError:
            pass


# ---- tile-cache checks (local disk only; no network) ----

def check_tile_cache_prune_removes_old_files():
    import time as _time
    from urban_pfr.visualization import static as static_mod
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "tiles"
        cache.mkdir()
        old = cache / "old_tile.png"
        old.write_bytes(b"x" * 10)
        new = cache / "new_tile.png"
        new.write_bytes(b"y" * 10)
        past = _time.time() - 200 * 86400
        os.utime(str(old), (past, past))
        assert static_mod.prune_tile_cache(max_age_days=90, cache_dir=cache) == (1, 10)
        assert not old.exists() and new.exists()
        assert static_mod.prune_tile_cache(cache_dir=Path(tmp) / "missing") == (0, 0)


def check_tile_cache_dir_is_owner_only():
    from urban_pfr.visualization import static as static_mod
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "tiles"
        root = static_mod._ensure_cache_dir(cache)
        assert root.is_dir()
        assert (root.stat().st_mode & 0o777) == 0o700, "cache dir must be owner-only"


def check_tile_cache_ttl_env_policy():
    import time as _time
    from urban_pfr.visualization import static as static_mod
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "tiles"
        cache.mkdir()
        old = cache / "old.png"
        old.write_bytes(b"x")
        past = _time.time() - 400 * 86400
        os.utime(str(old), (past, past))
        os.environ["URBAN_PFR_TILE_CACHE_MAX_AGE_DAYS"] = "90"
        try:
            static_mod._AUTO_PRUNED = False
            assert static_mod._auto_prune_if_configured(cache_dir=cache) == (1, 1)
            assert not old.exists()
            assert static_mod._auto_prune_if_configured(cache_dir=cache) is None, \
                "TTL prune runs once per process"
        finally:
            del os.environ["URBAN_PFR_TILE_CACHE_MAX_AGE_DAYS"]
            static_mod._AUTO_PRUNED = False
        keep = cache / "keep.png"
        keep.write_bytes(b"y")
        os.utime(str(keep), (past, past))
        static_mod._AUTO_PRUNED = False
        try:
            assert static_mod._auto_prune_if_configured(cache_dir=cache) is None
            assert keep.exists(), "unset TTL env must never delete anything"
        finally:
            static_mod._AUTO_PRUNED = False


# ---- canonical check registry ----

ALL_CHECKS = [
    # specification and classification
    check_golden_head_tail_breaks,
    check_small_dataset_classification_padding,
    check_no_finite_values_raises,
    check_bbox_validation_rules,
    check_tier_view_role_mappings,
    check_view_labels_and_palettes,
    check_risk_view_variable_mappings,
    check_bivariate_specification,
    check_renderer_policy_is_deterministic,
    check_bbox_axis_order_transform,
    check_empty_selection_reports_swap_hint,
    # style and breaks files
    check_style_file_loading,
    check_breaks_file_validation,
    check_precomputed_breaks_variable_matching,
    check_bivariate_precomputed_breaks,
    # rendering
    check_classification_equal_across_bboxes,
    check_pdf_render_produces_output,
    check_styled_precomputed_render_records_provenance,
    check_bivariate_rendering,
    check_basemap_retry_with_injected_fetcher,
    check_basemap_zoom_respects_tile_cap,
    check_boundary_overlay_renders,
    # tiered raw/smoothed risk contract
    check_public_risk_uses_smoothed_healpix,
    check_public_raw_risk_uses_raw_healpix,
    check_private_risk_uses_building_footprints,
    check_raw_and_smoothed_breaks_are_independent,
    check_bivariate_variables_per_tier,
    check_missing_smoothed_variable_does_not_fallback,
    # public/private isolation
    check_public_fails_closed_without_public_outputs,
    check_no_output_files_on_failure,
    check_public_auxiliary_file_is_rejected,
    check_public_error_does_not_disclose_private_source,
    check_symlinked_private_auxiliary_is_rejected,
    check_public_sidecar_has_no_private_leak,
    # provenance
    check_sidecar_source_fingerprint,
    # temporary files and atomic outputs
    check_temporary_paths_are_unique_per_invocation,
    check_failed_render_cleans_own_temp_files,
    check_failed_render_preserves_foreign_temp_files,
    check_output_pair_is_atomic,
    check_render_many_worker_bounds,
    # freshness
    check_identical_request_skips_rerender,
    check_request_change_invalidates_cached_figure,
    check_source_change_invalidates_cached_figure,
    check_corrupt_sidecar_invalidates_cached_figure,
    # input errors and API behavior
    check_corrupt_source_raises_sanitized_error,
    check_successful_render_is_silent_and_returns_result,
    check_invalid_input_raises_typed_not_systemexit,
    # tile cache
    check_tile_cache_prune_removes_old_files,
    check_tile_cache_dir_is_owner_only,
    check_tile_cache_ttl_env_policy,
]


def _check_id(check):
    name = check.__name__
    prefix = "check_"
    return name[len(prefix):] if name.startswith(prefix) else name


if pytest is not None:
    @pytest.mark.parametrize("check", ALL_CHECKS, ids=_check_id)
    def test_visualization(check):
        check()


# ---- local fallback runner (no pytest required) ----

def main():
    failures = []
    for check in ALL_CHECKS:
        try:
            check()
        except Exception as exc:
            failures.append((check.__name__, exc))
            print("FAIL  %s: %s: %s" % (check.__name__, type(exc).__name__, exc))
        else:
            print("PASS  %s" % check.__name__)
    print("%d/%d passed" % (len(ALL_CHECKS) - len(failures), len(ALL_CHECKS)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
