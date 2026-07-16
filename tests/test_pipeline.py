"""
Smoke tests for the urban_pfr pipeline using synthetic data.
No external data files required — all test data is generated inline.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import box, Polygon

try:
    import healpix_geo
    HAS_HEALPIX = True
except ImportError:
    HAS_HEALPIX = False


# ── Fixtures ──

@pytest.fixture
def synthetic_buildings():
    """Create 20 synthetic buildings in a 200m x 200m area."""
    np.random.seed(42)
    n = 20
    x0, y0 = 565000, 5935000  # UTM32N coords (Hamburg area)

    polys = []
    for i in range(n):
        x = x0 + (i % 5) * 40
        y = y0 + (i // 5) * 40
        polys.append(box(x, y, x + 15, y + 10))

    return gpd.GeoDataFrame({
        'ID': range(n),
        'StatisticalUnit': [1] * 10 + [2] * 10,
        'Residents': [100] * 10 + [200] * 10,
        'LivingArea': [5000.0] * 10 + [8000.0] * 10,
        'Floors': np.random.randint(1, 5, n),
        'Building_type': [1] * 15 + [2] * 5,
        'ES': np.random.randint(5, 50, n),
        'C': np.random.randint(5, 30, n),
        'WR': np.random.randint(0, 20, n),
        'EDQ': np.random.randint(0, 15, n),
    }, geometry=polys, crs='EPSG:25832')


@pytest.fixture
def synthetic_stats(synthetic_buildings):
    """Derive statistical units from buildings."""
    cols = ['StatisticalUnit', 'Residents', 'LivingArea', 'ES', 'C', 'WR', 'EDQ', 'geometry']
    return synthetic_buildings[cols].dissolve(by='StatisticalUnit', as_index=False)


@pytest.fixture
def synthetic_streets():
    """Create a street polygon covering the test area."""
    x0, y0 = 564990, 5935015
    street = Polygon([
        (x0, y0), (x0 + 220, y0), (x0 + 220, y0 + 5), (x0, y0 + 5)
    ])
    return gpd.GeoDataFrame(geometry=[street], crs='EPSG:25832')


@pytest.fixture
def synthetic_flood():
    """Create a flood polygon covering part of the test area."""
    x0, y0 = 565000, 5935000
    flood = Polygon([
        (x0, y0), (x0 + 100, y0), (x0 + 100, y0 + 80), (x0, y0 + 80)
    ])
    return gpd.GeoDataFrame(geometry=[flood], crs='EPSG:25832')


@pytest.fixture
def synthetic_flood_layers(synthetic_flood):
    """Create flood layers at multiple depths."""
    return {30: synthetic_flood, 50: synthetic_flood, 100: synthetic_flood}


@pytest.fixture
def config():
    return {
        'topsis_weights': {'sensitivity': [0.7, 0.3], 'coping_capacity': [0.5, 0.5], 'svi': [0.5, 0.5]},
        'topsis_entropy': {'sensitivity': True, 'coping_capacity': True, 'svi': True},
        'svpf_threshold': 0.25,
        'svpf_transform': 2.0,
    }


# ── Indicator tests ──

class TestIndicators:
    def test_topsis_basic(self):
        from urban_pfr.indicators import topsis
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
        scores, weights = topsis(df, weights=[0.5, 0.5], use_entropy=False)
        assert len(scores) == 5
        assert scores.min() >= 0
        assert scores.max() <= 1

    def test_topsis_with_entropy(self):
        from urban_pfr.indicators import topsis
        df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
        scores, _ = topsis(df, weights=[0.5, 0.5], use_entropy=True)
        assert len(scores) == 5

    def test_compute_social_vulnerability(self, synthetic_buildings, synthetic_stats, config):
        from urban_pfr.indicators import compute_social_vulnerability
        buildings, stats = compute_social_vulnerability(
            synthetic_buildings, synthetic_stats,
            sensitivity_fields=['ES', 'C'],
            coping_fields=['WR', 'EDQ'],
            config=config,
        )
        assert 'Sensitivity' in buildings.columns
        assert 'CopingCapacity' in buildings.columns
        assert 'SVI' in buildings.columns
        assert 'SVPF' in buildings.columns
        assert buildings['SVPF'].notna().all()
        # Stats should have aggregated values
        assert 'SVPF' in stats.columns


# ── Exposure tests ──

class TestExposure:
    def test_calculate_exposure_residents(self, synthetic_buildings, synthetic_stats):
        from urban_pfr.exposure import calculate_exposure_residents
        result = calculate_exposure_residents(synthetic_buildings, synthetic_stats)
        assert 'R' in result.columns
        assert 'Area_house' in result.columns
        assert (result['R'] >= 0).all()
        # Residential buildings should have residents
        residential = result[result['Building_type'] == 1]
        assert (residential['R'] > 0).any()

    def test_calculate_exposure_wellbeing(self, synthetic_buildings, synthetic_stats):
        from urban_pfr.exposure import calculate_exposure_residents, calculate_exposure_wellbeing
        buildings = calculate_exposure_residents(synthetic_buildings, synthetic_stats)
        result = calculate_exposure_wellbeing(buildings)
        assert 'R_G' in result.columns
        # Non-residential buildings should have R_G = 0
        non_res = result[result['Building_type'] == 2]
        assert (non_res['R_G'] == 0).all()


# ── Hazard tests ──

class TestHazard:
    def test_hma(self, synthetic_buildings, synthetic_streets, synthetic_flood):
        from urban_pfr.hazard import calculate_hazard_mobility_accessibility
        result = calculate_hazard_mobility_accessibility(
            synthetic_buildings, synthetic_streets, synthetic_flood,
            buffers=[5, 15], shape_param=0.25,
        )
        assert 'HMA' in result.columns
        assert result['HMA'].min() >= 0
        assert result['HMA'].max() <= 1

    def test_hwb(self, synthetic_buildings, synthetic_flood_layers):
        from urban_pfr.hazard import calculate_hazard_wellbeing
        result = calculate_hazard_wellbeing(
            synthetic_buildings, synthetic_flood_layers,
            buffer_distance=2, shape_param=0.25,
            depth_thresholds=[30, 50, 100],
        )
        assert 'HWB' in result.columns
        assert result['HWB'].min() >= 0
        # Unclipped by default (hwb_clip_max=None): each depth layer
        # contributes at most 1, so HWB is bounded by len(depth_thresholds).
        assert result['HWB'].max() <= 3

    def test_hwb_optional_clip(self, synthetic_buildings, synthetic_flood_layers):
        from urban_pfr.hazard import calculate_hazard_wellbeing
        result = calculate_hazard_wellbeing(
            synthetic_buildings, synthetic_flood_layers,
            buffer_distance=2, shape_param=0.25,
            depth_thresholds=[30, 50, 100],
            hwb_clip_max=1.0,
        )
        assert result['HWB'].max() <= 1  # paper-public mode clips to [0, 1]


# ── Risk tests ──

class TestRisk:
    def test_calculate_risk(self, synthetic_buildings):
        from urban_pfr.risk import calculate_risk
        buildings = synthetic_buildings.copy()
        buildings['HMA'] = np.random.rand(len(buildings))
        buildings['HWB'] = np.random.rand(len(buildings))
        buildings['R'] = np.random.rand(len(buildings)) * 10
        buildings['R_G'] = np.random.rand(len(buildings)) * 5
        buildings['SVI'] = np.random.rand(len(buildings))

        result = calculate_risk(buildings, vulnerability_col='SVI')
        assert 'PFRMA' in result.columns
        assert 'PFRWB' in result.columns
        assert (result['PFRMA'] >= 0).all()

    def test_delaunay_smoothing(self, synthetic_buildings):
        from urban_pfr.risk import delaunay_smoothing
        buildings = synthetic_buildings.copy()
        buildings['PFRMA'] = np.random.rand(len(buildings)) * 10

        smoothed = delaunay_smoothing(buildings, 'PFRMA', n_iterations=2)
        assert len(smoothed) == len(buildings)
        # Smoothing should reduce variance
        assert smoothed.std() <= buildings['PFRMA'].std() * 1.5


# ── HEALPix tests ──

@pytest.mark.skipif(not HAS_HEALPIX, reason="healpix-geo not installed")
class TestHEALPix:
    def test_aggregate_to_healpix(self, synthetic_buildings):
        from urban_pfr.healpix_agg import aggregate_to_healpix
        buildings = synthetic_buildings.copy()
        buildings['PFRMA'] = np.random.rand(len(buildings))
        buildings['HMA'] = np.random.rand(len(buildings))

        result = aggregate_to_healpix(
            buildings, depth=12, risk_columns=['PFRMA', 'HMA'], min_buildings=1,
        )
        assert 'healpix_id' in result.columns
        assert 'n_buildings' in result.columns
        assert 'PFRMA' in result.columns
        assert result.crs.to_epsg() == 4326
        assert len(result) > 0

    def test_min_buildings_filter(self, synthetic_buildings):
        from urban_pfr.healpix_agg import aggregate_to_healpix
        buildings = synthetic_buildings.copy()
        buildings['PFRMA'] = 1.0

        # With high min_buildings, fewer cells should survive
        result_low = aggregate_to_healpix(buildings, depth=15, risk_columns=['PFRMA'], min_buildings=1)
        result_high = aggregate_to_healpix(buildings, depth=15, risk_columns=['PFRMA'], min_buildings=10)
        assert len(result_high) <= len(result_low)


# ── Geometry lifecycle tests (canonical footprints vs Thiessen derivative) ──

@pytest.fixture
def analyzer_config(config):
    cfg = dict(config)
    cfg['project'] = {'city_name': 'synthetic', 'crs': 'EPSG:25832', 'paths': {}}
    cfg['schema'] = {'sensitivity_fields': ['ES', 'C'], 'coping_fields': ['WR', 'EDQ']}
    cfg['hazard_settings'] = {'hma_buffers': [5, 15], 'flood_depths': [30, 50, 100], 'flood_threshold': 0.3}
    cfg['risk_settings'] = {'smoothing_iterations': 2}
    return cfg


def _make_analyzer(cfg, buildings, stats, streets, flood_layers):
    from urban_pfr.analyzer import PFRAnalyzer
    analyzer = PFRAnalyzer(cfg)
    analyzer.buildings = buildings.copy()
    analyzer.statistical_units = stats.copy()
    analyzer.streets = streets.copy()
    analyzer.flood_layers = {k: v.copy() for k, v in flood_layers.items()}
    return analyzer


def _patch_healpix(monkeypatch, fake):
    """Replace ``urban_pfr.healpix_agg.aggregate_to_healpix`` with a test double.

    When the optional ``healpix_geo`` dependency is not installed, a minimal
    module stub is registered in ``sys.modules`` first so that
    ``urban_pfr.healpix_agg`` remains importable; the aggregation entry point
    is then monkeypatched with the supplied fake.
    """
    import sys, types, importlib
    if not HAS_HEALPIX and 'healpix_geo' not in sys.modules:
        pkg = types.ModuleType('healpix_geo')
        nested = types.ModuleType('healpix_geo.nested')
        pkg.nested = nested
        monkeypatch.setitem(sys.modules, 'healpix_geo', pkg)
        monkeypatch.setitem(sys.modules, 'healpix_geo.nested', nested)
    module = importlib.import_module('urban_pfr.healpix_agg')
    monkeypatch.setattr(module, 'aggregate_to_healpix', fake)


class TestGeometryLifecycle:
    def test_footprints_unchanged_after_pipeline(self, analyzer_config, synthetic_buildings,
                                                 synthetic_stats, synthetic_streets,
                                                 synthetic_flood_layers):
        orig_geom = synthetic_buildings.geometry.copy()
        analyzer = _make_analyzer(analyzer_config, synthetic_buildings, synthetic_stats,
                                  synthetic_streets, synthetic_flood_layers)
        analyzer.run_pipeline()
        assert len(analyzer.buildings) == len(orig_geom)
        assert list(analyzer.buildings.index) == list(orig_geom.index)
        assert analyzer.buildings.geometry.geom_equals(orig_geom).all()

    def test_thiessen_derivative_is_separate(self, analyzer_config, synthetic_buildings,
                                             synthetic_stats, synthetic_streets,
                                             synthetic_flood_layers):
        analyzer = _make_analyzer(analyzer_config, synthetic_buildings, synthetic_stats,
                                  synthetic_streets, synthetic_flood_layers)
        analyzer.run_pipeline()
        assert isinstance(analyzer.buildings_thiessen, gpd.GeoDataFrame)
        assert analyzer.buildings_thiessen is not analyzer.buildings
        assert not analyzer.buildings_thiessen.geometry.geom_equals(analyzer.buildings.geometry).all()

    def test_skip_thiessen_leaves_derivative_none(self, analyzer_config, synthetic_buildings,
                                                  synthetic_stats, synthetic_streets,
                                                  synthetic_flood_layers):
        analyzer = _make_analyzer(analyzer_config, synthetic_buildings, synthetic_stats,
                                  synthetic_streets, synthetic_flood_layers)
        analyzer.run_pipeline(skip_thiessen=True)
        assert analyzer.buildings_thiessen is None
        assert analyzer.buildings.geometry.geom_equals(synthetic_buildings.geometry).all()

    def test_save_results_uses_canonical_footprints(self, analyzer_config, synthetic_buildings,
                                                    synthetic_stats, synthetic_streets,
                                                    synthetic_flood_layers, tmp_path, monkeypatch):
        captured = {}

        def fake_agg(gdf, **kwargs):
            # Test double for aggregate_to_healpix: records the geometry that
            # save_results passes in, then returns a minimal schema-valid
            # HEALPix cell. The returned values are placeholders for I/O
            # testing only and carry no scientific meaning.
            captured['geometry'] = gdf.geometry.copy()
            return gpd.GeoDataFrame({'healpix_id': [1], 'n_buildings': [3], 'PFRMA': [0.1]},
                                    geometry=[box(9.9, 53.5, 10.0, 53.6)], crs='EPSG:4326')

        _patch_healpix(monkeypatch, fake_agg)
        analyzer = _make_analyzer(analyzer_config, synthetic_buildings, synthetic_stats,
                                  synthetic_streets, synthetic_flood_layers)
        analyzer.run_pipeline()
        paths = analyzer.save_results(output_dir=str(tmp_path))
        saved = gpd.read_file(paths['buildings_private'])
        assert len(saved) == len(synthetic_buildings)
        assert saved.geometry.geom_equals(synthetic_buildings.geometry).all()
        assert not saved.geometry.geom_equals(analyzer.buildings_thiessen.geometry).all()
        assert captured['geometry'].geom_equals(analyzer.buildings.geometry).all()

    def test_visualize_prefers_thiessen(self, analyzer_config, synthetic_buildings,
                                        synthetic_stats, synthetic_streets,
                                        synthetic_flood_layers, monkeypatch):
        import urban_pfr.viz as viz_mod
        called = {}

        def fake_viz(buildings_gdf, stats_gdf, **kwargs):
            called['buildings'] = buildings_gdf
            return None, None

        monkeypatch.setattr(viz_mod, 'create_risk_visualization', fake_viz)
        analyzer = _make_analyzer(analyzer_config, synthetic_buildings, synthetic_stats,
                                  synthetic_streets, synthetic_flood_layers)
        analyzer.run_pipeline()
        analyzer.visualize()
        assert called['buildings'] is analyzer.buildings_thiessen

    def test_visualize_falls_back_to_footprints(self, analyzer_config, synthetic_buildings,
                                                synthetic_stats, synthetic_streets,
                                                synthetic_flood_layers, monkeypatch):
        import urban_pfr.viz as viz_mod
        called = {}

        def fake_viz(buildings_gdf, stats_gdf, **kwargs):
            called['buildings'] = buildings_gdf
            return None, None

        monkeypatch.setattr(viz_mod, 'create_risk_visualization', fake_viz)
        analyzer = _make_analyzer(analyzer_config, synthetic_buildings, synthetic_stats,
                                  synthetic_streets, synthetic_flood_layers)
        analyzer.run_pipeline(skip_thiessen=True)
        analyzer.visualize()
        assert called['buildings'] is analyzer.buildings

    def test_apply_thiessen_numerically_neutral(self, analyzer_config, synthetic_buildings,
                                                synthetic_stats, synthetic_streets,
                                                synthetic_flood_layers):
        analyzer = _make_analyzer(analyzer_config, synthetic_buildings, synthetic_stats,
                                  synthetic_streets, synthetic_flood_layers)
        analyzer.run_pipeline(skip_thiessen=True)
        before = analyzer.buildings[['PFRMA', 'PFRWB', 'SVPF']].copy()
        frame_before = analyzer.buildings
        analyzer.apply_thiessen()
        assert analyzer.buildings is frame_before
        pd.testing.assert_frame_equal(before, analyzer.buildings[['PFRMA', 'PFRWB', 'SVPF']])
        assert analyzer.buildings_thiessen is not None


# ── TOPSIS granularity behaviour (feature untouched by the geometry fix) ──

class TestTopsisGranularity:
    def test_stat_unit_constant_within_unit(self, synthetic_buildings, synthetic_stats, config):
        from urban_pfr.indicators import compute_social_vulnerability
        cfg = dict(config)
        cfg['topsis_granularity'] = 'stat_unit'
        buildings, _ = compute_social_vulnerability(
            synthetic_buildings, synthetic_stats,
            sensitivity_fields=['ES', 'C'], coping_fields=['WR', 'EDQ'], config=cfg)
        assert (buildings.groupby('StatisticalUnit')['SVI'].nunique() <= 1).all()
        assert (buildings.groupby('StatisticalUnit')['SVPF'].nunique() <= 1).all()

    def test_building_mode_varies_within_unit(self, synthetic_buildings, synthetic_stats, config):
        from urban_pfr.indicators import compute_social_vulnerability
        cfg = dict(config)
        cfg['topsis_granularity'] = 'building'
        buildings, _ = compute_social_vulnerability(
            synthetic_buildings, synthetic_stats,
            sensitivity_fields=['ES', 'C'], coping_fields=['WR', 'EDQ'], config=cfg)
        assert (buildings.groupby('StatisticalUnit')['SVI'].nunique() > 1).any()
