"""
analyzer.py -- Main pipeline orchestrator for urban-pfr.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import warnings

from .indicators import compute_social_vulnerability
from .exposure import calculate_exposure_residents, calculate_exposure_wellbeing
from .hazard import calculate_hazard_mobility_accessibility, calculate_hazard_wellbeing
from .risk import calculate_risk, delaunay_smoothing

warnings.filterwarnings('ignore')


class PFRAnalyzer:
    """
    Pluvial Flood Risk Analyzer.
    Runs the full IPCC risk pipeline: Risk = Hazard x Exposure x Vulnerability.
    """

    def __init__(self, config):
        self.config = config
        self.buildings = None
        self.statistical_units = None
        self.streets = None
        self.flood_layers = None

        city = config.get('project', {}).get('city_name', 'Unknown')
        crs = config.get('project', {}).get('crs', 'Not specified')
        print(f"PFRAnalyzer initialized -- {city}, {crs}")

    def load_data(self, input_gdb=None, flood_dir=None,
                  buildings_layer=None, stats_layer=None, streets_layer=None,
                  buildings_file=None, stats_file=None, streets_file=None,
                  validate_first=False):
        """
        Load spatial data from GDB layers or individual files.

        Priority: individual files > GDB layers.
        Layer names default to config schema values, then fallback to generic names.
        """
        if validate_first:
            from .validation import validate_input_data
            report = validate_input_data(self.config, detailed=True)
            if not report['valid']:
                raise ValueError(
                    f"Validation failed with {len(report['errors'])} errors."
                )

        paths = self.config.get('project', {}).get('paths', {})
        schema = self.config.get('schema', {})

        if input_gdb is None:
            input_gdb = paths.get('input_gdb')
        if flood_dir is None:
            flood_dir = paths.get('flood_dir')

        # Layer names from config, then arguments, then generic defaults
        if buildings_layer is None:
            buildings_layer = schema.get('buildings_layer', 'Buildings')
        if stats_layer is None:
            stats_layer = schema.get('stats_layer', 'StatisticalUnits')
        if streets_layer is None:
            streets_layer = schema.get('streets_layer', 'Streets')

        # Individual file overrides from config
        if buildings_file is None:
            buildings_file = paths.get('buildings_file')
        if stats_file is None:
            stats_file = paths.get('stats_file')
        if streets_file is None:
            streets_file = paths.get('streets_file')

        # Load buildings
        if buildings_file and Path(buildings_file).exists():
            self.buildings = gpd.read_file(buildings_file)
        else:
            self.buildings = gpd.read_file(input_gdb, layer=buildings_layer)

        # Load statistical units
        if stats_file and Path(stats_file).exists():
            self.statistical_units = gpd.read_file(stats_file)
        else:
            self.statistical_units = gpd.read_file(input_gdb, layer=stats_layer)

        # Load streets
        if streets_file and Path(streets_file).exists():
            self.streets = gpd.read_file(streets_file)
        else:
            self.streets = gpd.read_file(input_gdb, layer=streets_layer)

        # Load flood layers -- tries .shp, .gpkg, .geojson for each depth
        self.flood_layers = {}
        flood_depths = self.config.get('hazard_settings', {}).get(
            'flood_depths', [20, 30, 40, 50, 60, 70, 80, 90, 100]
        )
        for depth in flood_depths:
            for ext in ['.shp', '.gpkg', '.geojson']:
                flood_file = Path(flood_dir) / f"Flood_{depth}{ext}"
                if flood_file.exists():
                    try:
                        self.flood_layers[depth] = gpd.read_file(flood_file)
                        break
                    except Exception:
                        continue

        print(f"Loaded: {len(self.buildings)} buildings, "
              f"{len(self.statistical_units)} stat units, "
              f"{len(self.streets)} streets, "
              f"{len(self.flood_layers)} flood layers")
        return self

    def compute_vulnerability(self):
        schema = self.config.get('schema', {})
        result = compute_social_vulnerability(
            self.buildings, self.statistical_units,
            sensitivity_fields=schema.get('sensitivity_fields'),
            coping_fields=schema.get('coping_fields'),
            config=self.config
        )
        if isinstance(result, tuple):
            self.buildings, self.statistical_units = result
        else:
            self.buildings = result
        return self

    def compute_exposure(self):
        schema = self.config.get('schema', {})
        self.buildings = calculate_exposure_residents(
            self.buildings, self.statistical_units,
            residents_col=schema.get('residents_col', 'Residents'),
            living_area_col=schema.get('living_area_col', 'LivingArea'),
            floors_col=schema.get('floors_col', 'Floors'),
            building_type_col=schema.get('building_type_col', 'Building_type')
        )
        self.buildings = calculate_exposure_wellbeing(
            self.buildings,
            floors_col=schema.get('floors_col', 'Floors'),
            building_type_col=schema.get('building_type_col', 'Building_type')
        )
        return self

    def compute_hazard(self):
        hazard_cfg = self.config.get('hazard_settings', {})

        flood_depth_cm = int(hazard_cfg.get('flood_threshold', 0.3) * 100)
        flood_for_hma = self.flood_layers.get(
            flood_depth_cm, list(self.flood_layers.values())[0]
        )

        self.buildings = calculate_hazard_mobility_accessibility(
            self.buildings, self.streets, flood_for_hma,
            buffers=hazard_cfg.get('hma_buffers', [5, 15, 30]),
            min_area_threshold=hazard_cfg.get('min_area_threshold', 0.0),
            shape_param=hazard_cfg.get('shape_param', 0.25)
        )

        self.buildings = calculate_hazard_wellbeing(
            self.buildings, self.flood_layers,
            buffer_distance=hazard_cfg.get('hwb_buffer', 2),
            shape_param=hazard_cfg.get('shape_param', 0.25)
        )
        return self

    def compute_risk(self):
        risk_cfg = self.config.get('risk_settings', {})
        self.buildings = calculate_risk(
            self.buildings,
            hazard_col_ma='HMA', hazard_col_wb='HWB',
            exposure_col_ma='R', exposure_col_wb='R_G',
            vulnerability_col='SVPF',
            weight_hazard=risk_cfg.get('weight_hazard', 1.0),
            weight_exposure=risk_cfg.get('weight_exposure', 1.0),
            weight_vulnerability=risk_cfg.get('weight_vulnerability', 1.0)
        )
        return self

    def apply_smoothing(self):
        risk_cfg = self.config.get('risk_settings', {})
        n_iter = risk_cfg.get('smoothing_iterations', 3)
        max_nb = risk_cfg.get('max_neighbors', 10)
        dist_thresh = risk_cfg.get('distance_threshold', 100)

        self.buildings['PFRMA_smoothed'] = delaunay_smoothing(
            self.buildings, risk_col='PFRMA', n_iterations=n_iter,
            max_neighbors=max_nb, distance_threshold=dist_thresh
        )
        self.buildings['PFRWB_smoothed'] = delaunay_smoothing(
            self.buildings, risk_col='PFRWB', n_iterations=n_iter,
            max_neighbors=max_nb, distance_threshold=dist_thresh
        )
        return self

    def run_pipeline(self, skip_smoothing=False):
        """Run all steps. Returns (buildings_gdf, statistical_units_gdf)."""
        print("Running PFR pipeline...")
        self.compute_vulnerability()
        self.compute_exposure()
        self.compute_hazard()
        self.compute_risk()

        if skip_smoothing:
            self.buildings['PFRMA_smoothed'] = self.buildings['PFRMA']
            self.buildings['PFRWB_smoothed'] = self.buildings['PFRWB']
        else:
            self.apply_smoothing()

        print("Pipeline complete.")
        return self.buildings, self.statistical_units

    def visualize(self, save_path=None, title_prefix=None):
        if self.buildings is None or self.statistical_units is None:
            raise ValueError("Must run pipeline before visualization")

        from .viz import create_risk_visualization

        if title_prefix is None:
            city = self.config.get('project', {}).get('city_name', '')
            title_prefix = f"{city} - " if city else ""

        n_classes = self.config.get('risk_settings', {}).get('n_classes', 5)

        return create_risk_visualization(
            self.buildings, self.statistical_units,
            pfrma_column='PFRMA_smoothed', pfrwb_column='PFRWB_smoothed',
            n_classes=n_classes, title_prefix=title_prefix, save_path=save_path,
            config=self.config
        )

    def save_results(self, output_dir=None, format='gpkg'):
        if self.buildings is None or self.statistical_units is None:
            raise ValueError("Must run pipeline before saving results")

        if output_dir is None:
            output_dir = self.config.get('project', {}).get('paths', {}).get('output_dir', './outputs')

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        drivers = {'gpkg': 'GPKG', 'shp': 'ESRI Shapefile', 'geojson': 'GeoJSON'}
        ext = format if format in drivers else 'gpkg'
        driver = drivers.get(ext, 'GPKG')

        b_path = f"{output_dir}/buildings_with_risk.{ext}"
        s_path = f"{output_dir}/statistical_units_with_vulnerability.{ext}"

        self.buildings.to_file(b_path, driver=driver)
        self.statistical_units.to_file(s_path, driver=driver)

        print(f"Results saved to {output_dir}")
        return {'buildings': b_path, 'statistical_units': s_path}