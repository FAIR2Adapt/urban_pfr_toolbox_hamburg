
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import warnings

# Allow loading large GeoJSON files (e.g. 651MB Streets.geojson)
os.environ.setdefault("OGR_GEOJSON_MAX_OBJ_SIZE", "0")

from .indicators import compute_social_vulnerability
from .exposure import calculate_exposure_residents, calculate_exposure_wellbeing
from .hazard import calculate_hazard_mobility_accessibility, calculate_hazard_wellbeing
from .risk import calculate_risk, delaunay_smoothing
from .thiessen import create_thiessen_polygons

warnings.filterwarnings('ignore')


class PFRAnalyzer:
    """
    Pluvial Flood Risk Analyzer.
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

    def _resolve_file_path(self, paths, *keys):
        for key in keys:
            val = paths.get(key)
            if val and Path(val).exists():
                return val
        return None

    def _ensure_crs(self, gdf, label="layer"):
        target_crs = self.config.get('project', {}).get('crs')
        if target_crs and gdf.crs and str(gdf.crs) != target_crs:
            print(f"  Reprojecting {label}: {gdf.crs} → {target_crs}")
            gdf = gdf.to_crs(target_crs)
        return gdf

    def load_data(self, input_gdb=None, flood_dir=None,
                  buildings_layer=None, stats_layer=None, streets_layer=None,
                  buildings_file=None, stats_file=None, streets_file=None,
                  validate_first=False):
        """
          - paths.buildings 
          - paths.statistical_units 
          - paths.streets 
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

        # Schema field names for dissolve fallback
        stat_unit_col = schema.get('stat_unit_col', 'StatisticalUnit')
        residents_col = schema.get('residents_col', 'Residents')
        living_area_col = schema.get('living_area_col', 'LivingArea')

        #  Resolve file paths
        if buildings_file is None:
            buildings_file = self._resolve_file_path(
                paths, 'buildings', 'buildings_file')
        if stats_file is None:
            stats_file = self._resolve_file_path(
                paths, 'statistical_units', 'stats_file', 'stats')
        if streets_file is None:
            streets_file = self._resolve_file_path(
                paths, 'streets', 'streets_file')

        # Load buildings 
        if buildings_file:
            self.buildings = gpd.read_file(buildings_file)
        elif input_gdb and Path(input_gdb).exists():
            self.buildings = gpd.read_file(input_gdb, layer=buildings_layer)
        else:
            raise FileNotFoundError(
                "No buildings source found. Set paths.buildings or paths.input_gdb in config.")
        self.buildings = self._ensure_crs(self.buildings, "buildings")

        # Load statistical units
        if stats_file:
            self.statistical_units = gpd.read_file(stats_file)
        elif input_gdb and Path(input_gdb).exists():
            self.statistical_units = gpd.read_file(input_gdb, layer=stats_layer)
        elif self.buildings is not None and stat_unit_col in self.buildings.columns:
            # Filter out buildings with null StatisticalUnit (e.g. outlier islands)
            n_null = self.buildings[stat_unit_col].isna().sum()
            if n_null > 0:
                print(f"  Dropping {n_null} buildings with null {stat_unit_col}")
                self.buildings = self.buildings[self.buildings[stat_unit_col].notna()].reset_index(drop=True)

            # Derive statistical units from buildings by dissolving
            print(f"  Deriving statistical units from buildings (dissolve by {stat_unit_col})...")
            stat_cols = [stat_unit_col, residents_col, living_area_col, 'geometry']
            # Add sensitivity/coping fields if present
            for f in schema.get('sensitivity_fields', []) + schema.get('coping_fields', []):
                if f in self.buildings.columns and f not in stat_cols:
                    stat_cols.append(f)
            available = [c for c in stat_cols if c in self.buildings.columns]
            self.statistical_units = self.buildings[available].dissolve(
                by=stat_unit_col, as_index=False)
            print(f"  Derived {len(self.statistical_units)} statistical units")
        else:
            raise FileNotFoundError(
                "No statistical units source found. Set paths.statistical_units or paths.input_gdb.")
        self.statistical_units = self._ensure_crs(self.statistical_units, "statistical_units")

        # Load streets 
        if streets_file:
            self.streets = gpd.read_file(streets_file)
        elif input_gdb and Path(input_gdb).exists():
            self.streets = gpd.read_file(input_gdb, layer=streets_layer)
        else:
            raise FileNotFoundError(
                "No streets source found. Set paths.streets or paths.input_gdb.")
        self.streets = self._ensure_crs(self.streets, "streets")

        # Load flood layers 
        self.flood_layers = {}
        flood_depths = self.config.get('hazard_settings', {}).get(
            'flood_depths', [20, 30, 40, 50, 60, 70, 80, 90, 100]
        )
        for depth in flood_depths:
            for ext in ['.fgb', '.shp', '.gpkg', '.geojson']:
                flood_file = Path(flood_dir) / f"Flood_{depth}{ext}"
                if flood_file.exists():
                    try:
                        fl = gpd.read_file(flood_file)
                        fl = self._ensure_crs(fl, f"Flood_{depth}")
                        self.flood_layers[depth] = fl
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
        exposure_cfg = self.config.get('exposure_settings', {})
        residential_types = exposure_cfg.get('residential_types', [1])
        floor_deduction = exposure_cfg.get('non_residential_floor_deduction', 1)

        self.buildings = calculate_exposure_residents(
            self.buildings, self.statistical_units,
            residents_col=schema.get('residents_col', 'Residents'),
            living_area_col=schema.get('living_area_col', 'LivingArea'),
            floors_col=schema.get('floors_col', 'Floors'),
            building_type_col=schema.get('building_type_col', 'Building_type'),
            residential_types=residential_types,
            non_residential_floor_deduction=floor_deduction,
        )
        self.buildings = calculate_exposure_wellbeing(
            self.buildings,
            floors_col=schema.get('floors_col', 'Floors'),
            building_type_col=schema.get('building_type_col', 'Building_type'),
            residential_types=residential_types,
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
            shape_param=hazard_cfg.get('shape_param', 0.25),
            #Added
            hwb_clip_max=hazard_cfg.get('hwb_clip_max', None)
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
            weight_vulnerability=risk_cfg.get('weight_vulnerability', 1.0),
            config=self.config
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

    def apply_thiessen(self):
        """Replace building footprints with Thiessen polygons for visualization."""
        paths = self.config.get('project', {}).get('paths', {})
        boundary_file = paths.get('boundary_file')

        boundary = None
        if boundary_file and Path(boundary_file).exists():
            print(f"  Loading boundary from {boundary_file}")
            boundary = gpd.read_file(boundary_file)
            boundary = self._ensure_crs(boundary, "boundary")

        self.buildings = create_thiessen_polygons(
            self.buildings, boundary=boundary
        )
        return self

    def run_pipeline(self, skip_smoothing=False, skip_thiessen=False):
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

        if not skip_thiessen:
            self.apply_thiessen()

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
            pfrma_column='PFRMA', pfrwb_column='PFRWB',
            n_classes=n_classes, title_prefix=title_prefix, save_path=save_path,
            config=self.config
        )

    def save_results(self, output_dir=None, format='gpkg', web_export=True):
        if self.buildings is None or self.statistical_units is None:
            raise ValueError("Must run pipeline before saving results")

        if output_dir is None:
            output_dir = self.config.get('project', {}).get('paths', {}).get('output_dir', './outputs')

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        drivers = {'gpkg': 'GPKG', 'shp': 'ESRI Shapefile', 'geojson': 'GeoJSON'}
        ext = format if format in drivers else 'gpkg'
        driver = drivers.get(ext, 'GPKG')

        paths = {}

        # ── Private outputs (for municipalities) ──
        # Full building-level data with all demographics and risk values
        private_dir = f"{output_dir}/private"
        Path(private_dir).mkdir(parents=True, exist_ok=True)

        b_path = f"{private_dir}/buildings_with_risk.{ext}"
        s_path = f"{private_dir}/statistical_units_with_vulnerability.{ext}"

        self.buildings.to_file(b_path, driver=driver)
        self.statistical_units.to_file(s_path, driver=driver)
        paths['buildings_private'] = b_path
        paths['statistical_units_private'] = s_path

        # Private FlatGeobuf (EPSG:4326, web-ready)
        b_fgb = f"{private_dir}/buildings_with_risk.fgb"
        s_fgb = f"{private_dir}/statistical_units_with_vulnerability.fgb"
        self.buildings.dropna(subset=['geometry']).to_crs('EPSG:4326').to_file(b_fgb, driver='FlatGeobuf')
        self.statistical_units.dropna(subset=['geometry']).to_crs('EPSG:4326').to_file(s_fgb, driver='FlatGeobuf')
        paths['buildings_private_fgb'] = b_fgb
        paths['statistical_units_private_fgb'] = s_fgb
        print(f"  Private results (full data): {private_dir}/")

        # ── Public outputs (safe to publish) ──
        public_dir = f"{output_dir}/public"
        Path(public_dir).mkdir(parents=True, exist_ok=True)

        output_cfg = self.config.get('output_settings', {})
        healpix_depth = output_cfg.get('healpix_depth', 15)
        min_buildings = output_cfg.get('min_buildings', 3)

        # Risk columns (no demographics)
        public_risk_cols = [
            'PFRMA', 'PFRWB', 'PFRMA_smoothed', 'PFRWB_smoothed',
            'HMA', 'HWB',
        ]
        available_risk = [c for c in public_risk_cols if c in self.buildings.columns]

        # HEALPix on WGS84 ellipsoid — equal-area, privacy-safe (default public output)
        from .healpix_agg import aggregate_to_healpix
        print(f"  Public risk: HEALPix depth {healpix_depth} (min {min_buildings} buildings/cell)")
        healpix_risk = aggregate_to_healpix(
            self.buildings, depth=healpix_depth,
            risk_columns=available_risk, min_buildings=min_buildings,
        )
        # Already in EPSG:4326
        risk_fgb = f"{public_dir}/risk_healpix.fgb"
        healpix_risk.to_file(risk_fgb, driver='FlatGeobuf')
        paths['risk_fgb'] = risk_fgb

        risk_gpkg = f"{public_dir}/risk_healpix.{ext}"
        healpix_risk.to_crs(self.buildings.crs).to_file(risk_gpkg, driver=driver)
        paths['risk_public'] = risk_gpkg

        # Public vulnerability — also on HEALPix grid
        vuln_columns = [c for c in ['Sensitivity', 'CopingCapacity', 'SVI', 'SVPF']
                        if c in self.buildings.columns]

        if vuln_columns:
            print(f"  Public vulnerability: HEALPix depth {healpix_depth}")
            healpix_vuln = aggregate_to_healpix(
                self.buildings, depth=healpix_depth,
                risk_columns=vuln_columns, min_buildings=min_buildings,
            )
            vuln_fgb = f"{public_dir}/vulnerability_healpix.fgb"
            healpix_vuln.to_file(vuln_fgb, driver='FlatGeobuf')
            paths['vulnerability_fgb'] = vuln_fgb

            vuln_gpkg = f"{public_dir}/vulnerability_healpix.{ext}"
            healpix_vuln.to_crs(self.buildings.crs).to_file(vuln_gpkg, driver=driver)
            paths['vulnerability_public'] = vuln_gpkg

        print(f"  Public results (HEALPix + vulnerability): {public_dir}/")
        print(f"Results saved to {output_dir}")
        return paths
