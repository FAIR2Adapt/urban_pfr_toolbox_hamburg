"""
validation.py – Data validation and preprocessing for urban-pfr.
Checks file existence, CRS consistency, required fields, data quality,
and spatial relationships before running the flood risk pipeline.
"""

import os
from pathlib import Path
import geopandas as gpd
import pandas as pd
from collections import defaultdict
import warnings
import tempfile
import shutil

warnings.filterwarnings('ignore')


def preprocess_and_fix_data(config_or_path, auto_fix=True, backup=True):
    """
    Detect and optionally fix CRS mismatches across all layers.
    Returns dict with 'fixed', 'issues_found', 'fixes_applied', 'backup_location'.
    """
    if isinstance(config_or_path, str):
        import yaml
        with open(config_or_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = config_or_path

    fix_report = {
        'fixed': False,
        'issues_found': [],
        'fixes_applied': [],
        'backup_location': None
    }

    try:
        input_gdb = config['project']['paths']['input_gdb']
        flood_dir = config['project']['paths']['flood_dir']
        target_crs = config['project']['crs']
    except KeyError as e:
        print(f"Missing configuration key: {e}")
        return fix_report

    if backup:
        backup_dir = Path("data_backup_original")
        if not backup_dir.exists():
            backup_dir.mkdir()
        fix_report['backup_location'] = str(backup_dir.absolute())

    schema = config.get('schema', {})
    layer_checks = [
        ('buildings', schema.get('buildings_layer', 'Buildings'), 'buildings_fixed.gpkg'),
        ('statistical_units', schema.get('stats_layer', 'StatisticalUnits'), 'statistical_units_fixed.gpkg'),
        ('streets', schema.get('streets_layer', 'Streets'), 'streets_fixed.gpkg'),
    ]

    for name, layer_id, fixed_name in layer_checks:
        try:
            gdf = gpd.read_file(input_gdb, layer=layer_id)
            if str(gdf.crs) != target_crs:
                fix_report['issues_found'].append(f"{name} CRS: {gdf.crs} -> {target_crs}")
                if auto_fix:
                    if backup:
                        gdf.to_file(backup_dir / f"{name}_original.gpkg", driver='GPKG')
                    gdf.to_crs(target_crs).to_file(fixed_name, driver='GPKG')
                    fix_report['fixes_applied'].append(f"{name} converted to {target_crs}")
        except Exception as e:
            fix_report['issues_found'].append(f"{name} error: {e}")

    flood_path = Path(flood_dir)
    flood_files = (list(flood_path.glob("Flood_*.shp")) +
                   list(flood_path.glob("Flood_*.gpkg")) +
                   list(flood_path.glob("Flood_*.geojson")))

    for flood_file in flood_files:
        try:
            gdf = gpd.read_file(flood_file)
            depth = flood_file.stem.split('_')[-1]
            if str(gdf.crs) != target_crs:
                fix_report['issues_found'].append(f"Flood_{depth} CRS: {gdf.crs} -> {target_crs}")
                if auto_fix:
                    if backup:
                        gdf.to_file(backup_dir / f"flood_{depth}_original.gpkg", driver='GPKG')
                    gdf.to_crs(target_crs).to_file(f"Flood_{depth}_fixed.shp")
                    fix_report['fixes_applied'].append(f"Flood_{depth} converted to {target_crs}")
        except Exception as e:
            fix_report['issues_found'].append(f"Flood layer error: {e}")

    if fix_report['fixes_applied']:
        fix_report['fixed'] = True

    n_issues = len(fix_report['issues_found'])
    n_fixes = len(fix_report['fixes_applied'])
    print(f"Preprocessing: {n_issues} issues found, {n_fixes} fixes applied")
    return fix_report


def show_data_requirements(city_name="YourCity", output_file=None):
    """Print or save a guide of required data structure for a new city."""

    requirements_text = f"""
{'='*70}
URBAN-PFR DATA REQUIREMENTS — {city_name.upper()}
{'='*70}

FOLDER STRUCTURE
  {city_name}_FloodRisk/
    Input.gdb/                    GDB or GeoPackage
      Buildings                   Footprints with attributes
      StatisticalUnits            Census/population polygons
      Streets                     Road network
    Floodlevels/
      Flood_30.shp                Min. required (30cm depth)
      Flood_20/40/.../100.shp     Additional depths 

LAYER REQUIREMENTS
  Buildings:          ID, Floors (>0), Building_type (1=Res/2=Mixed/3=Non-res), geometry
  Statistical Units:  StatisticalUnit (unique ID), Residents, LivingArea (m²), geometry
  Streets:            geometry only (LineString or Polygon)
  Flood layers:       geometry only, named Flood_[depth_cm].shp/.gpkg/.geojson

  Sensitivity/coping fields (e.g. WR, C, ES, EDQ) must exist on
  buildings or statistical units. Names are configurable in config.yaml.

CRS: All layers must share the same projected CRS (e.g. EPSG:25832).
     Mismatched CRS → zero spatial overlap → zero risk values.

VALIDATE
  from urban_pfr.validation import validate_input_data
  report = validate_input_data("config.yaml")
{'='*70}
"""
    print(requirements_text)

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(requirements_text)

    return requirements_text


def validate_input_data(config_or_path, detailed=True):
    #ToDO need to be complete with CS3 CS6
    """
    Validate inputs: file existence, CRS consistency, required fields,
    data quality, spatial relationships.
    """
    if isinstance(config_or_path, str):
        import yaml
        with open(config_or_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = config_or_path

    report = {'valid': True, 'errors': [], 'warnings': [], 'info': [], 'summary': {}}

    try:
        input_gdb = config['project']['paths']['input_gdb']
        flood_dir = config['project']['paths']['flood_dir']
        target_crs = config['project']['crs']
    except KeyError as e:
        report['errors'].append(f"Missing config key: {e}")
        report['valid'] = False
        return report

    schema = config.get('schema', {})

    # --- 1. File existence ---
    if not os.path.exists(input_gdb):
        report['errors'].append(f"Input GDB not found: {input_gdb}")
        report['valid'] = False
        return report
    if not os.path.exists(flood_dir):
        report['errors'].append(f"Flood directory not found: {flood_dir}")
        report['valid'] = False
        return report

    # --- 2. Load layers ---
    layers_to_check = {
        'buildings': schema.get('buildings_layer', 'Buildings'),
        'statistical_units': schema.get('stats_layer', 'StatisticalUnits'),
        'streets': schema.get('streets_layer', 'Streets')
    }
    loaded = {}
    for name, layer_id in layers_to_check.items():
        try:
            gdf = gpd.read_file(input_gdb, layer=layer_id)
            loaded[name] = gdf
            report['info'].append(f"{name}: {len(gdf)} features, CRS={gdf.crs}")
        except Exception as e:
            report['errors'].append(f"Failed to load {name} ({layer_id}): {e}")
            report['valid'] = False
            return report

    # load flood layers
    flood_layers = {}
    flood_files = (list(Path(flood_dir).glob("Flood_*.shp")) +
                   list(Path(flood_dir).glob("Flood_*.gpkg")) +
                   list(Path(flood_dir).glob("Flood_*.geojson")))

    if not flood_files:
        report['errors'].append(f"No flood layers found in {flood_dir}")
        report['valid'] = False
        return report

    for ff in flood_files:
        depth_str = ff.stem.split('_')[-1]
        try:
            depth = int(depth_str)
            flood_layers[depth] = gpd.read_file(ff)
        except Exception as e:
            report['warnings'].append(f"Could not load {ff.name}: {e}")

    if not flood_layers:
        report['errors'].append("No valid flood layers loaded")
        report['valid'] = False
        return report

    report['info'].append(f"Flood layers: {sorted(flood_layers.keys())} cm")

    if 30 not in flood_layers:
        report['warnings'].append("Flood_30 not found (recommended for HMA)")

    buildings = loaded['buildings']
    stats = loaded['statistical_units']

    # --- 3. CRS consistency ---
    ref_crs = buildings.crs
    crs_issues = []

    for name, gdf in loaded.items():
        if name == 'buildings':
            continue
        if gdf.crs != ref_crs:
            crs_issues.append(f"{name} CRS ({gdf.crs}) differs from buildings ({ref_crs})")

    for depth, gdf in flood_layers.items():
        if gdf.crs != ref_crs:
            crs_issues.append(f"Flood_{depth}cm CRS ({gdf.crs}) differs from buildings")

    if crs_issues:
        report['errors'].extend(crs_issues)
        report['valid'] = False

    # --- 4. Required fields ---
    for field in ['ID', 'Floors', 'Building_type']:
        if field not in buildings.columns:
            report['errors'].append(f"Buildings missing required field: {field}")
            report['valid'] = False

    stat_unit_col = schema.get('stat_unit_col', 'StatisticalUnit')
    if stat_unit_col not in buildings.columns:
        report['warnings'].append(
            f"Buildings missing '{stat_unit_col}' -- exposure will use spatial join (slower)"
        )

    stats_required = [
        stat_unit_col,
        schema.get('residents_col', 'Residents'),
        schema.get('living_area_col', 'LivingArea')
    ]
    for field in stats_required:
        if field not in stats.columns:
            report['errors'].append(f"Statistical units missing required field: {field}")
            report['valid'] = False

    # social vulnerability fields — read from schema, not topsis_weights
    sv_sensitivity = schema.get('sensitivity_fields', ['WR', 'C', 'ES', 'EDQ'])
    sv_coping = schema.get('coping_fields', ['Y', 'L', 'MM'])
    sv_fields = sv_sensitivity + sv_coping
    all_known = list(set(sv_fields + ['WR', 'C', 'ES', 'EDQ', 'Y', 'L', 'MM']))

    on_buildings = [f for f in sv_fields if f in buildings.columns]
    on_stats = [f for f in sv_fields if f in stats.columns]

    if len(on_buildings) == len(sv_fields):
        report['info'].append(f"SV fields on buildings: {on_buildings}")
    elif on_stats:
        report['warnings'].append(
            f"SV fields ({on_stats}) on statistical units but not buildings -- "
            f"spatial join needed before TOPSIS"
        )
    else:
        any_found = ([f for f in all_known if f in buildings.columns] +
                     [f for f in all_known if f in stats.columns])
        if any_found:
            report['warnings'].append(
                f"SV fields found ({any_found}) don't match config expects ({sv_fields}). "
                f"Update config.yaml schema section."
            )
        else:
            report['warnings'].append("No SV fields found -- default vulnerability will be used")

    # --- 5. Data quality ---
    if 'ID' in buildings.columns:
        n_null = buildings['ID'].isnull().sum()
        if n_null:
            report['errors'].append(f"{n_null} buildings have NULL ID")
            report['valid'] = False

        n_dup = buildings['ID'].duplicated().sum()
        if n_dup:
            report['errors'].append(f"{n_dup} duplicate building IDs")
            report['valid'] = False

    if 'Floors' in buildings.columns:
        bad = (buildings['Floors'] <= 0).sum()
        if bad:
            report['errors'].append(f"{bad} buildings have Floors <= 0")
            report['valid'] = False

    if 'Building_type' in buildings.columns:
        bad = (~buildings['Building_type'].isin([1, 2, 3])).sum()
        if bad:
            report['errors'].append(f"{bad} buildings have invalid Building_type (must be 1/2/3)")
            report['valid'] = False

    invalid_geom = (~buildings.geometry.is_valid).sum()
    if invalid_geom:
        report['warnings'].append(f"{invalid_geom} buildings have invalid geometries")

    residents_col = schema.get('residents_col', 'Residents')
    living_area_col = schema.get('living_area_col', 'LivingArea')

    if residents_col in stats.columns:
        bad = (stats[residents_col] < 0).sum()
        if bad:
            report['errors'].append(f"{bad} units have negative Residents")
            report['valid'] = False

    if living_area_col in stats.columns:
        bad = (stats[living_area_col] <= 0).sum()
        if bad:
            report['warnings'].append(f"{bad} units have LivingArea <= 0")

    # --- 6. Spatial relationships ---
    if detailed:
        test_flood = list(flood_layers.values())[0]
        flood_union = test_flood.unary_union
        n_overlap = buildings.geometry.intersects(flood_union).sum()
        pct = n_overlap / len(buildings) * 100

        if n_overlap == 0:
            report['errors'].append("ZERO buildings overlap flood layers -- check CRS and extent")
            report['valid'] = False
        elif pct < 5:
            report['warnings'].append(f"Only {pct:.1f}% buildings overlap flood -- verify this is expected")

        sample_n = min(100, len(buildings))
        sample = buildings.sample(sample_n)
        in_units = sum(1 for _, b in sample.iterrows()
                       if stats.geometry.intersects(b.geometry).any())
        containment = in_units / sample_n * 100
        if containment < 80:
            report['warnings'].append(
                f"Only {containment:.0f}% of buildings within statistical units"
            )

    # --- 7. Summary ---
    report['summary'] = {
        'buildings_count': len(buildings),
        'stats_units_count': len(stats),
        'streets_count': len(loaded['streets']),
        'flood_layers_count': len(flood_layers),
        'flood_depths': sorted(flood_layers.keys()),
        'crs_consistent': len(crs_issues) == 0,
        'errors_count': len(report['errors']),
        'warnings_count': len(report['warnings'])
    }

    status = "PASSED" if report['valid'] else "FAILED"
    print(f"Validation {status}: {len(report['errors'])} errors, "
          f"{len(report['warnings'])} warnings "
          f"({len(buildings)} buildings, {len(flood_layers)} flood layers)")
    if report['errors']:
        for e in report['errors']:
            print(f"  ERROR: {e}")
    if report['warnings']:
        for w in report['warnings']:
            print(f"  WARN:  {w}")

    return report


def generate_validation_report(report, output_file="validation_report.txt"):
    """Save validation report to text file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("URBAN-PFR VALIDATION REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("SUMMARY\n")
        for k, v in report['summary'].items():
            f.write(f"  {k}: {v}\n")

        f.write(f"\nSTATUS: {'PASSED' if report['valid'] else 'FAILED'}\n")

        if report['errors']:
            f.write(f"\nERRORS ({len(report['errors'])})\n")
            for i, e in enumerate(report['errors'], 1):
                f.write(f"  {i}. {e}\n")

        if report['warnings']:
            f.write(f"\nWARNINGS ({len(report['warnings'])})\n")
            for i, w in enumerate(report['warnings'], 1):
                f.write(f"  {i}. {w}\n")

        if report['info']:
            f.write(f"\nINFO ({len(report['info'])})\n")
            for i, info in enumerate(report['info'], 1):
                f.write(f"  {i}. {info}\n")

    return output_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Urban-PFR Data Validation')
    parser.add_argument('--requirements', type=str, metavar='CITY')
    parser.add_argument('--config', type=str, metavar='CONFIG')
    parser.add_argument('--output', type=str, metavar='FILE')

    args = parser.parse_args()

    if args.requirements:
        show_data_requirements(city_name=args.requirements, output_file=args.output)
    elif args.config:
        report = validate_input_data(args.config, detailed=not args.quick)
        if args.output:
            generate_validation_report(report, args.output)
    else:
        parser.print_help()