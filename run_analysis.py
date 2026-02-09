"""
Run pluvial flood risk analysis.

designed and under development to be set for different city:
    python run_analysis.py hamburg_config.yaml
    python run_analysis.py bremen_config.yaml
"""

import sys
import copy
import yaml
from pathlib import Path

from urban_pfr.validation import validate_input_data, preprocess_and_fix_data
from urban_pfr.analyzer import PFRAnalyzer

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"


def main(config_path):
    if not Path(config_path).exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    city = config.get('project', {}).get('city_name', 'Unknown')
    print(f"=== {city} pluvial flood risk analysis ===\n")

    # 1. Validate
    print("Step 1: Validating input data(make sure the ['input_gdb'] and ['flood_dir']  files are correctly set in config file)...")
    report = validate_input_data(config, detailed=True)

    if not report['valid']:
        crs_errors = [e for e in report['errors'] if 'CRS' in e]
        other_errors = [e for e in report['errors'] if 'CRS' not in e and 'overlap' not in e.lower()]

        if crs_errors and not other_errors:
            print(f"\n {RED}CRS mismatches found. Attempting auto-fix...{RESET}")
            fix_report = preprocess_and_fix_data(config, auto_fix=True)
            
            if fix_report['fixed']:
                print(f"{GREEN}✓ CRS fixed. CRS files...{RESET}\n")
                config = _patch_config_with_fixed_files(config)
            else:
                print(f"\n {RED}Auto-fix failed. Fix CRS manually.{RESET}")
                sys.exit(1)
        else:
            print("\nValidation failed. Fix errors above before running.")
            sys.exit(1)

    # 2. Run pipeline
    print("Step 2: Running pipeline...")
    analyzer = PFRAnalyzer(config)
    analyzer.load_data()
    buildings, stats = analyzer.run_pipeline()

    # 3. Save results
    print(f"\nStep 3: {GREEN}✓ Saving results...{RESET}")
    paths = analyzer.save_results()

    # 4. Visualize
    print("\nStep 4: Generating visualization...")
    print(f"\n {RED}Need to be check with CS3{RESET}")
    output_dir = config.get('project', {}).get('paths', {}).get('output_dir', './outputs')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_path = f"{output_dir}/{city.lower()}_risk_map.png"
    analyzer.visualize(save_path=save_path)
    print(f"Map saved to {save_path}")

    print(f"\n {GREEN}Done. Results in {output_dir}{RESET}")



def _patch_config_with_fixed_files(config):
    """
    After CRS fix, analyzer loads them instead of the original GDB layers.
    """
    patched = copy.deepcopy(config)
    paths = patched.setdefault('project', {}).setdefault('paths', {})

    if Path("buildings_fixed.gpkg").exists():
        paths['buildings_file'] = "buildings_fixed.gpkg"

    if Path("statistical_units_fixed.gpkg").exists():
        paths['stats_file'] = "statistical_units_fixed.gpkg"

    if Path("streets_fixed.gpkg").exists():
        paths['streets_file'] = "streets_fixed.gpkg"

    return patched


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python run_analysis.py <config.yaml>")
        print("Example: python run_analysis.py hamburg_config.yaml")
        sys.exit(1)

    main(sys.argv[1])