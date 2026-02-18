"""
Run pluvial flood risk analysis (wrapper-format).

- Reads config at:          /mnt/inputs/config.yaml
- Expects ZIP inputs at:    /mnt/inputs/pluvialfloodriskmap.gdb.zip
                            /mnt/inputs/Floodlevels.zip
- Writes outputs to:        /mnt/outputs
- Packages outputs as:      /mnt/outputs/outputs.zip

NOTE:
- YAML no longer needs project.paths.*. We force them here.

"""

import sys
import copy
import yaml
import zipfile
from pathlib import Path

from urban_pfr.validation import validate_input_data, preprocess_and_fix_data
from urban_pfr.analyzer import PFRAnalyzer

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"

CONFIG_PATH = Path("/mnt/inputs/config.yaml")
GDB_ZIP_PATH = Path("/mnt/inputs/pluvialfloodriskmap.gdb.zip")
FLOOD_ZIP_PATH = Path("/mnt/inputs/Floodlevels.zip")
OUTPUT_DIR = Path("/mnt/outputs")
OUTPUT_ZIP = OUTPUT_DIR / "outputs.zip"


def log(msg: str) -> None:
    print(f"{YELLOW}[WRAPPER]{RESET} {msg}")


def _unzip_to_dir(zip_path: Path, dest_dir: Path) -> Path:
    zip_path = Path(zip_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Not a zip file: {zip_path}")

    log(f"Unzipping {zip_path} -> {dest_dir}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)

    return dest_dir


def _force_fixed_paths(config: dict) -> dict:
    """
    Force wrapper I/O contract paths into the config, ignoring YAML paths in case it exist.
    """
    project = config.setdefault("project", {})
    paths = project.setdefault("paths", {})

    paths["input_gdb"] = str(GDB_ZIP_PATH)
    paths["flood_dir"] = str(FLOOD_ZIP_PATH)
    paths["output_dir"] = str(OUTPUT_DIR)

    log(f"Forced paths: {paths}")
    return config


def _resolve_zip_inputs_in_config(config: dict) -> dict:
    """
    Unzip the fixed ZIP inputs and patch config to point to extracted content.
    - input_gdb ZIP -> points to first *.gdb found inside
    - flood_dir ZIP -> points to folder that contains Flood_* files
    """
    project = config.setdefault("project", {})
    paths = project.setdefault("paths", {})

    input_gdb = paths.get("input_gdb")
    if input_gdb and str(input_gdb).lower().endswith(".zip"):
        zip_path = Path(input_gdb)
        unzip_dir = zip_path.with_suffix("")  # /mnt/inputs/pluvialfloodriskmap.gdb
        _unzip_to_dir(zip_path, unzip_dir)

        gdbs = list(unzip_dir.rglob("*.gdb"))
        if not gdbs:
            raise FileNotFoundError(f"No .gdb found after unzip: {zip_path}")
        paths["input_gdb"] = str(gdbs[0])
        log(f"Resolved input_gdb -> {paths['input_gdb']}")

    flood_dir = paths.get("flood_dir")
    if flood_dir and str(flood_dir).lower().endswith(".zip"):
        zip_path = Path(flood_dir)
        unzip_dir = zip_path.with_suffix("")  # /mnt/inputs/Floodlevels
        _unzip_to_dir(zip_path, unzip_dir)

        candidates = (
            list(unzip_dir.rglob("Flood_*.shp")) +
            list(unzip_dir.rglob("Flood_*.gpkg")) +
            list(unzip_dir.rglob("Flood_*.geojson"))
        )
        if not candidates:
            raise FileNotFoundError(f"No Flood_* layers found after unzip: {zip_path}")

        paths["flood_dir"] = str(candidates[0].parent)
        log(f"Resolved flood_dir -> {paths['flood_dir']}")

    return config


def _print_validation_report(report: dict) -> None:
    if report.get("errors"):
        for e in report["errors"]:
            print(f"  ERROR: {e}")
    if report.get("warnings"):
        for w in report["warnings"]:
            print(f"  WARN:  {w}")
    if report.get("info"):
        for i in report["info"]:
            print(f"  INFO:  {i}")


def _zip_dir(zip_path: Path, source_dir: Path) -> None:
    """
    Zip source_dir recursively into zip_path (Zip64 enabled).
    Avoids including the zip itself if zip_path is inside source_dir.
    """
    zip_path = Path(zip_path)
    source_dir = Path(source_dir)

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()

    log(f"Creating zip: {zip_path}")
    log(f"Zipping contents of: {source_dir}")

    with zipfile.ZipFile(
        zip_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        allowZip64=True,
    ) as zipf:
        for p in source_dir.rglob("*"):
            if not p.is_file():
                continue

            # don't include the zip itself
            try:
                if p.resolve() == zip_path.resolve():
                    continue
            except Exception:
                pass

            zipf.write(p, arcname=str(p.relative_to(source_dir)))

    log(f"Zip created OK: {zip_path}")


def _patch_config_with_fixed_files(config: dict) -> dict:
    """
    After CRS fix, analyzer loads them instead of the original GDB layers.
    NOTE: preprocess_and_fix_data writes these files to current working directory.
    """
    patched = copy.deepcopy(config)
    paths = patched.setdefault("project", {}).setdefault("paths", {})

    if Path("buildings_fixed.gpkg").exists():
        paths["buildings_file"] = "buildings_fixed.gpkg"

    if Path("statistical_units_fixed.gpkg").exists():
        paths["stats_file"] = "statistical_units_fixed.gpkg"

    if Path("streets_fixed.gpkg").exists():
        paths["streets_file"] = "streets_fixed.gpkg"

    log(f"Patched config with fixed files: {paths}")
    return patched


def main() -> None:
    log(f"Looking for config at {CONFIG_PATH}")
    if not CONFIG_PATH.exists():
        print(f"{RED}Config file not found:{RESET} {CONFIG_PATH}")
        sys.exit(1)

    
    if not GDB_ZIP_PATH.exists():
        print(f"{RED}Missing input:{RESET} {GDB_ZIP_PATH}")
        sys.exit(1)
    if not FLOOD_ZIP_PATH.exists():
        print(f"{RED}Missing input:{RESET} {FLOOD_ZIP_PATH}")
        sys.exit(1)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # Force wrapper contract paths (ignore YAML paths)
    config = _force_fixed_paths(config)

    # Unzip inputs (because config paths are ZIPs)
    try:
        config = _resolve_zip_inputs_in_config(config)
    except Exception as e:
        print(f"{RED}Input preparation failed:{RESET} {e}")
        sys.exit(1)

    city = config.get("project", {}).get("city_name", "Unknown")
    print(f"=== {city} pluvial flood risk analysis ===\n")

    # 1. Validate
    print("Step 1: Validating input data...")
    report = validate_input_data(config, detailed=True)

    if not report.get("valid", False):
        _print_validation_report(report)

        crs_errors = [e for e in report.get("errors", []) if "CRS" in e]
        other_errors = [
            e for e in report.get("errors", [])
            if "CRS" not in e and "overlap" not in e.lower()
        ]

        if crs_errors and not other_errors:
            print(f"\n{RED}CRS mismatches found. Attempting auto-fix...{RESET}")
            fix_report = preprocess_and_fix_data(config, auto_fix=True)

            if fix_report.get("fixed"):
                print(f"{GREEN}✓ CRS fixed. Using fixed files...{RESET}\n")
                config = _patch_config_with_fixed_files(config)
            else:
                print(f"\n{RED}Auto-fix failed. Fix CRS manually.{RESET}")
                sys.exit(1)
        else:
            print(f"\n{RED}Validation failed. Fix errors above before running.{RESET}")
            sys.exit(1)

    # 2. Run pipeline
    print("Step 2: Running pipeline...")
    analyzer = PFRAnalyzer(config)
    analyzer.load_data()
    analyzer.run_pipeline()

    # 3. Save results
    print(f"\nStep 3: {GREEN}✓ Saving results...{RESET}")
    analyzer.save_results()

    # 4. Visualize
    print("\nStep 4: Generating visualization...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_path = str(OUTPUT_DIR / f"{city.lower()}_risk_map.png")
    analyzer.visualize(save_path=save_path)
    print(f"Map saved to {save_path}")

    # 5. Zip outputs
    _zip_dir(zip_path=OUTPUT_ZIP, source_dir=OUTPUT_DIR)
    print(f"\n{GREEN}Done successfully.{RESET}")


if __name__ == "__main__":
    main()
