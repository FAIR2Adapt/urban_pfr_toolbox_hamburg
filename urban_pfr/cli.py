"""
Local mode:
    urban-pfr local hamburg_config.yaml
    urban-pfr local config.yaml --skip-validation

Docker mode (used by Dockerfile ENTRYPOINT):
    urban-pfr docker

Auto-detect (no subcommand, used by existing Dockerfile):
    python run_analysis.py
"""

import sys
import copy
import argparse
import zipfile
import yaml
from pathlib import Path

from urban_pfr.validation import validate_input_data, preprocess_and_fix_data
from urban_pfr.analyzer import PFRAnalyzer

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"

# Docker contract
CONFIG_PATH    = Path("/mnt/inputs/config.yaml")
GDB_ZIP_PATH   = Path("/mnt/inputs/pluvialfloodriskmap.gdb.zip")
FLOOD_ZIP_PATH = Path("/mnt/inputs/Floodlevels.zip")
OUTPUT_DIR     = Path("/mnt/outputs")
OUTPUT_ZIP     = OUTPUT_DIR / "outputs.zip"


# Shared pipeline
def _run_pipeline(config, output_dir, skip_validation=False):
    """Core pipeline logic. Both local and Docker call this."""
    city = config.get("project", {}).get("city_name", "Unknown")
    print(f"=== {city} pluvial flood risk analysis ===\n")

    # 1. Validate
    if not skip_validation:
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
                    print(f"{GREEN}CRS fixed. Using fixed files...{RESET}\n")
                    config = _patch_config_with_fixed_files(config)
                else:
                    print(f"\n{RED}Auto-fix failed. Fix CRS manually.{RESET}")
                    sys.exit(1)
            else:
                print(f"\n{RED}Validation failed. Fix errors above before running.{RESET}")
                sys.exit(1)
    else:
        print("Step 1: Skipping validation (--skip-validation)")

    # 2. Run pipeline
    print("Step 2: Running pipeline...")
    analyzer = PFRAnalyzer(config)
    analyzer.load_data()
    analyzer.run_pipeline()

    # 3. Save results
    print(f"\nStep 3: {GREEN}Saving results...{RESET}")
    analyzer.save_results()

    # 4. Visualize
    print("\nStep 4: Generating visualization...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_path = f"{output_dir}/{city.lower()}_risk_map.png"
    analyzer.visualize(save_path=save_path)
    print(f"Map saved to {save_path}")

    return output_dir


# Local mode
def _run_local(args):
    """Local mode: user provides config path on the command line."""
    config_path = args.config
    if not Path(config_path).exists():
        print(f"{RED}Config not found: {config_path}{RESET}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = config.get("project", {}).get("paths", {}).get("output_dir", "./outputs")
    _run_pipeline(config, output_dir, skip_validation=args.skip_validation)
    print(f"\n{GREEN}Done. Results in {output_dir}{RESET}")


# Docker mode
def _run_docker(args):
    """Docker mode: fixed mount paths, zip inputs/outputs."""
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

    _run_pipeline(config, str(OUTPUT_DIR), skip_validation=args.skip_validation)

    # 5. Zip outputs
    _zip_dir(zip_path=OUTPUT_ZIP, source_dir=OUTPUT_DIR)
    print(f"\n{GREEN}Done successfully.{RESET}")


# Docker helpers
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


# Argument parser
def _add_common_args(parser):
    """Flags shared by both local and docker subcommands."""
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip data validation step")


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="urban-pfr",
        description="Urban Pluvial Flood Risk Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  urban-pfr local hamburg_config.yaml                  # Local run
  urban-pfr local config.yaml --skip-validation        # Skip checks
  urban-pfr docker                                     # Inside Docker
""")

    sub = parser.add_subparsers(dest="mode")

    p_local = sub.add_parser("local", help="Run with a local config YAML file")
    p_local.add_argument("config", help="Path to config YAML")
    _add_common_args(p_local)

    p_docker = sub.add_parser("docker", help="Run inside Docker container")
    _add_common_args(p_docker)

    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.mode is None:
        if CONFIG_PATH.exists():
            args.mode = "docker"
            args.skip_validation = False
            _run_docker(args)
        else:
            parser.print_help()
            print(f"\n{YELLOW}Hint:{RESET} use 'local <config.yaml>' or 'docker'")
            sys.exit(1)
    elif args.mode == "docker":
        _run_docker(args)
    else:
        _run_local(args)
