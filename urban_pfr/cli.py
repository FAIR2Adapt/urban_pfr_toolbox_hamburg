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
INPUT_DIR      = Path("/mnt/inputs")
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


# FDO mode (RO-Crate in / RO-Crate out)
def _run_fdo(args):
    """FDO mode: resolve RO-Crate inputs, run pipeline, create output crate."""
    from fdo_resolver import FDOResolver

    workflow_crate = Path(args.workflow_crate)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve FDO inputs
    log("Reading Workflow RO-Crate profile...")
    resolver = FDOResolver.from_workflow_crate(str(workflow_crate))

    log(f"Resolving input RO-Crates from {input_dir}...")
    result = resolver.resolve(str(input_dir))

    if not result.is_complete:
        missing = [p.name for p in result.unmatched_params if p.value_required]
        print(f"{RED}Missing required inputs: {', '.join(missing)}{RESET}")
        print(result.summary())
        sys.exit(1)

    log("Resolved inputs:")
    print(result.summary())

    # 2. Build config from resolved paths
    config_path = result.paths.get("config")
    if not config_path:
        print(f"{RED}No config file resolved from input crates.{RESET}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    paths = config.setdefault("project", {}).setdefault("paths", {})

    # Clear existing paths and set from resolver
    for k in list(paths.keys()):
        if k != "output_dir":
            del paths[k]

    for param_name, resolved_path in result.paths.items():
        if param_name == "config":
            continue
        elif param_name == "flood_levels":
            paths["flood_dir"] = str(resolved_path)
        elif param_name == "buildings_gdb":
            paths["input_gdb"] = str(resolved_path)
        elif param_name == "boundary":
            paths["boundary_file"] = str(resolved_path)
        else:
            paths[param_name] = str(resolved_path)

    paths["output_dir"] = str(output_dir)

    # 3. Run pipeline
    _run_pipeline(config, str(output_dir), skip_validation=args.skip_validation)

    # 4. Create output Workflow Run Crate
    log("Creating output Workflow Run Crate...")
    city = config.get("project", {}).get("city_name", "Unknown")

    output_files = {}
    for f in sorted(output_dir.glob("*")):
        if f.is_file() and f.name != "ro-crate-metadata.json":
            # Use filename (not stem) as key to avoid duplicates between .gpkg and .fgb
            desc = f.stem.replace("_", " ").title()
            if f.suffix:
                desc += f" ({f.suffix.lstrip('.')})"
            output_files[desc] = f

    resolver.create_run_crate(
        str(output_dir),
        name=f"{city} Pluvial Flood Risk Assessment Results",
        description=(
            f"Output from the urban_pfr pipeline for {city}. "
            f"Contains flood risk indices (PFRMA, PFRWB) and social vulnerability (SVPF)."
        ),
        bindings=result,
        output_files=output_files,
    )

    log(f"Output RO-Crate written to {output_dir}/ro-crate-metadata.json")
    print(f"\n{GREEN}Done. Results in {output_dir}{RESET}")


# Docker mode
def _run_docker(args):
    """Docker mode: auto-detect inputs at /mnt/inputs/, output to /mnt/outputs/."""
    log(f"Looking for config at {CONFIG_PATH}")
    if not CONFIG_PATH.exists():
        print(f"{RED}Config file not found:{RESET} {CONFIG_PATH}")
        sys.exit(1)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # Auto-detect and resolve inputs
    try:
        config = _resolve_inputs(config)
    except Exception as e:
        print(f"{RED}Input preparation failed:{RESET} {e}")
        sys.exit(1)

    # Force output dir
    config.setdefault("project", {}).setdefault("paths", {})["output_dir"] = str(OUTPUT_DIR)

    _run_pipeline(config, str(OUTPUT_DIR), skip_validation=args.skip_validation)

    # Zip outputs
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


def _resolve_inputs(config: dict) -> dict:
    """
    Auto-detect input data at /mnt/inputs/ in this priority order:

    0. RO-Crate FDOs (directories with ro-crate-metadata.json)

    For the GDB (buildings, statistical units, streets):
      1. Config paths.input_gdb (if set and exists)
      2. *.gdb directory in INPUT_DIR
      3. *.gdb.zip in INPUT_DIR (extracted automatically)
      4. Individual files: paths.buildings, paths.statistical_units, paths.streets

    For flood layers:
      1. Config paths.flood_dir (if set and exists)
      2. Directory containing Flood_*.geojson/shp/gpkg in INPUT_DIR
      3. Floodlevels.zip in INPUT_DIR (extracted automatically)
      4. *.zip containing Flood_* files (extracted automatically)

    For boundary (optional):
      1. Config paths.boundary_file (if set and exists)
      2. *boundary*.gpkg in INPUT_DIR
    """
    project = config.setdefault("project", {})
    paths = project.setdefault("paths", {})

    # --- Try RO-Crate FDOs first ---
    try:
        from urban_pfr.rocrate_io import resolve_rocrate_inputs, resolve_to_config_paths
        resolved = resolve_rocrate_inputs(INPUT_DIR)
        if resolved:
            log("Inputs resolved from RO-Crate FDOs")
            crate_paths = resolve_to_config_paths(resolved)
            paths.update(crate_paths)
            return config
    except ImportError:
        pass  # rocrate not available, continue with other methods

    # --- Resolve GDB ---
    gdb_resolved = False
    input_gdb = paths.get("input_gdb")

    # Check if config path already works
    if input_gdb and Path(input_gdb).exists() and not str(input_gdb).endswith(".zip"):
        log(f"Using config input_gdb: {input_gdb}")
        gdb_resolved = True

    # Search for .gdb directory
    if not gdb_resolved:
        gdbs = list(INPUT_DIR.glob("*.gdb"))
        if gdbs:
            paths["input_gdb"] = str(gdbs[0])
            log(f"Found GDB directory: {gdbs[0]}")
            gdb_resolved = True

    # Search for .gdb.zip and extract
    if not gdb_resolved:
        gdb_zips = list(INPUT_DIR.glob("*.gdb.zip"))
        if not gdb_zips:
            gdb_zips = [f for f in INPUT_DIR.glob("*.zip")
                        if "flood" not in f.name.lower()]
        if gdb_zips:
            zip_path = gdb_zips[0]
            unzip_dir = zip_path.with_suffix("").with_suffix("")
            _unzip_to_dir(zip_path, unzip_dir)
            found = list(unzip_dir.rglob("*.gdb"))
            if found:
                paths["input_gdb"] = str(found[0])
                log(f"Extracted GDB: {found[0]}")
                gdb_resolved = True

    # Check for individual files (GeoJSON, GPKG, SHP)
    if not gdb_resolved:
        for key, patterns in [
            ("buildings", ["buildings.*", "Building*.*", "Gebaeude*.*"]),
            ("statistical_units", ["statistical*.*", "StatisticalUnit*.*"]),
            ("streets", ["streets.*", "Streets.*", "Strassen*.*"]),
        ]:
            if paths.get(key) and Path(paths[key]).exists():
                continue
            for pattern in patterns:
                found = list(INPUT_DIR.glob(pattern))
                found = [f for f in found if f.suffix in (".geojson", ".gpkg", ".shp")]
                if found:
                    paths[key] = str(found[0])
                    log(f"Found {key}: {found[0]}")
                    break

    if not gdb_resolved and not any(paths.get(k) for k in ["buildings", "statistical_units"]):
        raise FileNotFoundError(
            f"No GDB, zip, or individual data files found in {INPUT_DIR}. "
            f"Provide a .gdb directory, .gdb.zip, or individual .geojson/.gpkg/.shp files."
        )

    # --- Resolve flood layers ---
    flood_resolved = False
    flood_dir = paths.get("flood_dir")

    # Check if config path already works
    if flood_dir and Path(flood_dir).is_dir():
        flood_files = list(Path(flood_dir).glob("Flood_*"))
        if flood_files:
            log(f"Using config flood_dir: {flood_dir}")
            flood_resolved = True

    # Search for Flood_* files directly in INPUT_DIR
    if not flood_resolved:
        flood_files = (
            list(INPUT_DIR.rglob("Flood_*.geojson")) +
            list(INPUT_DIR.rglob("Flood_*.gpkg")) +
            list(INPUT_DIR.rglob("Flood_*.shp"))
        )
        if flood_files:
            paths["flood_dir"] = str(flood_files[0].parent)
            log(f"Found flood layers in: {flood_files[0].parent}")
            flood_resolved = True

    # Search for flood zip and extract
    if not flood_resolved:
        flood_zips = list(INPUT_DIR.glob("*[Ff]lood*.zip"))
        if not flood_zips:
            flood_zips = [f for f in INPUT_DIR.glob("*.zip")
                          if "gdb" not in f.name.lower()]
        if flood_zips:
            zip_path = flood_zips[0]
            unzip_dir = zip_path.with_suffix("")
            _unzip_to_dir(zip_path, unzip_dir)
            found = (
                list(unzip_dir.rglob("Flood_*.geojson")) +
                list(unzip_dir.rglob("Flood_*.gpkg")) +
                list(unzip_dir.rglob("Flood_*.shp"))
            )
            if found:
                paths["flood_dir"] = str(found[0].parent)
                log(f"Extracted flood layers: {found[0].parent}")
                flood_resolved = True

    if not flood_resolved:
        raise FileNotFoundError(
            f"No flood layers found in {INPUT_DIR}. "
            f"Provide Flood_*.geojson files, a Floodlevels/ directory, or a .zip archive."
        )

    # --- Resolve boundary (optional) ---
    boundary = paths.get("boundary_file")
    if not boundary or not Path(boundary).exists():
        boundary_files = list(INPUT_DIR.glob("*boundary*.*")) + list(INPUT_DIR.glob("*Boundary*.*"))
        boundary_files = [f for f in boundary_files if f.suffix in (".gpkg", ".geojson", ".shp")]
        if boundary_files:
            paths["boundary_file"] = str(boundary_files[0])
            log(f"Found boundary: {boundary_files[0]}")

    log(f"Resolved paths: {paths}")
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

    p_fdo = sub.add_parser("fdo", help="Run with RO-Crate FDO inputs/outputs")
    p_fdo.add_argument(
        "--workflow-crate", default=".",
        help="Path to the Workflow RO-Crate (default: current directory)",
    )
    p_fdo.add_argument(
        "--input-dir", default="/mnt/inputs",
        help="Directory containing input data RO-Crates (default: /mnt/inputs)",
    )
    p_fdo.add_argument(
        "--output-dir", default="/mnt/outputs",
        help="Directory for the output Workflow Run Crate (default: /mnt/outputs)",
    )
    _add_common_args(p_fdo)

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
    elif args.mode == "fdo":
        _run_fdo(args)
    else:
        _run_local(args)
