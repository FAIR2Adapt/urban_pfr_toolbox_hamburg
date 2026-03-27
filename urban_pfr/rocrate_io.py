"""
rocrate_io.py – RO-Crate FAIR Digital Object I/O.

Resolves RO-Crate inputs for the pipeline and creates RO-Crate outputs.

Each pipeline input can be provided as an RO-Crate directory containing
ro-crate-metadata.json and the actual data files. The module discovers
crates in the input directory and maps them to pipeline input slots.

Design: The matching logic is separated from pipeline-specific knowledge
so it can be reused by other tools. The input slots are defined by the
caller (or read from annotation.json), not hardcoded.
"""

import json
from pathlib import Path

# Encoding formats for geospatial data
GEOSPATIAL_FORMATS = {
    ".geojson": "application/geo+json",
    ".gpkg": "application/geopackage+sqlite3",
    ".gdb": "application/x-filegdb",
    ".shp": "application/x-shapefile",
    ".fgb": "application/flatgeobuf",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".csv": "text/csv",
    ".json": "application/json",
    ".yaml": "application/x-yaml",
    ".yml": "application/x-yaml",
    ".zip": "application/zip",
}


def read_crate_metadata(crate_dir):
    """
    Read an RO-Crate and return its metadata and data file paths.

    Parameters
    ----------
    crate_dir : str or Path
        Path to a directory containing ro-crate-metadata.json.

    Returns
    -------
    dict with keys:
        name : str - crate name
        description : str - crate description
        data_files : list of Path - paths to data files
        encoding_formats : list of str - MIME types of data files
        metadata : dict - raw parsed ro-crate-metadata.json
    """
    crate_dir = Path(crate_dir)
    meta_path = crate_dir / "ro-crate-metadata.json"

    if not meta_path.exists():
        return None

    # Try rocrate library first, fall back to raw JSON
    try:
        from rocrate.rocrate import ROCrate
        crate = ROCrate(str(crate_dir))
        root = crate.root_dataset
        data_files = []
        formats = []
        for entity in crate.data_entities:
            entity_path = crate_dir / entity.id
            if entity_path.exists() and entity.id != "ro-crate-metadata.json":
                data_files.append(entity_path)
                fmt = entity.get("encodingFormat", "")
                if isinstance(fmt, dict):
                    fmt = fmt.get("@id", "")
                formats.append(fmt or _guess_format(entity_path))
        return {
            "name": root.get("name", "") or "",
            "description": root.get("description", "") or "",
            "data_files": data_files,
            "encoding_formats": formats,
            "crate_dir": crate_dir,
        }
    except Exception:
        pass

    # Fallback: parse JSON directly
    try:
        with open(meta_path) as f:
            metadata = json.load(f)
    except Exception:
        return None

    graph = metadata.get("@graph", [])
    name, description = "", ""
    data_files, formats = [], []

    for entity in graph:
        eid = entity.get("@id", "")
        etype = entity.get("@type", "")
        if isinstance(etype, list):
            etype = etype[0] if etype else ""

        if eid == "./":
            name = entity.get("name", "")
            description = entity.get("description", "")

        if etype == "File":
            fpath = crate_dir / eid
            if fpath.exists():
                data_files.append(fpath)
                fmt = entity.get("encodingFormat", "")
                formats.append(fmt or _guess_format(fpath))

        # GDB directories
        if eid.rstrip("/").endswith(".gdb"):
            gdb_path = crate_dir / eid.rstrip("/")
            if gdb_path.exists():
                data_files.append(gdb_path)
                formats.append("application/x-filegdb")

    # If no files found in metadata, discover by extension
    if not data_files:
        data_files = _discover_data_files(crate_dir)
        formats = [_guess_format(f) for f in data_files]

    return {
        "name": name,
        "description": description,
        "data_files": data_files,
        "encoding_formats": formats,
        "crate_dir": crate_dir,
    }


def resolve_rocrate_inputs(input_dir, input_slots=None):
    """
    Scan input_dir for RO-Crate directories and match them to input slots.

    This is the generic resolver: given a set of named input slots with
    keywords, it finds RO-Crates in input_dir and maps each to a slot
    based on metadata matching.

    Parameters
    ----------
    input_dir : str or Path
        Directory to scan for RO-Crate subdirectories.
    input_slots : dict, optional
        Mapping of slot_name -> list of keywords for matching.
        Default slots are for the urban_pfr pipeline.

    Returns
    -------
    dict or None
        Mapping of slot_name -> {'path': str, 'crate': dict}.
        Returns None if no RO-Crates are found.
    """
    input_dir = Path(input_dir)

    if input_slots is None:
        input_slots = _default_input_slots()

    # Find all RO-Crates
    crates = []
    for meta_file in input_dir.rglob("ro-crate-metadata.json"):
        crate_info = read_crate_metadata(meta_file.parent)
        if crate_info:
            crates.append(crate_info)

    if not crates:
        return None

    print(f"  Found {len(crates)} RO-Crate(s) in {input_dir}")

    # Match each crate to a slot
    resolved = {}
    for crate in crates:
        slot = _match_crate_to_slot(crate, input_slots)
        if slot:
            print(f"    {crate['crate_dir'].name} -> {slot} "
                  f"({len(crate['data_files'])} data files)")
            resolved[slot] = crate

    return resolved if resolved else None


def resolve_to_config_paths(resolved_crates):
    """
    Convert resolved RO-Crate mappings to config paths for the pipeline.

    Parameters
    ----------
    resolved_crates : dict
        Output from resolve_rocrate_inputs().

    Returns
    -------
    dict
        Paths suitable for config['project']['paths'], e.g.:
        {'input_gdb': '...', 'flood_dir': '...', 'boundary_file': '...'}
    """
    paths = {}

    for slot, crate in resolved_crates.items():
        data_files = crate["data_files"]
        crate_dir = crate["crate_dir"]

        if slot == "flood":
            paths["flood_dir"] = str(crate_dir)
        elif slot == "boundary":
            if data_files:
                paths["boundary_file"] = str(data_files[0])
        elif slot == "buildings":
            gdb_dirs = [f for f in data_files if str(f).endswith(".gdb")]
            if gdb_dirs:
                paths["input_gdb"] = str(gdb_dirs[0])
            elif data_files:
                paths["buildings"] = str(data_files[0])
        elif slot == "statistical_units":
            if data_files:
                paths["statistical_units"] = str(data_files[0])
        elif slot == "streets":
            if data_files:
                paths["streets"] = str(data_files[0])

    return paths


def create_output_rocrate(output_dir, config, file_paths=None):
    """
    Create an RO-Crate for the pipeline output.

    Parameters
    ----------
    output_dir : str or Path
        Directory containing output files.
    config : dict
        Pipeline configuration.
    file_paths : dict, optional
        Mapping of description -> file path for files to include.
    """
    try:
        from rocrate.rocrate import ROCrate
    except ImportError:
        print("  rocrate library not available, skipping RO-Crate output")
        return None

    output_dir = Path(output_dir)
    city = config.get("project", {}).get("city_name", "Unknown")

    crate = ROCrate()
    crate.root_dataset["name"] = f"{city} Pluvial Flood Risk Assessment Results"
    crate.root_dataset["description"] = (
        f"Output from the urban_pfr pipeline for {city}. "
        f"Contains flood risk indices (PFRMA, PFRWB) as Thiessen polygons "
        f"and social vulnerability indicators (SVPF) at statistical unit level."
    )

    if file_paths:
        for desc, path in file_paths.items():
            path = Path(path)
            if path.exists():
                fmt = GEOSPATIAL_FORMATS.get(path.suffix)
                props = {"name": path.name, "description": desc}
                if fmt:
                    props["encodingFormat"] = fmt
                crate.add_file(str(path), dest_path=path.name, properties=props)

    crate.write(str(output_dir))
    print(f"  RO-Crate metadata written to {output_dir / 'ro-crate-metadata.json'}")
    return str(output_dir / "ro-crate-metadata.json")


# ── Private helpers ──

def _default_input_slots():
    """Default input slots for the urban_pfr pipeline."""
    return {
        "buildings": ["buildings", "building", "gebaeude", "building_example"],
        "statistical_units": ["statistical", "statistik", "stats", "census", "stadtteile"],
        "streets": ["streets", "street", "strassen", "roads", "road"],
        "flood": ["flood", "floodlevel", "hochwasser", "pluvial"],
        "boundary": ["boundary", "border", "grenze", "study_area", "admin"],
    }


def _match_crate_to_slot(crate, input_slots):
    """Match an RO-Crate to an input slot based on keywords."""
    text = (
        f"{crate['name']} {crate['description']} "
        f"{crate['crate_dir'].name} "
        f"{' '.join(f.name for f in crate['data_files'])}"
    ).lower()

    best_slot = None
    best_score = 0
    for slot, keywords in input_slots.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best_score = score
            best_slot = slot

    return best_slot


def _guess_format(path):
    """Guess encoding format from file extension."""
    return GEOSPATIAL_FORMATS.get(Path(path).suffix, "")


def _discover_data_files(crate_dir):
    """Find data files in a crate directory by extension."""
    files = []
    for ext in GEOSPATIAL_FORMATS:
        files.extend(crate_dir.glob(f"*{ext}"))
    files.extend(crate_dir.glob("*.gdb"))
    files = [f for f in files if f.name != "ro-crate-metadata.json"]
    return files
