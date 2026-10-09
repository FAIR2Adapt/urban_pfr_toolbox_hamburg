"""
rocrate_io.py – RO-Crate FAIR Digital Object I/O for urban_pfr.

Thin wrapper around the fdo_resolver library, adding pipeline-specific
input slot definitions and config path mapping.
"""

from fdo_resolver import FDOResolver

# Pipeline-specific input slots
URBAN_PFR_SLOTS = [
    {
        "name": "buildings",
        "encoding_format": "application/x-filegdb",
        "description": "Building footprints with demographic attributes",
    },
    {
        "name": "statistical_units",
        "encoding_format": "application/geo+json",
        "description": "Statistical units with social indicators",
    },
    {
        "name": "streets",
        "encoding_format": "application/geo+json",
        "description": "Street network outlines",
    },
    {
        "name": "flood",
        "encoding_format": "application/geo+json",
        "description": "Flood level scenario layers",
    },
    {
        "name": "boundary",
        "encoding_format": "application/geopackage+sqlite3",
        "description": "Study area boundary",
        "value_required": False,
    },
]


def resolve_rocrate_inputs(input_dir, input_slots=None):
    """
    Scan input_dir for RO-Crate directories and match them to input slots.

    Parameters
    ----------
    input_dir : str or Path
        Directory to scan for RO-Crate subdirectories.
    input_slots : list of dict, optional
        Parameter definitions. Defaults to URBAN_PFR_SLOTS.

    Returns
    -------
    dict or None
        Mapping of slot_name -> crate info dict.
    """
    slots = input_slots or URBAN_PFR_SLOTS
    resolver = FDOResolver.from_parameters(slots)
    result = resolver.resolve(input_dir)

    if not result.bindings:
        return None

    # Convert to legacy dict format for backward compatibility
    resolved = {}
    for name, binding in result.bindings.items():
        resolved[name] = {
            "name": binding.entity.name,
            "description": binding.entity.description,
            "data_files": [binding.entity.path],
            "encoding_formats": [binding.entity.encoding_format],
            "crate_dir": binding.entity.crate_dir,
        }

    return resolved


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
        Paths suitable for config['project']['paths'].
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
        Mapping of description -> file path.
    """
    city = config.get("project", {}).get("city_name", "Unknown")

    resolver = FDOResolver()
    meta_path = resolver.create_run_crate(
        output_dir,
        name=f"{city} Pluvial Flood Risk Assessment Results",
        description=(
            f"Output from the urban_pfr pipeline for {city}. "
            f"Contains flood risk indices (PFRMA, PFRWB) as Thiessen polygons "
            f"and social vulnerability indicators (SVPF) at statistical unit level."
        ),
        output_files=file_paths,
    )
    return str(meta_path) if meta_path else None
