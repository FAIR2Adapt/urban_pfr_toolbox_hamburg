# urban_pfr

### Urban Pluvial Flood Risk Assessment Toolbox

Python toolbox for urban pluvial flood risk assessment, implementing the methodology from [Urban Pluvial Flood Risk Mapping: A High-Resolution Assessment for the City of Hamburg](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5231006) (von Szombathely et al., 2025).

Originally developed as an ArcGIS workflow for the Hamburg case study (CS3) in the [FAIR2Adapt](https://fair2adapt.eu/) project, this package converts it into a reusable, config-driven Python pipeline that can be applied to other cities.

> **Note:** TOPSIS weights and SVPF parameters require verification by CS3
> (Hamburg team) — see comments in the config file.

## Methodology

The toolbox implements the **IPCC risk framework**:

**Risk = Hazard x Exposure x Vulnerability**

producing two pluvial flood risk indices:
- **PFRMA** — Pluvial Flood Risk for Mobility & Accessibility
- **PFRWB** — Pluvial Flood Risk for Well-Being

### Pipeline steps

```
run_analysis.py (CLI entry point)
  └─ PFRAnalyzer (analyzer.py) — orchestrates the pipeline:
       1. validation.py   → validate inputs, auto-fix CRS mismatches
       2. indicators.py   → social vulnerability (TOPSIS multi-criteria analysis)
       3. exposure.py     → distribute population to buildings
       4. hazard.py       → flood hazard metrics (HMA, HWB)
       5. risk.py         → combine into PFRMA/PFRWB, Delaunay smoothing
       6. thiessen.py     → Thiessen polygon tessellation for visualization
       7. viz.py          → publication-quality maps with head/tail classification
```

#### 1. Social Vulnerability (indicators.py)

Computes a Social Vulnerability to Pluvial Flooding index (SVPF) using TOPSIS with optional Shannon entropy weighting, in a 4-step chain:

**Sensitivity → Coping Capacity → SVI → SVPF**

- **Sensitivity**: Combines demographic indicators of flood vulnerability (e.g., share of elderly, children)
- **Coping Capacity**: Combines socioeconomic indicators (e.g., education, social benefits)
- **SVI** (Social Vulnerability Index): Weighted combination of Sensitivity and Coping Capacity
- **SVPF**: Flood-specific vulnerability transform of SVI

#### 2. Exposure (exposure.py)

Distributes statistical-unit population to individual buildings proportionally by residential floor area:

- **R** (EMA): Total residents per building, proportional to `(floors - (building_type - 1)) * building_area / unit_living_area * unit_residents`
- **R_G** (EWB): Ground-floor residents = `R / floors` (only for residential buildings, type 1)

Non-residential buildings (type 2) receive zero exposure.

#### 3. Hazard (hazard.py)

**HMA** (Hazard Mobility & Accessibility): For each building, creates ring buffers at 5m, 15m, and 30m. Each ring is intersected with streets, then the flooded fraction of street area within the ring is measured. The maximum flood fraction across all buffers is transformed via a lognormal CDF:

```
HMA = lognorm.cdf(max_flood_fraction * 4, shape=0.25)
```

Uses STRtree spatial indexing and street geometry tiling for performance (~55 min for 227k buildings).

**HWB** (Hazard Well-Being): For each building, creates a 2m ring buffer and measures flood coverage at each depth level (30–100cm). Each depth's flood fraction is transformed via lognormal CDF and summed:

```
HWB = sum(lognorm.cdf(flood_fraction_depth_i * 4, shape=0.25))
```

#### 4. Risk (risk.py)

Combines the three IPCC risk components with configurable weights (default: 1.0 each):

```
PFRMA = HMA^w_h * R^w_e * SVPF^w_v
PFRWB = HWB^w_h * R_G^w_e * SVPF^w_v
```

**Delaunay smoothing**: Risk values are spatially smoothed using Delaunay triangulation to identify building neighbors. Each iteration computes `(original + inv_dist_weighted_avg) / (n_neighbors + 1)`, matching the ArcGIS formula (3 iterations by default, no distance threshold or max neighbors limit). Uses sparse matrix multiplication for performance. Buildings without neighbors keep their original values.

#### 5. Thiessen polygons (thiessen.py)

Replaces building footprints with Voronoi/Thiessen polygons clipped to the study area boundary. This creates a continuous tessellation where each polygon inherits the risk value of its building centroid, matching the ArcGIS workflow output format used in the paper (Figures 4–6).

The boundary can be provided as a file (via `paths.boundary_file` in the config) or is derived from the convex hull of all buildings.

#### 6. Visualization (viz.py)

Generates a 3-panel publication map with head/tail breaks classification into 5 classes:
- Panel A: PFRMA (yellow-orange gradient)
- Panel B: PFRWB (purple gradient)
- Panel C: SVPF vulnerability (grey gradient)

## Installation

```bash
git clone https://github.com/FAIR2Adapt/urban_pfr_toolbox_hamburg.git
cd urban_pfr_toolbox_hamburg
pip install -e .
```

## Run

### Locally

```bash
pip install -e .
python run_analysis.py local config.yaml

# Skip validation (if CRS is already consistent)
python run_analysis.py local config.yaml --skip-validation
```

If the package is installed, you can also use the CLI command directly:
```bash
urban-pfr local config.yaml
```

### FDO mode (RO-Crate in / RO-Crate out)

```bash
urban-pfr fdo --input-dir ./data/inputs/ --output-dir ./outputs/hamburg/
```

In FDO mode, the pipeline:
1. Reads the Workflow RO-Crate (`ro-crate-metadata.json` in the project root) to discover expected input parameters
2. Scans the input directory for data RO-Crates and matches them to input slots by format, type, and [I-ADOPT](https://i-adopt.github.io/) `variableMeasured` annotations
3. Resolves column mappings automatically via `propertyID` (I-ADOPT nanopublication URIs) — no hardcoded column names needed
4. Runs the pipeline
5. Creates an output Workflow Run Crate with provenance

This is the recommended mode for FAIR reproducibility. See [fdo-resolver](https://github.com/FAIR2Adapt/fdo-resolver) for details on how variable matching works.

### Via Docker (LifeWatch platform)

Place inputs in `data/inputs/` and run:

```bash
./bin/build-image
./bin/execute
```

The Docker entry point auto-detects inputs at `/mnt/inputs/` and writes results to `/mnt/outputs/outputs.zip`. It accepts multiple input formats:

**Option A — Zipped archives (original LifeWatch format):**
```
data/inputs/
  config.yaml
  pluvialfloodriskmap.gdb.zip      # zipped GDB
  Floodlevels.zip                  # zipped flood layers
```

**Option B — Unzipped GDB + flood directory:**
```
data/inputs/
  config.yaml
  pluvialfloodriskmap.gdb/         # extracted GDB directory
  Floodlevels/                     # directory with Flood_*.geojson
    Flood_30.geojson
    ...
```

**Option C — Individual files (no GDB):**
```
data/inputs/
  config.yaml
  buildings.gpkg                   # or .geojson / .shp
  statistical_units.gpkg
  streets.gpkg
  Flood_30.geojson                 # flood layers directly in inputs/
  Flood_40.geojson
  ...
```

**Option D — RO-Crate FDOs (one per input):**
```
data/inputs/
  config.yaml
  buildings/
    ro-crate-metadata.json
    pluvialfloodriskmap.gdb/
  floodlevels/
    ro-crate-metadata.json
    Flood_30.geojson
    Flood_40.geojson
    ...
  streets/
    ro-crate-metadata.json
    streets.gpkg
  boundary/                        # optional
    ro-crate-metadata.json
    hamburg_boundary.gpkg
```

Each input is a [FAIR Digital Object](https://fairdo.org/) implemented as an [RO-Crate](https://www.researchobject.org/ro-crate/) — a directory containing a `ro-crate-metadata.json` file describing the data. The pipeline auto-detects RO-Crate inputs and maps them to the correct input slots based on the crate metadata (name, description, file types).

The boundary file (`*boundary*.gpkg`) is auto-detected if present in the inputs directory.

## Required Input Data

### For local runs

The config YAML points to individual paths. A typical layout:

```
data/inputs/
  config.yaml                          # pipeline configuration
  pluvialfloodriskmap.gdb/             # ArcGIS GDB with 3 layers (see below)
  Floodlevels/                         # flood depth layers
    Flood_30.geojson
    Flood_40.geojson
    ...
    Flood_100.geojson
  hamburg_boundary.gpkg                # optional: study area boundary
```

The GDB (or individual GeoPackage/GeoJSON files) must contain these layers:

| Layer | Required Fields | Purpose |
|-------|----------------|---------|
| Buildings | `ID`, `Floors`, `Building_type`, `StatisticalUnit`, geometry | Building footprints with structural attributes |
| Statistical units | `StatisticalUnit`, `Residents`, `LivingArea`, + sensitivity/coping fields, geometry | Census/population data at neighborhood level |
| Streets | geometry | Street network for accessibility analysis (HMA) |

Flood layers are separate files, one per depth threshold (cm):

| File | Description |
|------|-------------|
| `Flood_30.geojson` | Flood extent at 30cm depth |
| `Flood_40.geojson` ... `Flood_100.geojson` | Flood extents at increasing depths |

The **boundary file** (optional) defines the study area for Thiessen polygon clipping. If not provided, the convex hull of all buildings is used (which may extend beyond the city boundary). For Hamburg, this is the administrative boundary (~359 km²).

### For Docker / LifeWatch runs

Place **zipped** inputs in `data/inputs/`:

```
data/inputs/
  config.yaml                          # pipeline configuration
  pluvialfloodriskmap.gdb.zip          # zipped GDB archive
  Floodlevels.zip                      # zipped flood layers
```

The Docker entry point automatically extracts the archives and overrides the YAML paths to `/mnt/inputs/` and `/mnt/outputs/`.

### CRS requirements

All layers must share the same projected CRS (e.g., EPSG:25832 for Hamburg). The pipeline auto-detects CRS mismatches and reprojects flood layers to match the buildings CRS.

### Building types

| `Building_type` | Description | Exposure treatment |
|-----------------|-------------|-------------------|
| 1 | Residential | Full residential floor area; ground-floor residents computed |
| 2 | Non-residential | Reduced floor area (`floors - 1`); no ground-floor residents (R_G = 0) |

## Adapting to a Different City

To apply the toolbox to a new city, copy `data/inputs/config.yaml` and adjust:

### 1. Paths and CRS

```yaml
project:
  city_name: "Bremen"
  crs: "EPSG:25832"            # must match your input data
  paths:
    input_gdb: "path/to/your.gdb"
    flood_dir: "path/to/flood_layers/"
    output_dir: "outputs/bremen"
    boundary_file: "path/to/city_boundary.gpkg"  # optional
```

### 2. Column mapping

If your data uses different column names, map them in the `schema` section:

```yaml
schema:
  buildings_layer: "Gebaeude"            # layer name in GDB
  stats_layer: "Stadtteile"             # statistical units layer name
  streets_layer: "Strassen"             # streets layer name
  stat_unit_col: "Stadtteil_ID"         # joins buildings to statistical units
  residents_col: "Einwohner"            # population per statistical unit
  living_area_col: "Wohnflaeche"        # total living area per unit
  floors_col: "Stockwerke"             # number of floors
  building_type_col: "Gebaeudeart"      # 1=residential, 2=non-residential
  sensitivity_fields: ["Senioren", "Kinder"]    # vulnerability indicators
  coping_fields: ["Bildung", "Sozialleistungen"]
```

### 3. Weights and parameters

Adjust TOPSIS weights to match your city's vulnerability indicators. The weight arrays must have the same length as the corresponding field lists:

```yaml
topsis_weights:
  sensitivity: [0.6, 0.4]        # one weight per sensitivity field
  coping_capacity: [0.5, 0.5]    # one weight per coping field
  svi: [0.5, 0.5]                # [sensitivity_weight, coping_weight]
```

### 4. Building types

By default, building type `1` is residential and all others have one floor deducted for exposure. If your city uses different type codes, configure them:

```yaml
exposure_settings:
  residential_types: [1, 3]              # which type values are residential
  non_residential_floor_deduction: 1     # floors deducted for non-residential
```

For example, if your data uses `"R"` for residential and `"C"` for commercial, you would first recode them to integers and set `residential_types` accordingly.

### 5. Using FDO mode (recommended for FAIR reproducibility)

Instead of a config YAML, you can describe your input data as RO-Crates with [I-ADOPT](https://i-adopt.github.io/) variable annotations. This eliminates hardcoded column names — the pipeline discovers the correct column mappings from the metadata.

For each input dataset, create a directory with a `ro-crate-metadata.json`:

```json
{
  "@id": "buildings.fgb",
  "@type": "File",
  "encodingFormat": "application/flatgeobuf",
  "variableMeasured": [
    {"@id": "#var-elderly"},
    {"@id": "#var-children"}
  ]
}
```
```json
{
  "@id": "#var-elderly",
  "@type": "PropertyValue",
  "name": "Senioren",
  "propertyID": "https://w3id.org/np/RA-iadopt-elderly-singles-placeholder"
}
```

The `propertyID` is an I-ADOPT nanopublication URI that semantically identifies the variable. When the same `propertyID` appears in both the workflow's expected inputs and the data's `variableMeasured`, the column mapping is resolved automatically (e.g. workflow expects `elderly_singles` → data has it as `Senioren`).

Then run:
```bash
urban-pfr fdo --input-dir ./my-city-data/ --output-dir ./outputs/my-city/
```

See `Hamburg-data/fdo/` for a complete example of input RO-Crates.

### Current limitations for new cities

- **Flood layer naming** must follow `Flood_{depth_cm}.{ext}` (e.g., `Flood_30.geojson`, `Flood_50.fgb`).
- **Boundary file**: Without a study area boundary, Thiessen polygons extend to the convex hull of all buildings. For best results, provide the city administrative boundary.
- **Input format**: FlatGeobuf (.fgb) is recommended for large datasets (3-5x faster than GeoJSON).

## Configuration

All parameters are in a single YAML file. Key sections:

```yaml
project:
  city_name: "Hamburg"
  crs: "EPSG:25832"
  paths:
    input_gdb: "data/inputs/pluvialfloodriskmap.gdb"
    flood_dir: "data/inputs/Floodlevels_extracted/Floodlevels"
    output_dir: "data/outputs"
    boundary_file: "data/inputs/hamburg_boundary.gpkg"  # optional

schema:
  buildings_layer: "Building_ExampleLayer"
  stats_layer: "StatisticalExampleUnit"
  streets_layer: "Streets"
  sensitivity_fields: ["ES", "C"]       # ArcGIS: Elderly Singles, Children
  coping_fields: ["WR", "EDQ"]          # ArcGIS: Welfare Recipients, Education

topsis_entropy:                          # Shannon entropy for TOPSIS steps
  sensitivity: true
  coping_capacity: true
  svi: true

hazard_settings:
  hma_buffers: [5, 15, 30]       # ring buffer distances (meters)
  hwb_buffer: 2                   # building perimeter buffer (meters)
  flood_depths: [30, 40, 50, 60, 70, 80, 90, 100]  # cm
  flood_threshold: 0.3            # depth for HMA (30cm)
  shape_param: 0.25               # lognormal CDF shape parameter

risk_settings:
  weight_hazard: 1.0
  weight_exposure: 1.0
  weight_vulnerability: 1.0
  smoothing_iterations: 3

topsis_weights:
  sensitivity: [0.7, 0.3]
  coping_capacity: [0.5, 0.5]
  svi: [0.5, 0.5]
```

## Output

The pipeline produces two output tiers:

### Public outputs (safe to publish)

```
outputs/public/
  risk_healpix.fgb             # HEALPix cells — risk values, EPSG:4326 for web viewer
  risk_healpix.gpkg            # Same, projected CRS
  vulnerability_healpix.fgb    # HEALPix cells — aggregated vulnerability, EPSG:4326
  vulnerability_healpix.gpkg   # Same, projected CRS
```

| Column | File | Description |
|--------|------|-------------|
| `PFRMA` | risk_healpix | Pluvial Flood Risk MA (mean per cell) |
| `PFRWB` | risk_healpix | Pluvial Flood Risk WB (mean per cell) |
| `PFRMA_smoothed` | risk_healpix | Smoothed PFRMA (mean per cell) |
| `PFRWB_smoothed` | risk_healpix | Smoothed PFRWB (mean per cell) |
| `HMA` | risk_healpix | Hazard Mobility & Accessibility (0–1, mean) |
| `HWB` | risk_healpix | Hazard Well-Being (0–1, mean) |
| `n_buildings` | both | Number of buildings in the cell |
| `healpix_id` | both | HEALPix cell identifier (nested scheme) |
| `Sensitivity` | vulnerability_healpix | TOPSIS sensitivity (mean per cell) |
| `CopingCapacity` | vulnerability_healpix | TOPSIS coping capacity (mean per cell) |
| `SVI` | vulnerability_healpix | Social Vulnerability Index (mean per cell) |
| `SVPF` | vulnerability_healpix | Social Vulnerability to Pluvial Flooding (mean per cell) |

The public layers use **HEALPix cells on the WGS84 ellipsoid** ([healpix-geo](https://github.com/IAOCEA/healpix-geo)) — equal-area cells where each aggregates multiple buildings. No individual buildings can be identified. No demographic data is included. Cells with fewer than 3 buildings (configurable `min_buildings`) are excluded for privacy.

```yaml
output_settings:
  healpix_depth: 15          # depth 15 = ~200m cells (default)
  min_buildings: 3           # privacy threshold
```

### Private outputs (for municipalities)

```
outputs/private/
  buildings_with_risk.gpkg                       # Full building-level data + demographics
  buildings_with_risk.fgb                        # Same, EPSG:4326 FlatGeobuf (web-ready)
  statistical_units_with_vulnerability.gpkg      # Statistical units with all indicators
  statistical_units_with_vulnerability.fgb       # Same, EPSG:4326 FlatGeobuf (web-ready)
```

Contains building-level risk values plus demographics (`Residents`, `ES`, `C`, `WR`, `EDQ`, `R`, `R_G`, etc.). These files should **not be published** as they contain private demographic data at individual building level.

### Visualization

```
outputs/
  hamburg_risk_map.png         # 3-panel publication map (PFRMA, PFRWB, combined)
```

A MapLibre GL JS web viewer is included in `viewer/index.html` for interactive exploration of the public FlatGeobuf outputs.

## Architecture

```
urban_pfr/
  __init__.py          # package exports
  analyzer.py          # PFRAnalyzer orchestrator class
  cli.py               # CLI entry point (local/docker/fdo modes)
  validation.py        # input validation, CRS auto-fix
  indicators.py        # TOPSIS social vulnerability (SVPF)
  exposure.py          # population distribution to buildings
  hazard.py            # HMA (STRtree + street tiling), HWB
  risk.py              # PFRMA/PFRWB calculation, Delaunay smoothing
  thiessen.py          # Voronoi tessellation clipped to study area
  rocrate_io.py        # RO-Crate FDO I/O via fdo-resolver
  healpix_agg.py       # HEALPix aggregation on WGS84 ellipsoid (via healpix-geo)
  viz.py               # head/tail breaks classification, publication maps

ro-crate-metadata.json # Workflow RO-Crate with FormalParameter definitions
viewer/index.html      # MapLibre GL JS web viewer for FlatGeobuf outputs
```

## LifeWatch Integration

This branch supports execution on the [my.lifewatch.eu](https://my.lifewatch.eu) workflow platform (Argo-based):

- **Dockerfile**: `ghcr.io/osgeo/gdal:ubuntu-small-3.9.3` image (includes GDAL), runs `run_analysis.py` as entrypoint
- **I/O contract**: Inputs at `/mnt/inputs/`, outputs at `/mnt/outputs/` (YAML paths are overridden)
- **`annotation.json`**: LifeWatch service metadata (inputs/outputs/resources/tags)
- **`bin/build-image`**: Builds Docker image using name/version from `annotation.json`
- **`bin/execute`**: Validates metadata and runs the container with mounted volumes

## Related Resources

- Paper: [Urban Pluvial Flood Risk Mapping for Hamburg](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5231006)
- FAIR2Adapt project: [fair2adapt.eu](https://fair2adapt.eu/)
- [fdo-resolver](https://github.com/FAIR2Adapt/fdo-resolver) — RO-Crate input resolution with I-ADOPT variable matching
- [I-ADOPT](https://i-adopt.github.io/) — Framework for observable property descriptions
- Output data corresponds to Figures 4, 5, and 6 in the paper
