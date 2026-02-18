# urban_pfr 
### Urban Pluvial Flood Risk Assessment
Python package conversion of the ArcGIS workflow from [Urban Pluvial Flood Risk Mapping: A High-Resolution Assessment for the City of Hamburg](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5231006) (von Szombathely et al., 2025).

The package is designed to be reusable across cities through city-specific configuration files, for CS6 to transfer the Hamburg case study to target city.

> **Note:** This version is under  development and requires verification by CS3
> (Hamburg team) before finalizing the analysis pipeline and visualization.

**This version of the project encapsulates the code in Docker using the expected image structure, input/output paths, and metadata definition for replicating the execution environemnt at the [my.lifewatch.eu](my.lifewatch.eu) workflow platform.**

## Installation & Configuration
### Install

```bash
git clone https://github.com/FAIR2Adapt/urban_pfr_toolbox_hamburg.git
cd urban_pfr_toolbox_hamburg
git checkout feature/lifewatcheric
```
### Configure

Create a new YAML copying the following and replacing the desired fields. Save it as config.yaml and use it as an input (place it in data/inputs/config.yaml).

> **Note:** Configuration values such as TOPSIS weights and SVPF parameters
> require verification by CS3 (Hamburg team), as some are not explicitly
> defined in the original ArcGIS script files.
```yaml
# YAML

project:
  city_name: "YourCity"
  crs: "EPSG:25832"                     # must match your data

schema:
  buildings_layer: "Buildings"
  stats_layer: "StatisticalUnits"
  streets_layer: "Streets"
  stat_unit_col: "StatisticalUnit"
  residents_col: "Residents"
  living_area_col: "LivingArea"
  floors_col: "Floors"
  building_type_col: "Building_type"
  sensitivity_fields: ["WR", "C"]
  coping_fields: ['ES', 'EDQ']


hazard_settings:
  hma_buffers: [5, 15, 30]
  hwb_buffer: 2
  flood_depths: [20, 30, 40, 50, 60, 70, 80, 90, 100]
  flood_threshold: 0.3
  shape_param: 0.25
  min_area_threshold: 0.0

risk_settings:
  weight_hazard: 1.0
  weight_exposure: 1.0
  weight_vulnerability: 1.0
  smoothing_iterations: 3
  max_neighbors: 10
  distance_threshold: 100
  n_classes: 5

topsis_weights:
  sensitivity: [0.7, 0.3]        # Based on Old Singles and Children
  coping_capacity: [0.5, 0.5]    # Based on Social Benefits and Education
  svi: [0.5, 0.5]                

svpf_threshold: 0.25
svpf_transform: 2.0

visualization:
  figsize: [24, 8]
  dpi: 300
  colors:
    pfrma: ["#F5F5F5", "#FFF2CC", "#FFE599", "#FFD966", "#FF9900", "#CC7A00"]
    pfrwb: ["#F5F5F5", "#E6E6FA", "#DDA0DD", "#BA55D3", "#8B44AD", "#663399"]
```


## Required Input Data
 **Note:** Required verification by CS3 (Hamburg team).

The GDB (or individual shapefiles/GeoPackage files) must contain:

| Layer | Required Fields | Purpose |
|-------|----------------|---------|
| Buildings | `ID`, `Floors`, `Building_type`, geometry | Building footprints with structural attributes |
| Statistical units | `StatisticalUnit`, `Residents`, `LivingArea`, + sensitivity/coping fields, geometry | Census/population data at neighborhood level |
| Streets | geometry | Road network for evacuation route analysis (HMA) |
| Flood layers | geometry (one per depth level) | Pluvial flood extents at each depth threshold |

**CRS requirements:** All layers must be in the same coordinate reference system. The package will attempt to detect and fix mismatches automatically, but consistent input CRS is recommended. 

## Make sure to place **zipped** Floodlevels and pluvialfloodriskmap.gdb folders in the data/inputs folder

## Pre-run checks:

**data/inputs directory must contain (make sure the names match as follow):**
- config.yaml
- Floodlevels.zip
- pluvialfloodriskmap.gdb.zip

**Dockerfile exists**

The Dockerfile creates a Python based Docker image, installs dependencies from requirements.txt, copies the project files, and runs run_analysis.py by default when the container starts.

## RUN:
### ./bin/build-image
Builds a Docker image using the name and version defined in annotation.json.
### ./bin/execute
Validates annotation.json and data/execution-parameters.json (structure, allowed/required keys, and values) and then runs the Docker image defined in annotation.json with those parameters, mounting inputs/outputs.

## ANNEX:
**execution-parameters.json**: file where the parameters required for execution are defined. If none are needed (as in our case, it should be left empty). **It must not be removed**, as it is essential for Docker execution.

**annotation.json**: file that contains all the project data; it defines the inputs, outputs, names, and descriptions. It is used as the foundation for the entire service component definition in [MyLifeWatch](https://my.lifewatch.dev/)