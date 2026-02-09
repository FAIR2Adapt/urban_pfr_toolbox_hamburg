# urban_pfr 
### Urban Pluvial Flood Risk Assessment
This package is an  implementation of the research and methodological framework proposed by:
   von Szombathely, M., et al. (2025). Urban Pluvial Flood Risk Mapping: A High-Resolution Assessment for the City of Hamburg
This package  is designed to be reusable for cities through city-specific configuration files,
to eable CS6 to transfer the Hamburg case study to other locations.

> **Note:** This version is under  development and requires verification by CS3
> (Hamburg team) before finalizing the analysis pipeline and visualization.

## Installation & Configuration
### Install

```bash
pip install -e .
```
### Configure

One YAML file per city. Copy `config_template.yaml` and fill in paths and field names.

> **Note:** Configuration values such as TOPSIS weights and SVPF parameters
> require verification by CS3 (Hamburg team), as some are not explicitly
> defined in the original ArcGIS script files.
```yaml
project:
  city_name: "Hamburg"
  crs: "EPSG:25832"  # projected CRS in meters, not geographic (lat/lon)
  paths:
    input_gdb: "path/to/data.gdb"
    flood_dir: "path/to/Floodlevels"
    output_dir: "./outputs/hamburg"
  ...
```
See `hamburg_config.yaml` for a complete example and `config_template.yaml` for a blank starting point.

### Run
```bash
python run_analysis.py hamburg_config.yaml
```

Or from Python:
```python
from urban_pfr.analyzer import PFRAnalyzer

analyzer = PFRAnalyzer('hamburg_config.yaml')
analyzer.run()
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

# urban_pfr_toolbox_hamburg
