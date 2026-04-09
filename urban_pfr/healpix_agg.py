"""
healpix_agg.py – Aggregate building-level results to HEALPix cells.

Uses healpix-geo for HEALPix on the WGS84 ellipsoid (not spherical
approximation), consistent with the FAIR2Adapt DGGS ecosystem.

Produces a privacy-safe, equal-area grid output where each cell
aggregates risk values from all buildings within it.

Recommended depths for urban flood risk:
  Depth 14: ~158,000 m² (~400m cells) — coarse, high privacy
  Depth 15: ~40,000 m² (~200m cells) — good balance
  Depth 16: ~10,000 m² (~100m cells) — fine detail
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import healpix_geo.nested as hpx
from shapely.geometry import Polygon


ELLIPSOID = "WGS84"


def aggregate_to_healpix(buildings_gdf, depth=15, risk_columns=None,
                          agg_func='mean', min_buildings=1):
    """
    Aggregate building-level results to HEALPix cells on the WGS84 ellipsoid.

    Parameters
    ----------
    buildings_gdf : GeoDataFrame
        Buildings with risk columns, in any CRS.
    depth : int
        HEALPix depth (nside = 2^depth). Higher = finer cells.
    risk_columns : list of str, optional
        Columns to aggregate. Defaults to risk/hazard columns found in data.
    agg_func : str or dict
        Aggregation function: 'mean', 'max', 'sum', or dict per column.
    min_buildings : int
        Minimum buildings per cell to include (privacy threshold).

    Returns
    -------
    GeoDataFrame
        HEALPix cells with aggregated values in EPSG:4326.
    """
    if risk_columns is None:
        risk_columns = [c for c in [
            'PFRMA', 'PFRWB', 'PFRMA_smoothed', 'PFRWB_smoothed',
            'HMA', 'HWB',
        ] if c in buildings_gdf.columns]

    if not risk_columns:
        raise ValueError("No risk columns found to aggregate")

    # Convert to EPSG:4326
    bldg = buildings_gdf.copy()
    if bldg.crs and str(bldg.crs) != 'EPSG:4326':
        bldg = bldg.to_crs('EPSG:4326')

    # Building centroids in lon/lat
    centroids = bldg.geometry.centroid
    lon = centroids.x.values
    lat = centroids.y.values

    # Assign each building to a HEALPix cell
    pixels = hpx.lonlat_to_healpix(lon, lat, depth, ellipsoid=ELLIPSOID)
    bldg['_healpix'] = pixels

    # Aggregate risk columns per cell
    agg_dict = {col: agg_func if isinstance(agg_func, str) else agg_func.get(col, 'mean')
                for col in risk_columns}
    agg_dict['_healpix'] = 'count'

    grouped = bldg.groupby('_healpix').agg(agg_dict)
    grouped = grouped.rename(columns={'_healpix': 'n_buildings'})

    # Privacy filter
    grouped = grouped[grouped['n_buildings'] >= min_buildings]

    nside = 2 ** depth
    cell_area_m2 = 4 * np.pi * 6371000**2 / (12 * nside**2)

    print(f"  HEALPix depth {depth} (WGS84 ellipsoid): cell area ~{cell_area_m2:.0f} m²")
    print(f"  {len(grouped)} cells with >= {min_buildings} buildings")
    print(f"  Buildings per cell: median={grouped['n_buildings'].median():.0f}, "
          f"max={grouped['n_buildings'].max()}")

    # Generate cell polygons from vertices on WGS84 ellipsoid
    print(f"  Generating {len(grouped)} cell polygons ...")
    cell_ids = np.array(grouped.index, dtype=np.uint64)
    vlon, vlat = hpx.vertices(cell_ids, depth, ellipsoid=ELLIPSOID)

    geometries = []
    for i in range(len(cell_ids)):
        coords = list(zip(vlon[i], vlat[i]))
        coords.append(coords[0])  # close polygon
        geometries.append(Polygon(coords))

    result = gpd.GeoDataFrame(
        grouped.reset_index(),
        geometry=geometries,
        crs='EPSG:4326',
    )
    result = result.rename(columns={'_healpix': 'healpix_id'})

    return result
