"""
hazard.py – Flood hazard assessment (optimized).
HMA: mobility/accessibility via street-buffer analysis.
HWB: well-being via building-perimeter flood depth analysis.

Performance strategy:
  1. Split large geometries (streets) into a spatial grid of tiles
  2. Use STRtree spatial index for fast candidate lookup
  3. Use prepared geometries for fast intersection tests
  4. Vectorized buffer creation
  5. Every building is processed — none are skipped
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import prepared
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.strtree import STRtree
from scipy.stats import lognorm
import warnings

warnings.filterwarnings('ignore')

BATCH_LOG_SIZE = 10000
GRID_TILE_SIZE = 500  # meters — split large geometries into tiles of this size


def _split_to_tiles(gdf, tile_size=GRID_TILE_SIZE):
    """
    Split all geometries in a GeoDataFrame into a grid of tiles.
    Returns a new GeoDataFrame where each row is a piece of the original
    geometry clipped to a tile. This makes STRtree lookups and intersections
    much faster because each piece is small.
    """
    # Skip costly unary_union when there's only one feature (e.g., streets layer)
    if len(gdf) == 1:
        all_geom = gdf.geometry.iloc[0]
    else:
        all_geom = unary_union(gdf.geometry)
    minx, miny, maxx, maxy = all_geom.bounds

    tiles = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            tile_box = box(x, y, x + tile_size, y + tile_size)
            try:
                clipped = all_geom.intersection(tile_box)
                if not clipped.is_empty and clipped.area > 0:
                    tiles.append(clipped)
            except Exception:
                pass
            y += tile_size
        x += tile_size

    if not tiles:
        return gdf

    print(f"    Split into {len(tiles)} tiles ({tile_size}m grid)")
    return gpd.GeoDataFrame(geometry=tiles, crs=gdf.crs)


def _compute_flood_fraction(ring_geom, flood_tree, flood_geoms, flood_prep):
    """
    Compute fraction of ring_geom covered by flood polygons.
    Uses STRtree for candidate lookup, prepared geometry for fast tests.
    Returns float in [0, 1].
    """
    if ring_geom.is_empty:
        return 0.0

    ring_area = ring_geom.area
    if ring_area <= 0:
        return 0.0

    candidates = flood_tree.query(ring_geom)
    if len(candidates) == 0:
        return 0.0

    flooded_area = 0.0
    for idx in candidates:
        if flood_prep[idx].intersects(ring_geom):
            try:
                ix = ring_geom.intersection(flood_geoms[idx])
                flooded_area += ix.area
            except Exception:
                continue

    return min(flooded_area / ring_area, 1.0)


def _compute_street_area_in_ring(ring_geom, street_tree, street_geoms, street_prep):
    """
    Compute the intersection of a ring with street tiles.
    Returns the street-in-ring geometry and its total area.
    """
    candidates = street_tree.query(ring_geom)
    if len(candidates) == 0:
        return None, 0.0

    parts = []
    for idx in candidates:
        if street_prep[idx].intersects(ring_geom):
            try:
                ix = ring_geom.intersection(street_geoms[idx])
                if not ix.is_empty and ix.area > 0:
                    parts.append(ix)
            except Exception:
                continue

    if not parts:
        return None, 0.0

    merged = unary_union(parts)
    return merged, merged.area


def calculate_hazard_mobility_accessibility(buildings_gdf, streets_gdf, flood_gdf,
                                            buffers=[5, 15, 30],
                                            min_area_threshold=0.0,
                                            shape_param=0.25):
    """
    HMA per building: ring buffers at 5/15/30m, intersect with streets,
    measure flood %, take max across buffers, apply lognorm.cdf.

    HMA = lognorm.cdf(max_flood_fraction * 4, shape_param)
    """
    buildings_result = buildings_gdf.copy()
    n_total = len(buildings_result)

    if buildings_result.crs != streets_gdf.crs:
        streets_gdf = streets_gdf.to_crs(buildings_result.crs)
    if buildings_result.crs != flood_gdf.crs:
        flood_gdf = flood_gdf.to_crs(buildings_result.crs)

    # Split streets into tiles for fast per-building intersection
    print(f"  Preparing streets spatial index ...")
    streets_tiled = _split_to_tiles(streets_gdf, tile_size=GRID_TILE_SIZE)
    street_geoms = list(streets_tiled.geometry)
    street_tree = STRtree(street_geoms)
    street_prep = [prepared.prep(g) for g in street_geoms]

    # Prepare flood spatial index
    print(f"  Preparing flood spatial index ...")
    flood_geoms = list(flood_gdf.geometry)
    flood_tree = STRtree(flood_geoms)
    flood_prep = [prepared.prep(g) for g in flood_geoms]

    max_flood_pct = np.zeros(n_total)

    sorted_buffers = sorted(buffers)

    for buffer_dist in sorted_buffers:
        print(f"  HMA buffer {buffer_dist}m ({n_total} buildings) ...")

        # Vectorized buffer creation
        outer = buildings_result.geometry.buffer(buffer_dist)

        for i in range(n_total):
            if i > 0 and i % BATCH_LOG_SIZE == 0:
                print(f"    {i}/{n_total} ...")

            building_geom = buildings_result.geometry.iloc[i]
            ring = outer.iloc[i].difference(building_geom)

            if ring.is_empty or ring.area <= 0:
                continue

            if buffer_dist <= 5:
                # 5m buffer: direct flood fraction in the ring (no streets)
                flood_pct = _compute_flood_fraction(
                    ring, flood_tree, flood_geoms, flood_prep
                ) * 100
            else:
                # 15m/30m buffers: flood fraction on streets within the ring
                street_in_ring, total_street_area = _compute_street_area_in_ring(
                    ring, street_tree, street_geoms, street_prep
                )

                if street_in_ring is None or total_street_area < min_area_threshold:
                    continue

                flood_pct = _compute_flood_fraction(
                    street_in_ring, flood_tree, flood_geoms, flood_prep
                ) * 100

            if flood_pct > max_flood_pct[i]:
                max_flood_pct[i] = flood_pct

    # Apply lognormal CDF
    hma = np.where(
        max_flood_pct > 0,
        lognorm.cdf(max_flood_pct / 100 * 4, shape_param),
        0.0
    )
    buildings_result['HMA'] = hma

    n = (buildings_result['HMA'] > 0).sum()
    print(f"HMA: {n}/{len(buildings_result)} buildings affected, "
          f"max={buildings_result['HMA'].max():.4f}")
    return buildings_result


def calculate_hazard_wellbeing(buildings_gdf, flood_layers_dict,
                               buffer_distance=2,
                               shape_param=0.25,
                               depth_thresholds=[20, 30, 40, 50, 60, 70, 80, 90, 100]):
    """
    HWB per building: 2m ring buffer, for each flood depth compute
    flood fraction in buffer, apply lognorm.cdf, sum across depths.

    HWB = sum( lognorm.cdf(flood_pct_i * 4 / 100, shape) )
    """
    buildings_result = buildings_gdf.copy()
    n_total = len(buildings_result)

    available_depths = [d for d in depth_thresholds if d in flood_layers_dict]
    if not available_depths:
        buildings_result['HWB'] = 0
        return buildings_result

    # Vectorized: compute all outer buffers once
    print(f"  Computing {n_total} buffers ...")
    outer = buildings_result.geometry.buffer(buffer_distance)

    hwb_values = np.zeros(n_total)

    for depth in available_depths:
        print(f"  HWB depth {depth}cm ...")
        flood_layer = flood_layers_dict[depth]
        if buildings_result.crs != flood_layer.crs:
            flood_layer = flood_layer.to_crs(buildings_result.crs)

        # Build spatial index for this flood layer
        flood_geoms = list(flood_layer.geometry)
        flood_tree = STRtree(flood_geoms)
        flood_prep = [prepared.prep(g) for g in flood_geoms]

        for i in range(n_total):
            if i > 0 and i % BATCH_LOG_SIZE == 0:
                print(f"    {i}/{n_total} ...")

            building_geom = buildings_result.geometry.iloc[i]
            ring = outer.iloc[i].difference(building_geom)

            if ring.is_empty or ring.area <= 0:
                continue

            flood_frac = _compute_flood_fraction(ring, flood_tree, flood_geoms, flood_prep)

            if flood_frac > 0:
                hwb_values[i] += lognorm.cdf(flood_frac * 4, shape_param)

    # Clip to [0, 1] to match ArcGIS behavior
    buildings_result['HWB'] = np.clip(hwb_values, 0, 1)

    n = (buildings_result['HWB'] > 0).sum()
    print(f"HWB: {n}/{len(buildings_result)} buildings affected, "
          f"max={buildings_result['HWB'].max():.4f}")
    return buildings_result
