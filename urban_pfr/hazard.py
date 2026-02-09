"""
hazard.py – Flood hazard assessment.
HMA: mobility/accessibility via street-buffer analysis.
HWB: well-being via building-perimeter flood depth analysis.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import lognorm
import warnings

warnings.filterwarnings('ignore')


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

    if buildings_result.crs != streets_gdf.crs:
        streets_gdf = streets_gdf.to_crs(buildings_result.crs)
    if buildings_result.crs != flood_gdf.crs:
        flood_gdf = flood_gdf.to_crs(buildings_result.crs)

    hma_values = []

    for idx, building in buildings_result.iterrows():
        building_geom = building.geometry
        flood_percentages = []

        for buffer_dist in buffers:
            buffer_geom = building_geom.buffer(buffer_dist)
            ring_buffer = buffer_geom.difference(building_geom)

            if buffer_dist > 5:
                streets_in_buffer = streets_gdf[streets_gdf.intersects(ring_buffer)]
                if len(streets_in_buffer) == 0:
                    flood_percentages.append(0)
                    continue

                street_buffer_intersection = streets_in_buffer.intersection(ring_buffer)
                total_street_area = street_buffer_intersection.area.sum()

                if total_street_area < min_area_threshold:
                    flood_percentages.append(0)
                    continue

                flooded_streets = street_buffer_intersection.intersection(flood_gdf.unary_union)
                flooded_area = flooded_streets.area.sum() if hasattr(flooded_streets, 'area') else 0
                flood_pct = (flooded_area / total_street_area) * 100 if total_street_area > 0 else 0
            else:
                flood_in_buffer = ring_buffer.intersection(flood_gdf.unary_union)
                buffer_area = ring_buffer.area
                flooded_area = flood_in_buffer.area if hasattr(flood_in_buffer, 'area') else 0
                flood_pct = (flooded_area / buffer_area) * 100 if buffer_area > 0 else 0

            flood_percentages.append(flood_pct)

        max_flood_pct = max(flood_percentages) if flood_percentages else 0

        if max_flood_pct > 0:
            hma = lognorm.cdf(max_flood_pct / 100 * 4, shape_param)
        else:
            hma = 0

        hma_values.append(hma)

    buildings_result['HMA'] = hma_values

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

    available_depths = [d for d in depth_thresholds if d in flood_layers_dict]
    if not available_depths:
        buildings_result['HWB'] = 0
        return buildings_result

    hwb_values = np.zeros(len(buildings_result))

    for depth in available_depths:
        flood_layer = flood_layers_dict[depth]
        if buildings_result.crs != flood_layer.crs:
            flood_layer = flood_layer.to_crs(buildings_result.crs)

        flood_union = flood_layer.unary_union

        for i, (idx, building) in enumerate(buildings_result.iterrows()):
            buffer_geom = building.geometry.buffer(buffer_distance)
            ring_buffer = buffer_geom.difference(building.geometry)
            buffer_area = ring_buffer.area

            if buffer_area <= 0:
                continue

            flood_in_buffer = ring_buffer.intersection(flood_union)
            flooded_area = flood_in_buffer.area if hasattr(flood_in_buffer, 'area') else 0
            flood_pct = flooded_area / buffer_area

            if flood_pct > 0:
                hwb_values[i] += lognorm.cdf(flood_pct * 4, shape_param)

    buildings_result['HWB'] = hwb_values

    n = (buildings_result['HWB'] > 0).sum()
    print(f"HWB: {n}/{len(buildings_result)} buildings affected, "
          f"max={buildings_result['HWB'].max():.4f}")
    return buildings_result