"""
Flood hazard assessment.
HMA: mobility/accessibility via street-buffer analysis.
HWB: well-being via building-perimeter flood depth analysis.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import lognorm
import warnings
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Chunk size auto-selection
def _get_chunk_size(n_buildings, default=10_000):
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        chunk = min(50_000, max(2_000, int(available_gb * 1500)))
    except ImportError:
        chunk = default
    return min(chunk, n_buildings)


#HMA helpers 
def _compute_ring_flood_pct_direct(ring_gdf, flood_gdf, chunk_size):
    """
    For the 5m buffer
    """
    n = len(ring_gdf)
    if n == 0:
        return pd.Series(dtype=float)

    # Ensure CRS match
    if ring_gdf.crs != flood_gdf.crs:
        flood_gdf = flood_gdf.to_crs(ring_gdf.crs)

    ring_areas = ring_gdf.geometry.area
    flooded_areas = pd.Series(0.0, index=ring_gdf.index)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = ring_gdf.iloc[start:end]

        try:
            overlay = gpd.overlay(
                chunk[['_bldg_idx', 'geometry']],
                flood_gdf[['geometry']],
                how='intersection',
                keep_geom_type=False,
            )
            if len(overlay) > 0:
                overlay_areas = overlay.dissolve(by='_bldg_idx').geometry.area
                flooded_areas.update(overlay_areas)
        except Exception:
            # Fallback: per-feature intersection for this chunk
            flood_union = flood_gdf.unary_union
            for i in range(len(chunk)):
                idx = chunk.index[i]
                geom = chunk.geometry.iloc[i]
                try:
                    inter = geom.intersection(flood_union)
                    flooded_areas.iloc[start + i] = inter.area if hasattr(inter, 'area') else 0
                except Exception:
                    pass

    pct = pd.Series(0.0, index=ring_gdf.index)
    valid = ring_areas > 0
    pct[valid] = flooded_areas[valid] / ring_areas[valid] * 100
    return pct


def _compute_ring_flood_pct_streets(ring_gdf, streets_gdf, flood_gdf, chunk_size):
    """
    For 15m/30m buffers
    """
    n = len(ring_gdf)
    if n == 0:
        return pd.Series(dtype=float)

    if ring_gdf.crs != streets_gdf.crs:
        streets_gdf = streets_gdf.to_crs(ring_gdf.crs)
    if ring_gdf.crs != flood_gdf.crs:
        flood_gdf = flood_gdf.to_crs(ring_gdf.crs)

    pct_series = pd.Series(0.0, index=ring_gdf.index)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = ring_gdf.iloc[start:end]

        try:
            # Step 1: clip streets to ring buffers
            streets_in_rings = gpd.overlay(
                chunk[['_bldg_idx', 'geometry']],
                streets_gdf[['geometry']],
                how='intersection',
                keep_geom_type=False,
            )
            if len(streets_in_rings) == 0:
                continue

            # Total street area per building
            total_street_area = (
                streets_in_rings
                .assign(_area=streets_in_rings.geometry.area)
                .groupby('_bldg_idx')['_area']
                .sum()
            )

            # Step 2: clip those street segments to flood extent
            flooded_streets = gpd.overlay(
                streets_in_rings[['_bldg_idx', 'geometry']],
                flood_gdf[['geometry']],
                how='intersection',
                keep_geom_type=False,
            )

            if len(flooded_streets) > 0:
                flooded_area = (
                    flooded_streets
                    .assign(_area=flooded_streets.geometry.area)
                    .groupby('_bldg_idx')['_area']
                    .sum()
                )

                # Compute percentage for buildings that have both values
                common = total_street_area.index.intersection(flooded_area.index)
                valid = total_street_area[common] > 0
                pct_vals = (flooded_area[common[valid]] / total_street_area[common[valid]]) * 100
                pct_series.update(pct_vals)

        except Exception as e:
            logger.warning(f"Overlay failed for chunk {start}-{end}, falling back: {e}")
            # Fallback: per-feature for this chunk
            flood_union = flood_gdf.unary_union
            for i in range(len(chunk)):
                idx = chunk.index[i]
                ring_geom = chunk.geometry.iloc[i]
                try:
                    si = streets_gdf[streets_gdf.intersects(ring_geom)]
                    if len(si) == 0:
                        continue
                    st_clip = si.intersection(ring_geom)
                    total_a = st_clip.area.sum()
                    if total_a <= 0:
                        continue
                    flooded = st_clip.intersection(flood_union)
                    flood_a = flooded.area.sum() if hasattr(flooded, 'area') else 0
                    pct_series.at[idx] = (flood_a / total_a) * 100
                except Exception:
                    pass

    return pct_series


def calculate_hazard_mobility_accessibility(buildings_gdf, streets_gdf, flood_gdf,
                                            buffers=[5, 15, 30],
                                            min_area_threshold=0.0,
                                            shape_param=0.25):
    """
    HMA per building: ring buffers at 5/15/30m
    """
    buildings_result = buildings_gdf.copy()
    n = len(buildings_result)
    chunk_size = _get_chunk_size(n)

    if buildings_result.crs != streets_gdf.crs:
        streets_gdf = streets_gdf.to_crs(buildings_result.crs)
    if buildings_result.crs != flood_gdf.crs:
        flood_gdf = flood_gdf.to_crs(buildings_result.crs)

    print(f"  HMA: processing {n} buildings (chunk_size={chunk_size})")

    # Phase A: Vectorized ring buffer construction 
    sorted_buffers = sorted(buffers)
    building_geom = buildings_result.geometry

    ring_gdfs = {}
    prev_buffer_geom = building_geom 

    for buf_dist in sorted_buffers:
        outer = building_geom.buffer(buf_dist)
        ring = outer.difference(prev_buffer_geom)
        ring_gdf = gpd.GeoDataFrame(
            {'_bldg_idx': buildings_result.index, 'geometry': ring},
            crs=buildings_result.crs,
        )
        ring_gdf = ring_gdf.set_index('_bldg_idx')
        ring_gdfs[buf_dist] = ring_gdf

        if buf_dist == sorted_buffers[0]:
            prev_buffer_geom = building_geom  

    # Each ring = buffer(dist).difference(building_geom)
    ring_gdfs = {}
    for buf_dist in sorted_buffers:
        outer = building_geom.buffer(buf_dist)
        ring = outer.difference(building_geom)
        ring_gdf = gpd.GeoDataFrame(
            {'_bldg_idx': buildings_result.index, 'geometry': ring},
            crs=buildings_result.crs,
        )
        ring_gdf = ring_gdf.set_index('_bldg_idx')
        ring_gdfs[buf_dist] = ring_gdf

    # Compute flood percentages per buffer
    all_pcts = {}
    for buf_dist in sorted_buffers:
        rg = ring_gdfs[buf_dist].copy()
        rg['_bldg_idx'] = rg.index

        if buf_dist <= 5:
            pct = _compute_ring_flood_pct_direct(rg, flood_gdf, chunk_size)
        else:
            pct = _compute_ring_flood_pct_streets(rg, streets_gdf, flood_gdf, chunk_size)

        all_pcts[buf_dist] = pct.reindex(buildings_result.index, fill_value=0.0).values
        print(f"    Buffer {buf_dist}m: {(all_pcts[buf_dist] > 0).sum()} buildings with flood")

    pct_stack = np.column_stack(list(all_pcts.values()))
    max_flood_pct = np.max(pct_stack, axis=1)

    hma = np.zeros(n)
    affected = max_flood_pct > 0
    if affected.any():
        hma[affected] = lognorm.cdf(max_flood_pct[affected] / 100 * 4, shape_param)

    buildings_result['HMA'] = hma

    n_affected = affected.sum()
    print(f"HMA: {n_affected}/{n} buildings affected, "
          f"max={buildings_result['HMA'].max():.4f}")
    return buildings_result


def calculate_hazard_wellbeing(buildings_gdf, flood_layers_dict,
                               buffer_distance=2,
                               shape_param=0.25,
                               depth_thresholds=[20, 30, 40, 50, 60, 70, 80, 90, 100]):
    """
    HWB per building
    """
    buildings_result = buildings_gdf.copy()
    n = len(buildings_result)
    chunk_size = _get_chunk_size(n)

    available_depths = [d for d in depth_thresholds if d in flood_layers_dict]
    if not available_depths:
        buildings_result['HWB'] = 0
        return buildings_result

    building_geom = buildings_result.geometry
    outer = building_geom.buffer(buffer_distance)
    rings = outer.difference(building_geom)
    ring_areas = rings.area.values

    ring_gdf = gpd.GeoDataFrame(
        {'_bldg_idx': buildings_result.index, 'geometry': rings},
        crs=buildings_result.crs,
    )
    ring_gdf = ring_gdf.set_index('_bldg_idx')

    hwb_accum = np.zeros(n)

    for depth in available_depths:
        flood_layer = flood_layers_dict[depth]
        if buildings_result.crs != flood_layer.crs:
            flood_layer = flood_layer.to_crs(buildings_result.crs)

        flooded_area = np.zeros(n)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = ring_gdf.iloc[start:end].copy()
            chunk['_bldg_idx'] = chunk.index

            try:
                overlay = gpd.overlay(
                    chunk[['_bldg_idx', 'geometry']],
                    flood_layer[['geometry']],
                    how='intersection',
                    keep_geom_type=False,
                )
                if len(overlay) > 0:
                    areas = (
                        overlay
                        .assign(_a=overlay.geometry.area)
                        .groupby('_bldg_idx')['_a']
                        .sum()
                    )
                    for idx_val, area_val in areas.items():
                        pos = buildings_result.index.get_loc(idx_val)
                        flooded_area[pos] = area_val

            except Exception as e:
                logger.warning(f"HWB overlay failed depth={depth} chunk {start}-{end}: {e}")
                # Fallback for this chunk
                flood_union = flood_layer.unary_union
                for i in range(len(chunk)):
                    pos = start + i
                    ring_geom = rings.iloc[pos]
                    try:
                        inter = ring_geom.intersection(flood_union)
                        flooded_area[pos] = inter.area if hasattr(inter, 'area') else 0
                    except Exception:
                        pass

        valid = (ring_areas > 0) & (flooded_area > 0)
        if valid.any():
            pct = flooded_area[valid] / ring_areas[valid]
            hwb_accum[valid] += lognorm.cdf(pct * 4, shape_param)

        n_hit = (flooded_area > 0).sum()
        print(f"Depth {depth}cm: {n_hit} buildings intersected")

    buildings_result['HWB'] = hwb_accum

    n_affected = (buildings_result['HWB'] > 0).sum()

    return buildings_result