"""
exposure.py – Exposure calculation for pluvial flood risk.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import warnings

warnings.filterwarnings('ignore')


def calculate_exposure_residents(buildings_gdf, statistical_units_gdf,
                                 residents_col='Residents',
                                 living_area_col='LivingArea',
                                 floors_col='Floors',
                                 building_type_col='Building_type',
                                 stat_unit_col='StatisticalUnit'):
    """
    Distribute statistical-unit population to individual buildings
    proportionally by residential floor area.

    R = (Area_house / LivingArea_unit) * Residents_unit

    Area_house = (Floors - (BuildingType - 1)) * FootprintArea
      Type 1 (residential): all floors count
      Type 2 (mixed): ground floor excluded
      Type 3+: no residential area
    """
    buildings_result = buildings_gdf.copy()

    required_cols = [floors_col, building_type_col]
    missing = [c for c in required_cols if c not in buildings_result.columns]
    if missing:
        raise ValueError(f"Missing columns in buildings: {missing}")

    # residential floor area per building
    buildings_result['Area_house'] = (
        (buildings_result[floors_col] - (buildings_result[building_type_col] - 1))
        * buildings_result.geometry.area
    )
    buildings_result.loc[buildings_result['Area_house'] < 0, 'Area_house'] = 0

    if stat_unit_col in buildings_result.columns:
        # fast path: ID-based join
        unit_data = {}
        for idx, unit in statistical_units_gdf.iterrows():
            uid = unit.get(stat_unit_col, idx)
            unit_data[uid] = {
                'residents': unit.get(residents_col, 0),
                'living_area': unit.get(living_area_col, 1)
            }

        residents_list = []
        for idx, bld in buildings_result.iterrows():
            uid = bld.get(stat_unit_col)
            if pd.notna(uid) and uid in unit_data:
                info = unit_data[uid]
                if info['living_area'] > 0 and bld['Area_house'] > 0:
                    residents_list.append(
                        bld['Area_house'] / info['living_area'] * info['residents']
                    )
                else:
                    residents_list.append(0)
            else:
                residents_list.append(0)

        buildings_result['R'] = residents_list

    else:
        # fallback: spatial join via centroids
        centroids = buildings_result.copy()
        centroids.geometry = centroids.geometry.centroid

        joined = gpd.sjoin(
            centroids,
            statistical_units_gdf[[stat_unit_col, residents_col, living_area_col, 'geometry']],
            how='left', predicate='within'
        )

        residents_list = []
        for idx in buildings_result.index:
            if idx in joined.index:
                row = joined.loc[idx]
                area = buildings_result.loc[idx, 'Area_house']
                la = row.get(living_area_col, 0)
                res = row.get(residents_col, 0)
                residents_list.append((area / la) * res if la > 0 and area > 0 else 0)
            else:
                residents_list.append(0)

        buildings_result['R'] = residents_list

    print(f"Exposure EMA: {buildings_result['R'].sum():.0f} residents across "
          f"{(buildings_result['R'] > 0).sum()}/{len(buildings_result)} buildings")
    return buildings_result


def calculate_exposure_wellbeing(buildings_gdf,
                                 floors_col='Floors',
                                 building_type_col='Building_type',
                                 residents_col='R'):
    """
    Ground floor residents (EWB).
      Type 1: R_G = R / Floors
      Type 2+: R_G = 0 (ground floor is commercial or non-residential)
    """
    buildings_result = buildings_gdf.copy()

    required_cols = [floors_col, building_type_col, residents_col]
    missing = [c for c in required_cols if c not in buildings_result.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    r_g = []
    for idx, bld in buildings_result.iterrows():
        if bld[building_type_col] == 1 and bld[floors_col] > 0:
            r_g.append(bld[residents_col] / bld[floors_col])
        else:
            r_g.append(0)

    buildings_result['R_G'] = [max(0, v) for v in r_g]

    print(f"Exposure EWB: {buildings_result['R_G'].sum():.0f} ground-floor residents across "
          f"{(buildings_result['R_G'] > 0).sum()}/{len(buildings_result)} buildings")
    return buildings_result