"""
Exposure calculation for pluvial flood risk.
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
                                 stat_unit_col='StatisticalUnit',
                                 residential_types=None,
                                 non_residential_floor_deduction=1):
    """
    Distribute statistical-unit population to individual buildings
    proportionally by residential floor area.

    Parameters
    ----------
    residential_types : list, optional
        Building type values considered residential (default: [1]).
        Residential buildings use all floors; non-residential buildings
        have `non_residential_floor_deduction` floors deducted.
    non_residential_floor_deduction : int
        Number of floors deducted for non-residential buildings (default: 1).
    """
    buildings_result = buildings_gdf.copy()
    if residential_types is None:
        residential_types = [1]

    required_cols = [floors_col, building_type_col]
    missing = [c for c in required_cols if c not in buildings_result.columns]
    if missing:
        raise ValueError(f"Missing columns in buildings: {missing}")

    # Residential floor area: residential types use all floors,
    # non-residential types lose `non_residential_floor_deduction` floors
    is_residential = buildings_result[building_type_col].isin(residential_types)
    deduction = np.where(is_residential, 0, non_residential_floor_deduction)
    buildings_result['Area_house'] = (
        (buildings_result[floors_col] - deduction) * buildings_result.geometry.area
    )
    buildings_result.loc[buildings_result['Area_house'] < 0, 'Area_house'] = 0

    if stat_unit_col in buildings_result.columns:
        # ── Vectorized path: merge + array arithmetic ──
        unit_lookup = statistical_units_gdf[[stat_unit_col, residents_col, living_area_col]].copy()
        unit_lookup = unit_lookup.rename(columns={
            residents_col: '_unit_residents',
            living_area_col: '_unit_living_area',
        })

        merged = buildings_result.merge(
            unit_lookup, on=stat_unit_col, how='left'
        )

        mask = (
            merged['_unit_living_area'].notna()
            & (merged['_unit_living_area'] > 0)
            & (merged['Area_house'] > 0)
        )

        buildings_result['R'] = 0.0
        buildings_result.loc[mask, 'R'] = (
            merged.loc[mask, 'Area_house']
            / merged.loc[mask, '_unit_living_area']
            * merged.loc[mask, '_unit_residents']
        )

    else:
        # Fallback: spatial join via centroids (vectorized sjoin, not iterrows)
        centroids = buildings_result.copy()
        centroids.geometry = centroids.geometry.centroid

        joined = gpd.sjoin(
            centroids,
            statistical_units_gdf[[stat_unit_col, residents_col, living_area_col, 'geometry']],
            how='left', predicate='within'
        )

        # Handle potential duplicate matches by keeping the first
        joined = joined[~joined.index.duplicated(keep='first')]

        la = joined[living_area_col].fillna(0).values
        res = joined[residents_col].fillna(0).values
        area = buildings_result.loc[joined.index, 'Area_house'].values

        r_vals = np.where((la > 0) & (area > 0), (area / la) * res, 0.0)

        buildings_result['R'] = 0.0
        buildings_result.loc[joined.index, 'R'] = r_vals

    print(f"Exposure EMA: {buildings_result['R'].sum():.0f} residents across "
          f"{(buildings_result['R'] > 0).sum()}/{len(buildings_result)} buildings")
    return buildings_result


def calculate_exposure_wellbeing(buildings_gdf,
                                 floors_col='Floors',
                                 building_type_col='Building_type',
                                 residents_col='R',
                                 residential_types=None):
    """
    Ground floor residents (EWB).

    Parameters
    ----------
    residential_types : list, optional
        Building type values considered residential (default: [1]).
        Only residential buildings get ground-floor residents.
    """
    buildings_result = buildings_gdf.copy()
    if residential_types is None:
        residential_types = [1]

    required_cols = [floors_col, building_type_col, residents_col]
    missing = [c for c in required_cols if c not in buildings_result.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Only residential buildings get ground-floor residents
    mask = (buildings_result[building_type_col].isin(residential_types)) & (buildings_result[floors_col] > 0)

    buildings_result['R_G'] = 0.0
    buildings_result.loc[mask, 'R_G'] = (
        buildings_result.loc[mask, residents_col]
        / buildings_result.loc[mask, floors_col]
    ).clip(lower=0)

    print(f"Exposure EWB: {buildings_result['R_G'].sum():.0f} ground-floor residents across "
          f"{(buildings_result['R_G'] > 0).sum()}/{len(buildings_result)} buildings")
    return buildings_result