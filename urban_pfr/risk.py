"""
risk.py – Risk calculation and spatial smoothing.
Risk = Hazard^w * Exposure^w * Vulnerability^w (IPCC framework)
Delaunay triangulation for spatial smoothing.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist
import warnings

warnings.filterwarnings('ignore')


def calculate_risk(buildings_gdf,
                   hazard_col_ma='HMA',
                   hazard_col_wb='HWB',
                   exposure_col_ma='R',
                   exposure_col_wb='R_G',
                   vulnerability_col='svpf',
                   weight_hazard=1.0,
                   weight_exposure=1.0,
                   weight_vulnerability=1.0):
    """
    PFRMA = HMA^w * EMA^w * SVPF^w
    PFRWB = HWB^w * EWB^w * SVPF^w
    Default weights=1 so it simplifies to plain multiplication.
    """
    buildings_result = buildings_gdf.copy()

    # PFRMA
    cols_ma = [hazard_col_ma, exposure_col_ma, vulnerability_col]
    missing_ma = [c for c in cols_ma if c not in buildings_result.columns]

    if not missing_ma:
        buildings_result['PFRMA'] = (
            buildings_result[hazard_col_ma].fillna(0) ** weight_hazard *
            buildings_result[exposure_col_ma].fillna(0) ** weight_exposure *
            buildings_result[vulnerability_col].fillna(0) ** weight_vulnerability
        )
    else:
        buildings_result['PFRMA'] = 0

    # PFRWB
    cols_wb = [hazard_col_wb, exposure_col_wb, vulnerability_col]
    missing_wb = [c for c in cols_wb if c not in buildings_result.columns]

    if not missing_wb:
        buildings_result['PFRWB'] = (
            buildings_result[hazard_col_wb].fillna(0) ** weight_hazard *
            buildings_result[exposure_col_wb].fillna(0) ** weight_exposure *
            buildings_result[vulnerability_col].fillna(0) ** weight_vulnerability
        )
    else:
        buildings_result['PFRWB'] = 0

    nma = (buildings_result['PFRMA'] > 0).sum()
    nwb = (buildings_result['PFRWB'] > 0).sum()
    print(f"Risk: PFRMA>0 in {nma}/{len(buildings_result)}, "
          f"PFRWB>0 in {nwb}/{len(buildings_result)}")
    return buildings_result


def delaunay_smoothing(buildings_gdf, risk_col='PFRMA',
                       n_iterations=3,
                       max_neighbors=10,
                       distance_threshold=100):
    """
    Smooth risk values via Delaunay triangulation neighbor averaging.
    Each iteration: blend 80% neighbor-weighted-avg + 20% original.
    Weights = 1 / (distance + 1)^2.
    """
    if risk_col not in buildings_gdf.columns:
        return pd.Series(0, index=buildings_gdf.index)

    centroids = buildings_gdf.geometry.centroid
    coords = np.array([(p.x, p.y) for p in centroids])
    values = buildings_gdf[risk_col].values.copy()

    if len(coords) < 4:
        return pd.Series(values, index=buildings_gdf.index)

    try:
        tri = Delaunay(coords)

        # build neighbor graph from triangulation
        neighbors = [set() for _ in range(len(coords))]
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(3):
                    if i != j:
                        neighbors[simplex[i]].add(simplex[j])

        for iteration in range(n_iterations):
            new_values = values.copy()

            for i in range(len(coords)):
                if not neighbors[i]:
                    continue

                nb_idx = list(neighbors[i])
                if len(nb_idx) > max_neighbors:
                    dists = cdist([coords[i]], coords[nb_idx])[0]
                    closest = np.argsort(dists)[:max_neighbors]
                    nb_idx = [nb_idx[k] for k in closest]

                dists = cdist([coords[i]], coords[nb_idx])[0]
                mask = dists < distance_threshold
                if not np.any(mask):
                    continue

                valid_nb = [nb_idx[j] for j in range(len(nb_idx)) if mask[j]]
                valid_d = dists[mask]

                w = 1.0 / (valid_d + 1) ** 2
                w = w / w.sum()

                smoothed = np.sum(w * values[valid_nb])
                new_values[i] = 0.8 * smoothed + 0.2 * values[i]

            values = new_values

        return pd.Series(values, index=buildings_gdf.index)

    except Exception:
        return pd.Series(buildings_gdf[risk_col], index=buildings_gdf.index)