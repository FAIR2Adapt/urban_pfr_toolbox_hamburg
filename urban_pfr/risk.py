"""
Risk calculation (PFRMA, PFRWB) and Delaunay smoothing.
"""

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from scipy import sparse


def calculate_risk(buildings_gdf,
                   hazard_col_ma='HMA',
                   hazard_col_wb='HWB',
                   exposure_col_ma='R',
                   exposure_col_wb='R_G',
                   vulnerability_col='SVI',
                   weight_hazard=1.0,
                   weight_exposure=1.0,
                   weight_vulnerability=1.0,
                   normalize_exposure=False,
                   config=None):
    """
    Calculate PFRMA and PFRWB risk indices.

    Parameters
    ----------
    buildings_gdf : GeoDataFrame
        Buildings with vulnerability, exposure, and hazard columns.
    hazard_col_ma : str
        Column name for mobility/accessibility hazard (HMA).
    hazard_col_wb : str
        Column name for well-being hazard (HWB).
    exposure_col_ma : str
        Column name for MA exposure (R = total residents).
    exposure_col_wb : str
        Column name for WB exposure (R_G = ground-floor residents).
    vulnerability_col : str
        Column name for social vulnerability (SVI).
    weight_hazard, weight_exposure, weight_vulnerability : float
        IPCC risk component weights (default: 1.0 each).
    normalize_exposure : bool
        If True, normalize exposure to [0,1] before calculation.
        Default False — matches ArcGIS behavior (raw values).
        Set via config['risk_settings']['normalize_exposure'].
    config : dict, optional
        Full config dict. If provided, reads weights and normalize_exposure.

    Returns
    -------
    GeoDataFrame
        Buildings with PFRMA and PFRWB columns added.
    """
    buildings_result = buildings_gdf.copy()

    # --- Read settings from config if provided ---
    if config:
        risk_cfg = config.get('risk_settings', {})
        weight_hazard = risk_cfg.get('weight_hazard', weight_hazard)
        weight_exposure = risk_cfg.get('weight_exposure', weight_exposure)
        weight_vulnerability = risk_cfg.get('weight_vulnerability', weight_vulnerability)
        normalize_exposure = risk_cfg.get('normalize_exposure', normalize_exposure)

    w_h = weight_hazard
    w_e = weight_exposure
    w_v = weight_vulnerability

    print(f"  Weights: vulnerability={w_v}, exposure={w_e}, hazard={w_h}")
    print(f"  Normalize exposure: {normalize_exposure}")

    # --- Extract columns ---
    vuln = buildings_result[vulnerability_col].fillna(0).astype(float)
    haz_ma = buildings_result[hazard_col_ma].fillna(0).astype(float)
    haz_wb = buildings_result[hazard_col_wb].fillna(0).astype(float)
    exp_ma = buildings_result[exposure_col_ma].fillna(0).astype(float)
    exp_wb = buildings_result[exposure_col_wb].fillna(0).astype(float)

    # --- Exposure normalization (ONLY if explicitly requested) ---
    if normalize_exposure:
        print("  ⚠️  Normalizing exposure to [0,1] — values will be compressed!")
        max_ma = exp_ma.max()
        max_wb = exp_wb.max()
        if max_ma > 0:
            exp_ma = exp_ma / max_ma
        if max_wb > 0:
            exp_wb = exp_wb / max_wb
    else:
        print("  ✅ Using raw exposure values (matching ArcGIS)")

    # --- Check for missing columns ---
    required_ma = [vulnerability_col, exposure_col_ma, hazard_col_ma]
    required_wb = [vulnerability_col, exposure_col_wb, hazard_col_wb]

    missing_ma = [c for c in required_ma if c not in buildings_result.columns]
    missing_wb = [c for c in required_wb if c not in buildings_result.columns]

    # --- Calculate PFRMA ---
    if missing_ma:
        print(f"  ⚠️  Cannot calculate PFRMA — missing columns: {missing_ma}")
        buildings_result['PFRMA'] = 0
    else:
        buildings_result['PFRMA'] = (
            (vuln ** w_v) * (exp_ma ** w_e) * (haz_ma ** w_h)
        )
        pfrma_nonzero = (buildings_result['PFRMA'] > 0).sum()
        pfrma_max = buildings_result['PFRMA'].max()
        print(f"  ✅ PFRMA: {pfrma_nonzero}/{len(buildings_result)} buildings > 0, "
              f"max={pfrma_max:.4f}")

    # --- Calculate PFRWB ---
    if missing_wb:
        print(f"  ⚠️  Cannot calculate PFRWB — missing columns: {missing_wb}")
        buildings_result['PFRWB'] = 0
    else:
        buildings_result['PFRWB'] = (
            (vuln ** w_v) * (exp_wb ** w_e) * (haz_wb ** w_h)
        )
        pfrwb_nonzero = (buildings_result['PFRWB'] > 0).sum()
        pfrwb_max = buildings_result['PFRWB'].max()
        print(f"  ✅ PFRWB: {pfrwb_nonzero}/{len(buildings_result)} buildings > 0, "
              f"max={pfrwb_max:.4f}")

    # --- Scale verification ---
    if not normalize_exposure:
        pfrwb_max_val = buildings_result['PFRWB'].max()
        if pfrwb_max_val < 1.0 and pfrwb_max_val > 0:
            print(f"\n  ⚠️  WARNING: PFRWB max is {pfrwb_max_val:.4f} (< 1.0)")
            print(f"      This suggests exposure may still be normalized somewhere.")
            print(f"      ArcGIS Hamburg values should reach ~10–28.")
            print(f"      Check that R_G contains raw resident counts, not fractions.")
        elif pfrwb_max_val > 1.0:
            print(f"\n  ✅ PFRWB scale looks correct (max={pfrwb_max_val:.4f}, "
                  f"in ArcGIS range)")

    return buildings_result


def delaunay_smoothing(buildings_gdf, risk_col, n_iterations=3,
                       distance_threshold=100, max_neighbors=10,
                       blend_weight=0.8):
    """
    Smooth risk values using Delaunay triangulation and inverse-distance weighting.

    This implements the Thiessen polygon smoothing from the paper.

    Parameters
    ----------
    buildings_gdf : GeoDataFrame
        Buildings with risk column.
    risk_col : str
        Column to smooth (e.g. 'PFRMA' or 'PFRWB').
    n_iterations : int
        Number of smoothing passes.
    distance_threshold : float
        Maximum distance (meters) to consider neighbors.
    max_neighbors : int
        Maximum number of neighbors per building.
    blend_weight : float
        Weight for smoothed value (1 - blend_weight for original).
        Default 0.8 = 80% smoothed + 20% original.

    Returns
    -------
    np.ndarray
        Smoothed risk values.
    """
    values = buildings_gdf[risk_col].fillna(0).values.copy().astype(float)

    # Get centroids
    centroids = buildings_gdf.geometry.centroid
    coords = np.array([[p.x, p.y] for p in centroids])

    if len(coords) < 4:
        print(f"  ⚠️  Too few buildings ({len(coords)}) for Delaunay — skipping smoothing")
        return values

    # Build Delaunay triangulation
    try:
        tri = Delaunay(coords)
    except Exception as e:
        print(f"  ⚠️  Delaunay failed: {e} — skipping smoothing")
        return values

    # Build adjacency from triangulation using sparse matrix (vectorized)
    n = len(values)
    rows, cols = [], []
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(3):
                if i != j:
                    rows.append(simplex[i])
                    cols.append(simplex[j])

    rows = np.array(rows)
    cols = np.array(cols)

    # Compute distances and filter by threshold
    dists = np.sqrt(
        (coords[rows, 0] - coords[cols, 0]) ** 2 +
        (coords[rows, 1] - coords[cols, 1]) ** 2
    )
    mask = dists < distance_threshold
    rows, cols, dists = rows[mask], cols[mask], dists[mask]

    # Weights = 1 / (distance + epsilon)
    weights = 1.0 / (dists + 1e-10)

    # Build sparse weight matrix and row-normalize
    W = sparse.coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()
    row_sums = np.array(W.sum(axis=1)).flatten()
    has_neighbors = row_sums > 0  # track before modifying for normalization
    row_sums_safe = row_sums.copy()
    row_sums_safe[row_sums_safe == 0] = 1
    D_inv = sparse.diags(1.0 / row_sums_safe)
    W_norm = D_inv @ W

    # Iterative smoothing: blend_weight * neighbor_avg + (1 - blend_weight) * original
    # Only blend buildings that have neighbors; isolated buildings keep their value.
    for iteration in range(n_iterations):
        smoothed = W_norm @ values
        new_values = values.copy()
        new_values[has_neighbors] = (
            blend_weight * smoothed[has_neighbors] +
            (1 - blend_weight) * values[has_neighbors]
        )
        values = new_values

    return values