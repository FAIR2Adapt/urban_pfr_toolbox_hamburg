"""
indicators.py – Social vulnerability via TOPSIS (Steps 1-4 from ArcGIS).
  Step 1: Sensitivity    = TOPSIS(indicators, weights, entr=True)
  Step 2: CopingCapacity = TOPSIS(indicators, weights, entr=True)
  Step 3: SVI            = TOPSIS([Sen, CCap], weights, entr=False)
  Step 4: SVPF           = (SVI + threshold * mean(SVI)) ^ transform
"""

import numpy as np
import pandas as pd
import math
import warnings
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


def topsis(df, weights=None, use_entropy=True):
    """
    TOPSIS matching ArcGIS implementation, but one function diferent input and output
    """
    cols = list(df.columns)
    m, n = len(df), len(cols)

    if n == 0 or m == 0:
        return pd.Series(dtype=float), {}

    null_mask = df.isnull().any(axis=1)
    M = df.fillna(0).values.astype(float)

    if weights is not None:
        w = np.array(weights, dtype=float)
        if len(w) != n:
            raise ValueError(f"Expected {n} weights for {cols}, got {len(w)}")
        if abs(w.sum() - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {w.sum():.4f}")
    else:
        logger.warning(
            "No weights provided -- auto-computing . "
            "Results may differ from ArcGIS. Set topsis_weights in config.yaml."
        )
        w = _shannon_entropy_weights(M, n)

    weights_used = dict(zip(cols, w))

    # normalize and weight
    for i in range(n):
        temp = np.sum(np.square(M[:, i]))
        if temp == 0:
            M[:, i] = 0
        else:
            M[:, i] = M[:, i] / math.sqrt(temp) * w[i]

    Zmax = np.zeros_like(M)
    Zmin = np.zeros_like(M)
    for i in range(n):
        Zmax[:, i] = M[:, i] - np.max(M[:, i])
        Zmin[:, i] = M[:, i] - np.min(M[:, i])

    D_plus = np.array([math.sqrt(np.sum(np.square(Zmax[i, :]))) for i in range(m)])
    D_minus = np.array([math.sqrt(np.sum(np.square(Zmin[i, :]))) for i in range(m)])

    C = np.where(D_plus + D_minus != 0, D_minus / (D_plus + D_minus), 0.0)

    if use_entropy:
        Entropy = np.zeros(m)
        M_ent = M.copy()
        for i in range(m):
            if np.count_nonzero(M_ent[i, :]) == n:
                row_sum = np.sum(M_ent[i, :])
                if row_sum != 0:
                    M_ent[i, :] = M_ent[i, :] / row_sum
                x = sum(
                    M_ent[i, j] * math.log(M_ent[i, j])
                    for j in range(n) if M_ent[i, j] > 0
                )
                Entropy[i] = 2 + 1 / math.log(n) * x
            else:
                Entropy[i] = 2
        result = C * Entropy
    else:
        result = C

    scores = pd.Series(result, index=df.index)
    scores[null_mask] = np.nan
    return scores, weights_used


def _shannon_entropy_weights(M, n):
    #toDO Does NOT match ArcGIS pre-defined weights.
    weights = np.zeros(n)
    m = M.shape[0]

    for i in range(n):
        col = M[:, i].copy()
        if col.sum() == 0:
            continue
        p = col / col.sum()
        p = p + 1e-10
        p = p / p.sum()
        entropy = -np.sum(p * np.log(p)) / np.log(max(m, 2))
        weights[i] = 1 - entropy

    total = weights.sum()
    return weights / total if total > 0 else np.ones(n) / n


def flood_susceptibility_transform(svi_values, threshold=0.25, transform=2.0):
    """SVPF = (SVI + threshold * mean(SVI)) ^ transform"""
    svi = np.asarray(svi_values, dtype=float)
    valid = ~np.isnan(svi)
    if valid.sum() == 0:
        return svi

    svi_mean = np.nanmean(svi)
    svpf = np.full_like(svi, np.nan)
    svpf[valid] = (svi[valid] + threshold * svi_mean) ** transform
    return svpf


def topsis_with_shannon_entropy(df, weights=None, benefit_criteria=None,
                                 cost_criteria=None, use_entropy=True):
    return topsis(df, weights=weights, use_entropy=use_entropy)


def compute_social_vulnerability(buildings_gdf, statistical_units_gdf=None,
                                  sensitivity_fields=None, coping_fields=None,
                                  config=None):
    """
    Full SVPF pipeline (ArcGIS Steps 1-4).
    Returns (buildings_gdf, stats_gdf) or (buildings_gdf, None).
    """
    if config is None:
        config = {}

    buildings_result = buildings_gdf.copy()

    tw = config.get('topsis_weights', {}) or {}
    w_sensitivity = tw.get('sensitivity', None)
    w_coping = tw.get('coping_capacity', None)
    w_svi = tw.get('svi', None)

    svpf_threshold = config.get('svpf_threshold', 0.25)
    svpf_transform = config.get('svpf_transform', 2.0)

    if sensitivity_fields is None:
        sensitivity_fields = ['WR', 'C', 'ES', 'EDQ']
    available_sen = [c for c in sensitivity_fields if c in buildings_result.columns]

    if coping_fields is None:
        coping_fields = ['Y', 'L', 'MM']
    available_cop = [c for c in coping_fields if c in buildings_result.columns]

    # Step 1: Sensitivity
    if len(available_sen) >= 2:
        scores, _ = topsis(buildings_result[available_sen].copy(),
                           weights=w_sensitivity, use_entropy=True)
        buildings_result['Sensitivity'] = scores
    else:
        buildings_result['Sensitivity'] = 1.0

    # Step 2: Coping Capacity
    if len(available_cop) >= 2:
        scores, _ = topsis(buildings_result[available_cop].copy(),
                           weights=w_coping, use_entropy=True)
        buildings_result['CopingCapacity'] = scores
    else:
        buildings_result['CopingCapacity'] = 0.5

    # Step 3: SVI
    svi_scores, _ = topsis(
        buildings_result[['Sensitivity', 'CopingCapacity']].copy(),
        weights=w_svi, use_entropy=False
    )
    buildings_result['SVI'] = svi_scores

    # Step 4: SVPF
    buildings_result['SVPF'] = flood_susceptibility_transform(
        buildings_result['SVI'].values,
        threshold=svpf_threshold, transform=svpf_transform
    )

    print(f"SVPF: range={buildings_result['SVPF'].min():.4f}-"
          f"{buildings_result['SVPF'].max():.4f}, "
          f"mean={buildings_result['SVPF'].mean():.4f}")

    # aggregate to statistical units
    stats_result = None
    if statistical_units_gdf is not None:
        stats_result = statistical_units_gdf.copy()
        svpf_by_unit = {}
        for idx, unit in stats_result.iterrows():
            if unit.geometry is None or pd.isna(unit.geometry):
                svpf_by_unit[idx] = 0
                continue
            in_unit = buildings_result[buildings_result.geometry.intersects(unit.geometry)]
            svpf_by_unit[idx] = in_unit['SVPF'].mean() if len(in_unit) > 0 else 0
        stats_result['svpf'] = pd.Series(svpf_by_unit)

    return buildings_result, stats_result