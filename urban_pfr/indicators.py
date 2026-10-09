"""
Social vulnerability
"""

import numpy as np
import pandas as pd
import geopandas as gpd
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

    # Entropy flags — ArcGIS uses entropy=true for all three steps
    entropy_cfg = config.get('topsis_entropy', {}) or {}
    entropy_sensitivity = entropy_cfg.get('sensitivity', True)
    entropy_coping = entropy_cfg.get('coping_capacity', True)
    entropy_svi = entropy_cfg.get('svi', True)

    svpf_threshold = config.get('svpf_threshold', 0.25)
    svpf_transform = config.get('svpf_transform', 2.0)

    if sensitivity_fields is None:
        sensitivity_fields = ['WR', 'C', 'ES', 'EDQ']
    available_sen = [c for c in sensitivity_fields if c in buildings_result.columns]

    if coping_fields is None:
        coping_fields = ['Y', 'L', 'MM']
    available_cop = [c for c in coping_fields if c in buildings_result.columns]

    # Paper-replication path (Section 4.5 / 4.2): TOPSIS at stat-unit level on
    # demographic percentages (count / Residents), then propagate to buildings.
    # Default 'building' preserves the existing per-building behaviour.
    topsis_granularity = config.get('topsis_granularity', 'building')
    schema = config.get('schema', {}) or {}
    stat_unit_col = schema.get('stat_unit_col', 'StatisticalUnit')
    residents_col = schema.get('residents_col', 'Residents')

    if topsis_granularity == 'stat_unit':
        if stat_unit_col not in buildings_result.columns:
            raise ValueError(
                f"topsis_granularity='stat_unit' requires column '{stat_unit_col}' in buildings"
            )
        if residents_col not in buildings_result.columns:
            raise ValueError(
                f"topsis_granularity='stat_unit' requires column '{residents_col}' in buildings"
            )

        agg_cols = list(dict.fromkeys(available_sen + available_cop + [residents_col]))
        su_table = buildings_result.groupby(stat_unit_col)[agg_cols].first().reset_index()
        residents = su_table[residents_col].replace(0, np.nan)

        pct_sen, pct_cop = [], []
        for c in available_sen:
            pc = f"{c}_pct"
            su_table[pc] = su_table[c] / residents
            pct_sen.append(pc)
        for c in available_cop:
            pc = f"{c}_pct"
            su_table[pc] = su_table[c] / residents
            pct_cop.append(pc)

        # Step 1 at stat-unit level
        if len(pct_sen) >= 2:
            sen_su, _ = topsis(su_table[pct_sen].fillna(0).copy(),
                               weights=w_sensitivity, use_entropy=entropy_sensitivity)
        else:
            sen_su = pd.Series(1.0, index=su_table.index)
        su_table['Sensitivity'] = sen_su

        # Step 2 at stat-unit level
        if len(pct_cop) >= 2:
            cop_su, _ = topsis(su_table[pct_cop].fillna(0).copy(),
                               weights=w_coping, use_entropy=entropy_coping)
        else:
            cop_su = pd.Series(0.5, index=su_table.index)
        su_table['CopingCapacity'] = cop_su

        # Step 3: SVI at stat-unit level
        svi_su, _ = topsis(
            su_table[['CopingCapacity', 'Sensitivity']].copy(),
            weights=w_svi, use_entropy=entropy_svi,
        )
        su_table['SVI'] = svi_su
        su_table['SVPF'] = flood_susceptibility_transform(
            su_table['SVI'].values,
            threshold=svpf_threshold, transform=svpf_transform,
        )

        # Propagate stat-unit values to buildings
        for col in ('Sensitivity', 'CopingCapacity', 'SVI', 'SVPF'):
            mapping = dict(zip(su_table[stat_unit_col], su_table[col]))
            buildings_result[col] = buildings_result[stat_unit_col].map(mapping)

        print(
            f"Sensitivity (stat-unit, {len(su_table)} units): "
            f"median={su_table['Sensitivity'].median():.4f}"
        )
    else:
        # Step 1: Sensitivity (per-building)
        if len(available_sen) >= 2:
            scores, _ = topsis(buildings_result[available_sen].copy(),
                               weights=w_sensitivity, use_entropy=entropy_sensitivity)
            buildings_result['Sensitivity'] = scores
        else:
            buildings_result['Sensitivity'] = 1.0

        # Step 2: Coping Capacity (per-building)
        if len(available_cop) >= 2:
            scores, _ = topsis(buildings_result[available_cop].copy(),
                               weights=w_coping, use_entropy=entropy_coping)
            buildings_result['CopingCapacity'] = scores
        else:
            buildings_result['CopingCapacity'] = 0.5

        # Step 3: SVI — ArcGIS uses [CopingCapacity, Sensitivity] order
        svi_scores, _ = topsis(
            buildings_result[['CopingCapacity', 'Sensitivity']].copy(),
            weights=w_svi, use_entropy=entropy_svi
        )
        buildings_result['SVI'] = svi_scores

        # Step 4: SVPF
        buildings_result['SVPF'] = flood_susceptibility_transform(
            buildings_result['SVI'].values,
            threshold=svpf_threshold, transform=svpf_transform
        )

    # ── Aggregate to statistical units via sjoin + groupby ──
    stats_result = None
    if statistical_units_gdf is not None:
        stats_result = statistical_units_gdf.copy()

        # Aggregate all vulnerability indicators to statistical unit level
        agg_cols = ['Sensitivity', 'CopingCapacity', 'SVI', 'SVPF']
        available_agg = [c for c in agg_cols if c in buildings_result.columns]

        bld_for_join = buildings_result[['geometry'] + available_agg].copy()
        bld_for_join = bld_for_join.dropna(subset=available_agg, how='all')

        if len(bld_for_join) > 0 and len(stats_result) > 0:
            try:
                joined = gpd.sjoin(
                    bld_for_join,
                    stats_result[['geometry']],
                    how='inner',
                    predicate='intersects',
                )
                for col in available_agg:
                    agg = joined.groupby('index_right')[col].mean()
                    stats_result[col] = stats_result.index.map(agg).fillna(0)
            except Exception as e:
                logger.warning(f"sjoin aggregation failed, using fallback: {e}")
                for idx, unit in stats_result.iterrows():
                    if unit.geometry is None or pd.isna(unit.geometry):
                        for col in available_agg:
                            stats_result.at[idx, col] = 0
                        continue
                    in_unit = buildings_result[
                        buildings_result.geometry.intersects(unit.geometry)
                    ]
                    for col in available_agg:
                        stats_result.at[idx, col] = in_unit[col].mean() if len(in_unit) > 0 else 0
        else:
            for col in available_agg:
                stats_result[col] = 0

    return buildings_result, stats_result