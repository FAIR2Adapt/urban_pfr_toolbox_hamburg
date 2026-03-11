"""
urban_pfr/viz.py
=================
Visualization for pluvial flood risk assessment.

Supports two classification modes (set in config):
  - "fixed":     Use exact ArcGIS .lyrx thresholds (for paper replication)
  - "head_tail": Compute breaks dynamically (for new cities / different data)

The mode is set in hamburg_config.yaml → visualization.classification_mode
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch, Rectangle
import warnings

warnings.filterwarnings('ignore')


# =====================================================================
#  DEFAULT COLORS — used when classification_mode = "head_tail"
# =====================================================================

# 5 classes: [no risk / very low, low, medium, high, very high]
DEFAULT_PFRMA_COLORS = [
    '#FFFFFF',   # no risk:   White
    '#FFF2CC',   # low:       Light yellow      (from ArcGIS .lyrx)
    '#FFD966',   # medium:    Yellow
    '#FF9900',   # high:      Orange
    '#CC7A00',   # very high: Dark orange/amber
]

DEFAULT_PFRWB_COLORS = [
    '#FFFFFF',   # no risk:   White
    '#C18EAE',   # low:       Light purple  (from ArcGIS .lyrx)
    '#A8608D',   # medium:    Medium purple
    '#841D5D',   # high:      Dark purple
    '#5C1441',   # very high: Deep wine/maroon
]

DEFAULT_SVPF_COLORS = [
    '#FFFFFF',   # very low:  White
    '#D0D0D0',   # low:       Light grey
    '#909090',   # medium:    Medium grey
    '#505050',   # high:      Dark grey
    '#202020',   # very high: Very dark grey
]

DEFAULT_LABELS = ['No risk', 'Low', 'Medium', 'High', 'Very high']


# =====================================================================
#  HEAD/TAIL BREAKS — dynamic classification
# =====================================================================

def head_tail_breaks(values, n_iterations=3):
    """
    Head/tail breaks classification.

    Matches Step6_Calculating_classes_for_visualization.py from ArcGIS:
    iteratively compute mean, keep values > mean, repeat.

    Parameters
    ----------
    values : array-like
        Non-zero, non-NaN risk values.
    n_iterations : int
        Number of iterations (3 → 4 breaks → 5 classes including zero).

    Returns
    -------
    breaks : list of float
        Break points (length = n_iterations or fewer).
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]

    if len(values) == 0:
        return []

    breaks = []
    current = values.copy()

    for _ in range(n_iterations):
        if len(current) < 2:
            break
        mean_val = np.mean(current)
        breaks.append(mean_val)
        current = current[current > mean_val]

    return sorted(breaks)


# =====================================================================
#  CLASSIFICATION — unified interface for both modes
# =====================================================================

def classify_values(values, risk_type='PFRWB', config=None):
    """
    Classify risk values into discrete classes.

    Reads classification_mode from config to choose between fixed
    ArcGIS thresholds or dynamic head/tail breaks.

    Parameters
    ----------
    values : array-like
        Raw risk values (NOT normalized to 0–1 if using fixed mode).
    risk_type : str
        'PFRMA', 'PFRWB', or 'SVPF'. Used to look up fixed thresholds.
    config : dict, optional
        Full config dict with visualization settings.

    Returns
    -------
    classes : np.ndarray of int
        Class index for each value (0 = no risk, 1–4 = low–very high).
    bounds : list of float
        The boundary values used (including 0 and max).
    colors : list of str
        HEX color for each class.
    labels : list of str
        Label for each class.
    """
    values = np.asarray(values, dtype=float)
    viz_cfg = config.get('visualization', {}) if config else {}
    mode = viz_cfg.get('classification_mode', 'head_tail')

    # ----- FIXED MODE: use ArcGIS .lyrx thresholds -----
    if mode == 'fixed':
        fixed = viz_cfg.get('fixed_thresholds', {}).get(risk_type)
        if fixed:
            bounds = fixed['bounds']
            colors = fixed['colors']
            labels = fixed['labels']

            classes = np.digitize(values, bounds[1:])
            classes = np.clip(classes, 0, len(colors) - 1)

            print(f"    Mode: FIXED (ArcGIS .lyrx thresholds)")
            print(f"    Bounds: {bounds}")
            for i, lbl in enumerate(labels):
                count = (classes == i).sum()
                print(f"      {lbl}: {count} buildings")

            return classes, bounds, colors, labels
        else:
            print(f"    ⚠️  No fixed thresholds for '{risk_type}' — "
                  f"falling back to head_tail")

    # ----- QUANTILE MODE: equal-count bins (best for small datasets) -----
    if mode == 'quantile':
        nonzero = values[values > 0]
        if len(nonzero) >= 4:
            q25, q50, q75 = np.percentile(nonzero, [25, 50, 75])
            breaks = [q25, q50, q75]
        elif len(nonzero) > 0:
            breaks = [np.mean(nonzero)]
        else:
            breaks = []

        max_val = np.nanmax(values) if len(values) > 0 else 1.0
        bounds = [0, 0.0001] + breaks + [max_val * 1.01]
        while len(bounds) < 6:
            bounds.insert(-1, bounds[-1])
        bounds = bounds[:6]

        color_cfg = viz_cfg.get('colors', {})
        if risk_type == 'PFRMA':
            colors = color_cfg.get('pfrma', DEFAULT_PFRMA_COLORS)
        elif risk_type == 'PFRWB':
            colors = color_cfg.get('pfrwb', DEFAULT_PFRWB_COLORS)
        elif risk_type == 'SVPF':
            colors = color_cfg.get('svpf', DEFAULT_SVPF_COLORS)
        else:
            colors = DEFAULT_PFRWB_COLORS

        labels = DEFAULT_LABELS[:len(colors)]
        classes = np.digitize(values, bounds[1:-1])
        classes = np.clip(classes, 0, len(colors) - 1)

        print(f"    Mode: QUANTILE (percentile-based)")
        print(f"    Bounds: {[f'{b:.4f}' for b in bounds]}")
        for i, lbl in enumerate(labels):
            count = (classes == i).sum()
            print(f"      {lbl}: {count} buildings")

        return classes, bounds, colors, labels

    # ----- DYNAMIC MODE: head/tail breaks -----
    n_iter = viz_cfg.get('n_iterations', 3)
    breaks = head_tail_breaks(values, n_iterations=n_iter)

    # Build bounds: [0, break1, break2, ..., max]
    max_val = np.nanmax(values) if len(values) > 0 else 1.0
    bounds = [0] + breaks + [max_val * 1.01]

    # Ensure we have exactly 5 classes (pad or trim breaks)
    while len(bounds) < 6:
        bounds.insert(-1, bounds[-1])
    bounds = bounds[:6]

    # Get colors from config or defaults
    color_cfg = viz_cfg.get('colors', {})
    if risk_type == 'PFRMA':
        colors = color_cfg.get('pfrma', DEFAULT_PFRMA_COLORS)
    elif risk_type == 'PFRWB':
        colors = color_cfg.get('pfrwb', DEFAULT_PFRWB_COLORS)
    elif risk_type == 'SVPF':
        colors = color_cfg.get('svpf', DEFAULT_SVPF_COLORS)
    else:
        colors = DEFAULT_PFRWB_COLORS

    labels = DEFAULT_LABELS[:len(colors)]

    # Classify
    classes = np.digitize(values, bounds[1:-1])
    classes = np.clip(classes, 0, len(colors) - 1)

    print(f"    Mode: HEAD_TAIL (n_iterations={n_iter})")
    print(f"    Breaks: {[f'{b:.4f}' for b in breaks]}")
    for i, lbl in enumerate(labels):
        count = (classes == i).sum()
        print(f"      {lbl}: {count} buildings")

    return classes, bounds, colors, labels


def get_cmap_and_norm(bounds, colors_hex):
    """
    Build matplotlib ListedColormap + BoundaryNorm.

    Parameters
    ----------
    bounds : list of float
        Class boundaries, e.g. [0, 0.0001, 2.876, 10.0295, 15.5199, 28.4086].
    colors_hex : list of str
        HEX color per class.

    Returns
    -------
    cmap : ListedColormap
    norm : BoundaryNorm
    """
    cmap = mcolors.ListedColormap(colors_hex)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


# =====================================================================
#  SINGLE-PANEL PLOT FUNCTIONS
# =====================================================================

def plot_risk_panel(buildings_gdf, risk_column, risk_type,
                    ax, config=None, title=None,
                    statistical_units_gdf=None,
                    edgecolor='black', linewidth=0.5,
                    show_basemap=False):
    """
    Plot a single risk panel (PFRMA, PFRWB, or SVPF).

    Parameters
    ----------
    buildings_gdf : GeoDataFrame
        Buildings with the risk column.
    risk_column : str
        Column name to plot (e.g. 'PFRWB', 'PFRMA_smoothed').
    risk_type : str
        'PFRMA', 'PFRWB', or 'SVPF' — determines thresholds and colors.
    ax : matplotlib Axes
        Axes to draw on.
    config : dict, optional
        Config with visualization settings.
    title : str, optional
        Panel title.
    statistical_units_gdf : GeoDataFrame, optional
        If provided, draws unit boundaries as background.
    """
    values = buildings_gdf[risk_column].fillna(0).values

    # Classify
    print(f"\n  Classifying {risk_column} as {risk_type}:")
    classes, bounds, colors, labels = classify_values(
        values, risk_type=risk_type, config=config
    )

    # Background: statistical units
    if statistical_units_gdf is not None:
        statistical_units_gdf.plot(
            ax=ax,
            facecolor='#f5f5f5',
            edgecolor='#cccccc',
            linewidth=0.3,
        )

    # Build colormap
    cmap, norm = get_cmap_and_norm(bounds, colors)

    # Plot buildings
    buildings_gdf.plot(
        column=risk_column,
        cmap=cmap,
        norm=norm,
        ax=ax,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )

    # Optional basemap
    if show_basemap:
        try:
            import contextily as ctx
            ctx.add_basemap(
                ax,
                crs=buildings_gdf.crs,
                source=ctx.providers.CartoDB.Positron,
                alpha=0.3,
            )
        except Exception:
            pass

    # Legend
    legend_elements = [
        Patch(facecolor=c, edgecolor='black', linewidth=0.5, label=lbl)
        for c, lbl in zip(colors, labels)
    ]

    # Subscript label
    if risk_type == 'PFRWB':
        legend_title = 'PFR$_{WB}$'
    elif risk_type == 'PFRMA':
        legend_title = 'PFR$_{MA}$'
    else:
        legend_title = risk_type

    ax.legend(
        handles=legend_elements,
        title=legend_title,
        loc='lower right',
        fontsize=8,
        title_fontsize=9,
        framealpha=0.9,
    )

    if title:
        ax.set_title(title, fontsize=13, fontweight='bold')
    ax.axis('off')

    # Attribution
    ax.text(0.01, 0.02, '(C) OpenStreetMap contributors\n(C) CARTO',
            transform=ax.transAxes, fontsize=6, color='grey',
            verticalalignment='bottom')

    return classes


# =====================================================================
#  THREE-PANEL VISUALIZATION
# =====================================================================

def create_risk_visualization(buildings_gdf, statistical_units_gdf,
                              pfrma_column='PFRMA_smoothed',
                              pfrwb_column='PFRWB_smoothed',
                              svpf_column='svpf',
                              sensitivity_column='Sensitivity',
                              coping_column='CopingCapacity',
                              n_iterations=3,
                              n_classes=None,
                              figsize=(24, 8),
                              title_prefix="",
                              save_path=None,
                              dpi=300,
                              config=None):
    """
    Create 3-panel visualization:

    Panel A: PFRMA risk (yellow/orange classes)
    Panel B: PFRWB risk (purple/wine classes)
    Panel C: Combined bivariate overlay

    Supports both fixed (ArcGIS) and dynamic (head/tail) classification
    via config['visualization']['classification_mode'].

    Parameters
    ----------
    buildings_gdf : GeoDataFrame
        Buildings with risk columns.
    statistical_units_gdf : GeoDataFrame
        Statistical units for background.
    pfrma_column : str
        PFRMA column to plot. Use 'PFRMA' for unsmoothed.
    pfrwb_column : str
        PFRWB column to plot. Use 'PFRWB' for unsmoothed.
    config : dict, optional
        Config with visualization settings.
    """
    # Backward compatibility
    if n_classes is not None:
        n_iterations = n_classes

    # Get figsize/dpi from config if available
    viz_cfg = config.get('visualization', {}) if config else {}
    figsize = viz_cfg.get('figsize', figsize)
    if isinstance(figsize, list):
        figsize = tuple(figsize)
    dpi = viz_cfg.get('dpi', dpi)

    city = ""
    if config:
        city = config.get('project', {}).get('city_name', '')

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # ── Panel A: PFRMA ──
    print(f"\n{'='*60}")
    print(f"PANEL A: {pfrma_column}")
    print(f"{'='*60}")
    plot_risk_panel(
        buildings_gdf, pfrma_column, 'PFRMA',
        ax=axes[0], config=config,
        title=f'{city} - (A) PFR$_{{MA}}$ Risk\n(Mobility & Accessibility)',
        statistical_units_gdf=statistical_units_gdf,
    )

    # ── Panel B: PFRWB ──
    print(f"\n{'='*60}")
    print(f"PANEL B: {pfrwb_column}")
    print(f"{'='*60}")
    plot_risk_panel(
        buildings_gdf, pfrwb_column, 'PFRWB',
        ax=axes[1], config=config,
        title=f'{city} - (B) PFR$_{{WB}}$ Risk\n(Well-being)',
        statistical_units_gdf=statistical_units_gdf,
    )

    # ── Panel C: Combined bivariate ──
    print(f"\n{'='*60}")
    print(f"PANEL C: Combined bivariate")
    print(f"{'='*60}")
    _plot_combined_panel(
        buildings_gdf, pfrma_column, pfrwb_column,
        ax=axes[2], config=config,
        title=f'{city} - (C) Combined PFR Risk',
        statistical_units_gdf=statistical_units_gdf,
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"\n✅ Saved to: {save_path}")

    return fig, axes


def _plot_combined_panel(buildings_gdf, pfrma_column, pfrwb_column,
                         ax, config=None, title=None,
                         statistical_units_gdf=None):
    """
    Panel C: Combined bivariate risk (PFRMA × PFRWB blended).

    Buildings are colored by blending their PFRMA (yellow/orange) and
    PFRWB (purple/wine) classifications.
    """
    ma_vals = buildings_gdf[pfrma_column].fillna(0).values
    wb_vals = buildings_gdf[pfrwb_column].fillna(0).values

    # Classify each independently
    ma_classes, _, ma_colors, _ = classify_values(
        ma_vals, risk_type='PFRMA', config=config
    )
    wb_classes, _, wb_colors, _ = classify_values(
        wb_vals, risk_type='PFRWB', config=config
    )

    # Background
    if statistical_units_gdf is not None:
        statistical_units_gdf.plot(
            ax=ax, facecolor='#f5f5f5', edgecolor='#cccccc', linewidth=0.3,
        )

    # Blend colors per building
    n = len(buildings_gdf)
    facecolors = np.ones((n, 4))  # RGBA, start white

    for i in range(n):
        ma_c = mcolors.to_rgba(ma_colors[ma_classes[i]])
        wb_c = mcolors.to_rgba(wb_colors[wb_classes[i]])

        # Additive blend — if both risks present, color darkens
        if ma_classes[i] == 0 and wb_classes[i] == 0:
            facecolors[i] = [1, 1, 1, 1]  # white
        elif ma_classes[i] == 0:
            facecolors[i] = wb_c
        elif wb_classes[i] == 0:
            facecolors[i] = ma_c
        else:
            # Blend: average the two risk colors
            r = (ma_c[0] + wb_c[0]) / 2
            g = (ma_c[1] + wb_c[1]) / 2
            b = (ma_c[2] + wb_c[2]) / 2
            # Darken slightly when both present
            darken = 0.85
            facecolors[i] = [r * darken, g * darken, b * darken, 1.0]

    # Plot buildings one by one with blended colors
    for idx, (_, row) in enumerate(buildings_gdf.iterrows()):
        geom = row.geometry
        # Handle both Polygon and MultiPolygon
        if geom.geom_type == 'MultiPolygon':
            for part in geom.geoms:
                ax.fill(
                    *part.exterior.xy,
                    facecolor=facecolors[idx],
                    edgecolor='black',
                    linewidth=0.5,
                )
        elif geom.geom_type == 'Polygon':
            ax.fill(
                *geom.exterior.xy,
                facecolor=facecolors[idx],
                edgecolor='black',
                linewidth=0.5,
            )

    # Bivariate legend (simplified 3×3 grid)
    _add_bivariate_legend(ax, ma_colors, wb_colors)

    if title:
        ax.set_title(title, fontsize=13, fontweight='bold')
    ax.axis('off')

    ax.text(0.01, 0.02, '(C) OpenStreetMap contributors\n(C) CARTO',
            transform=ax.transAxes, fontsize=6, color='grey',
            verticalalignment='bottom')


def _add_bivariate_legend(ax, ma_colors, wb_colors):
    """Add a small bivariate legend grid to the axes."""
    # Build 3×3 mini-grid (Low/Med/High for each axis)
    legend_indices = [1, 2, 4]  # low, medium, very high from the 5-class scheme

    bbox = ax.get_position()
    lx = bbox.x0 + 0.01
    ly = bbox.y0 + 0.01
    lw, lh = 0.12, 0.12

    leg_ax = ax.figure.add_axes([lx, ly, lw, lh])

    grid = np.ones((3, 3, 4))  # 3×3 RGBA
    for row in range(3):      # PFRWB axis (y)
        for col in range(3):  # PFRMA axis (x)
            ma_c = mcolors.to_rgba(ma_colors[legend_indices[col]])
            wb_c = mcolors.to_rgba(wb_colors[legend_indices[row]])
            # Blend
            r = (ma_c[0] + wb_c[0]) / 2
            g = (ma_c[1] + wb_c[1]) / 2
            b = (ma_c[2] + wb_c[2]) / 2
            grid[row, col] = [r * 0.85, g * 0.85, b * 0.85, 1.0]

    leg_ax.imshow(grid, origin='lower', aspect='auto', interpolation='nearest')
    leg_ax.set_xticks([0, 2])
    leg_ax.set_xticklabels(['Low', 'High'], fontsize=6)
    leg_ax.set_yticks([0, 2])
    leg_ax.set_yticklabels(['Low', 'High'], fontsize=6)
    leg_ax.set_xlabel('PFR$_{MA}$', fontsize=7, labelpad=2)
    leg_ax.set_ylabel('PFR$_{WB}$', fontsize=7, labelpad=2)
    leg_ax.tick_params(length=0)
    for spine in leg_ax.spines.values():
        spine.set_linewidth(0.5)


# =====================================================================
#  SINGLE PFRWB PLOT — for direct comparison with ArcGIS experiment
# =====================================================================

def plot_pfrwb_building_level(buildings_gdf, pfrwb_column='PFRWB',
                               ax=None, figsize=(10, 10),
                               title='Hamburg - (B) PFR$_{WB}$ Risk\n(Well-being)',
                               config=None,
                               statistical_units_gdf=None,
                               edgecolor='black', linewidth=0.5,
                               show_basemap=True,
                               save_path=None, dpi=300):
    """
    Plot PFRWB at building level — for direct comparison with ArcGIS output.

    Use pfrwb_column='PFRWB' (not smoothed) to match the author's
    "experiment without smooth" visualization.

    Parameters
    ----------
    buildings_gdf : GeoDataFrame
        Buildings with PFRWB column (raw, unnormalized values).
    pfrwb_column : str
        Column to plot. 'PFRWB' for unsmoothed, 'PFRWB_smoothed' for smoothed.
    config : dict, optional
        Config with fixed_thresholds for exact ArcGIS match.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    print(f"\n{'='*60}")
    print(f"PFRWB Building-Level Plot: {pfrwb_column}")
    print(f"{'='*60}")

    plot_risk_panel(
        buildings_gdf, pfrwb_column, 'PFRWB',
        ax=ax, config=config,
        title=title,
        statistical_units_gdf=statistical_units_gdf,
        edgecolor=edgecolor,
        linewidth=linewidth,
        show_basemap=show_basemap,
    )

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"\n✅ Saved to: {save_path}")

    return fig, ax