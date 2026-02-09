import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import warnings

warnings.filterwarnings('ignore')

try:
    import contextily as ctx
    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False

CLASS_LABELS = ['very low', 'low', 'medium', 'high', 'very high']

SVPF_COLORS = [
    '#ffffff',  # very low: white
    '#d0d0d0',  # low: light grey
    '#909090',  # medium: medium grey
    '#505050',  # high: dark grey
    '#202020',  # very high: very dark grey
]

PFRMA_COLOR_MAP = [
    (0, 0, 0, 0),            # 0: transparent
    (1.0, 0.95, 0.6, 0.9),   # 1: light yellow
    (1.0, 0.82, 0.3, 0.9),   # 2: yellow
    (0.95, 0.65, 0.1, 0.9),  # 3: dark yellow / amber
    (0.85, 0.45, 0.0, 0.9),  # 4: deep orange-yellow
]

PFRWB_COLOR_MAP = [
    (0, 0, 0, 0),            # 0: transparent
    (0.85, 0.75, 0.92, 0.9), # 1: light purple
    (0.68, 0.50, 0.78, 0.9), # 2: purple
    (0.52, 0.28, 0.65, 0.9), # 3: dark purple
    (0.38, 0.10, 0.55, 0.9), # 4: deep purple
]

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"


def _prepare_viz_gdf(gdf):
    """
    Mis match for visualize, create a copy in Web Mercator (EPSG:3857) for basemap compatibility.
    """
    viz_gdf = gdf.copy()
    if viz_gdf.crs is None:
        viz_gdf.set_crs(epsg=25832, inplace=True)
    return viz_gdf.to_crs(epsg=3857)


def _add_basemap(ax, zoom='auto', alpha=0.5):
    """
    Add a light grey basemap.
    """
    if not HAS_CONTEXTILY:
        return

    try:
        ctx.add_basemap(
            ax,
            crs='EPSG:3857',
            source=ctx.providers.CartoDB.Positron,
            zoom=zoom,
            alpha=alpha,
        )
    except Exception as e:
        # Fallback: try without specific zoom
        try:
            ctx.add_basemap(
                ax,
                crs='EPSG:3857',
                source=ctx.providers.CartoDB.Positron,
                alpha=alpha,
            )
        except Exception:
            print(f"Basemap unavailable: {e}")


# --- Classification ---

def head_tail_breaks(values, n_iterations=3):
    """
    Head/tail breaks classification (Step6 in ArcGIS workflow).
    """
    values = np.array(values, dtype=float)
    values = values[~np.isnan(values)]
    values = values[values > 0]

    if len(values) == 0:
        return []

    means = []
    current = values.copy()

    for _ in range(n_iterations):
        if len(current) == 0:
            break
        mean_val = np.mean(current)
        means.append(mean_val)
        current = current[current > mean_val]

    return means


def classify_risk_values(values, n_iterations=3):
    """
    Classify values into 5 classes using head/tail breaks.

    Classes:
        0 = very low (zero/null -> transparent)
        1 = low (0 < val <= break1)
        2 = medium (break1 < val <= break2)
        3 = high (break2 < val <= break3)
        4 = very high (val > break3)
    """
    values = np.array(values, dtype=float)
    breaks = head_tail_breaks(values, n_iterations)

    classes = np.zeros(len(values), dtype=int)
    for i, val in enumerate(values):
        if np.isnan(val) or val <= 0:
            classes[i] = 0
        else:
            assigned = False
            for c, threshold in enumerate(breaks):
                if val <= threshold:
                    classes[i] = c + 1
                    assigned = True
                    break
            if not assigned:
                classes[i] = len(breaks) + 1

    classes = np.minimum(classes, 4)
    breaks = [0] + breaks
    return classes, breaks


# --- Color helpers ---

def _normalize_values(values):
    """Normalize positive values to 0-1 range for color mapping."""
    values = np.array(values, dtype=float)
    positive = values[values > 0]
    if len(positive) == 0:
        return np.zeros_like(values)
    vmin, vmax = positive.min(), positive.max()
    if vmax == vmin:
        return np.where(values > 0, 0.5, 0.0)
    normed = np.where(values > 0, (values - vmin) / (vmax - vmin), 0.0)
    return np.clip(normed, 0, 1)


def _yellow_color(intensity):
    """Yellow color for PFRMA at given intensity (0-1)."""
    return [1.0, 0.85 - 0.4 * intensity, 0.2 * (1 - intensity),
            0.3 + 0.6 * intensity]


def _purple_color(intensity):
    """Purple color for PFRWB at given intensity (0-1)."""
    return [0.55 + 0.1 * intensity, 0.25 * (1 - intensity),
            0.5 + 0.2 * intensity, 0.3 + 0.5 * intensity]


# --- Plot helpers ---

def _plot_risk_layers(ax, buildings_gdf, pfrma_norm, pfrwb_norm,
                      has_pfrma, has_pfrwb, plot_no_risk=True):
    """Plot yellow PFRMA + purple PFRWB layers with transparency blending."""
    no_risk = ~(has_pfrma | has_pfrwb)

    if plot_no_risk and no_risk.any():
        buildings_gdf[no_risk].plot(ax=ax, color='#e8e8e8',
                                   edgecolor='none', linewidth=0, alpha=0.3)

    # Layer 1: PFRMA (yellow)
    if has_pfrma.any():
        gdf = buildings_gdf[has_pfrma]
        intensities = pfrma_norm[has_pfrma]
        colors = [_yellow_color(i) for i in intensities]
        for idx in range(len(gdf)):
            gdf.iloc[idx:idx + 1].plot(ax=ax, color=colors[idx],
                                       edgecolor='none', linewidth=0)

    # Layer 2: PFRWB (purple) on top
    if has_pfrwb.any():
        gdf = buildings_gdf[has_pfrwb]
        intensities = pfrwb_norm[has_pfrwb]
        colors = [_purple_color(i) for i in intensities]
        for idx in range(len(gdf)):
            gdf.iloc[idx:idx + 1].plot(ax=ax, color=colors[idx],
                                       edgecolor='none', linewidth=0)


def _add_bivariate_legend(ax, position='upper right', size=0.12):
    """
    Add 2D bivariate legend matching Figure 4 in the paper.
    Yellow (PFRMA) on x-axis, purple (PFRWB) on y-axis.
    """
    n = 4
    legend_data = np.zeros((n, n, 4))

    for iy in range(n):
        for ix in range(n):
            yellow = np.array([1.0, 0.85, 0.2, ix / (n - 1) * 0.8])
            purple = np.array([0.55, 0.25, 0.7, iy / (n - 1) * 0.8])
            a_out = purple[3] + yellow[3] * (1 - purple[3])
            if a_out > 0:
                rgb = (purple[:3] * purple[3] +
                       yellow[:3] * yellow[3] * (1 - purple[3])) / a_out
            else:
                rgb = np.array([1, 1, 1])
            legend_data[iy, ix] = [*rgb, max(a_out, 0.15)]

    bbox = ax.get_position()
    fig_x = bbox.x0 + 0.98 * bbox.width
    fig_y = bbox.y0 + 0.98 * bbox.height

    leg_w, leg_h = size, size
    leg_ax = ax.figure.add_axes([fig_x - leg_w - 0.01,
                                  fig_y - leg_h - 0.01, leg_w, leg_h])
    leg_ax.imshow(legend_data, origin='lower', aspect='auto',
                  interpolation='nearest')
    leg_ax.set_xticks([0, n - 1])
    leg_ax.set_xticklabels(['Low', 'High'], fontsize=6)
    leg_ax.set_yticks([0, n - 1])
    leg_ax.set_yticklabels(['Low', 'High'], fontsize=6)
    leg_ax.set_xlabel('PFR$_{MA}$', fontsize=7, labelpad=2)
    leg_ax.set_ylabel('PFR$_{WB}$', fontsize=7, labelpad=2)
    leg_ax.tick_params(length=0)
    leg_ax.set_title('Pluvial flood risk (PFR)', fontsize=7, pad=3)

    leg_ax.annotate('to mobility &\naccessibility (MA)',
                    xy=(1, -0.18), xycoords='axes fraction',
                    fontsize=5, ha='right', va='top', color='#d4a017')
    leg_ax.annotate('to well-\nbeing (WB)',
                    xy=(-0.18, 1), xycoords='axes fraction',
                    fontsize=5, ha='right', va='top', color='#7b3fa0',
                    rotation=90)

    return leg_ax


# --- Standalone figure functions ---

def create_figure4(buildings_gdf, n_iterations=3, figsize=(12, 10),
                   pfrma_column='PFRMA_smoothed',
                   pfrwb_column='PFRWB_smoothed',
                   title=None, save_path=None, dpi=300,
                   basemap=True):
    #ToDO papers figure mismatch??
    """
    Create Figure 4: Combined pluvial flood risk map (PFRMA & PFRWB).
    """
    # Reproject for basemap
    bld = _prepare_viz_gdf(buildings_gdf)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_facecolor('#f0f0f0')

    pfrma_vals = bld[pfrma_column].fillna(0).values
    pfrwb_vals = bld[pfrwb_column].fillna(0).values

    pfrma_classes, pfrma_breaks = classify_risk_values(pfrma_vals, n_iterations)
    pfrwb_classes, pfrwb_breaks = classify_risk_values(pfrwb_vals, n_iterations)

    pfrma_norm = _normalize_values(pfrma_vals)
    pfrwb_norm = _normalize_values(pfrwb_vals)

    has_pfrma = pfrma_vals > 0
    has_pfrwb = pfrwb_vals > 0

    print(f"PFRMA breaks: {[f'{b:.4f}' for b in pfrma_breaks]}")
    print(f"PFRWB breaks: {[f'{b:.4f}' for b in pfrwb_breaks]}")
    # print(f"PFRMA class dist: "
    #       f"{dict(zip(*np.unique(pfrma_classes, return_counts=True)))}")
    # print(f"PFRWB class dist: "
    #       f"{dict(zip(*np.unique(pfrwb_classes, return_counts=True)))}")

    _plot_risk_layers(ax, bld, pfrma_norm, pfrwb_norm,
                      has_pfrma, has_pfrwb)

    if basemap:
        _add_basemap(ax)

    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    _add_bivariate_legend(ax)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"{GREEN}Saved to: {save_path}{RESET}\n")

    return fig, ax


def create_figure5(buildings_gdf, statistical_units_gdf,
                   pfrma_column='PFRMA_smoothed',
                   pfrwb_column='PFRWB_smoothed',
                   svpf_column='svpf', n_iterations=3,
                   figsize=(12, 10), title=None, save_path=None, dpi=300,
                   basemap=True):
    #ToDO papers figure mismatch??
    """
    Create Figure 5? Social vulnerability (SVPF) with risk overlay.
    """
    bld = _prepare_viz_gdf(buildings_gdf)
    stu = _prepare_viz_gdf(statistical_units_gdf)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_facecolor('#f0f0f0')

    # Background: SVPF grey gradient
    if svpf_column in stu.columns:
        svpf_vals = stu[svpf_column].fillna(0).values
        svpf_classes, svpf_breaks = classify_risk_values(svpf_vals, n_iterations)
        print(f"SVPF breaks: {[f'{b:.4f}' for b in svpf_breaks]}")

        for cls in range(5):
            mask = svpf_classes == cls
            if mask.any():
                stu[mask].plot(ax=ax, color=SVPF_COLORS[cls],
                               edgecolor='#aaaaaa', linewidth=0.5, alpha=0.7)
    else:
        stu.plot(ax=ax, color='#e0e0e0',
                 edgecolor='#aaaaaa', linewidth=0.5, alpha=0.5)

    # Overlay: risk layers
    pfrma_vals = bld[pfrma_column].fillna(0).values
    pfrwb_vals = bld[pfrwb_column].fillna(0).values
    pfrma_norm = _normalize_values(pfrma_vals)
    pfrwb_norm = _normalize_values(pfrwb_vals)
    has_pfrma = pfrma_vals > 0
    has_pfrwb = pfrwb_vals > 0

    _plot_risk_layers(ax, bld, pfrma_norm, pfrwb_norm,
                      has_pfrma, has_pfrwb, plot_no_risk=False)

    if basemap:
        _add_basemap(ax, alpha=0.3)

    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    sv_legend = [Patch(facecolor=SVPF_COLORS[i], edgecolor='grey',
                       label=CLASS_LABELS[i]) for i in range(5)]
    ax.legend(handles=sv_legend, loc='lower left',
              title='Social vulnerability (SV$_{PF}$)',
              fontsize=8, title_fontsize=9)

    _add_bivariate_legend(ax)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"{GREEN}Saved to: {save_path}{RESET}\n")


    return fig, ax


def create_figure6(buildings_gdf, statistical_units_gdf,
                   sensitivity_column='Sensitivity',
                   coping_column='CopingCapacity',
                   pfrma_column='PFRMA_smoothed',
                   pfrwb_column='PFRWB_smoothed',
                   n_iterations=3, figsize=(12, 10),
                   title=None, save_path=None, dpi=300,
                   basemap=True):
    """
    Create Figure 6 from paper???
    """
    bld = _prepare_viz_gdf(buildings_gdf)
    stu = _prepare_viz_gdf(statistical_units_gdf)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_facecolor('#f0f0f0')

    has_sens = sensitivity_column in stu.columns
    has_cope = coping_column in stu.columns

    if has_sens and has_cope:
        sens_vals = stu[sensitivity_column].fillna(0).values
        cope_vals = stu[coping_column].fillna(0).values

        sens_classes, sens_breaks = classify_risk_values(sens_vals, n_iterations)
        cope_classes, cope_breaks = classify_risk_values(cope_vals, n_iterations)

        print(f"Sensitivity breaks: {[f'{b:.4f}' for b in sens_breaks]}")
        print(f"Coping capacity breaks: {[f'{b:.4f}' for b in cope_breaks]}")

        sens_norm = _normalize_values(sens_vals)
        cope_norm = _normalize_values(cope_vals)

        for idx in range(len(stu)):
            s = sens_norm[idx]
            c = cope_norm[idx]
            r = 0.9 - 0.5 * max(s, c)
            g = 0.9 - 0.6 * s + 0.1 * c
            b = 0.9 - 0.6 * c + 0.1 * s
            color = [np.clip(r, 0, 1), np.clip(g, 0, 1),
                     np.clip(b, 0, 1), 0.7]
            stu.iloc[idx:idx + 1].plot(
                ax=ax, color=color, edgecolor='#aaaaaa', linewidth=0.5)
    else:
        stu.plot(ax=ax, color='#e0e0e0',
                 edgecolor='#aaaaaa', linewidth=0.5, alpha=0.5)
        if not has_sens:
            print(f"Warning: '{sensitivity_column}' not found")
        if not has_cope:
            print(f"Warning: '{coping_column}' not found")

    # Overlay: risk layers
    pfrma_vals = bld[pfrma_column].fillna(0).values
    pfrwb_vals = bld[pfrwb_column].fillna(0).values
    pfrma_norm = _normalize_values(pfrma_vals)
    pfrwb_norm = _normalize_values(pfrwb_vals)
    has_pfrma = pfrma_vals > 0
    has_pfrwb = pfrwb_vals > 0

    _plot_risk_layers(ax, bld, pfrma_norm, pfrwb_norm,
                      has_pfrma, has_pfrwb, plot_no_risk=False)

    if basemap:
        _add_basemap(ax, alpha=0.3)

    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    # Bivariate legend for background
    if has_sens and has_cope:
        n = 4
        bg_legend = np.zeros((n, n, 4))
        for iy in range(n):
            for ix in range(n):
                s = ix / (n - 1)
                c = iy / (n - 1)
                r = 0.9 - 0.5 * max(s, c)
                g = 0.9 - 0.6 * s + 0.1 * c
                b_val = 0.9 - 0.6 * c + 0.1 * s
                bg_legend[iy, ix] = [np.clip(r, 0, 1), np.clip(g, 0, 1),
                                     np.clip(b_val, 0, 1), 0.85]

        bbox = ax.get_position()
        lx = bbox.x0 + 0.01
        ly = bbox.y0 + 0.01
        lw, lh = 0.1, 0.1
        leg_ax = ax.figure.add_axes([lx, ly, lw, lh])
        leg_ax.imshow(bg_legend, origin='lower', aspect='auto',
                      interpolation='nearest')
        leg_ax.set_xticks([0, n - 1])
        leg_ax.set_xticklabels(['Low', 'High'], fontsize=6)
        leg_ax.set_yticks([0, n - 1])
        leg_ax.set_yticklabels(['Low', 'High'], fontsize=6)
        leg_ax.set_xlabel('Coping capacity', fontsize=7, labelpad=2)
        leg_ax.set_ylabel('Sensitivity', fontsize=7, labelpad=2)
        leg_ax.tick_params(length=0)

    _add_bivariate_legend(ax)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"{GREEN}Saved to: {save_path}{RESET}\n")


    return fig, ax


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
                              config=None,
                              basemap=True):
    """
    Create 3-panel visualization:
    """
    # Backward compatibility
    if n_classes is not None:
        n_iterations = n_classes

    # Reproject for basemap (internal copies, originals unchanged)
    bld = _prepare_viz_gdf(buildings_gdf)
    stu = _prepare_viz_gdf(statistical_units_gdf)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.subplots_adjust(bottom=0.15, wspace=0.05)

    pfrma_vals = bld[pfrma_column].fillna(0).values
    pfrwb_vals = bld[pfrwb_column].fillna(0).values

    pfrma_classes, pfrma_breaks = classify_risk_values(pfrma_vals, n_iterations)
    pfrwb_classes, pfrwb_breaks = classify_risk_values(pfrwb_vals, n_iterations)

    has_pfrma = pfrma_vals > 0
    has_pfrwb = pfrwb_vals > 0

    print(f"PFRMA breaks: {[f'{b:.4f}' for b in pfrma_breaks]}")
    print(f"PFRWB breaks: {[f'{b:.4f}' for b in pfrwb_breaks]}")
    # print(f"PFRMA class dist: "
    #       f"{dict(zip(*np.unique(pfrma_classes, return_counts=True)))}")
    # print(f"PFRWB class dist: "
    #       f"{dict(zip(*np.unique(pfrwb_classes, return_counts=True)))}")

    # ---- Panel A: PFRMA only (yellow) ----
    ax1 = axes[0]
    ax1.set_facecolor('#f0f0f0')

    no_risk_ma = pfrma_classes == 0
    if no_risk_ma.any():
        bld[no_risk_ma].plot(ax=ax1, color='#e8e8e8',
                             edgecolor='none', linewidth=0, alpha=0.2)

    for cls in range(1, 5):
        mask = pfrma_classes == cls
        if mask.any():
            bld[mask].plot(ax=ax1, color=PFRMA_COLOR_MAP[cls],
                           edgecolor='black', linewidth=0.3)

    if basemap:
        _add_basemap(ax1)

    ax1.set_title(f'{title_prefix}(A) PFR$_{{MA}}$ Risk\n(Mobility & Accessibility)',
                  fontsize=12, fontweight='bold')
    ax1.axis('off')

    legend_a = [Patch(facecolor=PFRMA_COLOR_MAP[i], edgecolor='black',
                      label=CLASS_LABELS[i].title())
                for i in range(1, 5) if (pfrma_classes == i).sum() > 0]
    if legend_a:
        ax1.legend(handles=legend_a, loc='lower right', fontsize=8,
                   title='PFR$_{MA}$', title_fontsize=9)

    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()

    # ---- Panel B: PFRWB only (purple) ----
    ax2 = axes[1]
    ax2.set_facecolor('#f0f0f0')

    no_risk_wb = pfrwb_classes == 0
    if no_risk_wb.any():
        bld[no_risk_wb].plot(ax=ax2, color='#e8e8e8',
                             edgecolor='none', linewidth=0, alpha=0.2)

    for cls in range(1, 5):
        mask = pfrwb_classes == cls
        if mask.any():
            bld[mask].plot(ax=ax2, color=PFRWB_COLOR_MAP[cls],
                           edgecolor='black', linewidth=0.3)

    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)

    if basemap:
        _add_basemap(ax2)

    ax2.set_title(f'{title_prefix}(B) PFR$_{{WB}}$ Risk\n(Well-being)',
                  fontsize=12, fontweight='bold')
    ax2.axis('off')

    legend_b = [Patch(facecolor=PFRWB_COLOR_MAP[i], edgecolor='black',
                      label=CLASS_LABELS[i].title())
                for i in range(1, 5) if (pfrwb_classes == i).sum() > 0]
    if legend_b:
        ax2.legend(handles=legend_b, loc='lower right', fontsize=8,
                   title='PFR$_{WB}$', title_fontsize=9)

    # ---- Panel C: Combined  ----
    ax3 = axes[2]
    ax3.set_facecolor('#f0f0f0')

    # Background: SVPF grey if available
    if svpf_column in stu.columns:
        svpf_vals = stu[svpf_column].fillna(0).values
        svpf_classes, _ = classify_risk_values(svpf_vals, n_iterations)
        for cls in range(5):
            mask = svpf_classes == cls
            if mask.any():
                stu[mask].plot(ax=ax3, color=SVPF_COLORS[cls],
                               edgecolor='#aaaaaa', linewidth=0.5, alpha=0.6)
    else:
        stu.plot(ax=ax3, color='#e0e0e0',
                 edgecolor='#aaaaaa', linewidth=0.5, alpha=0.5)

    # Overlay: dual-layer blending (yellow PFRMA + purple PFRWB)
    pfrma_norm = _normalize_values(pfrma_vals)
    pfrwb_norm = _normalize_values(pfrwb_vals)
    _plot_risk_layers(ax3, bld, pfrma_norm, pfrwb_norm,
                      has_pfrma, has_pfrwb, plot_no_risk=False)

    ax3.set_xlim(xlim)
    ax3.set_ylim(ylim)

    if basemap:
        _add_basemap(ax3, alpha=0.3)

    ax3.set_title(f'{title_prefix}(C) Combined PFR$_{{MA}}$ & PFR$_{{WB}}$\n+ SV$_{{PF}}$',
                  fontsize=12, fontweight='bold')
    ax3.axis('off')

    # Combined legend
    sv_legend = [Patch(facecolor=SVPF_COLORS[i], edgecolor='grey',
                       label=CLASS_LABELS[i].title())
                 for i in range(1, 5)]
    ax3.legend(handles=sv_legend, loc='lower left', title='SV$_{PF}$',
               fontsize=7, title_fontsize=8)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"{GREEN}Saved to: {save_path}{RESET}\n")


    print(f"PFRMA: {has_pfrma.sum()} buildings with risk")
    print(f"PFRWB: {has_pfrwb.sum()} buildings with risk")
    print(f"Both: {(has_pfrma & has_pfrwb).sum()} buildings with both risks")
    #TODO need to check with CS3
    print(f"{RED}#TODO need to check with CS3{RESET}\n")
    print(f"Only PFRMA: {(has_pfrma & ~has_pfrwb).sum()} buildings")
    print(f"Only PFRWB: {(~has_pfrma & has_pfrwb).sum()} buildings")

    return fig, axes