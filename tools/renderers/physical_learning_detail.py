"""Render physical learning detail with the shared website theme.

Scientific inputs are frozen in tools/data/physical_learning_detail.npz.
Use tools/theme.py to rebuild both display modes into a preview directory.
Input provenance and refresh instructions are in docs/THEMING.md.
"""

from site_theme.theme import asset_style, data_path, output_dir, font_files

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, Rectangle

WEBSITE_IMAGES = str(output_dir())

FAMILIES = [
    {'key': 'rand-del', 'label': 'Delaunay',         'is_spatial': True},
    {'key': 'ws',       'label': 'Watts-Strogatz',   'is_spatial': False},
    {'key': 'ba',       'label': 'Barabási-Albert',  'is_spatial': False},
    {'key': 'er',       'label': 'Erdős-Rényi',      'is_spatial': False},
]

# ---------------------------------------------------------------------------
# Typography: Poppins, registered explicitly so it works without a font cache.
# ---------------------------------------------------------------------------
for _f in font_files():
    fm.fontManager.addfont(_f)
plt.rcParams['font.family'] = 'Poppins'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Poppins'
plt.rcParams['mathtext.it'] = 'Poppins:italic'
plt.rcParams['mathtext.bf'] = 'Poppins:bold'

# ---------------------------------------------------------------------------
PALETTES = {mode: asset_style("physical_learning_detail", mode) for mode in ("normal", "accessible")}
for _style in PALETTES.values():
    _style["edge_cmap"] = LinearSegmentedColormap.from_list("edge_weight", _style.pop("edge_ramp"), N=256)

# Type sizes (pt). Figure is drawn 10 in wide and shown ~900 CSS px wide.
FS_TITLE = 16
FS_LABEL = 14
FS_TICK = 12
FS_LEGEND = 13

N_BINS_EDGES = 38  # histogram bin edges on [0, 1]


def draw_network(ax, pos, segs, weights, palette,
                 periodic_box=False, top_mask=None, node_size=22):
    lw = palette['lw']
    if periodic_box:
        ax.add_patch(Rectangle((0, 0), 1, 1, fill=False,
                               edgecolor=palette['axis'], linestyle=(0, (3, 3)),
                               linewidth=0.6 * lw, alpha=0.6, zorder=0))

    segs = np.asarray(segs)
    weights = np.asarray(weights)
    rmax = max(weights.max(), 1e-12)
    rn = weights / rmax

    if top_mask is None:
        priority = np.zeros(len(rn), dtype=int)
    else:
        priority = np.asarray(top_mask, dtype=bool).astype(int)

    # Sort low-weight first so high-weight edges render on top.
    order = np.lexsort((rn, priority))
    segs_o = segs[order]
    rn_o = rn[order]

    rgba = palette['edge_cmap'](rn_o)
    ax.add_collection(LineCollection(segs_o, linewidths=1.1 * lw,
                                     colors=rgba, capstyle='round', zorder=2))

    ax.scatter(pos[:, 0], pos[:, 1],
               s=node_size, c=palette['node'],
               edgecolors=palette['node_edge'], linewidths=0.6 * lw,
               zorder=3)

    ax.set_aspect('equal')
    if periodic_box:
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
    else:
        ax.autoscale_view()
        ax.margins(0.07)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.patch.set_visible(False)
    for s in ax.spines.values():
        s.set_visible(False)


def draw_window(ax, x0, x1, palette, legend=False):
    """Target band-gap window: a peach region behind the bars (outlined and
    hatched in the accessible variant)."""
    lw = palette['lw']
    ax.axvspan(x0, x1, facecolor=palette['window'], edgecolor='none',
               linewidth=0, zorder=0)
    if palette['window_hatch'] is not None:
        ax.axvspan(x0, x1, facecolor='none', edgecolor=palette['window_hatch'],
                   hatch='////', linewidth=0, zorder=0.2)
    if palette['window_edge'] is not None:
        for x in (x0, x1):
            ax.axvline(x, color=palette['window_edge'], linewidth=0.8 * lw,
                       zorder=0.4)


def draw_dos(ax, arr, ytop, palette,
             show_xticklabels=False, show_yticklabels=False):
    lw = palette['lw']
    f0 = np.asarray(arr['f0'])
    ff = np.asarray(arr['ff'])
    wlo = float(arr['wlo'])
    whi = float(arr['whi'])

    f0_pos = f0[f0 > 1e-6]
    ff_pos = ff[ff > 1e-6]
    fmax = max(f0_pos.max(), ff_pos.max()) * 1.02
    f0_norm = f0_pos / fmax
    ff_norm = ff_pos / fmax
    wlo_norm = wlo / fmax
    whi_norm = whi / fmax
    bins = np.linspace(0, 1.0, N_BINS_EDGES)
    c_before, _ = np.histogram(f0_norm, bins=bins)
    c_after, _ = np.histogram(ff_norm, bins=bins)

    draw_window(ax, wlo_norm, whi_norm, palette)

    # Draw order: light "after" fill, opaque "before" bars, "after" outline.
    # The before bars are an opaque tint, so nothing blends where they overlap.
    ax.stairs(c_after, bins, fill=True, color=palette['after_fill'],
              linewidth=0, zorder=1)
    ax.stairs(c_before, bins, fill=True, color=palette['before'],
              linewidth=0, zorder=2)
    ax.stairs(c_after, bins, fill=False, baseline=0, color=palette['after_line'],
              linewidth=1.6 * lw, zorder=3, clip_on=False)

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, ytop)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(['0', '0.5', '1'] if show_xticklabels else [])
    ax.set_yticks([0, ytop])
    ax.set_yticklabels(['0', f'{ytop:d}'] if show_yticklabels else [])

    ax.tick_params(axis='both', length=3, width=0.8 * lw, pad=3,
                   color=palette['axis'], labelcolor=palette['axis'],
                   labelsize=FS_TICK)

    # Did this run clear the window (no modes left inside it)?  The label sits
    # in the top-right corner, right of the window and above the bars in every
    # panel; "not" gets its own line so the label stays narrower than the gap
    # between the Erdos-Renyi window and the axis edge.
    cleared = int(arr['n_in_final']) == 0
    ax.text(1.0, 1.0, 'window\ncleared' if cleared else 'window\nnot\ncleared',
            transform=ax.transAxes, ha='right', va='top', ma='right',
            fontsize=FS_TICK, linespacing=1.1, color=palette['axis'], zorder=4)
    ax.patch.set_visible(False)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['bottom', 'left']:
        ax.spines[s].set_color(palette['axis'])
        ax.spines[s].set_linewidth(0.8 * lw)


# ---------------------------------------------------------------------------
# Layout, in inches (figure is FIG_W x FIG_H; origin bottom-left).
# ---------------------------------------------------------------------------
FIG_W, FIG_H = 10.0, 6.05
MARGIN_X = 0.12          # left/right page margin
COL_GAP = 0.50           # gap between the two panel columns
NET = 1.75               # network drawing: square side
NET_HIST_GAP = 0.80      # room for y tick labels and the shared y label
ROW_H = NET              # histogram height = network height
ROW_Y = [3.04, 0.74]     # bottom edge of the top and bottom rows
BAND_CY = FIG_H - 0.45   # centre line of the legend / colorbar band
TITLE_PAD = 0.10         # family title baseline above the row
DPI = 230


def _axes_in(fig, x, y, w, h):
    return fig.add_axes([x / FIG_W, y / FIG_H, w / FIG_W, h / FIG_H])


def render_detail(out_path, palette_key, paths, ytop, reps):
    palette = PALETTES[palette_key]
    lw = palette['lw']
    text_c = palette['text']
    axis_c = palette['axis']

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=palette['bg'])

    panel_w = (FIG_W - 2 * MARGIN_X - COL_GAP) / 2
    hist_w = panel_w - NET - NET_HIST_GAP

    hist_axes = []
    for k, family in enumerate(FAMILIES):
        row, col = divmod(k, 2)
        x0 = MARGIN_X + col * (panel_w + COL_GAP)
        y0 = ROW_Y[row]
        ax_net = _axes_in(fig, x0, y0, NET, ROW_H)
        ax_dos = _axes_in(fig, x0 + NET + NET_HIST_GAP, y0, hist_w, ROW_H)
        hist_axes.append(ax_dos)

        path = paths[k]
        if path is None:
            for a in (ax_net, ax_dos):
                a.text(0.5, 0.5, '(no data)', ha='center', va='center',
                       fontsize=FS_TICK, color=text_c, transform=a.transAxes)
                a.axis('off')
            continue

        small_pos, small_segs, small_weights, small_top = reps[family['key']]
        draw_network(ax_net, small_pos, small_segs, small_weights, palette,
                     periodic_box=family['is_spatial'],
                     top_mask=small_top, node_size=24 * (1.0 if lw == 1.0 else 1.15))

        draw_dos(ax_dos, path, ytop, palette,
                 show_xticklabels=(row == 1),
                 show_yticklabels=True)

        fig.text((x0 + NET / 2) / FIG_W, (y0 + ROW_H + TITLE_PAD) / FIG_H,
                 family['label'], ha='center', va='bottom',
                 fontsize=FS_TITLE, fontweight='600', color=text_c)

    # One shared y label for the left-column histograms (same y scale everywhere).
    left_hist_x = MARGIN_X + NET + NET_HIST_GAP
    fig.text((left_hist_x - 0.47) / FIG_W,
             (ROW_Y[1] + ROW_Y[0] + ROW_H) / 2 / FIG_H,
             'mode count', rotation=90, ha='center', va='center',
             fontsize=FS_LABEL, color=text_c)

    # Shared x label under the bottom row.
    fig.text(0.5, 0.20 / FIG_H,
             r'normalized eigenfrequency, $\omega / \omega_{\max}$',
             ha='center', va='center', fontsize=FS_LABEL, color=text_c)

    # --- top band: legend (left) and edge-weight colorbar (right) ---------
    ax_legend = _axes_in(fig, MARGIN_X, BAND_CY - 0.2, 6.2, 0.4)
    ax_legend.axis('off')
    win_kw = dict(facecolor=palette['window'], linewidth=0)
    if palette['window_edge'] is not None:
        win_kw = dict(facecolor=palette['window'], edgecolor=palette['window_edge'],
                      linewidth=0.8 * lw, hatch='////')
    legend_handles = [
        Patch(facecolor=palette['before'], edgecolor='none', label='before training'),
        Patch(facecolor=palette['after_fill'], edgecolor=palette['after_line'],
              linewidth=1.6 * lw, label='after training'),
        Patch(label='target window', **win_kw),
    ]
    leg = ax_legend.legend(handles=legend_handles, loc='center left',
                           bbox_to_anchor=(0.0, 0.5),
                           ncol=3, fontsize=FS_LEGEND, frameon=False,
                           handlelength=1.3, handleheight=1.0,
                           columnspacing=1.4, borderpad=0.0, borderaxespad=0.0,
                           handletextpad=0.5)
    for txt in leg.get_texts():
        txt.set_color(text_c)

    cbar_w = 1.55
    cbar_h = 0.13
    cbar_x = FIG_W - MARGIN_X - 0.48 - cbar_w
    cbar_y = BAND_CY - cbar_h / 2
    ax_cbar = _axes_in(fig, cbar_x, cbar_y, cbar_w, cbar_h)
    sm = cm.ScalarMappable(cmap=palette['edge_cmap'], norm=Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cbar, orientation='horizontal')
    cb.set_ticks([])
    cb.outline.set_edgecolor(axis_c)
    cb.outline.set_linewidth(0.6 * lw)
    ax_cbar.text(-0.06, 0.5, 'low', transform=ax_cbar.transAxes,
                 ha='right', va='center', fontsize=FS_TICK, color=axis_c)
    ax_cbar.text(1.06, 0.5, 'high', transform=ax_cbar.transAxes,
                 ha='left', va='center', fontsize=FS_TICK, color=axis_c)
    ax_cbar.annotate('trained edge weight', xy=(0.5, 1.0), xycoords='axes fraction',
                     xytext=(0, 5), textcoords='offset points',
                     ha='center', va='bottom', fontsize=FS_LABEL, color=text_c)

    fig.savefig(out_path, dpi=DPI, facecolor=palette['bg'])
    plt.close(fig)
    print(f'wrote {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=WEBSITE_IMAGES)
    args = ap.parse_args()

    with np.load(data_path("physical_learning_detail"), allow_pickle=False) as data:
        paths, reps = [], {}
        for family in FAMILIES:
            key = family["key"]
            paths.append({field: data[f"{key}_{field}"] for field in ("f0", "ff", "wlo", "whi", "n_in_final")})
            top = data[f"{key}_top"]
            reps[key] = (data[f"{key}_pos"], data[f"{key}_segs"], data[f"{key}_weights"], top if len(top) else None)
        ytop = int(data["ytop"])
    render_detail(os.path.join(args.out_dir, "physical_learning_detail.png"), "normal", paths, ytop, reps)
    render_detail(os.path.join(args.out_dir, "physical_learning_detail_accessible.png"), "accessible", paths, ytop, reps)


if __name__ == '__main__':
    main()
