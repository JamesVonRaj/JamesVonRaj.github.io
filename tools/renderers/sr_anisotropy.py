"""Render sr anisotropy with the shared website theme.

Scientific inputs are frozen in tools/data/sr_anisotropy.npz.
Use tools/theme.py to rebuild both display modes into a preview directory.
Input provenance and refresh instructions are in docs/THEMING.md.
"""

from site_theme.theme import asset_style, data_path, output_dir, font_files


import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib import font_manager as fm


# Poppins, registered explicitly so it works regardless of the font cache.
# Poppins has no Greek glyphs, so alpha is written as mathtext and falls back
# to STIX Sans.
for _f in font_files():
    fm.fontManager.addfont(_f)
plt.rcParams["font.family"] = "Poppins"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Poppins"
plt.rcParams["mathtext.it"] = "Poppins:italic"
plt.rcParams["mathtext.bf"] = "Poppins:bold"
plt.rcParams["mathtext.fallback"] = "stixsans"

OUT_DIR = str(output_dir())

ANGLES_DEG = [0, 30, 60, 90, 120, 150]
REFF_COLS = [f"reff_{a}deg" for a in ANGLES_DEG]

# Geometry parameters
BOX_SIZE = 1.0

# Figure geometry (inches); about 10 in wide so it reads at ~800 CSS px
FIG_W, FIG_H = 10.0, 4.3
DPI = 230


def draw_geometry_circle(ax, segments, st):
    """Draw SR beam geometry, rotated 90 degrees and cropped to a circle."""
    ax.set_facecolor(st["bg"])

    center = BOX_SIZE / 2
    radius = BOX_SIZE / 2

    ax.set_xlim(center - radius * 1.04, center + radius * 1.04)
    ax.set_ylim(center - radius * 1.04, center + radius * 1.04)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


    if not segments:
        return

    thicknesses = np.array([s[2] for s in segments])
    norm = mcolors.LogNorm(vmin=max(thicknesses.min(), 1e-6), vmax=thicknesses.max())
    cmap = mcolors.LinearSegmentedColormap.from_list("beams", st["beam_ramp"])

    clip = mpatches.Circle((center, center), radius, transform=ax.transData)

    # Thin beams first, so thick beams sit on top
    for start, end, t in sorted(segments, key=lambda s: s[2]):
        lw = max(st["beam_lw_min"], t * st["beam_lw_scale"])
        line, = ax.plot([start[0], end[0]], [start[1], end[1]],
                        color=cmap(norm(t)), linewidth=lw, solid_capstyle='butt')
        line.set_clip_path(clip)

    circle_border = mpatches.Circle((center, center), radius, transform=ax.transData,
                                    facecolor='none', edgecolor=st["ink_2"],
                                    linewidth=st["frame_lw"], zorder=10)
    ax.add_patch(circle_border)


def get_single_polar_data(row):
    """Get directional R_eff from a single row, mirrored to the full circle."""
    means = row[REFF_COLS].values.astype(float)

    angles_rad = np.array(ANGLES_DEG + [a + 180 for a in ANGLES_DEG]) * np.pi / 180
    means_full = np.concatenate([means, means])

    order = np.argsort(angles_rad)
    angles_rad = angles_rad[order]
    means_full = means_full[order]

    # Close the loop
    angles_rad = np.append(angles_rad, angles_rad[0])
    means_full = np.append(means_full, means_full[0])

    return angles_rad, means_full


def draw_polar(ax, row, st):
    """Draw a minimal polar plot of directional R_eff from a single realization."""
    ax.set_facecolor(st["bg"])

    angles, reff = get_single_polar_data(row)
    rmax = reff.max()

    ax.fill(angles, reff, color=st["main"], alpha=0.16, linewidth=0, zorder=2)
    ax.plot(angles, reff, color=st["main"], linewidth=st["data_lw"], zorder=3,
            solid_joinstyle='round')

    # Mark the directions of largest and smallest R_eff (they set the ratio)
    vals = row[REFF_COLS].values.astype(float)
    i_max, i_min = int(np.argmax(vals)), int(np.argmin(vals))
    for i in (i_max, i_min):
        th = np.deg2rad(ANGLES_DEG[i])
        ax.plot([th, th + np.pi], [vals[i], vals[i]], linestyle='none', marker='o',
                markersize=st["ms"], markerfacecolor=st["accent"],
                markeredgecolor=st["bg"], markeredgewidth=st["mew"], zorder=4)
    ax.annotate("max", xy=(np.deg2rad(ANGLES_DEG[i_max]), vals[i_max]),
                xytext=(-9, 0), textcoords='offset points', ha='right', va='center',
                fontsize=12, color=st["ink_2"], zorder=5)
    ax.annotate("min", xy=(np.deg2rad(ANGLES_DEG[i_min]), vals[i_min]),
                xytext=(-4, 15), textcoords='offset points', ha='right', va='bottom',
                fontsize=12, color=st["ink_2"], zorder=5)

    ax.set_ylim(0, rmax * 1.06)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines['polar'].set_color(st["rule"])
    ax.spines['polar'].set_linewidth(st["frame_lw"])

    ax.set_thetagrids([0, 90, 180, 270], labels=['0°', '90°', '180°', '270°'],
                      fontsize=12, color=st["ink_2"])
    ax.tick_params(axis='x', pad=2)


def get_anisotropy_vs_alpha(df):
    """Get anisotropy ratio (max/min R_eff) vs alpha: mean and SEM."""
    grouped = df.groupby('alpha')['reff_anisotropy_ratio']
    means = grouped.mean()
    stds = grouped.std()
    counts = grouped.count()
    sems = stds / np.sqrt(counts)
    return means.index.values, means.values, sems.values


def draw_alpha_curve(ax, df, st):
    ax.set_facecolor(st["bg"])

    alphas, means, sems = get_anisotropy_vs_alpha(df)

    ax.axhline(y=1.0, color=st["ink_2"], linestyle=(0, (4, 3)),
               linewidth=st["spine_lw"], zorder=1)
    ax.text(1.06, 1.12, "isotropic", ha='right', va='bottom', fontsize=12,
            color=st["ink_2"])

    ax.errorbar(alphas, means, yerr=sems, color=st["main"],
                linewidth=st["data_lw"], capsize=0,
                elinewidth=st["spine_lw"],
                marker='o', markersize=st["ms"], markerfacecolor=st["main"],
                markeredgecolor=st["bg"], markeredgewidth=st["mew"], zorder=3)

    ax.set_xlim(0.04, 1.06)
    ax.set_ylim(0, 8.2)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([1, 3, 5, 7])
    ax.set_xlabel(r"Heterogeneity $\alpha$", fontsize=14, color=st["ink"])
    ax.set_ylabel(r"max / min $R_\mathrm{eff}$", fontsize=14, color=st["ink"])
    ax.tick_params(colors=st["ink_2"], labelsize=12, width=st["spine_lw"], length=4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(st["ink_2"])
        ax.spines[side].set_linewidth(st["spine_lw"])


def make_figure(filename, st, seg_thick_dict, polar_row, df):
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=st["bg"])

    # Explicit axes placement (figure fractions) so the output size is fixed.
    top, bottom = 0.86, 0.17
    h = top - bottom
    w_sq = h * FIG_H / FIG_W  # square panels

    ax_geo = fig.add_axes([0.015, bottom, w_sq, h])
    draw_geometry_circle(ax_geo, seg_thick_dict, st)

    # Polar panel slightly inset so its theta labels clear the title
    shrink = 0.92
    ax_polar = fig.add_axes([0.355 + w_sq * (1 - shrink) / 2, bottom + h * (1 - shrink) / 2,
                             w_sq * shrink, h * shrink], projection='polar')
    draw_polar(ax_polar, polar_row, st)

    ax_line = fig.add_axes([0.745, bottom, 0.24, h])
    draw_alpha_curve(ax_line, df, st)

    # Panel titles on one baseline (the polar theta labels would push a
    # set_title higher than the others)
    titles = [(ax_geo, r"Scale-rich network ($\alpha = 1$)"),
              (ax_polar, r"Directional $R_\mathrm{eff}$"),
              (ax_line, r"Anisotropy vs. $\alpha$")]
    for ax, text in titles:
        pos = ax.get_position()
        fig.text(pos.x0 + pos.width / 2, 0.935, text, ha='center', va='baseline',
                 fontsize=16, fontweight='semibold', color=st["ink"])

    pos = ax_polar.get_position()
    ratio = polar_row['reff_anisotropy_ratio']
    fig.text(pos.x0 + pos.width / 2, 0.045, f"single realization, max / min ≈ {ratio:.0f}",
             ha='center', va='baseline', fontsize=13, color=st["ink_2"])

    fig.savefig(filename, dpi=DPI, facecolor=st["bg"])
    plt.close(fig)
    print(f"Saved: {filename}")


NORMAL_STYLE = asset_style("sr_anisotropy", "normal")
ACCESSIBLE_STYLE = asset_style("sr_anisotropy", "accessible")

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()
    with np.load(data_path("sr_anisotropy"), allow_pickle=False) as data:
        df = pd.DataFrame({"alpha": data["alpha"], "reff_anisotropy_ratio": data["ratio"]})
        polar_row = pd.Series(dict(zip(REFF_COLS, data["polar"])))
        polar_row["reff_anisotropy_ratio"] = float(data["polar_ratio"])
        geometry = [(start, end, thickness) for (start, end), thickness in zip(data["segments"], data["thicknesses"])]
    make_figure(os.path.join(args.out_dir, "sr_anisotropy.png"), NORMAL_STYLE, geometry, polar_row, df)
    make_figure(os.path.join(args.out_dir, "sr_anisotropy_accessible.png"), ACCESSIBLE_STYLE, geometry, polar_row, df)


if __name__ == "__main__":
    main()
