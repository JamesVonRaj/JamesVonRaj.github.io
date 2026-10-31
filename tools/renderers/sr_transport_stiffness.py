"""Render sr transport stiffness with the shared website theme.

Scientific inputs are frozen in tools/data/sr_transport_stiffness.npz.
Use tools/theme.py to rebuild both display modes into a preview directory.
Input provenance and refresh instructions are in docs/THEMING.md.

Layout (two rows, about 10 in wide):
  top:    a larger scale-rich network with a square sample cut at angle theta |
          the sample as a resistor network (current in each beam, top and bottom
          faces held at fixed voltages) | the same sample under uniaxial
          compression (von Mises stress from the FEM)
  bottom: conductance and stiffness of this network at all 12 cut angles, each
          relative to its largest value | median anisotropy (largest / smallest
          over the 12 angles) of both against alpha, with interquartile bands
"""

from site_theme.theme import asset_style, data_path, output_dir, font_files


import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib import font_manager as fm


# Poppins, registered explicitly so it works regardless of the font cache.
# Poppins has no Greek glyphs, so alpha and theta are mathtext and fall back
# to Computer Modern, as in the physical-learning figure.
for _f in font_files():
    fm.fontManager.addfont(_f)
plt.rcParams["font.family"] = "Poppins"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Poppins"
plt.rcParams["mathtext.it"] = "Poppins:italic"
plt.rcParams["mathtext.bf"] = "Poppins:bold"

OUT_DIR = str(output_dir())

FIG_W, FIG_H = 10.0, 7.9
DPI = 230

# Type sizes (pt), the same as in the physical-learning figure
FS_TITLE = 16
FS_LABEL = 14
FS_TICK = 12
FS_LEGEND = 13


# ----------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------

def path_patch(vertices, codes, **kw):
    return mpatches.PathPatch(MplPath(vertices, codes), **kw)


def bare_axes(ax, lim):
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def draw_sample_frame(ax, st):
    """Thin frame around the unit sample and coral bars on the loaded faces."""
    ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, fill=False, edgecolor=st["ink_2"],
                                    linewidth=st["frame_lw"], zorder=5))
    for y in (1.0 + st["bar_h"] / 2, -st["bar_h"] / 2):
        ax.add_patch(mpatches.Rectangle((-0.02, y - st["bar_h"] / 2), 1.04, st["bar_h"],
                                        facecolor=st["accent"], edgecolor="none",
                                        zorder=6, clip_on=False))


def draw_cut(ax, d, st):
    """The larger network, with the sample window at the shown angle picked out."""
    side = float(d["master_side"])
    ax.add_patch(path_patch(d["master_vertices"], d["master_codes"],
                            facecolor=st["faded"], edgecolor="none"))
    ax.add_patch(path_patch(d["sample_vertices"], d["sample_codes"], facecolor=st["beam"],
                            edgecolor="none", zorder=2))
    ax.add_patch(mpatches.Rectangle((0, 0), side, side, fill=False,
                                    edgecolor=st["faded_frame"], linewidth=st["frame_lw"]))
    xy = d["window"]
    ax.plot(xy[:, 0], xy[:, 1], color=st["ink_2"], linewidth=st["frame_lw"], zorder=3)

    # Loaded faces (the window's own top and bottom) as coral bars
    for p0, p1 in d["faces"]:
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=st["accent"],
                linewidth=st["bar_lw_cut"], solid_capstyle="butt", zorder=4)

    # Angle between the horizontal and a loaded face, at the corner chosen
    # when the inputs were captured; the panel limits include the marker
    bare_axes(ax, (0, 1))
    ax.set_xlim(*d["cut_limits"][0])
    ax.set_ylim(*d["cut_limits"][1])
    corner, ref_end, lab, arc = (d["marker_corner"], d["marker_reference"],
                                 d["marker_label"], d["marker_arc"])
    ax.plot([corner[0], ref_end[0]], [corner[1], ref_end[1]], color=st["ink"],
            linewidth=st["spine_lw"], linestyle=(0, (3, 2)), zorder=4)
    ax.plot(arc[:, 0], arc[:, 1], color=st["ink"], linewidth=st["spine_lw"] * 1.4, zorder=4)
    ax.text(lab[0], lab[1], r"$\theta$", fontsize=15, color=st["ink"], ha="center",
            va="center", zorder=5,
            path_effects=[pe.withStroke(linewidth=3, foreground=st["bg"])])


def field_cmap(st):
    return mcolors.LinearSegmentedColormap.from_list("field", st["field_ramp"])


def draw_current(ax, d, st):
    bare_axes(ax, (-0.03, 1.03))
    I = d["current"]
    norm = mcolors.LogNorm(st["decades_floor"], 1.0)
    order = np.argsort(I)                      # strongest current drawn on top
    ax_w_pt = ax.get_position().width * FIG_W * 72 / 1.06
    segs = d["segments"][order].copy()
    u = segs[:, 1] - segs[:, 0]
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
    segs[:, 0] -= 0.002 * u                    # tiny overlap hides seams at joins
    segs[:, 1] += 0.002 * u
    lc = LineCollection(segs,
                        colors=field_cmap(st)(norm(np.clip(I[order], st["decades_floor"], 1))),
                        linewidths=np.maximum(d["thickness"][order] * ax_w_pt, st["min_lw"]),
                        capstyle="butt", zorder=2)
    ax.add_collection(lc)
    lc.set_clip_path(mpatches.Rectangle((0, 0), 1, 1, transform=ax.transData))
    draw_sample_frame(ax, st)
    # Battery terminals: top face at the higher voltage
    for y, sign in ((1.0 + st["bar_h"] / 2, "+"), (-st["bar_h"] / 2, "−")):
        ax.text(-0.045, y, sign, fontsize=16, fontweight="semibold", color=st["ink"],
                ha="right", va="center", clip_on=False)


def draw_stress(ax, d, st):
    bare_axes(ax, (-0.03, 1.03))
    nodes = d["mesh_nodes"].astype(float)
    stress = np.clip(d["stress"].astype(float), st["decades_floor"], 1)
    norm = mcolors.LogNorm(st["decades_floor"], 1.0)
    ax.tripcolor(nodes[:, 0], nodes[:, 1], d["mesh_elements"], stress, shading="gouraud",
                 cmap=field_cmap(st), norm=norm, zorder=2)
    draw_sample_frame(ax, st)
    # As in the FEM: the top plate is pushed down and the bottom plate is held
    # fixed, drawn as a support (hatching) so nothing hangs below the panel
    for x in (0.22, 0.5, 0.78):
        ax.annotate("", xy=(x, 1.0 + st["bar_h"] + 0.005), xytext=(x, 1.0 + st["bar_h"] + 0.09),
                    arrowprops=dict(arrowstyle="-|>", color=st["ink_2"],
                                    lw=st["spine_lw"] * 1.6, mutation_scale=11),
                    annotation_clip=False)
    y0 = -st["bar_h"]
    for x in np.arange(0.0, 1.03, 0.06):
        ax.plot([x, x - 0.03], [y0, y0 - 0.035], color=st["ink_2"],
                linewidth=st["spine_lw"], clip_on=False, solid_capstyle="butt")


def style_axes(ax, st):
    ax.set_facecolor(st["bg"])
    ax.tick_params(colors=st["ink"], labelsize=FS_TICK, width=st["spine_lw"], length=3,
                   pad=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(st["ink"])
        ax.spines[side].set_linewidth(st["spine_lw"])


def series_handle(colour, marker, st, band=False):
    """Legend key: the line and marker, over its interquartile band if any."""
    line = Line2D([], [], color=colour, linewidth=st["data_lw"], marker=marker,
                  markersize=st["ms"], markerfacecolor=colour, markeredgecolor=st["bg"],
                  markeredgewidth=st["mew"])
    if not band:
        return line
    patch = mpatches.Patch(facecolor=colour, alpha=st["band_alpha"], edgecolor="none")
    return (patch, line)


def draw_legend(ax, handles, labels, loc):
    leg = ax.legend(handles, labels, loc=loc, frameon=False, fontsize=FS_LEGEND,
                    handlelength=1.8, handleheight=1.0, handletextpad=0.5, labelspacing=0.2,
                    borderaxespad=0.2, handler_map={tuple: HandlerTuple(ndivide=1)})
    for txt in leg.get_texts():
        txt.set_color(ax.xaxis.label.get_color())


def draw_profiles(ax, d, st):
    style_axes(ax, st)
    angles = d["angles"]
    shown = int(d["shown_angle"])
    th = np.r_[angles, 180]                    # 180 degrees is the 0-degree sample
    for vals, colour, marker in ((d["conductance"], st["main"], "o"),
                                 (d["stiffness"], st["accent"], "s")):
        rel = np.r_[vals, vals[0]] / vals.max()
        ax.plot(th, rel, color=colour, linewidth=st["data_lw"], marker=marker,
                markersize=st["ms"], markerfacecolor=colour, markeredgecolor=st["bg"],
                markeredgewidth=st["mew"], zorder=3)
    ax.axvline(shown, color=st["ink_2"], linewidth=st["spine_lw"],
               linestyle=(0, (4, 3)), zorder=1)
    ax.text(shown + 3, 0.0225, "sample\nshown", fontsize=FS_TICK, color=st["ink_2"],
            ha="left", va="bottom", linespacing=0.75)
    ax.set_yscale("log")
    ax.set_ylim(0.02, 1.6)
    ax.set_yticks([0.03, 0.1, 0.3, 1])
    ax.set_yticklabels(["0.03", "0.1", "0.3", "1"])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(-6, 186)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xticklabels(["0°", "45°", "90°", "135°", "180°"])
    ax.set_xlabel(r"cut angle, $\theta$", fontsize=FS_LABEL, color=st["ink"])
    ax.set_ylabel("relative to largest value", fontsize=FS_LABEL, color=st["ink"])
    draw_legend(ax, [series_handle(st["main"], "o", st), series_handle(st["accent"], "s", st)],
                ["conductance", "stiffness"], "upper left")


def draw_anisotropy(ax, d, st):
    style_axes(ax, st)
    a = d["alphas"]
    ax.axhline(1.0, color=st["ink_2"], linestyle=(0, (4, 3)), linewidth=st["spine_lw"])
    ax.text(0.16, 1.1, "isotropic", ha="left", va="bottom", fontsize=FS_TICK,
            color=st["ink_2"])
    for q, colour, marker in ((d["stiffness_ratio_quartiles"], st["accent"], "s"),
                              (d["conductance_ratio_quartiles"], st["main"], "o")):
        ax.fill_between(a, q[:, 0], q[:, 2], color=colour, alpha=st["band_alpha"],
                        linewidth=0, zorder=1)
        ax.plot(a, q[:, 1], color=colour, linewidth=st["data_lw"], marker=marker,
                markersize=st["ms"], markerfacecolor=colour, markeredgecolor=st["bg"],
                markeredgewidth=st["mew"], zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(0.8, 200)
    ax.set_yticks([1, 10, 100])
    ax.set_yticklabels(["1", "10", "100"])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(0.12, 1.08)
    ax.set_xticks(a)
    ax.set_xticklabels([f"{v:.1f}" for v in a])
    ax.set_xlabel(r"heterogeneity, $\alpha$", fontsize=FS_LABEL, color=st["ink"])
    ax.set_ylabel("largest / smallest", fontsize=FS_LABEL, color=st["ink"])
    draw_legend(ax, [series_handle(st["main"], "o", st, band=True),
                     series_handle(st["accent"], "s", st, band=True)],
                ["conductance", "stiffness"], "upper left")


# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------

def make_figure(filename, st, d):
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=st["bg"])

    # Top row: one sample (cut, current, stress), three equal squares
    w = 0.29                                   # square panel width (figure fraction)
    h = w * FIG_W / FIG_H
    gap = (1 - 3 * w - 0.02) / 2
    title1, top1 = 0.955, 0.91
    xs = [0.01 + i * (w + gap) for i in range(3)]
    ax_cut = fig.add_axes([xs[0], top1 - h, w, h])
    ax_cur = fig.add_axes([xs[1], top1 - h, w, h])
    ax_str = fig.add_axes([xs[2], top1 - h, w, h])
    for ax in (ax_cut, ax_cur, ax_str):
        ax.set_facecolor(st["bg"])
    draw_cut(ax_cut, d, st)
    draw_current(ax_cur, d, st)
    draw_stress(ax_str, d, st)

    # One colour key for both fields, in the gap between their panels
    pb, pc = ax_cur.get_position(), ax_str.get_position()
    kx = (pb.x1 + pc.x0) / 2
    key = fig.add_axes([kx - 0.006, pb.y0 + 0.30 * pb.height, 0.012, 0.40 * pb.height])
    key.imshow(np.linspace(1, 0, 256)[:, None], aspect="auto", extent=(0, 1, 0, 1),
               cmap=field_cmap(st))
    key.set_xticks([])
    key.set_yticks([])
    for sp in key.spines.values():
        sp.set_visible(False)
    key.text(0.5, 1.04, "high", transform=key.transAxes, ha="center", va="bottom",
             fontsize=FS_TICK, color=st["ink_2"])
    key.text(0.5, -0.04, "low", transform=key.transAxes, ha="center", va="top",
             fontsize=FS_TICK, color=st["ink_2"])

    # Bottom row: results, on its own two-column grid (so its titles do not sit
    # under the top row's panels) and separated by a clear band of space
    bottom2, h2 = 0.08, 0.30
    title2 = bottom2 + h2 + 0.035
    ax_prof = fig.add_axes([0.085, bottom2, 0.40, h2])
    ax_ani = fig.add_axes([0.615, bottom2, 0.37, h2])
    draw_profiles(ax_prof, d, st)
    draw_anisotropy(ax_ani, d, st)

    titles = [(ax_cut, r"Sample cut at angle $\theta$", title1),
              (ax_cur, "Electrical current", title1),
              (ax_str, "Compressive stress", title1),
              (ax_prof, "This network at every angle", title2),
              (ax_ani, r"Anisotropy vs. $\alpha$", title2)]
    for ax, text, y in titles:
        pos = ax.get_position()
        fig.text(pos.x0 + pos.width / 2, y, text, ha="center", va="baseline",
                 fontsize=FS_TITLE, fontweight="semibold", color=st["ink"])

    fig.savefig(filename, dpi=DPI, facecolor=st["bg"])
    plt.close(fig)
    print(f"Saved: {filename}")


NORMAL_STYLE = asset_style("sr_transport_stiffness", "normal")
ACCESSIBLE_STYLE = asset_style("sr_transport_stiffness", "accessible")

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default=OUT_DIR)
    args = parser.parse_args()
    with np.load(data_path("sr_transport_stiffness"), allow_pickle=False) as data:
        d = {key: data[key] for key in data.files}
    make_figure(os.path.join(args.out_dir, "sr_transport_stiffness.png"), NORMAL_STYLE, d)
    make_figure(os.path.join(args.out_dir, "sr_transport_stiffness_accessible.png"), ACCESSIBLE_STYLE, d)


if __name__ == "__main__":
    main()
