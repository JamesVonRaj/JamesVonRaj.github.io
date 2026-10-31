"""Render huppi comparison with the shared website theme.

Scientific inputs are frozen in tools/data/huppi_comparison.npz.
Use tools/theme.py to rebuild both display modes into a preview directory.
Input provenance and refresh instructions are in docs/THEMING.md.
"""

from site_theme.theme import asset_style, data_path, output_dir, font_files


import os

# One BLAS thread per worker process; the parallelism is across realizations.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.collections as mc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter

DEFAULT_OUTPUT_DIR = output_dir()

# Use at most 8 cores to be respectful on shared machine

# Large point patterns for good S(k) statistics
SK_SIZE = 150        # 150x150 = 22500 points per realization
SK_C = 1.0
SK_N_MAX = 60        # wavevectors k = 2 pi (n_x, n_y) / L with |n_x|, |n_y| <= SK_N_MAX

# Smaller patterns for the visual panels (readable at figure size)

# --------------------------------------------------------------------------
NORMAL = asset_style("huppi_comparison", "normal")
ACCESSIBLE = asset_style("huppi_comparison", "accessible")

TITLE_SIZE = 16
LABEL_SIZE = 15
TICK_SIZE = 13


def setup_fonts():
    """Register Poppins explicitly so the figure does not depend on the font cache."""
    for f in font_files():
        fm.fontManager.addfont(f)
    plt.rcParams["font.family"] = "Poppins"
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Poppins"
    plt.rcParams["mathtext.it"] = "Poppins:italic"
    plt.rcParams["mathtext.bf"] = "Poppins:bold"
    # Poppins has no arrows; take \to from STIX sans rather than Computer Modern.
    plt.rcParams["mathtext.fallback"] = "stixsans"
    plt.rcParams["axes.unicode_minus"] = True


# --------------------------------------------------------------------------
# Visual point patterns and networks (drawn once, shared by both versions)
# --------------------------------------------------------------------------


def network_segments(nodes, edges, box_size):
    """Edge segments, skipping the long periodic-boundary edges."""
    lines = []
    for e in edges:
        p1, p2 = nodes[int(e[0])], nodes[int(e[1])]
        if np.linalg.norm(p1 - p2) < box_size * 0.5:
            lines.append([p1, p2])
    return lines


def draw_network(ax, nodes, edges, color, pal, box_size):
    """Draw network edges (uniform width) and nodes in the row colour."""
    lc = mc.LineCollection(network_segments(nodes, edges, box_size), colors=color,
                           linewidths=0.6 * pal["lw"], alpha=pal["edge_alpha"],
                           zorder=1, capstyle="round")
    ax.add_collection(lc)
    ax.scatter(nodes[:, 0], nodes[:, 1], s=pal["node_size"], c=color,
               linewidths=0, zorder=2)


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def _style_pattern_axis(ax, pal, box_size):
    ax.set_facecolor(pal["surface"])
    pad = 0.4
    ax.set_xlim(-pad, box_size + pad)
    ax.set_ylim(-pad, box_size + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _style_sk_axis(ax, pal):
    ax.set_facecolor(pal["surface"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(pal["ink_2"])
        ax.spines[side].set_linewidth(0.8 * pal["lw"])
    ax.tick_params(which="both", colors=pal["ink_2"], labelcolor=pal["ink_2"],
                   labelsize=TICK_SIZE, width=0.8 * pal["lw"])
    ax.tick_params(which="major", length=5)
    ax.tick_params(which="minor", length=2.5)
    ax.yaxis.set_major_locator(FixedLocator([1e-4, 1e-2, 1e0]))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(1.0,), numticks=12))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_locator(FixedLocator([1e-1, 1e0]))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=12))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _plot_sk(ax, k, S, S_std, color, pal, k_max):
    keep = k <= k_max
    k, S, S_std = k[keep], S[keep], S_std[keep]
    lower = np.clip(S - S_std, 1e-7, None)
    ax.fill_between(k, lower, S + S_std, color=color, alpha=pal["band_alpha"],
                    linewidth=0, zorder=1)
    ax.plot(k, S, "-", color=color, linewidth=2.0 * pal["lw"], zorder=2,
            solid_capstyle="round")


def make_figure(pal, filename, patterns, sk):
    """Generate the 2x3 comparison figure with a given palette."""
    box_size = patterns["box_size"]

    # Manual layout in inches: two square pattern columns and a wider S(k) column.
    W = 10.0
    left, right = 0.08, 0.12
    gap_pat = 0.28              # between the two pattern columns
    gap_sk = 0.98               # before the S(k) column (room for its y labels)
    a = 2.42                    # square pattern panels
    b = W - left - right - 2 * a - gap_pat - gap_sk   # S(k) panel width
    top_title = 0.44            # room for the top-row titles
    row_gap = 0.52              # between rows (bottom-row titles)
    bottom = 0.64               # room for the shared x label
    H = top_title + a + row_gap + a + bottom

    fig = plt.figure(figsize=(W, H), facecolor=pal["surface"])

    def add_ax(x_in, y_in, w_in, h_in):
        return fig.add_axes([x_in / W, y_in / H, w_in / W, h_in / H])

    xs = [left, left + a + gap_pat, left + 2 * a + gap_pat + gap_sk]
    ys = [bottom + a + row_gap, bottom]          # top row, bottom row
    axes = np.empty((2, 3), dtype=object)
    for r in range(2):
        axes[r, 0] = add_ax(xs[0], ys[r], a, a)
        axes[r, 1] = add_ax(xs[1], ys[r], a, a)
    axes[1, 2] = add_ax(xs[2], ys[1], b, a)
    axes[0, 2] = add_ax(xs[2], ys[0], b, a)
    axes[0, 2].sharex(axes[1, 2])
    axes[0, 2].sharey(axes[1, 2])

    title_kw = dict(fontsize=TITLE_SIZE, fontweight="medium", color=pal["ink"], pad=9)
    label_kw = dict(fontsize=LABEL_SIZE, color=pal["ink"])

    for r in range(2):
        for c in range(2):
            _style_pattern_axis(axes[r, c], pal, box_size)
        _style_sk_axis(axes[r, 2], pal)

    rows = [
        dict(points=patterns["po_points"], nodes=patterns["po_nodes"],
             edges=patterns["po_edges"], color=pal["poisson"],
             titles=("Poisson points", "PoPPI network",
                     r"$S(k) \to \mathrm{const}$ as $k \to 0$"),
             k=sk["k_po"], S=sk["S_po"], S_std=sk["S_po_std"]),
        dict(points=patterns["hu_points"], nodes=patterns["hu_nodes"],
             edges=patterns["hu_edges"], color=pal["hu"],
             titles=("Hyperuniform points", "HuPPI network",
                     r"$S(k) \to 0$ as $k \to 0$"),
             k=sk["k_hu"], S=sk["S_hu"], S_std=sk["S_hu_std"]),
    ]

    # Plot S(k) only over complete |k| shells (beyond 2 pi n_max / L the square
    # reciprocal grid covers just the corners of each shell).
    k_max = 2 * np.pi * SK_N_MAX / (SK_SIZE * SK_C) * 1.0001

    for r, row in enumerate(rows):
        ax_pts, ax_net, ax_sk = axes[r]
        ax_pts.scatter(row["points"][:, 0], row["points"][:, 1], s=pal["point_size"],
                       c=row["color"], linewidths=0, zorder=2)
        draw_network(ax_net, row["nodes"], row["edges"], row["color"], pal, box_size)
        _plot_sk(ax_sk, row["k"], row["S"], row["S_std"], row["color"], pal, k_max)
        for ax, title in zip(axes[r], row["titles"]):
            ax.set_title(title, **title_kw)
        # The S(k) titles mix text and mathtext; mathtext is regular weight, so
        # set the text regular too and the title reads as one weight.
        ax_sk.title.set_fontweight("normal")
        ax_sk.set_ylabel(r"$S(k)$", labelpad=6, **label_kw)

    axes[1, 2].set_xlabel(r"$k$", labelpad=4, **label_kw)
    axes[0, 2].tick_params(labelbottom=False)

    k_lo = min(sk["k_po"].min(), sk["k_hu"].min())
    axes[1, 2].set_xlim(k_lo * 0.85, k_max * 1.1)
    axes[1, 2].set_ylim(1e-5, 6.0)

    fig.savefig(filename, dpi=230, facecolor=pal["surface"])
    plt.close(fig)
    print(f"Saved: {filename}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="directory for the two rendered PNG files")
    args = parser.parse_args()
    with np.load(data_path("huppi_comparison"), allow_pickle=False) as data:
        sk = {key[3:]: data[key] for key in data.files if key.startswith("sk_")}
        patterns = {key: data[key] for key in data.files if not key.startswith("sk_")}
    setup_fonts()
    args.output_dir.mkdir(parents=True, exist_ok=True)


    print("Generating normal figure...")
    make_figure(NORMAL, args.output_dir / "huppi_comparison.png", patterns, sk)
    print("Generating accessible figure...")
    make_figure(ACCESSIBLE, args.output_dir / "huppi_comparison_accessible.png", patterns, sk)
    print("Done!")


if __name__ == "__main__":
    main()
