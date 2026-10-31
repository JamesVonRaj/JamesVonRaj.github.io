"""Render huppi card with the shared website theme.

Scientific inputs are frozen in tools/data/huppi_card.npz.
Use tools/theme.py to rebuild both display modes into a preview directory.
Input provenance and refresh instructions are in docs/THEMING.md.
"""

from site_theme.theme import asset_style, data_path, output_dir


import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

DEFAULT_OUTPUT_DIR = output_dir()

FIG_W, FIG_H, DPI = 8.0, 2.0, 200   # 1600 x 400 px
VIEW_W, VIEW_H = 24.0, 6.0          # window, in lattice spacings (4:1)

NORMAL = asset_style("huppi_card", "normal")
ACCESS = asset_style("huppi_card", "accessible")


def render(path, style, nodes, edges):
    seg = nodes[edges]
    length = np.linalg.norm(seg[:, 1] - seg[:, 0], axis=1)
    lo, hi = np.percentile(length, 5), np.percentile(length, 97)
    t = np.clip((length - lo) / (hi - lo), 0, 1)
    order = np.argsort(t)                                    # long, dark edges on top
    cmap = LinearSegmentedColormap.from_list("len", style["ramp"])
    w0, w1 = style["lw"]
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor=style["bg"])
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, VIEW_W)
    ax.set_ylim(0, VIEW_H)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.add_collection(LineCollection(seg[order], colors=cmap(t[order]), linewidths=w0 + (w1 - w0) * t[order],
                                     capstyle="round", zorder=1))
    vis = (nodes[:, 0] > -0.5) & (nodes[:, 0] < VIEW_W + 0.5) & (nodes[:, 1] > -0.5) & (nodes[:, 1] < VIEW_H + 0.5)
    ax.scatter(*nodes[vis].T, s=style["node_size"], c=style["node"], linewidths=0, zorder=2)
    fig.savefig(path, dpi=DPI, facecolor=style["bg"])
    plt.close(fig)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    with np.load(data_path("huppi_card"), allow_pickle=False) as data:
        nodes, edges = data["nodes"], data["edges"]


    render(args.out_dir / "huppi_card.png", NORMAL, nodes, edges)
    render(args.out_dir / "huppi_card_accessible.png", ACCESS, nodes, edges)


if __name__ == "__main__":
    main()
