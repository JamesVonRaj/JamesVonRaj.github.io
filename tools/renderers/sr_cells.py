"""Render sr cells with the shared website theme.

Scientific inputs are frozen in tools/data/sr_cells.npz.
Use tools/theme.py to rebuild both display modes into a preview directory.
Input provenance and refresh instructions are in docs/THEMING.md.
"""

from site_theme.theme import asset_style, data_path, output_dir

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection

NORMAL = asset_style("sr_cells", "normal")
ACCESS = asset_style("sr_cells", "accessible")

# Scale-rich construction

# Banner: an x-range of the full square and a y-range a quarter of its height (4:1)
STRIP = ((0.0, 1.0), (0.375, 0.625))
FIG_W, FIG_H, DPI = 8.0, 2.0, 200    # 1600 x 400 px


def render(path, geometry, fills, line_color, min_line_pt):
    lig_polys, centres, cell_polys, colour_idx = geometry
    colours = [fills[i] for i in colour_idx]
    (x0, x1), (y0, y1) = STRIP
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor=line_color)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_axis_off()
    ax.add_collection(PolyCollection(cell_polys, facecolors=colours, edgecolors=colours,
                                     linewidths=0.3, zorder=1))
    ax.add_collection(PolyCollection(lig_polys, facecolors=line_color, edgecolors="none", zorder=2))
    ax.add_collection(LineCollection(centres, colors=line_color, linewidths=min_line_pt,
                                     capstyle="butt", zorder=3))
    fig.savefig(path, dpi=DPI, facecolor=line_color)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default=str(output_dir()))
    args = parser.parse_args()
    with np.load(data_path("sr_cells"), allow_pickle=False) as data:
        def polygons(key):
            offsets = data[key + "_offsets"]
            return [data[key][a:b] for a, b in zip(offsets[:-1], offsets[1:])]
        geometry = (polygons("ligaments"), data["centres"], polygons("cells"), data["color_indices"])
    for mode, style in (("", NORMAL), ("_accessible", ACCESS)):
        render(os.path.join(args.out_dir, "sr_cells" + mode + ".png"), geometry,
               style["fills"], style["line"], style["line_width"])


if __name__ == "__main__":
    main()
