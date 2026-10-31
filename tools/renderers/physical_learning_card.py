"""Render physical learning card with the shared website theme.

Scientific inputs are frozen in tools/data/physical_learning_card.npz.
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
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

NPZ = data_path("physical_learning_card")
WEBSITE_IMAGES = str(output_dir())

NORMAL = asset_style("physical_learning_card", "normal")
ACCESS = asset_style("physical_learning_card", "accessible")
for _style in (NORMAL, ACCESS):
    _style["cmap"] = LinearSegmentedColormap.from_list("stiffness", _style.pop("ramp"))

# Banner frame: 8 x 2 in at 200 dpi = 1600 x 400 px (4:1).
FIG_W, FIG_H, DPI = 8.0, 2.0, 200
L = 20.0          # side of the periodic square
TILES = 1         # one periodic tile across the width
VIEW_H = L / 4.0  # strip height, so the view is L x L/4 (4:1)
R_MIN, R_MAX = 0.5, 2.0
LW_MIN, LW_MAX = 0.9, 3.4      # line width (pt) at R_MIN and R_MAX
NODE_SIZE = 7.0                # scatter size (pt^2); 0 omits the nodes


def load_network(path=NPZ):
    d = np.load(path)
    pos = d["pos"].astype(float)
    edges = d["edges"].astype(int)
    radii = d["radii"].astype(float)
    disp = pos[edges[:, 1]] - pos[edges[:, 0]]
    disp -= L * np.round(disp / L)                  # minimum image
    return pos, edges, radii, disp


def stiffness(radii):
    """Trained radius rescaled to [0, 1] (0 = softest, 1 = stiffest)."""
    return np.clip((radii - R_MIN) / (R_MAX - R_MIN), 0.0, 1.0)


def best_shift(pos, edges, disp, t, n=200, margin=2.0):
    """Shift of the periodic box that puts the frame edges in the quietest places.

    y: the top and bottom edges of the frame (one periodic seam) go where the larger of
    the two stiff-edge weights within ``margin`` inside the top edge and inside the bottom
    edge is smallest, so neither edge cuts through a cluster of stiff edges. x: the
    left/right seam goes where it cuts the fewest stiff edges.
    """
    w = t ** 2
    mid = (pos[edges[:, 0]] + 0.5 * disp) % L
    grid = np.linspace(0.0, L, n, endpoint=False)

    def edge_load(s):
        y = (mid[:, 1] - s) % L
        return max(np.sum(w[y < margin]), np.sum(w[y > L - margin]))

    def cut_cost(s):
        a = (pos[edges[:, 0], 0] - s) % L
        b = a + disp[:, 0]
        return np.sum(w[(b < 0.0) | (b >= L)])

    sx = grid[np.argmin([cut_cost(s) for s in grid])]
    sy = grid[np.argmin([edge_load(s) for s in grid])]
    return np.array([sx, sy])


def tiled_segments(pos, edges, disp, shift):
    """Segments of the 4 x 1 tiling, and the index of the edge each one draws."""
    p = (pos - shift) % L
    a = p[edges[:, 0]]
    b = a + disp
    crosses = np.any((b < 0.0) | (b >= L), axis=1)
    # Non-crossing edges once (from end 0); crossing edges from both ends.
    starts = np.vstack([a, p[edges[crosses, 1]]])
    ends = np.vstack([b, p[edges[crosses, 1]] - disp[crosses]])
    owner = np.concatenate([np.arange(len(edges)), np.flatnonzero(crosses)])
    segs, idx = [], []
    for k in range(-1, TILES + 1):                 # one extra tile each side for the seams
        off = np.array([k * L, 0.0])
        segs.append(np.stack([starts + off, ends + off], axis=1))
        idx.append(owner)
    segs = np.concatenate(segs)
    idx = np.concatenate(idx)
    # Keep only segments that touch the frame.
    lo = segs.min(axis=1)
    hi = segs.max(axis=1)
    vis = (hi[:, 0] >= 0) & (lo[:, 0] <= TILES * L) & (hi[:, 1] >= 0) & (lo[:, 1] <= L)
    nodes = np.vstack([p + [k * L, 0.0] for k in range(TILES)])
    return segs[vis], idx[vis], nodes


def best_strip(pos, edges, disp, t, shift, n=200, margin=0.6):
    """Bottom of the L x L/4 strip with the most stiff-edge weight inside it."""
    w = t ** 2
    mid_y = ((pos[edges[:, 0]] - shift) % L + 0.5 * disp)[:, 1]
    grid = np.linspace(0.0, L - VIEW_H, n)
    score = [np.sum(w[(mid_y > y0 + margin) & (mid_y < y0 + VIEW_H - margin)]) for y0 in grid]
    return grid[int(np.argmax(score))]


def render(out_path, palette, segs, t_seg, nodes, y0):
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor=palette["bg"])
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_facecolor(palette["bg"])
    ax.set_xlim(0.0, TILES * L)
    ax.set_ylim(y0, y0 + VIEW_H)
    ax.set_axis_off()

    order = np.argsort(t_seg, kind="stable")      # stiff edges on top
    t_o = t_seg[order]
    colors = palette["cmap"](t_o)
    widths = palette["lw"] * (LW_MIN + (LW_MAX - LW_MIN) * t_o)
    ax.add_collection(LineCollection(segs[order], colors=colors, linewidths=widths,
                                     capstyle="round", zorder=1))
    if NODE_SIZE > 0:
        ax.scatter(nodes[:, 0], nodes[:, 1], s=NODE_SIZE * palette["lw"],
                   c=palette["node"], linewidths=0, zorder=2)

    fig.savefig(out_path, dpi=DPI, facecolor=palette["bg"])
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default=WEBSITE_IMAGES)
    args = parser.parse_args()

    pos, edges, radii, disp = load_network()
    t = stiffness(radii)
    shift = best_shift(pos, edges, disp, t)
    segs, owner, nodes = tiled_segments(pos, edges, disp, shift)
    y0 = best_strip(pos, edges, disp, t, shift)
    t_seg = t[owner]
    print(f"  {len(pos)} nodes, {len(edges)} edges, radii {radii.min():.2f} to "
          f"{radii.max():.2f}; box shift {shift.round(2)}; {len(segs)} segments drawn")

    render(os.path.join(args.out_dir, "physical_learning_card.png"), NORMAL, segs, t_seg, nodes, y0)
    render(os.path.join(args.out_dir, "physical_learning_card_accessible.png"), ACCESS,
           segs, t_seg, nodes, y0)


if __name__ == "__main__":
    main()
