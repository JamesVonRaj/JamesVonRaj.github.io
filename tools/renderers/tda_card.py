"""Render tda card with the shared website theme.

Scientific inputs are frozen in tools/data/tda_card.npz.
Use tools/theme.py to rebuild both display modes into a preview directory.
Input provenance and refresh instructions are in docs/THEMING.md.
"""

from __future__ import annotations

from site_theme.theme import asset_style, data_path, output_dir


import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay

OUT_DIR = output_dir()

W, H = 4.0, 1.0                  # frame, in units of its height
FIG_W, FIG_H, DPI = 8.0, 2.0, 200
EPS = 0.058                      # ball radius of the complex

NORMAL = asset_style("tda_card", "normal")
ACCESS = asset_style("tda_card", "accessible")


def tri_area(p):
    return 0.5 * np.abs((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                        - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))


def complex_and_voids(pts, eps):
    """Delaunay triangulation, complex mask, entry scale of each triangle, exact voids."""
    tri = Delaunay(pts)
    s = tri.simplices
    side = lambda i, j: np.linalg.norm(pts[s[:, i]] - pts[s[:, j]], axis=1)
    sides = np.column_stack([side(1, 2), side(0, 2), side(0, 1)])   # side k is opposite vertex k
    birth = sides.max(axis=1) / 2
    in_cx = birth < eps
    rows, cols = [], []
    outer = np.zeros(len(s), bool)
    for t in np.flatnonzero(~in_cx):
        for k in range(3):
            if sides[t, k] < 2 * eps:
                continue                      # the shared edge is in the complex: a wall
            u = tri.neighbors[t, k]
            if u == -1:
                outer[t] = True               # open to the outside of the hull
            elif not in_cx[u]:
                rows.append(t)
                cols.append(u)
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(s), len(s)))
    _, lab = connected_components(graph, directed=False)
    voids = []
    for c in np.unique(lab[~in_cx]):
        members = np.flatnonzero((lab == c) & ~in_cx)
        if not outer[members].any():
            voids.append(members)
    return tri, in_cx, birth, voids


def render(path, style, pts, tri, in_cx, birth, voids):
    s = tri.simplices
    tri_cmap = LinearSegmentedColormap.from_list("entry", style["tri_ramp"])
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor=style["bg"])
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.add_collection(PolyCollection(pts[s[in_cx]], facecolors=tri_cmap(np.clip(birth[in_cx] / EPS, 0, 1)),
                                     edgecolors=style["bg"], linewidths=style["gap_lw"], zorder=1))
    areas = np.array([tri_area(pts[s[v]]).sum() for v in voids])
    la = np.log(areas)
    t = (la - la.min()) / max(la.max() - la.min(), 1e-9)
    void_cmap = LinearSegmentedColormap.from_list("area", style["void_ramp"])
    for v, x in zip(voids, t):
        colour = void_cmap(x)
        ax.add_collection(PolyCollection(pts[s[v]], facecolors=[colour], edgecolors=[colour],
                                         linewidths=0.3, zorder=1.2))
    fig.savefig(path, dpi=DPI, facecolor=style["bg"])
    plt.close(fig)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    pts = np.load(data_path("tda_card"), allow_pickle=False)["points"]
    tri, in_cx, birth, voids = complex_and_voids(pts, EPS)
    print(f"{len(pts)} points, {in_cx.sum()} triangles in the complex, {len(voids)} voids")
    render(args.out_dir / "tda_card.png", NORMAL, pts, tri, in_cx, birth, voids)
    render(args.out_dir / "tda_card_accessible.png", ACCESS, pts, tri, in_cx, birth, voids)


if __name__ == "__main__":
    main()
