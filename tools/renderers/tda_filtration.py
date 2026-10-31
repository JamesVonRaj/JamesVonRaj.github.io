"""Render tda filtration with the shared website theme.

Scientific inputs are frozen in tools/data/tda_filtration.npz.
Use tools/theme.py to rebuild both display modes into a preview directory.
Input provenance and refresh instructions are in docs/THEMING.md.
"""

from __future__ import annotations

from site_theme.theme import asset_style, data_path, output_dir, font_files


import argparse
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator, NullLocator
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist, squareform

OUT_DIR = output_dir()

# Point pattern, in units of the window height (the window is W x H = 4:3).
W, H = 4.0 / 3.0, 1.0

# Filtration and timing.
EPS_END = 0.10                  # final ball radius
FPS = 30
HOLD_START, MOTION, HOLD_END = 1.0, 12.0, 2.0   # seconds
EASE = 1.5                      # epsilon = EPS_END u^EASE for u in [0, 1]: slower early
N_DOTS = 20                     # direct counts, evenly spaced in (0, EPS_END]
COLOUR_POWER = 0.7              # edges and triangles: colour = (entry radius / EPS_END)^COLOUR_POWER
VOID_LO_QUANTILE = 0.1          # void colour scale: from this quantile of all void areas to the largest

HEAD_MIN_WEIGHT = 100           # merge-history stems get a dot at the top when a b >= this
POSTER_FRAME = 300              # the posters show this frame (epsilon about 0.065, voids still open)

# Figure: 10 x 5 in at 160 dpi = 1600 x 800 px.
FIG_W, FIG_H, DPI = 10.0, 5.0, 160

STYLES = {mode: asset_style("tda_filtration", mode) for mode in ("normal", "accessible")}

def setup_fonts() -> None:
    """Register Poppins explicitly, so the figure does not depend on the font cache."""
    for f in font_files():
        fm.fontManager.addfont(f)
    # Poppins has no Greek; DejaVu Sans supplies the epsilons.
    plt.rcParams["font.family"] = ["Poppins", "DejaVu Sans"]
    plt.rcParams["mathtext.fontset"] = "custom"
    plt.rcParams["mathtext.rm"] = "Poppins"
    plt.rcParams["mathtext.it"] = "Poppins:italic"
    plt.rcParams["mathtext.bf"] = "Poppins:bold"
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


# ----------------------------------------------------------------------------- data


def tri_area(p: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                        - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))


class Filtration:
    """The Delaunay-Rips filtration of a point pattern, precomputed once."""

    def __init__(self, pts: np.ndarray):
        self.pts = pts
        self.n = len(pts)
        self.dist = squareform(pdist(pts))   # every length below comes from this matrix
        tri = Delaunay(pts)
        s = tri.simplices
        self.simplices = s
        # Sides of each triangle; side k is opposite vertex k.
        ends = [(1, 2), (0, 2), (0, 1)]
        sides = np.column_stack([self.dist[s[:, i], s[:, j]] for i, j in ends])
        self.tri_birth = sides.max(axis=1) / 2.0
        self.tri_area = tri_area(pts[s])
        # Delaunay edges, each once.
        pairs = np.vstack([np.sort(s[:, [i, j]], axis=1) for i, j in ends])
        self.edges = np.unique(pairs, axis=0)
        self.edge_birth = self.dist[self.edges[:, 0], self.edges[:, 1]] / 2.0
        # Dual graph: pairs of triangles sharing an edge, and triangles on the hull,
        # each with the length of the edge in question.
        t_idx, k_idx = np.nonzero(tri.neighbors >= 0)
        u_idx = tri.neighbors[t_idx, k_idx]
        keep = t_idx < u_idx
        self.adj_t, self.adj_u = t_idx[keep], u_idx[keep]
        self.adj_len = sides[t_idx[keep], k_idx[keep]]
        t_out, k_out = np.nonzero(tri.neighbors < 0)
        self.hull_t, self.hull_len = t_out, sides[t_out, k_out]
        # Drawing order: by entry radius, so the complex at epsilon is a prefix.
        self.tri_order = np.argsort(self.tri_birth, kind="stable")
        self.edge_order = np.argsort(self.edge_birth, kind="stable")

    def voids(self, eps: float) -> list[np.ndarray]:
        """The voids at radius eps: bounded components of the complement, as triangle sets.

        Two triangles outside the complex are joined when the Delaunay edge they share is
        outside the complex too (longer than 2 eps; a triangle with such a side is never
        in the complex).  A component is a void unless it reaches the convex hull through
        an edge outside the complex.
        """
        m = len(self.simplices)
        in_cx = self.tri_birth <= eps
        open_edge = self.adj_len > 2.0 * eps
        graph = coo_matrix((np.ones(open_edge.sum()), (self.adj_t[open_edge], self.adj_u[open_edge])),
                           shape=(m, m))
        _, lab = connected_components(graph, directed=False)
        outer_labels = np.unique(lab[self.hull_t[self.hull_len > 2.0 * eps]])
        free = ~in_cx & ~np.isin(lab, outer_labels)
        return [np.flatnonzero(lab == c) for c in np.unique(lab[free])]

    def void_pairs(self) -> np.ndarray:
        """(birth, death) radius of every void (H1 class), by duality with the complement.

        Run the filtration backwards: removing a triangle creates a component of the
        complement, and removing an edge joins the two faces on either side of it (a
        triangle or the outside).  When two components join, the one created at the
        smaller radius (the younger one; the outside is the oldest) is a void, which in
        the forward filtration opens when that edge enters and fills when its last
        triangle enters.  Zero-length pairs (a triangle entering with its longest side)
        are dropped.
        """
        m = len(self.simplices)
        outer = m
        parent = np.arange(m + 1)
        created = np.append(self.tri_birth, np.inf)

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        # Edge radius and the faces on either side: interior edges, then hull edges.
        e_rad = np.concatenate([self.adj_len, self.hull_len]) / 2.0
        e_t = np.concatenate([self.adj_t, self.hull_t])
        e_u = np.concatenate([self.adj_u, np.full(len(self.hull_t), outer)])
        # Triangles are all present before any edge joins them (a triangle enters no
        # earlier than its sides), so only the edges need ordering.
        pairs = []
        for k in np.argsort(-e_rad, kind="stable"):
            a, b = find(e_t[k]), find(e_u[k])
            if a == b:
                continue
            if created[a] < created[b]:
                a, b = b, a           # b is the younger component
            if created[b] > e_rad[k]:
                pairs.append((e_rad[k], created[b]))
            parent[b] = a
        return np.array(pairs)

    def h0_merge_history(self) -> tuple[np.ndarray, np.ndarray]:
        """Radii of the H0 mergers and the weight a b of each (Kruskal with union-find).

        The edge between points at distance d enters at radius d / 2.  A merger of
        components with a and b points connects a b pairs of points for the first time.
        """
        parent = np.arange(self.n)
        size = np.ones(self.n, dtype=np.int64)

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        radii, weights = [], []
        for k in self.edge_order:
            a, b = find(self.edges[k, 0]), find(self.edges[k, 1])
            if a == b:
                continue
            radii.append(self.edge_birth[k])
            weights.append(size[a] * size[b])
            if size[a] < size[b]:
                a, b = b, a
            parent[b] = a
            size[a] += size[b]
        return np.array(radii), np.array(weights, dtype=np.int64)

    def sum_sq_direct(self, eps: np.ndarray) -> np.ndarray:
        """sum_i s_i^2 at each radius, from the components of the full Rips 1-skeleton."""
        out = []
        for e in eps:
            _, labels = connected_components(self.dist <= 2.0 * e, directed=False)
            out.append(int(np.sum(np.bincount(labels).astype(np.int64) ** 2)))
        return np.array(out, dtype=np.int64)


def sum_sq_from_merges(radii: np.ndarray, weights: np.ndarray, n: int, eps: np.ndarray) -> np.ndarray:
    """sum_i s_i^2 at each radius in `eps`, from the weighted merge history (integers)."""
    cumulative = np.concatenate([[0], np.cumsum(weights)])
    n_merged = np.searchsorted(radii, eps, side="right")  # mergers with radius <= eps
    return n + 2 * cumulative[n_merged]


def frame_radii() -> np.ndarray:
    """Epsilon at every frame: a hold at 0, an eased rise to EPS_END, a hold at EPS_END."""
    n_motion = round(MOTION * FPS)
    u = np.linspace(0.0, 1.0, n_motion)
    rise = EPS_END * u**EASE
    return np.concatenate([np.zeros(round(HOLD_START * FPS)), rise,
                           np.full(round(HOLD_END * FPS), EPS_END)])


def precompute(filt: Filtration) -> dict:
    """Everything the frames need, shared by both versions."""
    n = filt.n
    radii, weights = filt.h0_merge_history()
    assert len(radii) == n - 1, "the Delaunay graph should be connected"
    assert radii[-1] <= EPS_END, "every cluster should have merged by EPS_END"

    dot_eps = EPS_END * np.arange(1, N_DOTS + 1) / N_DOTS
    check = np.unique(np.concatenate([np.linspace(0.0, EPS_END, 2401), radii, dot_eps]))
    from_merges = sum_sq_from_merges(radii, weights, n, check)
    direct = filt.sum_sq_direct(check)
    assert np.array_equal(from_merges, direct), "merge history and direct count disagree"
    dot_direct = filt.sum_sq_direct(dot_eps)
    assert np.array_equal(dot_direct, sum_sq_from_merges(radii, weights, n, dot_eps))

    # Voids two ways: frame by frame (drawn) and as persistence pairs (counted).
    pairs = filt.void_pairs()

    eps = frame_radii()
    frames = []
    void_areas = []
    for e in eps:
        voids = filt.voids(e)
        areas = np.array([filt.tri_area[v].sum() for v in voids])
        void_areas.append(areas)
        alive = int(np.sum((pairs[:, 0] <= e) & (e < pairs[:, 1])))
        assert alive == len(voids), f"void count mismatch at epsilon = {e}: {alive} vs {len(voids)}"
        frames.append({
            "eps": e,
            "n_tri": int(np.searchsorted(filt.tri_birth[filt.tri_order], e, side="right")),
            "n_edge": int(np.searchsorted(filt.edge_birth[filt.edge_order], e, side="right")),
            "n_merge": int(np.searchsorted(radii, e, side="right")),
            "n_dot": int(np.searchsorted(dot_eps, e, side="right")),
            "voids": voids,
            "void_areas": areas,
        })
    # One log-area scale for the whole movie; the smallest voids are clipped to its low end.
    all_areas = np.concatenate([a for a in void_areas if len(a)])
    log_lo, log_hi = np.log(np.quantile(all_areas, VOID_LO_QUANTILE)), np.log(all_areas.max())
    for f in frames:
        t = (np.log(f["void_areas"]) - log_lo) / (log_hi - log_lo) if len(f["voids"]) else np.array([])
        f["void_t"] = np.clip(t, 0.0, 1.0)
    return {
        "radii": radii,
        "weights": weights,
        "s_after": sum_sq_from_merges(radii, weights, n, radii) / n,   # S just after each merger
        "dot_eps": dot_eps,
        "dot_s": dot_direct / n,
        "n_checked": len(check),
        "frames": frames,
        "area_range": (all_areas.min(), all_areas.max()),
        "void_pairs": pairs,
    }


# -------------------------------------------------------------------------- drawing


def style_axis(ax, style: dict) -> None:
    lw = 0.8 * style["scale"]
    ax.set_facecolor(style["bg"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(style["ink_2"])
        ax.spines[side].set_linewidth(lw)
    ax.tick_params(axis="both", which="both", length=3.5, width=lw, pad=3, color=style["ink_2"],
                   labelcolor=style["ink_2"], labelsize=12)


class Movie:
    """A figure whose artists are updated in place, frame by frame.

    Layout, in inches of the 10 x 5 figure: the complex fills a 4:3 window 5.9 in wide
    on the left (about 60 percent of the width), with the epsilon readout below its left
    corner; the two plots share the right, with room between them.
    """

    def __init__(self, filt: Filtration, data: dict, style: dict):
        self.filt, self.data, self.style = filt, data, style
        sc, bg = style["scale"], style["bg"]
        pts, s = filt.pts, filt.simplices
        self.fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor=bg)
        fig = self.fig

        def axes(left, bottom, width, height):
            return fig.add_axes((left / FIG_W, bottom / FIG_H, width / FIG_W, height / FIG_H))

        # Left: the complex.
        win_left, win_bottom, win_w = 0.1, 0.46, 5.9
        win_h = win_w * H / W
        ax = axes(win_left, win_bottom, win_w, win_h)
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect("equal")
        ax.set_axis_off()
        tri_cmap = LinearSegmentedColormap.from_list("entry", style["tri_ramp"])
        self.void_cmap = LinearSegmentedColormap.from_list("area", style["void_ramp"])
        self.tri_verts = pts[s[filt.tri_order]]
        tri_t = np.clip(filt.tri_birth[filt.tri_order] / EPS_END, 0, 1)
        self.tri_colours = tri_cmap(tri_t**COLOUR_POWER)
        self.edge_segs = pts[filt.edges[filt.edge_order]]
        edge_t = np.clip(filt.edge_birth[filt.edge_order] / EPS_END, 0, 1)
        self.edge_colours = tri_cmap(edge_t**COLOUR_POWER)
        # Each void triangle is also stroked in its own colour, which hides antialiasing seams.
        self.voids = PolyCollection([], linewidths=0.6 * sc, zorder=1)
        self.edge_lines = LineCollection([], linewidths=0.75 * sc, capstyle="round", zorder=2)
        # Triangles are stroked in the background colour: the stained-glass gaps.
        self.tris = PolyCollection([], edgecolors=bg, linewidths=0.9 * sc, zorder=3)
        for artist in (self.voids, self.edge_lines, self.tris):
            ax.add_collection(artist)
        ax.scatter(pts[:, 0], pts[:, 1], s=4.0 * sc, c=style["ink"], lw=0, zorder=4)
        self.readout = fig.text(win_left / FIG_W, 0.13 / FIG_H, "", fontsize=14,
                                color=style["ink_2"], ha="left", va="baseline")

        # Right: merge history (top) and mean cluster size (bottom), sharing epsilon.
        right_left, right_w = 6.86, 2.92
        title_kw = {"fontsize": 16, "fontweight": "semibold", "color": style["ink"], "loc": "left", "pad": 8}
        top = axes(right_left, 3.08, right_w, 1.42)
        bot = axes(right_left, 0.66, right_w, 1.3)
        for a in (top, bot):
            style_axis(a, style)
            a.set_xlim(0.0, EPS_END)
        ticks = np.round(np.arange(0.0, EPS_END + 1e-9, 0.02), 2)
        for a in (top, bot):
            a.xaxis.set_major_locator(FixedLocator(ticks))
            a.xaxis.set_minor_locator(NullLocator())
        top.set_xticklabels([])
        bot.set_xticklabels([f"{t:g}" for t in ticks])

        # Merge history: thin stems coloured by a b on a log scale (light for small
        # mergers, dark for large), largest drawn last; heads only for a b >= HEAD_MIN_WEIGHT.
        w = data["weights"]
        top.set_yscale("log")
        top.set_ylim(0.6, 10 ** np.ceil(np.log10(w.max() * 1.3)))
        decades = np.arange(0, int(np.log10(top.get_ylim()[1])) + 1, 2)
        top.yaxis.set_major_locator(FixedLocator(10.0**decades))
        top.yaxis.set_minor_locator(NullLocator())
        top.set_yticklabels(["1" if d == 0 else f"$10^{{{d}}}$" for d in decades])
        top.set_ylabel("a·b", fontsize=14, color=style["ink"], labelpad=6, fontstyle="italic")
        top.set_title("Weighted merge history", **title_kw)
        stem_cmap = LinearSegmentedColormap.from_list("weight", style["stem_ramp"])
        self.stem_colours = stem_cmap(np.log(w) / np.log(w.max()))
        self.stem_base = top.get_ylim()[0]
        self.stems = LineCollection([], linewidths=0.8 * sc, zorder=2)
        top.add_collection(self.stems)
        self.heads = top.scatter([], [], s=(3.4 * sc**0.5) ** 2, lw=0, zorder=3)

        # Mean cluster size.
        n = filt.n
        bot.set_yscale("log")
        bot.set_ylim(0.8, n * 1.5)
        s_ticks = [1, 10, 100]
        bot.yaxis.set_major_locator(FixedLocator(s_ticks))
        bot.yaxis.set_minor_locator(NullLocator())
        bot.set_yticklabels([f"{t}" for t in s_ticks])
        bot.set_ylabel("S(ε)", fontsize=14, color=style["ink"], labelpad=6)
        bot.set_xlabel("ε", fontsize=14, color=style["ink"], labelpad=2)
        bot.set_title("Mean cluster size", **{**title_kw, "pad": 30})
        (self.curve,) = bot.plot([], [], color=style["curve"], lw=2.2 * sc, drawstyle="steps-post", zorder=2,
                                 label="merge history", solid_joinstyle="round", solid_capstyle="butt")
        (self.dots,) = bot.plot([], [], ls="none", marker="o", ms=5.5 if sc == 1.0 else 6.5,
                                mfc=style["count"], mec=bg, mew=0.8 * sc, zorder=3,
                                label="direct count", clip_on=False)
        # The legend is one row between the title and the plot, outside the axes, so
        # neither the curve nor the sweeping marker ever reaches it.
        legend = bot.legend(loc="lower left", bbox_to_anchor=(0.0, 1.05), ncol=2, fontsize=12, frameon=False,
                            handlelength=1.5, handletextpad=0.45, columnspacing=1.0, borderaxespad=0.0,
                            borderpad=0.0)
        for text in legend.get_texts():
            text.set_color(style["ink"])
        self.markers = [a.axvline(0.0, color=style["ink_2"], lw=0.8 * sc, zorder=1, alpha=0.7)
                        for a in (top, bot)]

    def update(self, frame: dict) -> None:
        d, eps = self.data, frame["eps"]
        k = frame["n_tri"]
        self.tris.set_verts(self.tri_verts[:k])
        self.tris.set_facecolor(self.tri_colours[:k])
        k = frame["n_edge"]
        self.edge_lines.set_segments(self.edge_segs[:k])
        self.edge_lines.set_color(self.edge_colours[:k])
        verts, colours = [], []
        s, pts = self.filt.simplices, self.filt.pts
        for v, t in zip(frame["voids"], frame["void_t"], strict=True):
            verts.append(pts[s[v]])
            colours.append(np.repeat([self.void_cmap(t)], len(v), axis=0))
        if verts:
            self.voids.set_verts(np.concatenate(verts))
            colours = np.concatenate(colours)
            self.voids.set_facecolor(colours)
            self.voids.set_edgecolor(colours)
        else:
            self.voids.set_verts([])
        self.readout.set_text(f"ε = {eps:.3f}")

        k = frame["n_merge"]
        order = np.argsort(d["weights"][:k], kind="stable")     # small first, large on top
        r, w = d["radii"][order], d["weights"][order]
        self.stems.set_segments([[(x, self.stem_base), (x, y)] for x, y in zip(r, w, strict=True)])
        self.stems.set_color(self.stem_colours[order])
        big = w >= HEAD_MIN_WEIGHT
        self.heads.set_offsets(np.column_stack([r[big], w[big]]) if big.any() else np.empty((0, 2)))
        self.heads.set_facecolor(self.stem_colours[order][big])
        x = np.concatenate([[0.0], d["radii"][:k], [eps]])
        y = np.concatenate([[1.0], d["s_after"][:k], [d["s_after"][k - 1] if k else 1.0]])
        self.curve.set_data(x, y)
        k = frame["n_dot"]
        self.dots.set_data(d["dot_eps"][:k], d["dot_s"][:k])
        for m in self.markers:
            m.set_xdata([eps, eps])


def render(filt: Filtration, data: dict, name: str, out_dir: Path, preview: Path | None,
           preview_frames: list[int], crf: int) -> list[str]:
    setup_fonts()
    style = STYLES[name]
    movie = Movie(filt, data, style)
    frames = data["frames"]
    written = []
    if preview is not None:
        preview.mkdir(parents=True, exist_ok=True)
        for i in preview_frames:
            movie.update(frames[i])
            path = preview / f"frame{style['suffix']}_{i:03d}.png"
            movie.fig.savefig(path, dpi=DPI, facecolor=style["bg"])
            written.append(str(path))
        return written
    ffmpeg = shutil.which("ffmpeg") or os.path.expanduser("~/.local/bin/ffmpeg")
    plt.rcParams["animation.ffmpeg_path"] = ffmpeg
    writer = FFMpegWriter(fps=FPS, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", "slow",
                                      "-movflags", "+faststart"])
    path = out_dir / f"tda_filtration{style['suffix']}.mp4"
    with writer.saving(movie.fig, str(path), dpi=DPI):
        for f in frames:
            movie.update(f)
            writer.grab_frame(facecolor=style["bg"])
    written.append(str(path))
    movie.update(frames[POSTER_FRAME])
    poster = out_dir / f"tda_filtration_poster{style['suffix']}.png"
    movie.fig.savefig(poster, dpi=DPI, facecolor=style["bg"])
    written.append(str(poster))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--preview", type=Path, default=None,
                        help="write a few frames as PNGs to this directory instead of the movies")
    parser.add_argument("--frames", type=int, nargs="*", default=None, help="frames to preview")
    parser.add_argument("--crf", type=int, default=23)
    args = parser.parse_args()

    pts = np.load(data_path("tda_filtration"), allow_pickle=False)["points"]
    filt = Filtration(pts)
    data = precompute(filt)
    frames = data["frames"]
    n_voids = np.array([len(f["voids"]) for f in frames])
    peak = int(n_voids.argmax())
    print(f"{filt.n} points, {len(filt.edges)} Delaunay edges, {len(filt.simplices)} Delaunay triangles")
    radii = data["radii"]
    print(f"{len(radii)} H0 mergers, from epsilon = {radii[0]:.4f} to {radii[-1]:.4f}; "
          f"largest a b = {data['weights'].max()}")
    print(f"merge history vs direct count at {data['n_checked']} radii in [0, {EPS_END:g}] and "
          f"{N_DOTS} dots: identical (integers)")
    pairs = data["void_pairs"]
    opened = pairs[pairs[:, 0] <= EPS_END]
    print(f"voids: {len(opened)} open by epsilon = {EPS_END:g} and {np.sum(opened[:, 1] <= EPS_END)} of them "
          f"fill by then; at most {n_voids.max()} at once (epsilon = {frames[peak]['eps']:.4f}), "
          f"{n_voids[-1]} at the end; void counts match the persistence pairs in every frame")
    print(f"void areas {data['area_range'][0]:.2e} to {data['area_range'][1]:.2e} (window area {W * H:.3f})")
    print(f"{len(frames)} frames at {FPS} fps = {len(frames) / FPS:g} s; poster = frame {POSTER_FRAME} "
          f"(epsilon = {frames[POSTER_FRAME]['eps']:.4f}, {n_voids[POSTER_FRAME]} voids); "
          f"{np.sum(data['weights'] >= HEAD_MIN_WEIGHT)} mergers with a b >= {HEAD_MIN_WEIGHT}")
    preview_frames = args.frames if args.frames else [0, 60, 120, 180, 240, 300, len(frames) - 1]
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(render, filt, data, name, args.out_dir, args.preview, preview_frames, args.crf)
                for name in STYLES]
        for job in jobs:
            for path in job.result():
                print("wrote", path)


if __name__ == "__main__":
    main()
