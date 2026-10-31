"""Render physical learning sweep with the shared website theme.

Scientific inputs are frozen in tools/data/physical_learning_sweep.npz.
Use tools/theme.py to rebuild both display modes into a preview directory.
Input provenance and refresh instructions are in docs/THEMING.md.
"""
from __future__ import annotations

from site_theme.theme import asset_style, data_path, output_dir, font_files


import argparse
import logging
import os
from pathlib import Path
from types import SimpleNamespace

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")                       # small dense algebra; don't spin up every core


import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator  # noqa: E402

from _oblique import Oblique
FPS = 30
SITE_IMAGES = output_dir()
STEM = "physical_learning_sweep"
plt.rcParams["axes.linewidth"] = 1.5

# ---------------------------------------------------------------- type (Poppins, registered explicitly)
for _f in font_files():
    fm.fontManager.addfont(_f)
plt.rcParams.update({
    "font.family": "Poppins", "mathtext.fontset": "custom", "mathtext.rm": "Poppins",
    "mathtext.it": "Poppins:italic", "mathtext.bf": "Poppins:bold", "mathtext.cal": "Poppins", "mathtext.sf": "Poppins",
    "font.size": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,      # reset what make_videos sets for slides
    "axes.spines.top": False, "axes.spines.right": False,
})
FS_LABEL, FS_TICK = 14, 12
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# ---------------------------------------------------------------- frame and view
W_IN, H_IN, DPI = 10.0, 5.0, 160                         # 1600 x 800 px
R_MIN, R_MAX = 0.5, 2.0                                  # trained radius range (as on the card banner)
LW_MIN, LW_MAX = 0.6, 2.4                                # edge width (pt) at R_MIN and R_MAX, normal version
NODE_SIZE = 4.5                                          # scatter size (pt^2) at unit perspective scale
WINDOW_TOP = 0.74                                        # the window in the response panel stops here (axes
LABEL_Y = 0.79                                           # fraction), and "inside the window" sits above it
# H.264 settings: a network that moves every frame is expensive to encode. crf 27 with preset veryslow and tune
# animation gives about 3.4 MB (normal) and 3.6 MB (accessible), under the 4 MB target; crf 30 smears the light
# (soft) edges while the network shakes, and crf 26 puts the accessible version just over 4 MB.
CRF, PRESET, TUNE = 27, "veryslow", "animation"

STYLES = {mode: asset_style("physical_learning_sweep", mode) for mode in ("normal", "accessible")}

# ---------------------------------------------------------------- physics (as in make_videos.sweep_video)

def compute_sweep():
    """Load the frozen sweep; recoloring never reruns training or response calculations."""
    with np.load(data_path("physical_learning_sweep"), allow_pickle=False) as data:
        values = {key: data[key] for key in data.files}
    run = SimpleNamespace(wlo=float(values.pop("wlo")), whi=float(values.pop("whi")), n=int(values.pop("n_nodes")))
    view = Oblique(values["pos"], values.pop("edges"), values["box"], elev=float(values.pop("elevation")))
    return SimpleNamespace(run=run, view=view, **values)


# ---------------------------------------------------------------- drawing

def build(sw, st):
    """The figure for one version; returns it and a draw(w, z, az, amp, trace_to) function."""
    run, view, s, bg = sw.run, sw.view, st["lw"], st["bg"]
    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI, facecolor=bg)

    # left: the network, oblique view, edges coloured and sized by trained stiffness
    axn = fig.add_axes([0.0, 0.0, 0.595, 1.0])
    pad = 0.35
    axn.set_xlim(sw.lim[0] - pad, sw.lim[1] + pad); axn.set_ylim(sw.lim[2] - pad, sw.lim[3] + pad)
    axn.set_aspect("equal"); axn.axis("off")
    cmap = LinearSegmentedColormap.from_list(f"seq_{st['suffix']}", st["seq"])
    t = sw.piece_stiff
    colors, widths = cmap(t), s * (LW_MIN + (LW_MAX - LW_MIN) * t)
    tier = np.minimum((t * 3).astype(int), 2)                      # soft, middle, stiff: stiff drawn on top
    mx = view.pos[view.node, 0] + 0.5 * view.dx[:, 0] - view.b / 2
    my = view.pos[view.node, 1] + 0.5 * view.dx[:, 1] - view.b / 2
    floor = plt.Polygon(view.floor(sw.az0), closed=True, fc=st["floor"], ec="none", zorder=0)
    axn.add_patch(floor)
    lines = LineCollection([], capstyle="round", zorder=1)
    axn.add_collection(lines)
    nodes = axn.scatter(np.zeros(run.n), np.zeros(run.n), s=np.zeros(run.n), c=st["node"], linewidths=0, zorder=3)

    # right column
    x0, wd = 0.675, 0.30
    fb = sw.f1[(sw.f1 > sw.lo) & (sw.f1 < sw.hi)]

    def window(ax, top, transform):
        """Peach window with thin coral edges, from the bottom of the panel up to `top`."""
        ax.fill_between([run.wlo, run.whi], 0, top, transform=transform, fc=st["window"], lw=0, zorder=0)
        for x in (run.wlo, run.whi):
            ax.plot([x, x], [0, top], transform=transform, color=st["window_edge"], lw=0.8 * s, zorder=1,
                    solid_capstyle="butt")

    # top right: the trained frequencies, with the drive
    axb = fig.add_axes([x0, 0.72, wd, 0.14])
    axb.set_xlim(sw.lo, sw.hi); axb.set_ylim(0, 1); axb.axis("off")
    window(axb, 1.0, axb.transData)
    axb.add_collection(LineCollection([[(x, 0), (x, 1)] for x in fb], colors=st["bar"], linewidths=0.6 * s,
                                      alpha=st["bar_alpha"], zorder=2))
    fig.text(x0, 0.885, "trained frequencies", fontsize=FS_LABEL, color=st["ink"], va="bottom")
    drive_b = axb.plot([sw.lo, sw.lo], [-0.22, 1.12], color=st["drive"], lw=2.4 * s, clip_on=False, zorder=4,
                       solid_capstyle="butt")[0]

    # bottom right: response amplitude against drive frequency
    axr = fig.add_axes([x0, 0.19, wd, 0.44])
    axr.set_facecolor(bg)
    axr.set_xlim(sw.lo, sw.hi); axr.set_yscale("log"); axr.set_ylim(0.1, 10)
    window(axr, WINDOW_TOP, axr.get_xaxis_transform())
    trace, = axr.plot([], [], color=st["curve"], lw=2.5 * s, zorder=3, solid_capstyle="round")
    dot, = axr.plot([], [], "o", color=st["dot"], ms=9 * (1 + 0.5 * (s - 1)), mec=bg, mew=1.4 * s, zorder=5,
                    clip_on=False)
    axr.yaxis.set_major_locator(FixedLocator([0.1, 1, 10]))
    axr.yaxis.set_major_formatter(FixedFormatter(["0.1", "1", "10"]))
    axr.yaxis.set_minor_locator(NullLocator())
    axr.set_xticks([run.wlo, run.whi]); axr.set_xticklabels([r"$\omega_{\rm lo}$", r"$\omega_{\rm hi}$"])
    for side in ("left", "bottom"):
        axr.spines[side].set_color(st["ink2"]); axr.spines[side].set_linewidth(0.8 * s)
    axr.tick_params(labelsize=FS_TICK, colors=st["ink2"], width=0.8 * s, length=5)
    axr.tick_params(axis="x", labelsize=FS_LABEL)            # omega_lo / omega_hi: mathtext subscripts run small
    axr.set_xlabel("drive frequency", fontsize=FS_LABEL, color=st["ink"], labelpad=4)
    axr.set_ylabel("response amplitude", fontsize=FS_LABEL, color=st["ink"], labelpad=6)
    status = axr.text(sw.mid, LABEL_Y, "inside the window", transform=axr.get_xaxis_transform(), ha="center",
                      va="bottom", fontsize=FS_LABEL, color=st["status"], zorder=6)

    def draw(w, z, az, amp, trace_to):
        floor.set_xy(view.floor(az))
        depth = mx * np.sin(az) + my * np.cos(az)                  # larger is farther
        order = np.lexsort((-depth, tier))                         # by tier, then back to front within a tier
        lines.set_segments(view.segments(z, az)[order])
        lines.set_color(colors[order]); lines.set_linewidth(widths[order])
        X, Y, sz = view.project(sw.pos[:, 0], sw.pos[:, 1], z, az)
        back = view.back_to_front(az)
        nodes.set_offsets(np.stack([X, Y], 1)[back]); nodes.set_sizes((NODE_SIZE * s * sz**2)[back])
        drive_b.set_xdata([w, w])
        sel = sw.grid <= trace_to
        trace.set_data(sw.grid[sel], sw.r1[sel])
        dot.set_data([w], [amp])
        status.set_visible(run.wlo < w < run.whi)

    return fig, draw


def frame(sw, n):
    return sw.ws[n], sw.Z[n], sw.azs[n], sw.amp[n], sw.ws[n]


def poster(sw):
    """Drive at the window centre (a state the clip passes through) with the whole response drawn."""
    return sw.mid, sw.z_mid, 0.5 * (sw.az0 + sw.az1), sw.amp_mid, sw.hi


def make_writer():
    return FFMpegWriter(fps=FPS, codec="libx264",
                        extra_args=["-pix_fmt", "yuv420p", "-crf", str(CRF), "-preset", PRESET, "-tune", TUNE,
                                    "-movflags", "+faststart"])


def render(sw, which, out_dir):
    st = STYLES[which]
    fig, draw = build(sw, st)
    path = out_dir / f"{STEM}{st['suffix']}.mp4"
    w = make_writer()
    with w.saving(fig, str(path), dpi=DPI):
        for n in range(len(sw.ws)):
            draw(*frame(sw, n)); w.grab_frame(facecolor=st["bg"])
    draw(*poster(sw))
    fig.savefig(out_dir / f"{STEM}_poster{st['suffix']}.png", dpi=DPI, facecolor=st["bg"])
    plt.close(fig)
    print(f"{path.name}: {len(sw.ws) / FPS:.1f} s, {path.stat().st_size / 1e6:.2f} MB")


def stills(sw, which, out_dir):
    st = STYLES[which]
    fig, draw = build(sw, st)
    run = sw.run
    picks = {"start": 15, "below": None, "inside": None, "above": None}
    for name, target in (("below", 0.5 * (sw.lo + run.wlo)), ("inside", sw.mid), ("above", 0.5 * (run.whi + sw.hi))):
        picks[name] = int(np.argmin(np.abs(sw.ws - target)))
    for name, n in picks.items():
        draw(*frame(sw, n)); fig.savefig(out_dir / f"still_{which}_{name}.png", dpi=DPI, facecolor=st["bg"])
    draw(*poster(sw)); fig.savefig(out_dir / f"still_{which}_poster.png", dpi=DPI, facecolor=st["bg"])
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=SITE_IMAGES)
    parser.add_argument("--stills", type=Path, default=None, metavar="DIR",
                        help="write a few PNG frames of both versions into DIR instead of the videos")
    args = parser.parse_args()
    sw = compute_sweep()
    jin = (sw.grid > sw.run.wlo) & (sw.grid < sw.run.whi)
    print(f"{len(sw.ws)} frames; response inside the window {sw.r1[jin].min():.3g} to {sw.r1[jin].max():.3g}, "
          f"outside median {np.median(sw.r1[~jin]):.3g}")
    for which in STYLES:
        if args.stills:
            args.stills.mkdir(parents=True, exist_ok=True)
            stills(sw, which, args.stills)
        else:
            render(sw, which, args.out_dir)
