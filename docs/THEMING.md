# Website colors and scientific media

`theme/theme.json` is the source of color choices for both pages, all seven static research figures/banners, both animations and their posters, and the three favicon files. The normal and accessible modes have independent palettes and can also have different role assignments.

The normal theme uses warm cream (`#fffbea`) wherever the original cream was used, including banners, animation backgrounds, and the pale scale-rich cells. The animation floor uses a slightly darker related cream (`#f9f4df`). Accessible mode retains its white backgrounds. The research descriptions, page layout, scientific data, and animation schedules have not been redesigned.

## Choose colors, then assign roles

There are three levels:

| Location in `theme/theme.json` | Purpose | Examples |
| --- | --- | --- |
| `palettes.normal` and `palettes.accessible` | Named color swatches | `cream`, `ink`, `cerulean`, `coral` |
| `roles.web` and `roles.media` | Assign swatches to shared purposes | Page background, plot background, text, data series, gradient stops |
| `assets` | Assign colors to each figure's actual marks | Before/after histograms, target window, drive marker, triangle birth scale, void area |
| `overrides.accessible` | Change a role or treatment only in accessible mode | Outlines, hatching, thicker lines, larger nodes, alternate ramps |

A reference such as `"@palette.coral"` uses that swatch in the current mode. `"@roles.media.series-primary"` follows a shared role. `"@assets.physical_learning_detail.after_line"` follows a particular mark in a particular figure.

Palette names describe the starting colors, not permanent constraints. You can add a new named swatch to both palettes and assign any role to it. Both palettes must contain the same swatch names. A role can use a different swatch in accessible mode through `overrides.accessible`.

### Change page and plot backgrounds independently

For example, these assignments make the page white and the plots cream:

```json
"roles": {
  "web": {
    "page-background": "@palette.white"
  },
  "media": {
    "figure-background": "@palette.cream"
  }
}
```

These are excerpts: edit the corresponding entries in the existing file and retain its other roles. `banner-background` and `video-background` are separate roles. The web video placeholder has its own `roles.web.video-background`. If a new design should match these surfaces, assign them to the same swatch.

### Exchange series colors without changing headings or navigation

```json
"series-primary": "@palette.coral",
"series-secondary": "@palette.cerulean"
```

These media roles affect the appropriate plotted series. Website roles remain independent. The meaning of the series stays the same: the hyperuniform row is still the hyperuniform row, for example, regardless of its color.

### Make one figure different

Change `assets.physical_learning_detail.after_line` to `"@palette.coral"`. Its translucent-looking, opaque histogram fill follows that line through a reference. Other figures retain their assignments.

The histogram fills are explicit mixtures:

```json
"after_fill": {
  "mix": [
    "@assets.physical_learning_detail.after_line",
    "@assets.physical_learning_detail.bg"
  ],
  "weight": 0.16
}
```

The first color contributes 16 percent; the actual background contributes the rest. This recalculates the fill when either changes and avoids accidental blends between overlapping histogram bars. Node halos also follow the figure background. Shadows and borders use explicit `alpha`/`opacity` expressions.

Gradients are ordered lists of swatch references. Preserve the direction of the scientific encoding when recoloring: low to high stiffness, short to long edge, early to late triangle entry, or small to large void area. Related banners, figures, and videos share the corresponding ramp role, and individual assets can override it. The eight scale-rich cell fills are decorative; their frozen assignments preserve the same geometry and color distribution.

## Rebuild workflow

Use Python 3.12. The rendering dependencies are pinned because Matplotlib and font versions affect the rasterized result.

```sh
uv venv --python 3.12 .venv-theme
uv pip install --python .venv-theme/bin/python -r requirements-theme.txt
.venv-theme/bin/python tools/theme.py validate
.venv-theme/bin/python tools/theme.py build --stills --output .theme-build/preview
.venv-theme/bin/python -m http.server 8000 --bind 127.0.0.1
```

Open `http://127.0.0.1:8000/.theme-build/preview/theme-preview.html`. The gallery shows both modes, the palette, every figure, and the video posters. The copied main page and music page use the same generated stylesheet. A still preview replaces videos with posters; sampled animation frames are also saved under `preview-frames/`. Music previews reference the existing local music directory.

`ffmpeg` and `ffprobe` must be on `PATH`. Poppins TTF files must be installed under `~/.local/share/fonts/Poppins/`, including Regular, SemiBold, Bold, and Italic. Set `WEBSITE_FONT_DIR` to another Poppins directory when needed. CairoSVG also needs the operating system's Cairo library. These resources are already available on James's workstation.

A full build includes both MP4 variants and their corresponding posters:

```sh
.venv-theme/bin/python tools/theme.py build --output .theme-build/full
.venv-theme/bin/python tools/theme.py apply .theme-build/full
.venv-theme/bin/python tools/theme.py check
```

`apply` updates generated CSS, icons, figures, videos, and posters together. It does not commit, push, or deploy. It refuses incomplete builds, changed theme/code/inputs, modified build outputs, and destination files edited after the build. It saves the previous generated files under the build's `before-apply/` directory.

For a palette-only change, save the current theme before editing it. A full build can then retain existing artwork wherever the resolved drawing styles have not changed:

```sh
mkdir -p .theme-build
cp theme/theme.json .theme-build/previous-theme.json
# Edit theme/theme.json, then:
.venv-theme/bin/python tools/theme.py build --keep-unchanged-from .theme-build/previous-theme.json --output .theme-build/recolored
.venv-theme/bin/python tools/theme.py apply .theme-build/recolored
```

This checks that the saved theme matches the applied CSS and media, compares every asset's resolved styles independently in normal and accessible mode, and records the hashes of preserved files. It protects unrelated artwork, including local edits adopted during the initial migration, while rebuilding all affected figures and videos. Use this option only when changing colors or role assignments; omit it when updating renderer code, fonts, or scientific inputs so those changes are rendered too. The output is still a complete build and is applied as one set.

Builds validate PNG dimensions, H.264/yuv420p video encoding, durations, fast-start metadata, and the existing 4 MB video budget. A theme is frozen for the duration of a build so editing its source during video encoding cannot mix palettes. Verified unchanged outputs can be reused on subsequent builds in the same directory.

To explore a separate theme without selecting it:

```sh
.venv-theme/bin/python tools/theme.py --theme path/to/alternative.json build --stills --output .theme-build/alternative
```

To select that design, put its contents in `theme/theme.json`, then build and apply from that file. Never hand-edit `assets/theme.css` or recolor PNGs/MP4s directly. A theme change needs a media rebuild because colors are baked into those files.

## Accessible mode and visual review

Accessible mode retains its own colors, thicker lines, larger markers, stronger outlines, and the histogram hatching. The existing page toggle switches images, videos, posters, and linked full-size figures as before.

`validate` and the preview report selected text contrast ratios. These checks are advisory, not a claim of full accessibility or color-vision suitability. They cannot establish that two nearby series are distinguishable. Inspect overlaps, gradient direction, text, thin edges, and animated frames in both modes.

The palette has three existing text-pair warnings: the normal music play controls use cream on coral (about 2.8:1), and accessible navigation hover uses dark ink on dark coral (3.35:1). Their separate roles allow these combinations to be adjusted independently.

## Where the rendering code lives

| Asset family | Renderer | Frozen inputs |
| --- | --- | --- |
| Physical-learning banner | `tools/renderers/physical_learning_card.py` | Exemplar positions, edges, trained radii |
| Physical-learning detail | `tools/renderers/physical_learning_detail.py` | Four selected spectra and schematic layouts |
| Physical-learning animation | `tools/renderers/physical_learning_sweep.py` | Frozen network, displacements, drive schedule, response curve |
| Scale-rich banner | `tools/renderers/sr_cells.py` | Ligament/cell polygons and decorative fill assignments |
| Scale-rich transport and stiffness | `tools/renderers/sr_transport_stiffness.py` | Master and sample geometry, angle marker, beam currents, FEM mesh and nodal stress, angular profiles, ensemble anisotropy quartiles |
| TDA banner | `tools/renderers/tda_card.py` | Filamentary point pattern |
| TDA animation | `tools/renderers/tda_filtration.py` | Point pattern; filtration and exact checks remain in the renderer |
| Hyperuniform banner | `tools/renderers/huppi_card.py` | Centroidal network geometry |
| Hyperuniform comparison | `tools/renderers/huppi_comparison.py` | Patterns, networks, cached structure-factor ensemble |

These are the authoritative website renderers. The old research-repository scripts remain available as sources for refreshing scientific inputs, but should no longer be used to recolor or publish website artwork. No research-repository files were modified.

The nine compressed snapshots in `tools/data/` total about 2.3 MB. Each has a JSON provenance record containing the source script path/hash, research commit, relevant input hashes, snapshot hash, and array shapes. Regular builds need no research checkout and do not rerun training, select new representative runs, or regenerate network realizations. The TDA renderer retains its exact comparisons between weighted merge history and direct cluster counts, and between void counts and persistence pairs.

`tools/asset-manifest.json` records every paired output, allowed style fields, dimensions, and animation duration. Validation checks it against the HTML, so a new themed asset cannot be silently omitted from the rebuild.

### Refresh scientific content separately

Use `tools/capture_inputs.py` with the relevant research project's Python environment and the source paths in `tools/research-sources.example.json`:

```sh
/path/to/research/.venv/bin/python tools/capture_inputs.py physical_learning_sweep --output .theme-build/new-inputs
```

Inspect the new inputs and provenance before replacing the matching pair of files in `tools/data/`. Then rebuild and review both modes. Use `--sources path/to/local-sources.json` if the research repositories moved. Capturing inputs imports the existing research scripts and requires their dependencies; normal theme builds do not.

## Verification and deployment

```sh
.venv-theme/bin/python -m unittest discover -s tests -v
.venv-theme/bin/python tools/theme.py check
.venv-theme/bin/python tools/theme.py package --output _site
```

Packaging copies only public website files. It excludes research inputs, Python tools, tests, documentation, build reports, and local preview directories. Choose a fresh output directory; packaging refuses to overwrite an existing directory. GitHub Pages runs the unit checks and packaging step before deployment. It publishes `_site/`, not the whole repository. Plotting dependencies are not needed in CI because generated assets are committed.

The UCLA mirror still needs a manual sync after a push. Sync the packaged site, including its new `assets/theme.css` file, while excluding domain-specific files:

```sh
rsync -rlptz --chmod=D755,F644 --exclude CNAME --exclude robots.txt --exclude sitemap.xml --exclude 'googlef*.html' _site/ james@gateway.pa.ucla.edu:public_html/
```

The initial migration adopted the existing artwork without claiming it was regenerated. `assets/theme-state.json` now records the applied theme and exact output hashes. A palette-only update also records which files were preserved from the preceding theme. These fingerprints prevent later palette changes from being published with stale media. Independent preexisting local image edits are retained when their drawing styles are unaffected.

For browser comparisons of a refactor:

```sh
uv pip install --python .venv-theme/bin/python -r requirements-theme-dev.txt
PLAYWRIGHT_SKIP_BROWSER_GC=1 .venv-theme/bin/playwright install chromium
.venv-theme/bin/python tools/verify_browser.py --baseline http://127.0.0.1:8000/baseline --candidate http://127.0.0.1:8000 --output .theme-build/browser
```

This checks both pages at desktop and mobile widths in both modes, compares computed styles and screenshots, and records JavaScript errors. It is intended for appearance-preserving refactors; a deliberately new palette should change the screenshots.

During this migration, all eight page/mode/viewport comparisons were pixel-identical. Rebuilt banners and video posters matched the existing artwork. Three older detail figures were originally rasterized with different Matplotlib versions (3.7.5, 3.10.8, and 3.9.2), so a rebuild with the pinned 3.11.2 environment changes some font/antialiasing pixels. The original and migrated generators produce identical pixels when run in matching environments. The existing website images were kept intact.
