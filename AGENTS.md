# Website theming

- Read `docs/THEMING.md` before changing colors or regenerating website artwork.
- `theme/theme.json` is the color source for both pages and all themed media. Keep palette swatches separate from their semantic roles and per-asset assignments.
- Website renderers live in `tools/renderers/` and use frozen inputs in `tools/data/`. Older website scripts in the research repositories are input-refresh sources, not the current recoloring workflow.
- Preserve layout, scientific content, accessible line widths/hatching, and existing local asset edits when changing theme plumbing.
- Build into `.theme-build/`, review both modes, and apply the complete output set with `tools/theme.py`. Do not manually recolor image pixels or edit generated `assets/theme.css`.
- Run the standard-library tests and `tools/theme.py check`. Deployment packages public files with `tools/theme.py package`; it must include `assets/theme.css` on both hosting targets.
- For mathematical notation, put one literal space on each side of an equality sign, including in labels and captions.
