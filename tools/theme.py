#!/usr/bin/env python3
"""Validate, preview, rebuild, and apply one website color theme."""
from __future__ import annotations

import argparse
import hashlib
from html import escape
from html.parser import HTMLParser
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from site_theme.theme import DEFAULT_THEME, MODES, ROOT, Theme, contrast

REGISTRY = json.loads((ROOT / 'tools/asset-manifest.json').read_text())
GENERATED = ['assets/theme.css', 'favicon.svg', 'favicon-96.png', 'apple-touch-icon.png'] + [
    filename for asset in REGISTRY.values() for filename in asset['outputs']]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hashes(base, names):
    return {name: sha(base / name) if (base / name).is_file() else None for name in names}


def code_digest():
    files = (sorted((ROOT / 'tools/renderers').glob('*.py'))
             + sorted((ROOT / 'tools/site_theme').glob('*.py'))
             + [ROOT / 'tools/theme.py', ROOT / 'tools/asset-manifest.json', ROOT / 'theme/favicon.svg.template'])
    return hashlib.sha256(''.join(str(p.relative_to(ROOT)) + sha(p) for p in files).encode()).hexdigest()


def validate(theme):
    if set(theme.data['assets']) != set(REGISTRY) | {'favicon'}:
        raise ValueError('The theme must define every registered asset and the favicon')
    for name, asset in REGISTRY.items():
        for mode in MODES:
            style = theme.style(name, mode)
            if set(style) != set(asset['style_keys']):
                raise ValueError(f'{name}/{mode}: style keys must match tools/asset-manifest.json')
            for key, value in style.items():
                kind = ('null' if value is None else 'number' if isinstance(value, (int, float))
                        else 'colors' if isinstance(value, list) and all(isinstance(v, str) for v in value)
                        else 'numbers' if isinstance(value, list) else 'color' if isinstance(value, str) and value.startswith('#')
                        else 'string')
                if kind not in asset['style_types'][key]:
                    raise ValueError(f'{name}/{mode}/{key}: expected {asset["style_types"][key]}, got {kind}')
                if kind == 'color' and not re.fullmatch(r'#[0-9a-fA-F]{6}', value):
                    raise ValueError(f'{name}/{mode}/{key}: invalid color')
                if kind in ('number', 'numbers'):
                    values = value if kind == 'numbers' else [value]
                    if not values or any(not isinstance(v, (int, float)) or v < 0 for v in values):
                        raise ValueError(f'{name}/{mode}/{key}: expected nonnegative numbers')
                if key.endswith(('ramp', 'seq', 'fills')):
                    if not isinstance(value, list) or len(value) < 2 or not all(re.fullmatch(r'#[0-9a-fA-F]{6}', v) for v in value):
                        raise ValueError(f'{name}/{mode}/{key}: expected a list of colors')
            if 'suffix' in style and style['suffix'] != ('_accessible' if mode == 'accessible' else ''):
                raise ValueError('Output suffixes are fixed by the asset manifest')
            if name == 'sr_cells' and len(style['fills']) != 8:
                raise ValueError('sr_cells needs eight decorative roles to match the frozen cell assignments')
        data = ROOT / 'tools/data' / f'{name}.npz'
        provenance = json.loads(data.with_suffix('.json').read_text())
        if sha(data) != provenance['snapshot_sha256']:
            raise ValueError(f'Input snapshot changed: {data}. Refresh its provenance explicitly.')
    for path in (ROOT / 'index.html', ROOT / 'music.html'):
        text = path.read_text()
        tokens = set(re.findall(r'var\(--([a-z0-9-]+)\)', text))
        missing = tokens - set(theme.resolved['normal']['roles']['web'])
        if missing:
            raise ValueError(f'{path.name}: undefined color roles: {sorted(missing)}')
        styles = '\n'.join(part for match in re.findall(r'<style>(.*?)</style>|style="([^"]*)"', text, re.S) for part in match)
        if re.search(r'#[0-9a-fA-F]{3,8}\b|\brgba?\(', styles):
            raise ValueError(f'{path.name}: literal colors belong in the central theme')
    class Media(HTMLParser):
        def __init__(self):
            super().__init__()
            self.paths = set()
        def handle_starttag(self, tag, attributes):
            for key, value in attributes:
                if key in ('data-normal', 'data-accessible', 'data-poster-normal', 'data-poster-accessible'):
                    self.paths.add(value)
    media = Media()
    media.feed((ROOT / 'index.html').read_text())
    if media.paths != set(GENERATED) - set(GENERATED[:4]):
        raise ValueError('The asset manifest and the HTML media references disagree')


def contrast_report(theme):
    pairs = [('text', 'page-background'), ('heading', 'page-background'),
             ('text', 'research-card-background'), ('header-text', 'header-background'),
             ('nav-text', 'nav-background'), ('nav-hover-text', 'nav-hover-background'),
             ('toggle-text', 'toggle-background'), ('link-hover', 'page-background'),
             ('play-button-text', 'play-button-active'), ('player-button-text', 'player-button-background')]
    report = []
    for mode in MODES:
        web = theme.resolved[mode]['roles']['web']
        for foreground, background in pairs:
            ratio = contrast(web[foreground], web[background])
            report.append({'mode': mode, 'foreground': foreground, 'background': background,
                           'ratio': round(ratio, 2), 'target': 4.5, 'passes': ratio >= 4.5})
        for name in REGISTRY:
            style = theme.style(name, mode)
            surface = style.get('bg', style.get('surface'))
            ink = style.get('ink', style.get('text'))
            if surface and ink:
                ratio = contrast(ink, surface)
                report.append({'mode': mode, 'foreground': name + '.text', 'background': name + '.surface',
                               'ratio': round(ratio, 2), 'target': 4.5, 'passes': ratio >= 4.5})
    return report


def dependencies():
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'matplotlib', 'pandas', 'Pillow', 'CairoSVG')}
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        raise ValueError('Install ffmpeg and ffprobe before rebuilding videos')
    font_dir = Path(os.environ.get('WEBSITE_FONT_DIR', '~/.local/share/fonts/Poppins')).expanduser()
    fonts = sorted(font_dir.glob('*.ttf'))
    required = {'Poppins-Regular.ttf', 'Poppins-SemiBold.ttf', 'Poppins-Bold.ttf', 'Poppins-Italic.ttf'}
    missing = required - {path.name for path in fonts}
    if missing:
        raise ValueError(f'Poppins fonts are required: {sorted(missing)}. Set WEBSITE_FONT_DIR to their directory.')
    versions['fonts'] = {p.name: sha(p) for p in fonts}
    versions['ffmpeg'] = subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0]
    return versions


def verify_output(path, entry):
    if path.suffix == '.png':
        from PIL import Image
        with Image.open(path) as picture:
            if list(picture.size) != entry['dimensions']:
                raise ValueError(f'Unexpected image dimensions: {path.name}: {picture.size}')
            picture.verify()
    else:
        info = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams', '-show_format', '-of', 'json', str(path)
        ], text=True))
        stream = info['streams'][0]
        if (stream['codec_name'] != 'h264' or stream['pix_fmt'] != 'yuv420p'
                or [stream['width'], stream['height']] != entry['dimensions']
                or abs(float(info['format']['duration']) - entry['duration']) > 0.05):
            raise ValueError(f'Unexpected video format or duration: {path.name}')
        if path.stat().st_size > 4_000_000:
            raise ValueError(f'{path.name} exceeds the existing 4 MB video budget')
        encoded = path.read_bytes()
        if not (0 <= encoded.find(b'moov') < encoded.find(b'mdat')):
            raise ValueError(f'{path.name} is missing fast-start metadata')


def write_static(theme, output):
    import cairosvg
    (output / 'assets').mkdir(exist_ok=True)
    (output / 'images').mkdir(exist_ok=True)
    (output / 'assets/theme.css').write_text(theme.css())
    svg = (ROOT / 'theme/favicon.svg.template').read_text()
    for role, color in theme.style('favicon').items():
        svg = svg.replace('{{' + role + '}}', color)
    if '{{' in svg:
        raise ValueError('Unresolved favicon role')
    (output / 'favicon.svg').write_text(svg)
    for name, size in [('favicon-96.png', 96), ('apple-touch-icon.png', 180)]:
        cairosvg.svg2png(bytestring=svg.encode(), write_to=str(output / name), output_width=size, output_height=size)
    for name in ['index.html', 'music.html', 'cv.pdf', 'images/me.jpg']:
        shutil.copy2(ROOT / name, output / name)
    if not (output / 'music').exists():
        (output / 'music').symlink_to(ROOT / 'music', target_is_directory=True)


def preview_gallery(theme, output, report):
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
             '<title>Website theme preview</title><style>body{font:16px sans-serif;max-width:1100px;margin:2rem auto;padding:1rem} '
             'img,video{max-width:100%;height:auto} td,th{padding:.4rem;text-align:left} table{border-collapse:collapse} '
             '.swatch{display:inline-block;width:2rem;height:1rem;border:1px solid;margin-right:.5rem} '
             '.modes{display:grid;grid-template-columns:1fr 1fr;gap:1rem}</style>',
             f'<h1>{escape(theme.data["name"])} theme preview</h1><p><a href="index.html">Main page</a> · <a href="music.html">Music page</a></p>',
             '<p>Color ratios below check selected text pairs only. Inspect gradients, overlapping marks, hatching, and video frames visually.</p>']
    for mode in MODES:
        parts.append(f'<h2>{mode.title()} palette</h2><table>')
        for name, color in theme.data['palettes'][mode].items():
            parts.append(f'<tr><td><span class="swatch" style="background:{color}"></span>{escape(name)}</td><td>{color}</td></tr>')
        parts.append('</table>')
    for name, entry in REGISTRY.items():
        parts.append(f'<h2>{escape(name.replace("_", " "))}</h2><div class="modes">')
        for suffix in ('', '_accessible'):
            base = f'images/{name}{suffix}'
            if entry['video']:
                poster = f'images/{name}_poster{suffix}.png'
                parts.append(f'<video controls preload="none" poster="{poster}" src="{base}.mp4"></video>' if (output / (base + '.mp4')).exists() else f'<img alt="{name}{suffix}" src="{poster}">')
            else:
                parts.append(f'<img alt="{name}{suffix}" src="{base}.png">')
        parts.append('</div>')
    parts.append('<h2>Selected text contrast ratios</h2><table><tr><th>Mode</th><th>Text</th><th>Background</th><th>Ratio</th><th>4.5 target</th></tr>')
    for item in report:
        parts.append('<tr>' + ''.join(f'<td>{escape(str(item[k]))}</td>' for k in ('mode','foreground','background','ratio','passes')) + '</tr>')
    parts.append('</table></html>')
    (output / 'theme-preview.html').write_text('\n'.join(parts))


def unchanged_outputs(theme, previous_theme):
    """Keep existing artwork during an explicit palette-only update.

    The previous theme must describe the currently applied CSS and media.
    Compare resolved styles separately for each mode, including indirect
    palette references, so a change never reuses artwork in an affected mode.
    """
    check(previous_theme)
    preserved = {}
    for name, entry in REGISTRY.items():
        for relative in entry['outputs']:
            mode = 'accessible' if '_accessible' in Path(relative).stem else 'normal'
            if theme.style(name, mode) == previous_theme.style(name, mode):
                preserved[relative] = sha(ROOT / relative)
    if theme.style('favicon') == previous_theme.style('favicon'):
        for relative in GENERATED[1:4]:
            preserved[relative] = sha(ROOT / relative)
    return preserved


def build(theme, output, stills=False, previous_theme=None):
    validate(theme)
    preserved = unchanged_outputs(theme, previous_theme) if previous_theme else {}
    versions = dependencies()
    output = output.resolve()
    if output == ROOT or ROOT.is_relative_to(output) or any(output == ROOT / p for p in ('images','assets','theme','tools','music')):
        raise ValueError('Build into a separate preview directory, such as .theme-build/preview')
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'build.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    fingerprint = {'theme': theme.digest, 'code': code_digest(), 'versions': versions,
                   'data': hashes(ROOT, [f'tools/data/{name}.npz' for name in REGISTRY]), 'stills': stills}
    if previous_theme:
        fingerprint['palette_only_update'] = {'previous_theme': previous_theme.digest, 'preserved': preserved}
    source_files = hashes(ROOT, GENERATED)
    write_static(theme, output)
    def copy_preserved(relative):
        if sha(ROOT / relative) != preserved[relative]:
            raise ValueError(f'{relative} changed during this build; rebuild to preserve that work.')
        shutil.copy2(ROOT / relative, output / relative)
    for relative in GENERATED[1:4]:
        if relative in preserved:
            copy_preserved(relative)
    # Subprocesses must all see the same theme, even if the user edits the source
    # while a long video is encoding.
    frozen_theme = output / 'build-theme.json'
    frozen_theme.write_text(json.dumps(theme.data, indent=2) + '\n')
    environment = os.environ.copy()
    environment.update(WEBSITE_THEME=str(frozen_theme), PYTHONPATH=str(ROOT / 'tools'),
                       OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    jobs = {}
    def checkpoint():
        written = [p for p in GENERATED if (output / p).is_file()]
        manifest_path.write_text(json.dumps({
            'complete': False, 'fingerprint': fingerprint, 'source_files': source_files,
            'jobs': jobs, 'outputs': hashes(output, written),
        }, indent=2) + '\n')
    # A failed or interrupted build is never eligible for apply.
    manifest_path.write_text(json.dumps({'complete': False}, indent=2) + '\n')
    for name, entry in REGISTRY.items():
        expected = [p for p in entry['outputs'] if not (stills and p.endswith('.mp4'))]
        jobs[name] = {
            'styles': {mode: theme.style(name, mode) for mode in MODES},
            'versions': versions, 'data': sha(ROOT / 'tools/data' / (name + '.npz')),
            'renderer': sha(ROOT / entry['script']), 'stills': stills,
            'shared': hashes(ROOT, ['tools/site_theme/theme.py', 'tools/renderers/_oblique.py']),
        }
        retained = {p: preserved[p] for p in expected if p in preserved}
        if retained:
            jobs[name]['preserved'] = retained
        if len(retained) == len(expected):
            for relative in expected:
                copy_preserved(relative)
                verify_output(output / relative, entry)
            print(f'{name}: palette unchanged in both modes, preserving existing artwork', flush=True)
            checkpoint()
            continue
        matches = previous.get('jobs', {}).get(name) == jobs[name] or previous.get('fingerprint') == fingerprint
        if matches and all((output / p).is_file() and sha(output / p) == previous.get('outputs', {}).get(p) for p in expected):
            for relative in expected:
                verify_output(output / relative, entry)
            print(f'{name}: unchanged, using verified build outputs', flush=True)
            checkpoint()
            continue
        with tempfile.TemporaryDirectory(prefix=name + '-', dir=output.parent) as temporary:
            destination = Path(temporary)
            command = [sys.executable, str(ROOT / entry['script'])]
            command += ['--output-dir' if name == 'huppi_comparison' else '--out-dir', str(destination)]
            if stills and name == 'physical_learning_sweep':
                command += ['--stills', str(destination)]
            if stills and name == 'tda_filtration':
                command += ['--preview', str(destination), '--frames', '0', '120', '300', '449']
            print(f'Rendering {name}...', flush=True)
            subprocess.run(command, check=True, env=environment)
            if stills and entry['video']:
                for mode, suffix in (('normal',''), ('accessible','_accessible')):
                    source = destination / (f'still_{mode}_poster.png' if name == 'physical_learning_sweep' else f'frame{suffix}_300.png')
                    shutil.copy2(source, destination / f'{name}_poster{suffix}.png')
                frame_dir = output / 'preview-frames' / name
                frame_dir.mkdir(parents=True, exist_ok=True)
                for path in destination.glob('*.png'):
                    shutil.copy2(path, frame_dir / path.name)
            for relative in expected:
                if relative in retained:
                    copy_preserved(relative)
                    verify_output(output / relative, entry)
                    continue
                source = destination / Path(relative).name
                if not source.is_file() or not source.stat().st_size:
                    raise ValueError(f'{name} did not produce {source.name}')
                verify_output(source, entry)
                shutil.copy2(source, output / relative)
        checkpoint()
    if stills:
        # A still preview must not display a stale video from an earlier full build.
        for relative in GENERATED:
            if relative.endswith('.mp4'):
                (output / relative).unlink(missing_ok=True)
        path = output / 'index.html'
        text = path.read_text()
        def poster_image(match):
            attributes = dict(re.findall(r'([\w-]+)="([^"]*)"', match[0]))
            return (f'<img class="research-detail-figure" alt="Animation poster" src="{attributes["poster"]}" '
                    f'data-normal="{attributes["data-poster-normal"]}" data-accessible="{attributes["data-poster-accessible"]}">')
        text = re.sub(r'<video\b[^>]*></video>', poster_image, text)
        text = re.sub(r'<button class="video-toggle".*?</button>', '', text)
        path.write_text(text)
    report = contrast_report(theme)
    (output / 'contrast.json').write_text(json.dumps(report, indent=2) + '\n')
    preview_gallery(theme, output, report)
    outputs = [p for p in GENERATED if not (stills and p.endswith('.mp4'))]
    if Theme(theme.path).digest != theme.digest or code_digest() != fingerprint['code']:
        raise ValueError('Theme or code changed during this build. Rebuild before applying.')
    manifest = {'complete': not stills, 'fingerprint': fingerprint, 'source_files': source_files,
                'outputs': hashes(output, outputs), 'jobs': jobs}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Preview: {output / "theme-preview.html"}')
    print('Stills only; run a full build before apply.' if stills else 'Complete build verified. Use apply to copy these generated assets into the website.')


def apply(theme, output):
    validate(theme)
    output = output.resolve()
    manifest = json.loads((output / 'build.json').read_text())
    if not manifest.get('complete') or set(manifest.get('outputs', {})) != set(GENERATED):
        raise ValueError('Only a complete build containing every asset can be applied')
    if manifest['fingerprint']['theme'] != theme.digest or manifest['fingerprint']['code'] != code_digest():
        raise ValueError('Theme or rendering code changed since the build; rebuild first')
    if manifest['fingerprint']['data'] != hashes(ROOT, [f'tools/data/{name}.npz' for name in REGISTRY]):
        raise ValueError('Rendering inputs changed since the build; rebuild first')
    if theme.path != DEFAULT_THEME.resolve():
        raise ValueError('Preview themes must be selected as theme/theme.json and rebuilt before applying')
    for relative in GENERATED:
        if sha(output / relative) != manifest['outputs'][relative]:
            raise ValueError(f'Build output changed: {relative}')
        actual = sha(ROOT / relative) if (ROOT / relative).exists() else None
        if actual not in (manifest['source_files'][relative], manifest['outputs'][relative]):
            raise ValueError(f'{relative} changed in the website after this build. Rebuild to preserve that work.')
    backup = output / 'before-apply'
    backup.mkdir(exist_ok=True)
    for relative in GENERATED:
        destination = ROOT / relative
        if destination.exists():
            previous = backup / relative
            previous.parent.mkdir(parents=True, exist_ok=True)
            if not previous.exists():
                shutil.copy2(destination, previous)
    try:
        for relative in GENERATED:
            destination = ROOT / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output / relative, destination)
    except OSError:
        for relative in GENERATED:
            if (backup / relative).exists():
                shutil.copy2(backup / relative, ROOT / relative)
        raise
    state = {'theme': theme.digest, 'kind': 'rendered', 'outputs': hashes(ROOT, GENERATED)}
    if 'palette_only_update' in manifest['fingerprint']:
        state['palette_only_update'] = manifest['fingerprint']['palette_only_update']
    (ROOT / 'assets/theme-state.json').write_text(json.dumps(state, indent=2) + '\n')
    print('Applied the theme CSS, icons, figures, videos, and posters. No deployment was performed.')


def check(theme):
    validate(theme)
    if (ROOT / 'assets/theme.css').read_text() != theme.css():
        raise ValueError('Theme CSS is stale. Build and apply the updated theme.')
    state = json.loads((ROOT / 'assets/theme-state.json').read_text())
    if state['theme'] != theme.digest:
        raise ValueError('The media were built for a different theme. Build and apply all assets together.')
    if state.get('kind') == 'adopted-existing':
        # Initial migration keeps the existing artwork, including independent
        # local edits. Future theme applies record exact hashes instead.
        if set(state['files']) != set(GENERATED) or any(not (ROOT / name).is_file() or not (ROOT / name).stat().st_size for name in GENERATED):
            raise ValueError('The adopted baseline must include every generated asset')
        print('Theme and CSS match the adopted artwork. A full theme apply will record output hashes.')
        return
    if state.get('kind') != 'rendered' or set(state['outputs']) != set(GENERATED):
        raise ValueError('Theme state must record every generated asset')
    for name, digest in state['outputs'].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f'Generated asset differs from its recorded version: {name}')
    print('Theme, CSS, and all recorded media are consistent.')


def package(theme, output):
    check(theme)
    output = output.resolve()
    if output.exists():
        raise ValueError('Choose a new package directory or remove the previous _site build first')
    output.mkdir(parents=True)
    for name in ('index.html', 'music.html', 'cv.pdf', 'favicon.svg', 'favicon-96.png', 'apple-touch-icon.png',
                 'CNAME', 'robots.txt', 'sitemap.xml', 'googlef4233ba5409d69b0.html', 'assets/theme.css'):
        (output / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, output / name)
    for directory in ('images', 'music'):
        shutil.copytree(ROOT / directory, output / directory)
    print(f'Static website packaged at {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--theme', type=Path, default=DEFAULT_THEME)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('validate')
    sub.add_parser('check')
    building = sub.add_parser('build')
    building.add_argument('--output', type=Path, default=ROOT / '.theme-build/preview')
    building.add_argument('--stills', action='store_true')
    building.add_argument('--keep-unchanged-from', type=Path, metavar='PREVIOUS_THEME',
                          help='For palette-only updates, preserve artwork whose resolved styles match the applied theme')
    applying = sub.add_parser('apply')
    applying.add_argument('build', type=Path)
    packaging = sub.add_parser('package')
    packaging.add_argument('--output', type=Path, default=ROOT / '_site')
    args = parser.parse_args()
    try:
        theme = Theme(args.theme)
        if args.command == 'validate':
            validate(theme)
            print('Theme references, HTML roles, media coverage, and input snapshots are valid.')
            for item in contrast_report(theme):
                if not item['passes']:
                    print(f"Contrast review: {item['mode']} {item['foreground']} on {item['background']}: {item['ratio']}:1 (target 4.5:1)")
        elif args.command == 'check':
            check(theme)
        elif args.command == 'build':
            previous_theme = Theme(args.keep_unchanged_from) if args.keep_unchanged_from else None
            build(theme, args.output, args.stills, previous_theme)
        elif args.command == 'package':
            package(theme, args.output)
        else:
            apply(theme, args.build)
    except (ValueError, KeyError, FileNotFoundError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Theme error: {error}\n')


if __name__ == '__main__':
    main()
