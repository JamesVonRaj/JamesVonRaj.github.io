"""Regression and role-independence checks; no plotting packages are required."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from site_theme.theme import MODES, Theme, contrast
import theme as command

FIXTURES = Path(__file__).parent / 'fixtures'


class ThemeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'theme.json'
        self.data = json.loads((FIXTURES / 'original-theme.json').read_text())

    def resolve(self):
        self.path.write_text(json.dumps(self.data))
        return Theme(self.path)

    def test_original_rendering_styles_are_preserved(self):
        original = json.loads((FIXTURES / 'original-styles.json').read_text())
        theme = self.resolve()
        for asset, styles in original.items():
            for mode in MODES:
                with self.subTest(asset=asset, mode=mode):
                    self.assertEqual(theme.style(asset, mode), styles[mode])

    def test_current_theme_covers_both_pages_and_every_asset(self):
        command.validate(Theme())

    def test_changing_web_background_does_not_change_figure_background(self):
        self.data['roles']['web']['page-background'] = '@palette.ink'
        theme = self.resolve()
        self.assertEqual(theme.resolved['normal']['roles']['web']['page-background'], '#1a2332')
        self.assertEqual(theme.style('physical_learning_detail')['bg'], '#ffffff')
        self.assertEqual(theme.style('physical_learning_card')['bg'], '#fdfcdc')

    def test_plot_series_can_be_swapped_without_changing_page_colors(self):
        previous = self.resolve()
        self.data['roles']['media']['series-primary'] = '@palette.coral'
        self.data['roles']['media']['series-secondary'] = '@palette.cerulean'
        theme = self.resolve()
        self.assertEqual(theme.css(), previous.css())
        self.assertEqual(theme.style('huppi_comparison')['hu'], '#f07167')
        self.assertEqual(theme.style('huppi_comparison')['poisson'], '#0081a7')
        self.assertNotEqual(theme.style('physical_learning_detail')['before'], previous.style('physical_learning_detail')['before'])

    def test_asset_override_updates_its_tint_without_affecting_other_figures(self):
        previous = self.resolve()
        self.data['assets']['physical_learning_detail']['after_line'] = '@palette.coral'
        theme = self.resolve()
        self.assertEqual(theme.style('physical_learning_detail')['after_line'], '#f07167')
        self.assertNotEqual(theme.style('physical_learning_detail')['after_fill'], previous.style('physical_learning_detail')['after_fill'])
        self.assertEqual(theme.style('huppi_comparison'), previous.style('huppi_comparison'))
        self.assertEqual(theme.css(), previous.css())

    def test_tints_and_node_halos_follow_the_actual_plot_background(self):
        previous = self.resolve()
        self.data['assets']['physical_learning_detail']['bg'] = '@palette.cream'
        style = self.resolve().style('physical_learning_detail')
        self.assertEqual(style['node_edge'], '#fdfcdc')
        self.assertNotEqual(style['before'], previous.style('physical_learning_detail')['before'])
        self.assertNotEqual(style['after_fill'], previous.style('physical_learning_detail')['after_fill'])

    def test_accessible_role_override_is_independent(self):
        original = self.resolve()
        self.data['overrides']['accessible']['roles'] = {'web': {'page-background': '@palette.card-peach'}}
        theme = self.resolve()
        self.assertEqual(theme.resolved['normal'], original.resolved['normal'])
        self.assertEqual(theme.resolved['accessible']['roles']['web']['page-background'], '#fde6d2')
        self.assertEqual(theme.style('physical_learning_detail', 'accessible')['lw'], 1.4)
        self.assertIsNotNone(theme.style('physical_learning_detail', 'accessible')['window_hatch'])

    def test_bad_references_and_cycles_fail_before_rendering(self):
        for reference in ['@palette.missing', '@roles.web.page-background']:
            with self.subTest(reference=reference):
                self.data['roles']['web']['page-background'] = reference
                with self.assertRaises(ValueError):
                    self.resolve()

    def test_unknown_overrides_are_rejected(self):
        self.data['overrides']['accessible']['assets']['physical_learning_detail']['typo'] = 2
        with self.assertRaises(ValueError):
            self.resolve()

    def test_invalid_colors_and_mixes_are_rejected(self):
        self.data['palettes']['normal']['coral'] = '#fff; color:red'
        with self.assertRaises(ValueError):
            self.resolve()
        self.data['palettes']['normal']['coral'] = '#f07167'
        self.data['assets']['physical_learning_detail']['before']['weight'] = 1.5
        with self.assertRaises(ValueError):
            self.resolve()

    def test_changed_output_suffix_is_rejected(self):
        self.data['assets']['physical_learning_sweep']['suffix'] = '../unintended'
        with self.assertRaisesRegex(ValueError, 'suffixes'):
            command.validate(self.resolve())

    def test_contrast_ratio_reference_values(self):
        self.assertAlmostEqual(contrast('#ffffff', '#000000'), 21)
        self.assertAlmostEqual(contrast('#005f7d', '#005f7d'), 1)

    def test_partial_build_cannot_be_applied(self):
        build = Path(self.directory.name) / 'build'
        build.mkdir()
        (build / 'build.json').write_text(json.dumps({'complete': False}))
        with self.assertRaisesRegex(ValueError, 'complete build'):
            command.apply(Theme(), build)

    def test_stale_theme_cannot_be_applied(self):
        build = Path(self.directory.name) / 'build'
        build.mkdir()
        manifest = {'complete': True, 'outputs': dict.fromkeys(command.GENERATED),
                    'fingerprint': {'theme': 'old', 'code': command.code_digest()}}
        (build / 'build.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'changed since'):
            command.apply(Theme(), build)

    def applied_site(self, previous):
        site = Path(self.directory.name) / 'site'
        site.mkdir()
        for relative in command.GENERATED:
            path = site / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'existing artwork with local edits')
        (site / 'assets/theme.css').write_text(previous.css())
        state = {'theme': previous.digest, 'kind': 'adopted-existing', 'files': command.GENERATED}
        (site / 'assets/theme-state.json').write_text(json.dumps(state))
        for page in ('index.html', 'music.html'):
            shutil.copy2(ROOT / page, site / page)
        shutil.copytree(ROOT / 'tools/data', site / 'tools/data')
        return site

    def test_palette_update_preserves_only_unaffected_media_and_modes(self):
        previous = self.resolve()
        site = self.applied_site(previous)
        self.data['palettes']['normal'].update(cream='#fffbea', floor='#f9f4df')
        self.data['palettes']['normal']['pale-cell'] = '#fffbea'
        current = self.resolve()
        with patch.object(command, 'ROOT', site):
            preserved = command.unchanged_outputs(current, previous)
        expected = {p for p in command.GENERATED if '_accessible' in p}
        expected.update(command.GENERATED[1:4])
        expected.update(f'images/{name}.png' for name in (
            'physical_learning_detail', 'sr_transport_stiffness', 'huppi_comparison'))
        self.assertEqual(set(preserved), expected)
        for path, digest in preserved.items():
            self.assertEqual(digest, command.sha(site / path))

    def test_palette_update_rejects_a_previous_theme_that_was_not_applied(self):
        previous = self.resolve()
        site = self.applied_site(previous)
        state_path = site / 'assets/theme-state.json'
        state = json.loads(state_path.read_text())
        state['theme'] = 'unrelated theme'
        state_path.write_text(json.dumps(state))
        with patch.object(command, 'ROOT', site):
            with self.assertRaisesRegex(ValueError, 'different theme'):
                command.unchanged_outputs(previous, previous)

    def test_palette_update_detects_changes_to_previously_recorded_artwork(self):
        previous = self.resolve()
        site = self.applied_site(previous)
        state = {'theme': previous.digest, 'kind': 'rendered',
                 'outputs': command.hashes(site, command.GENERATED)}
        (site / 'assets/theme-state.json').write_text(json.dumps(state))
        (site / 'images/physical_learning_detail.png').write_bytes(b'new local edit')
        with patch.object(command, 'ROOT', site):
            with self.assertRaisesRegex(ValueError, 'differs from its recorded version'):
                command.unchanged_outputs(previous, previous)


if __name__ == '__main__':
    unittest.main()
