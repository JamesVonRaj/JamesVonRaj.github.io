"""Explicitly refresh website rendering inputs from the research repositories.

Run each asset with its research environment. This is separate from recoloring:
regular theme builds never import research repositories or recompute experiments.
The output directory is required so refreshing scientific content is deliberate.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '4')

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capture(name, source, output):
    spec = importlib.util.spec_from_file_location('website_input_source', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    m = module
    arrays, inputs = {}, []
    if name == 'physical_learning_card':
        arrays = dict(zip(('pos', 'edges', 'radii', 'disp'), m.load_network()))
        inputs = [m.NPZ]
    elif name == 'physical_learning_detail':
        paths = [m.pick_representative(f['key'], m.THUMB_REG) for f in m.FAMILIES]
        if any(p is None for p in paths):
            raise ValueError('All four families need research data')
        arrays['ytop'] = int(np.ceil(m.compute_global_ymax(paths) / 10)) * 10
        for family, path in zip(m.FAMILIES, paths):
            key = family['key']
            with np.load(path, allow_pickle = False) as data:
                for field in ('f0', 'ff', 'wlo', 'whi', 'n_in_final'):
                    arrays[f'{key}_{field}'] = data[field]
            pos, segs, weights, top = m.make_small_representative(key)
            arrays.update({f'{key}_pos': pos, f'{key}_segs': segs, f'{key}_weights': weights,
                           f'{key}_top': np.asarray([] if top is None else top, dtype = bool),
                           f'{key}_success': m.collect_success_count(key, m.THUMB_REG)})
        inputs = paths
    elif name == 'sr_cells':
        with contextlib.redirect_stdout(io.StringIO()):
            polygons, centres, cells, indices = m.build()
        for key, groups in [('ligaments', polygons), ('cells', cells)]:
            arrays[key] = np.concatenate(groups)
            arrays[key + '_offsets'] = np.r_[0, np.cumsum([len(p) for p in groups])]
        arrays['centres'] = np.asarray(centres)
        arrays['color_indices'] = indices
        arrays['color_weights'] = m.WEIGHTS
    elif name == 'sr_anisotropy':
        data = m.load_data()
        selected = data[(data.reff_anisotropy_ratio >= 12) & (data.reff_anisotropy_ratio <= 20)
                        & (data.alpha == 1.0)].iloc[0]
        with contextlib.redirect_stdout(io.StringIO()):
            seed = m.find_high_anisotropy_seed(n_tries = 50)
            geometry = m.generate_geometry(seed)
        segments, thicknesses = [], []
        for poly in geometry.values():
            middle = poly.middle_segment
            start, end = np.asarray(middle.start), np.asarray(middle.end)
            length = np.linalg.norm(end - start)
            if length > 0:
                centre = m.BOX_SIZE / 2
                rotate = lambda point: np.array([centre + (point[1] - centre), centre - (point[0] - centre)])
                segments.append([rotate(start), rotate(end)])
                thicknesses.append(poly.area() / length)
        arrays = {'segments': np.asarray(segments), 'thicknesses': np.asarray(thicknesses), 'geometry_seed': seed,
                  'alpha': data.alpha.to_numpy(), 'ratio': data.reff_anisotropy_ratio.to_numpy(),
                  'polar': selected[m.REFF_COLS].to_numpy(dtype = float),
                  'polar_ratio': selected.reff_anisotropy_ratio}
        inputs = [m.DATA_PATH]
    elif name == 'huppi_comparison':
        arrays = m.make_patterns()
        with np.load(m.DEFAULT_CACHE, allow_pickle = False) as data:
            if not np.array_equal(data['params'], m._cache_params()):
                raise ValueError('Research S(k) cache does not match the figure parameters')
            arrays.update({f'sk_{key}': data[key] for key in data.files})
        inputs = [m.DEFAULT_CACHE]
    elif name == 'physical_learning_sweep':
        sw = m.compute_sweep()
        arrays = {key: value for key, value in vars(sw).items() if key not in ('run', 'view')}
        arrays.update({'edges': sw.run.edges, 'wlo': sw.run.wlo, 'whi': sw.run.whi,
                       'n_nodes': sw.run.n, 'elevation': m.ELEV})
        inputs = [Path(source).parent / 'data/randdel_run.npz']
    elif name == 'tda_card':
        arrays['points'] = m.point_pattern(np.random.default_rng(m.SEED))
    elif name == 'tda_filtration':
        arrays['points'] = m.point_pattern(np.random.default_rng(np.random.SeedSequence(m.SEED)))
    elif name == 'huppi_card':
        points = m.url_points(np.random.default_rng(m.SEED), m.DISORDER)
        nodes, edges = m.centroidal_network(points)
        arrays = {'points': points, 'nodes': nodes, 'edges': edges}
    else:
        raise ValueError(f'Unknown asset: {name}')
    output.mkdir(parents = True, exist_ok = True)
    destination = output / f'{name}.npz'
    np.savez_compressed(destination, **arrays)
    try:
        commit = subprocess.check_output(['git', '-C', str(Path(source).parent), 'rev-parse', 'HEAD'], text = True).strip()
    except subprocess.CalledProcessError:
        commit = None
    provenance = {
        'asset': name, 'source': str(source).replace(str(Path.home()), '~'),
        'source_sha256': sha(source), 'research_commit': commit,
        'inputs': [{'path': str(p).replace(str(Path.home()), '~'), 'sha256': sha(p)} for p in inputs],
        'snapshot_sha256': sha(destination),
        'arrays': {key: {'shape': list(np.asarray(value).shape), 'dtype': str(np.asarray(value).dtype)}
                   for key, value in arrays.items()},
    }
    (output / f'{name}.json').write_text(json.dumps(provenance, indent = 2) + '\n')
    print(f'{name}: captured {len(arrays)} arrays, {destination.stat().st_size:,} bytes', flush = True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('asset')
    parser.add_argument('--sources', type = Path, default = Path(__file__).with_name('research-sources.example.json'))
    parser.add_argument('--output', type = Path, required = True)
    arguments = parser.parse_args()
    sources = json.loads(arguments.sources.read_text())
    capture(arguments.asset, Path(sources[arguments.asset]).expanduser(), arguments.output)
