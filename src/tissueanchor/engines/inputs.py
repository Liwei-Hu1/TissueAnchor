"""Construct the registered input-camera scene for each supported data format."""
from pathlib import Path

from tissueanchor.datasets.drends import DRENDSSharedScene
from tissueanchor.datasets.endonerf import EndoNeRFSharedScene
from tissueanchor.datasets.imed import ImedSharedScene
from tissueanchor.datasets.stereomis import SharedScene

DATASETS = ('stereomis', 'drends', 'endonerf', 'imed')


def open_scene(dataset, name, root, *, adapt=False):
    root = Path(root).resolve()
    if dataset == 'stereomis':
        return SharedScene(name, root=root, source_only=not adapt, load_test_rgb=False)
    if dataset == 'drends':
        return DRENDSSharedScene(
            name, root=root, width=960, eval_frames=0, load_target_rgb=False,
            include_adaptation_targets=adapt, adaptation_target_count=100,
            tmi_resize=True, clip_prior_like_tmi=True,
        )
    if dataset == 'endonerf':
        return EndoNeRFSharedScene(
            name, root=root, include_adaptation_targets=adapt, load_test_rgb=False,
        )
    if dataset == 'imed':
        return ImedSharedScene(
            name, root=root, include_adaptation_targets=adapt, load_target_rgb=False,
        )
    raise ValueError(f'Unknown dataset: {dataset}')


def validate_cache(cache, scene):
    if cache.get('schema') != 'shared_query_cache_v1':
        raise ValueError('Expected a shared_query_cache_v1 cache')
    for key, tail in {'tracks': (48, 1024, 3), 'latent': (1024, 1280),
                      'reliability': (48, 1024)}.items():
        if tuple(cache[key].shape) != tail:
            raise ValueError(f'Cache shape mismatch for {key}: {tuple(cache[key].shape)}')
    saved = dict(cache['manifest'])
    live = scene.manifest()
    for manifest in (saved, live):
        manifest.pop('source', None)
    if live != saved:
        raise ValueError(f'Cache manifest does not match the scene: {scene.name}')
