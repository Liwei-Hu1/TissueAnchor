"""Adapt a saved source field optionally, and render a recording's held-out input view."""
import argparse
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch

from tissueanchor.datasets.stereomis import atomic_json
from tissueanchor.engines.adapt import adapt
from tissueanchor.engines.inputs import DATASETS, open_scene, validate_cache
from tissueanchor.models.shared_adapter import SharedQueryNeRF, render_shared


@torch.inference_mode()
def render_frame(field, scene, cache, index, device, ray_batch):
    field.eval()
    y, x = torch.meshgrid(torch.arange(scene.h), torch.arange(scene.w), indexing='ij')
    pixels = torch.stack((y.flatten(), x.flatten()), dim=-1)
    source = scene.context_tensors(device)
    time = scene.times[index].to(device)
    views = (source['times'] - time).abs().argsort()[:field.source_views]
    prepared = views, field.source_features(source, views), field.anchor_state(cache, time)
    rgb, depth = [], []
    for batch in pixels.split(ray_batch):
        prediction = render_shared(field, scene, cache, index, batch, device, 48, prepared=prepared)
        rgb.append(prediction['rgb'].cpu())
        depth.append(prediction['depth'].cpu())
    image = (torch.cat(rgb).reshape(scene.h, scene.w, 3).clamp(0, 1) * 255).round().to(torch.uint8)
    depth_map = torch.cat(depth).reshape(scene.h, scene.w) * scene.scale_cm
    return image.numpy(), depth_map.numpy()


def default_adaptation_rays(dataset, steps):
    return 512 if dataset == 'stereomis' and steps == 1000 else 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('data-root', 'cache-root', 'checkpoint', 'output'):
        parser.add_argument(f'--{key}', required=True, type=Path)
    parser.add_argument('--dataset', choices=DATASETS, required=True)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--steps', type=int, choices=(0, 100, 1000), default=0)
    parser.add_argument('--adapt-rays', type=int)
    parser.add_argument('--render-rays', type=int, default=2048)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--frame-id', help='Render a single held-out frame using its original ID')
    args = parser.parse_args()
    rays = args.adapt_rays if args.adapt_rays is not None else default_adaptation_rays(args.dataset, args.steps)
    if rays < 1 or args.render_rays < 1:
        parser.error('Ray batch sizes must be positive')

    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable. Use a compatible PyTorch build or --device cpu')
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    scene = open_scene(args.dataset, args.scene, args.data_root, adapt=bool(args.steps))
    frames = [i for i in scene.test if args.frame_id is None or scene.meta['frame_ids'][i] == args.frame_id]
    if not frames:
        parser.error('The requested frame is absent from the held-out partition')
    cache_path = args.cache_root/'cache'/scene.name/'query_cache.pt'
    cache = torch.load(cache_path, weights_only=True, map_location='cpu')
    validate_cache(cache, scene)
    cache = {k: value.to(device) if torch.is_tensor(value) else value for k, value in cache.items()}
    field = SharedQueryNeRF().to(device)
    state = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
    field.load_state_dict(state['field'], strict=True)

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'renders').mkdir()
    (args.output/'depth').mkdir()
    report = {'steps': 0}
    if args.steps:
        if not scene.targets:
            raise ValueError('This scene has no adaptation frames')
        report = adapt(field, scene, cache, device, args.steps, rays, 48)
        torch.save({'field': {k: v.detach().cpu() for k, v in field.state_dict().items()}},
                   args.output/'adapted.pt')
    for index in frames:
        key = scene.meta['frame_ids'][index]
        image, depth = render_frame(field, scene, cache, index, device, args.render_rays)
        Image.fromarray(image).save(args.output/'renders'/f'{key}.png')
        np.save(args.output/'depth'/f'{key}.npy', depth)
    depth_units = dict(stereomis='cm', drends='cm', imed='pseudo-cm', endonerf='released stereo prior units')
    atomic_json(args.output/'render_manifest.json', {
        'frame_ids': [scene.meta['frame_ids'][i] for i in frames],
        'dataset': args.dataset, 'scene': args.scene, 'depth_units': depth_units[args.dataset],
        'adaptation_rays': rays, 'adaptation': report, 'checkpoint': str(args.checkpoint),
    })


if __name__ == '__main__':
    main()
