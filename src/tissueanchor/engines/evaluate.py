"""Score saved RGB using same-name references and a common tissue mask."""
import argparse
from pathlib import Path
from statistics import mean

import numpy as np
from PIL import Image
import torch

from tissueanchor.datasets.stereomis import atomic_json
from tissueanchor.metrics.image import masked_inputs, ssim


def read_image(path, mode, device):
    with Image.open(path) as image:
        return torch.from_numpy(np.array(image.convert(mode))).to(device)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('predictions', 'references', 'masks', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    frames = sorted(p.name for p in args.predictions.glob('*.png'))
    if not frames:
        parser.error('No PNG predictions found')
    for folder in (args.references, args.masks):
        if sorted(p.name for p in folder.glob('*.png')) != frames:
            parser.error(f'Frame IDs differ in {folder}')
    if args.output.exists():
        parser.error('Choose a fresh output JSON file')

    import lpips
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    perceptual = lpips.LPIPS(net='alex').to(args.device).eval()
    rows = []
    for name in frames:
        pred = read_image(args.predictions/name, 'RGB', args.device)
        ref = read_image(args.references/name, 'RGB', args.device)
        mask = read_image(args.masks/name, 'L', args.device) > 127
        psnr, x, y = masked_inputs(pred, ref, mask)
        rows.append({'frame_id': Path(name).stem, 'PSNR': float(psnr), 'SSIM': float(ssim(x, y)),
                     'LPIPS': float(perceptual(2*x-1, 2*y-1).mean()), 'mask_pixels': int(mask.sum())})
    atomic_json(args.output, {
        'metrics': {key: mean(row[key] for row in rows) for key in ('PSNR', 'SSIM', 'LPIPS')},
        'frames': len(rows), 'rows': rows,
    })


if __name__ == '__main__':
    main()
