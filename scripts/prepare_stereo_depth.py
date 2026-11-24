"""Run the recorded DRENDS-Eval RAFT-Stereo adapter on prepared StereoMIS pairs."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import tifffile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--drends-eval', required=True, type=Path)
    parser.add_argument('--scenes', nargs='+', required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.drends_eval.resolve()))
    from evaluation.config import configure_model_caches
    from evaluation.models import get_model

    configure_model_caches()
    model = get_model('raft_stereo')
    model.setup()
    for name in args.scenes:
        scene = args.data_root/name
        config = json.loads((scene/'input_manifest.json').read_text())
        destination = scene/'stereo_depth'
        destination.mkdir(exist_ok=True)
        model.set_calibration(config['K'][0][0], config['baseline_mm'])
        for i in config['train_indices']:
            frame = config['frame_ids'][i]
            output = destination/f'{frame}.tiff'
            if output.is_file():
                continue
            inputs = []
            for eye in ('rect_left', 'rect_right'):
                with Image.open(scene/eye/f'{frame}.png') as image:
                    inputs.append(np.asarray(image.convert('RGB')))
            depth = np.asarray(model.predict(*inputs), dtype=np.float32)
            if depth.shape != (config['height'], config['width']):
                raise ValueError(f'Unexpected RAFT-Stereo depth shape for {name}/{frame}')
            temp = destination/f'{frame}.partial.tiff'
            tifffile.imwrite(temp, depth)
            temp.replace(output)


if __name__ == '__main__':
    main()
