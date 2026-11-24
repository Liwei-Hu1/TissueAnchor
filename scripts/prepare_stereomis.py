"""Create StereoMIS input metadata and links to a local tracking release."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import yaml

from tissueanchor.datasets.stereomis import atomic_json, TRAIN_SCENES, VAL_SCENES, TEST_SCENES


def prepare(source, calibration, dest):
    images = sorted((source/'video_frames').glob('*l.png'))
    pose_file = source/'groundtruth.txt'
    poses = np.loadtxt(pose_file)
    lookup = {int(row[0]): row for row in poses}
    ids = [image.stem[:-1] for image in images]
    if len(lookup) != len(poses) or len(ids) != 200 or not all(int(key) in lookup for key in ids):
        raise ValueError(f'Expected 200 left frames with aligned, unique poses in {source}')
    rows = np.stack([lookup[int(key)] for key in ids])
    c2w = np.tile(np.eye(4), (len(rows), 1, 1))
    c2w[:, :3, :3] = Rotation.from_quat(rows[:, 4:8]).as_matrix()
    c2w[:, :3, 3] = rows[:, 1:4]
    c2w = np.linalg.inv(c2w[0])[None] @ c2w
    c2w[:, :3, 3] *= 100
    time = (rows[:, 0] - rows[0, 0]) / (rows[-1, 0] - rows[0, 0])
    camera = yaml.safe_load(calibration.read_text())['cam']
    for key in ('rect_left', 'rect_right', 'tissue_masks', 'stereo_depth'):
        (dest/key).mkdir(parents=True, exist_ok=True)

    for frame, left in zip(ids, images):
        with Image.open(left) as image:
            if image.size != (640, 512):
                raise ValueError(f'Unexpected image dimensions in {left}')
        for folder, original in (
            ('rect_left', left), ('rect_right', source/'video_frames'/f'{frame}r.png'),
            ('tissue_masks', source/'masks'/left.name),
        ):
            target = dest/folder/f'{frame}.png'
            if not original.is_file():
                raise FileNotFoundError(original)
            if target.is_symlink() and target.resolve() == original.resolve():
                continue
            target.symlink_to(original.resolve())
    test = list(range(1, len(ids), 8))
    np.save(dest/'c2w_cm.npy', c2w.astype(np.float32))
    np.save(dest/'times.npy', time.astype(np.float32))
    atomic_json(dest/'input_manifest.json', {
        'frame_ids': ids, 'width': 640, 'height': 512,
        'K': [[camera['fx'], 0, camera['cx']], [0, camera['fy'], camera['cy']], [0, 0, 1]],
        'baseline_mm': camera['stereo_baseline']/camera['fx'],
        'train_indices': [i for i in range(len(ids)) if i not in test],
        'test_indices': test, 'raw_pose_sha256': hashlib.sha256(pose_file.read_bytes()).hexdigest(),
        'calibration_sha256': hashlib.sha256(calibration.read_bytes()).hexdigest(),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', required=True, type=Path)
    parser.add_argument('--camera-yaml-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--scenes', nargs='+', choices=TRAIN_SCENES+VAL_SCENES+TEST_SCENES,
                        default=TRAIN_SCENES+VAL_SCENES+TEST_SCENES)
    parser.add_argument('--p2-2-calibration-pair', nargs=2, type=Path, metavar=('P2_0_INI', 'P2_2_INI'))
    args = parser.parse_args()
    if 'P2_2' in args.scenes and not (args.camera_yaml_dir/'P2_2.yaml').is_file():
        pair = args.p2_2_calibration_pair
        if not pair or pair[0].read_bytes() != pair[1].read_bytes():
            parser.error('To reuse P2_0.yaml, supply the identical official P2_0/P2_2 calibration INI files')
    for scene in args.scenes:
        yaml_path = args.camera_yaml_dir/f'{scene}.yaml'
        if scene == 'P2_2' and not yaml_path.is_file():
            yaml_path = args.camera_yaml_dir/'P2_0.yaml'
        prepare(args.raw_root/scene, yaml_path, args.output/scene)


if __name__ == '__main__':
    main()
