"""Source/target-isolated StereoMIS inputs for shared conditional training."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image
import tifffile
import torch

TRAIN_SCENES = ['P1_1', 'P2_0', 'P2_1', 'P2_2']
VAL_SCENES = ['P2_6']
TEST_SCENES = ['P3_1', 'P3_2']


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


class SharedScene:
    def __init__(self, name, device='cpu', source_only=False, *, root, load_test_rgb=True):
        self.name = name
        self.root = Path(root) / name
        self.meta = json.loads((self.root / 'input_manifest.json').read_text())
        self.h, self.w = self.meta['height'], self.meta['width']
        self.train = self.meta['train_indices']
        self.test = self.meta['test_indices']
        self.context = np.asarray(self.train)[np.linspace(0, len(self.train)-1, 48).round().astype(int)].tolist()
        self.targets = [i for i in self.train if i not in self.context]
        assert len(set(self.context)) == 48
        assert not set(self.context) & (set(self.targets) | set(self.test))
        self.K = torch.tensor(self.meta['K'], dtype=torch.float32)
        self.times = torch.from_numpy(np.load(self.root / 'times.npy').astype(np.float32))
        self.images = []
        self.masks = []
        self.depths = []
        for index,fid in enumerate(self.meta['frame_ids']):
            if (source_only and index not in self.context) or (not load_test_rgb and index in self.test):
                self.images.append(torch.zeros(self.h,self.w,3,dtype=torch.uint8))
                self.masks.append(torch.zeros(self.h,self.w,dtype=torch.bool))
                self.depths.append(torch.zeros(self.h,self.w,dtype=torch.float32))
                continue
            self.images.append(torch.from_numpy(np.array(Image.open(self.root/'rect_left'/f'{fid}.png').convert('RGB'))))
            self.masks.append(torch.from_numpy(np.array(Image.open(self.root/'tissue_masks'/f'{fid}.png')) > 0))
            d = np.zeros((self.h, self.w), dtype=np.float32)
            if index not in self.test:
                d = np.asarray(tifffile.imread(self.root/'stereo_depth'/f'{fid}.tiff'), dtype=np.float32)/10
            self.depths.append(torch.from_numpy(np.nan_to_num(d, nan=0., posinf=0., neginf=0.)))
        self.images = torch.stack(self.images)
        self.masks = torch.stack(self.masks)
        depth = torch.stack(self.depths)
        valid = self.masks[self.context] & (depth[self.context] > 1) & (depth[self.context] < 100)
        vals = depth[self.context][valid]
        if not len(vals):
            raise ValueError(f'{name}: no valid tissue depth in the context frames')
        self.scale_cm = float(vals.median())
        self.depths = depth / self.scale_cm
        self.depth_valid = (depth > 1) & (depth < 100)
        self.c2w = torch.from_numpy(np.load(self.root/'c2w_cm.npy').astype(np.float32))
        self.c2w[:, :3, 3] /= self.scale_cm
        self.w2c = torch.linalg.inv(self.c2w)
        self.near = max(.02, float(torch.quantile(vals, .005))*.65/self.scale_cm)
        self.far = float(torch.quantile(vals, .995))*1.35/self.scale_cm
        self.context_times = self.times[self.context].contiguous()
        self.device = torch.device(device)
        self.pixels = [torch.nonzero(m, as_tuple=False) for m in self.masks]
        self._gpu_context = None

    def context_tensors(self, device):
        # Materialized CUDA tensors report cuda:0, whereas torch.device('cuda')
        # has index=None. Resolve aliases before comparing cached storage.
        device = torch.device(device)
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        if self._gpu_context is None or self._gpu_context['images'].device != torch.device(device):
            self._gpu_context = {
                'images': self.images[self.context].to(device).permute(0,3,1,2).float()/255,
                'masks': self.masks[self.context].to(device).float()[:,None],
                'depths': self.depths[self.context].to(device)[:,None],
                'depth_valid': self.depth_valid[self.context].to(device).float()[:,None],
                'c2w': self.c2w[self.context].to(device),
                'w2c': self.w2c[self.context].to(device), 'K': self.K.to(device),
                'times': self.context_times.to(device),
            }
        return self._gpu_context

    def rays(self, index, pixels, device):
        y, x = pixels[:,0].to(device), pixels[:,1].to(device)
        K = self.K.to(device)
        pose = self.c2w[index].to(device)
        cam = torch.stack([(x-K[0,2])/K[0,0], (y-K[1,2])/K[1,1], torch.ones_like(x)], -1)
        return pose[:3,3].expand(len(x),-1), cam @ pose[:3,:3].T

    def manifest(self):
        result = {
            'scene': self.name, 'animal': self.name.split('_')[0],
            'frame_ids': self.meta['frame_ids'], 'context_indices': self.context,
            'shared_train_or_adaptation_target_indices': self.targets,
            'original_test_indices': self.test, 'context_target_disjoint': True,
            'source_depth_scale_cm': self.scale_cm,
            'normalization': 'camera translations and priors divided by source-context median depth in cm',
            'near': self.near, 'far': self.far,
            'depth': 'RAFT-Stereo pseudo prior, not independent GT; context depth only for normalization/features',
            'mask': 'released automatic tissue masks; no manual segmentation GT',
            'source': str(self.root), 'manifest_sha256': hashlib.sha256((self.root/'input_manifest.json').read_bytes()).hexdigest(),
            'no_target_rgb_in_encoder_query_patch_or_HR_branch': True,
            'source_processing': 'per-pair stereo estimates and released per-frame masks; no RIFE frames used',
        }
        return result
