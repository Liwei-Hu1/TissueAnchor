"""Context-isolated iMED v2 input for the unchanged shared NeRF renderer.

Only source-camera (endoscope1,left) observations condition the model. The right
view contributes training-only stereo pseudo-depth; endoscope2 is never loaded.
48context/25test/127adapt partitions are fixed by the CPU preparation manifest.
Depth is nominal-baseline pseudo-depth, not independent geometry ground truth.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from tissueanchor.datasets.stereomis import SharedScene



def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ImedSharedScene:
    def __init__(self,case_id,root,include_adaptation_targets=False,
                 load_target_rgb=True,source_only=False):
        self.name='iMED_'+case_id;self.root=Path(root)/case_id;self.camera=self.root/'endoscope1'
        self.record=json.loads((self.root/'manifest.json').read_text())
        self.split=json.loads((self.root/'split_manifest.json').read_text())
        self.calibration_path=self.camera/'calibration.json'
        self.calibration=json.loads(self.calibration_path.read_text())
        if self.record.get('preprocessing_profile')!='author_stereo_v2' or self.calibration.get('additional_rectification_applied') is not False:
            raise ValueError('Only author_stereo_v2 published-pixel inputs are permitted')
        expected={'all':200,'test':25,'training':175,'context':48,'adaptation':127}
        if self.split['counts']!=expected:raise ValueError('iMED split mismatch')
        self.context=list(self.split['context']);self.test=list(self.split['test'])
        self.permitted_train=list(self.split['training']);self.adaptation_pool=list(self.split['adaptation'])
        assert not set(self.context)&set(self.test)
        assert not set(self.context)&set(self.adaptation_pool)
        assert set(self.context)|set(self.adaptation_pool)==set(self.permitted_train)
        assert not set(self.permitted_train)&set(self.test)
        self.targets=list(self.adaptation_pool) if include_adaptation_targets and not source_only else []
        self.train=self.context+self.targets
        needed=set(self.context)|set(self.targets)
        missing=[i for i in sorted(needed) if not (self.camera/'stereo_depth_mm'/f'{i:05d}.npy').is_file()]
        missing_valid=[i for i in sorted(needed) if not (self.camera/'stereo_depth_valid'/f'{i:05d}.png').is_file()]
        if missing or missing_valid:
            raise FileNotFoundError(f'iMED blocked: permitted stereo pseudo-priors are missing for {case_id}; depth={len(missing)}, valid={len(missing_valid)}. First depth indices={missing[:8]}. Generate context/training priors only; test depths must remain absent.')
        contaminated=[i for i in self.test if (self.camera/'stereo_depth_mm'/f'{i:05d}.npy').exists()]
        if contaminated:raise RuntimeError('Heldout-depth contamination: '+str(contaminated))
        self.depth_manifest=json.loads((self.camera/'depth_manifest.json').read_text())
        self.depth_valid_range_mm=tuple(self.depth_manifest['depth_valid_range_for_shared_loader_mm'])
        if self.depth_valid_range_mm!=(2.,200.):raise ValueError('Undeclared shared-loader prior validity range')
        self.w,self.h=self.record['image_wh'];n=self.record['frames']
        self.K=torch.tensor(self.calibration['K'],dtype=torch.float32)
        self.meta={'frame_ids':[f'{i:05d}' for i in range(n)],'width':self.w,'height':self.h,
                   'K':self.K.tolist(),'train_indices':self.permitted_train,'test_indices':self.test}
        self.original_positions=list(range(n));self.original_frame_count=n
        self.adaptation_original_positions=list(self.adaptation_pool)
        self.times=torch.from_numpy(np.load(self.root/'times.npy').astype('float32'))
        if self.times.shape!=(n,):raise ValueError('Time shape mismatch')
        roi=np.asarray(Image.open(self.camera/'rectification_valid_roi_left.png'))>127
        if roi.shape!=(self.h,self.w):raise ValueError('ROI shape mismatch')
        rgb_needed=set(self.context)|set(self.targets)
        if load_target_rgb and not source_only:rgb_needed|=set(self.test)
        images=[];masks=[];depth=[];valids=[]
        for i in range(n):
            fid=f'{i:05d}'
            if i in rgb_needed:
                image=np.array(Image.open(self.camera/'left'/f'{fid}.png').convert('RGB'))
                mask=(np.asarray(Image.open(self.camera/'tissue_masks_left'/f'{fid}.png'))>127)&roi
                if image.shape!=(self.h,self.w,3) or mask.shape!=(self.h,self.w):raise ValueError('Image/mask shape mismatch')
            else:
                image=np.zeros((self.h,self.w,3),dtype='uint8');mask=np.zeros((self.h,self.w),dtype=bool)
            # The held-out branch never calls np.load on any prior, even if one
            # appears on disk later; contamination is separately rejected above.
            if i in needed:
                assert i not in self.test
                d=np.load(self.camera/'stereo_depth_mm'/f'{fid}.npy').astype('float32')
                raw_valid=np.asarray(Image.open(self.camera/'stereo_depth_valid'/f'{fid}.png'))>127
                valid=np.isfinite(d)&raw_valid&(d>=2.)&(d<=200.)&roi
                d=np.where(valid,d/10.,0.).astype('float32')
            else:d=np.zeros((self.h,self.w),dtype='float32');valid=np.zeros((self.h,self.w),dtype=bool)
            images.append(torch.from_numpy(image));masks.append(torch.from_numpy(mask))
            depth.append(torch.from_numpy(d));valids.append(torch.from_numpy(valid))
        self.images=torch.stack(images);self.masks=torch.stack(masks)
        depth=torch.stack(depth);self.depth_valid=torch.stack(valids)
        source_valid=self.masks[self.context]&self.depth_valid[self.context]
        values=depth[self.context][source_valid]
        if not len(values):raise ValueError('No valid context tissue pseudo-depth in declared2-200mm range')
        self.scale_cm=float(values.median());self.depths=depth/self.scale_cm
        sample=values.numpy().reshape(-1)
        if len(sample)>1_000_000:sample=sample[np.linspace(0,len(sample)-1,1_000_000,dtype=np.int64)]
        q005,q995=np.quantile(sample,[.005,.995])
        self.near=max(.02,float(q005)*.65/self.scale_cm);self.far=float(q995)*1.35/self.scale_cm
        poses=np.load(self.camera/'c2w_m.npy').astype('float32')
        if poses.shape!=(n,4,4):raise ValueError('Pose shape mismatch')
        poses[:,:3,3]*=100./self.scale_cm # Source poses metres -> normalized centimetres.
        self.c2w=torch.from_numpy(poses);self.w2c=torch.linalg.inv(self.c2w)
        self.context_times=self.times[self.context].contiguous()
        self.pixels=[torch.nonzero(mask,as_tuple=False) for mask in self.masks]
        self.depth_output_folder='depth_pseudo_cm';self._gpu_context=None
        self._source_depth_sha={f'{i:05d}':file_sha(self.camera/'stereo_depth_mm'/f'{i:05d}.npy') for i in self.context}

    context_tensors=SharedScene.context_tensors
    rays=SharedScene.rays

    def manifest(self):
        # Independent of source_only/adaptation/test-RGB load flags: all entries
        # below describe the same48conditioning observations and fixed split.
        return {'scene':self.name,'dataset':'iMED v2 selected subset','source':str(self.root),
            'session':self.record['session'],'organ_label':self.record['organ_label'],
            'tissue_type':self.record['tissue_type'],'independent_subject_id':self.record['independent_subject_id'],
            'frame_ids':self.meta['frame_ids'],'image_wh':[self.w,self.h],
            'context_indices':self.context,'original_test_indices':self.test,
            'permitted_adaptation_indices':self.adaptation_pool,'source_context_count':48,
            'test_count':25,'adaptation_pool_count':127,'context_target_disjoint':True,
            'K':self.K.tolist(),'calibration_sha256':file_sha(self.calibration_path),
            'split_manifest_sha256':file_sha(self.root/'split_manifest.json'),
            'pose_sha256':file_sha(self.camera/'c2w_m.npy'),'timestamps_sha256':file_sha(self.root/'times.npy'),
            'source_depth_sha256':self._source_depth_sha,'source_depth_scale_pseudo_cm':self.scale_cm,
            'depth':'RAFT-Stereo pseudo-prior with nominal4.1mm baseline and author45px padding at640; not independent metric geometry truth',
            'depth_valid_range_mm':[2.,200.],'near':self.near,'far':self.far,
            'mask':'Released tissue masks, white=valid tissue, intersected with left-eye ROI',
            'camera':'endoscope1-left only; fixed source scene frame; source poses converted metres to pseudo-centimetre normalization',
            'time':'Released L timestamps normalized to selected clip duration',
            'heldout_RGB_in_conditioning':False,'heldout_depth_predicted_or_loaded':False,
            'second_endoscope_used':False,'preprocessing_profile':'author_stereo_v2'}
