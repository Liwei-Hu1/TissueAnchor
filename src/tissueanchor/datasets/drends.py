"""DRENDS development adapter for frozen-source conditional reconstruction.

This loader reads RGB, estimated stereo depth, SAM2 masks and camera/timestamp
metadata only. Independent sensor references are opened by the evaluator only.
"""
import csv
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import tifffile
import torch
from tissueanchor.datasets.stereomis import SharedScene


def _resize_rgb(array,w,h,tmi_match):
    if array.shape[1::-1]==(w,h):return array
    if tmi_match:return cv2.resize(array,(w,h),interpolation=cv2.INTER_AREA)
    return np.array(Image.fromarray(array).resize((w,h),Image.Resampling.BOX))


def _resize_mask(array,w,h,tmi_match):
    if array.shape[1::-1]==(w,h):return array
    if tmi_match:return cv2.resize(array,(w,h),interpolation=cv2.INTER_NEAREST)
    return np.array(Image.fromarray(array).resize((w,h),Image.Resampling.NEAREST))


def _resize_depth_mm(array,w,h,tmi_match,clip_like_tmi):
    if array.shape!=(h,w):
        if tmi_match:array=cv2.resize(array,(w,h),interpolation=cv2.INTER_LINEAR)
        else:array=np.array(Image.fromarray(array).resize((w,h),Image.Resampling.NEAREST))
    valid=np.isfinite(array)&(array>0)
    array=array.astype(np.float32,copy=True)
    if clip_like_tmi and valid.any():
        lo,hi=np.percentile(array[valid],(3.0,99.8))
        array[valid]=np.clip(array[valid],lo,hi)
    return np.where(valid,array/10.,0.).astype(np.float32)


class DRENDSSharedScene:
    def __init__(self,sequence='Vid00_Colon_Ext',width=640,eval_frames=8,
                 load_target_rgb=True,include_adaptation_targets=False,
                 adaptation_target_count=None,tmi_resize=False,
                 clip_prior_like_tmi=False, *, root):
        self.name='DRENDS_'+sequence+'_dev'
        self.tmi_resize=tmi_resize;self.clip_prior_like_tmi=clip_prior_like_tmi
        self.root=Path(root)/sequence
        paths=sorted((self.root/'rect_left').glob('*.png'))
        if not paths:raise FileNotFoundError(self.root/'rect_left')
        original_ids=[p.stem for p in paths];n=len(paths)
        train=[i for i in range(n) if (i-1)%8!=0]
        test=[i for i in range(n) if (i-1)%8==0]
        source=np.asarray(train)[np.linspace(0,len(train)-1,48).round().astype(int)].tolist()
        held=(np.asarray(test)[np.linspace(0,len(test)-1,eval_frames).round().astype(int)].tolist()
              if eval_frames else list(test))
        assert not set(source)&set(held) and len(set(source))==48
        self.original_positions=source+held
        self.context=list(range(48));self.train=self.context;self.targets=[]
        self.test=list(range(48,48+len(held)))
        self.calibration_path=self.root.parent/'calibration.json'
        calib=json.loads(self.calibration_path.read_text())
        ow,oh=calib['img_size'];self.w=width;self.h=round(oh*width/ow)
        sx,sy=self.w/ow,self.h/oh
        K=np.array(calib['left_P'],dtype=np.float32)[:,:3].copy()
        K[0,0]*=sx;K[1,1]*=sy
        K[0,2]=sx*(K[0,2]+.5)-.5;K[1,2]=sy*(K[1,2]+.5)-.5
        self.K=torch.from_numpy(K)
        with (self.root/'timestamps.csv').open() as f:rows=list(csv.DictReader(f))
        assert len(rows)>=n
        ts=np.array([float(r['left']) for r in rows[:n]],dtype=np.float64)
        ts=(ts-ts[0])/max(ts[-1]-ts[0],1e-8)
        self.times=torch.tensor(ts[self.original_positions],dtype=torch.float32)
        self.context_times=self.times[self.context].contiguous()
        images=[];masks=[];depths=[]
        for slot,original_index in enumerate(self.original_positions):
            fid=original_ids[original_index]
            if slot<48 or load_target_rgb:
                with Image.open(paths[original_index]) as im:
                    rgb=np.array(im.convert('RGB'))
                rgb=_resize_rgb(rgb,self.w,self.h,self.tmi_resize)
            else:rgb=np.zeros((self.h,self.w,3),dtype=np.uint8)
            if slot < 48 or load_target_rgb:
                with Image.open(self.root/'tool_masks_left'/f'{fid}.png') as im:
                    tool = _resize_mask(np.array(im.convert('L')), self.w, self.h, self.tmi_resize)
            else:
                tool = np.full((self.h, self.w), 255, dtype=np.uint8)
            images.append(torch.from_numpy(rgb));masks.append(torch.from_numpy(tool<128))
            # No held-out prior is needed to build rays, normalize or condition.
            if slot<48:
                d=np.asarray(tifffile.imread(self.root/'stereo_depth'/f'{fid}.tiff'),dtype=np.float32)
                d=np.nan_to_num(d,nan=0.,posinf=0.,neginf=0.)
                d=_resize_depth_mm(d,self.w,self.h,self.tmi_resize,self.clip_prior_like_tmi)
            else:d=np.zeros((self.h,self.w),dtype=np.float32)
            depths.append(torch.from_numpy(d.astype(np.float32)))
        self.images=torch.stack(images);self.masks=torch.stack(masks)
        depth=torch.stack(depths)
        valid=self.masks[:48]&(depth[:48]>1)&(depth[:48]<100)
        values=depth[:48][valid]
        self.scale_cm=float(values.median())
        self.depths=depth/self.scale_cm;self.depth_valid=(depth>1)&(depth<100)
        # torch.quantile has a hard-size limit on this many source pixels.
        # A deterministic evenly spaced sample gives the same robust bounds
        # without ever touching an independent ToF reference.
        values_np=values.numpy().reshape(-1)
        if len(values_np)>1_000_000:
            qidx=np.linspace(0,len(values_np)-1,1_000_000,dtype=np.int64)
            values_np=values_np[qidx]
        self.depth_bound_quantile_sample_size=len(values_np)
        q005,q995=np.quantile(values_np,(.005,.995))
        self.near=max(.02,float(q005)*.65/self.scale_cm)
        self.far=float(q995)*1.35/self.scale_cm
        self.c2w=torch.eye(4).repeat(len(self.original_positions),1,1)
        self.w2c=self.c2w.clone();self._gpu_context=None
        self.pixels=[torch.nonzero(m,as_tuple=False) for m in self.masks]
        self.meta={'frame_ids':[original_ids[i] for i in self.original_positions]}
        self.original_frame_count=n;self.full_heldout_count=len(test)
        self.adaptation_original_positions=[]
        if include_adaptation_targets:
            # Append only non-source, non-test training frames. The existing
            # manifest remains source/test-only, so the frozen teacher cache is
            # reproducible without opening held-out sensor references.
            targets=[i for i in train if i not in set(source)]
            if adaptation_target_count and len(targets)>adaptation_target_count:
                selected=np.linspace(0,len(targets)-1,adaptation_target_count).round().astype(int)
                targets=[targets[i] for i in selected]
            new_images=[];new_masks=[];new_depths=[];new_valid=[]
            for original_index in targets:
                fid=original_ids[original_index]
                with Image.open(paths[original_index]) as im:
                    rgb=np.array(im.convert('RGB'))
                rgb=_resize_rgb(rgb,self.w,self.h,self.tmi_resize)
                with Image.open(self.root/'tool_masks_left'/f'{fid}.png') as im:
                    tool=np.array(im.convert('L'))
                tool=_resize_mask(tool,self.w,self.h,self.tmi_resize)
                d=np.asarray(tifffile.imread(self.root/'stereo_depth'/f'{fid}.tiff'),dtype=np.float32)
                d=np.nan_to_num(d,nan=0.,posinf=0.,neginf=0.)
                d=_resize_depth_mm(d,self.w,self.h,self.tmi_resize,self.clip_prior_like_tmi)
                new_images.append(torch.from_numpy(rgb))
                new_masks.append(torch.from_numpy(tool<128))
                new_depths.append(torch.from_numpy(d.astype(np.float32)/self.scale_cm))
                new_valid.append(torch.from_numpy((d>1.)&(d<100.)))
            target_start=len(self.images)
            if targets:
                self.images=torch.cat([self.images,torch.stack(new_images)],dim=0)
                self.masks=torch.cat([self.masks,torch.stack(new_masks)],dim=0)
                self.depths=torch.cat([self.depths,torch.stack(new_depths)],dim=0)
                self.depth_valid=torch.cat([self.depth_valid,torch.stack(new_valid)],dim=0)
                self.times=torch.cat([self.times,torch.tensor(ts[targets],dtype=torch.float32)])
                target_poses=torch.eye(4).repeat(len(targets),1,1)
                self.c2w=torch.cat([self.c2w,target_poses],dim=0)
                self.w2c=torch.cat([self.w2c,target_poses.clone()],dim=0)
                self.pixels.extend(torch.nonzero(m,as_tuple=False) for m in new_masks)
                self.targets=list(range(target_start,target_start+len(targets)))
                self.adaptation_original_positions=targets

    context_tensors=SharedScene.context_tensors
    rays=SharedScene.rays

    def manifest(self):
        return {'scene':self.name,'dataset':'DRENDS ExVivo','sequence':self.root.name,
            'source':str(self.root),'frame_ids':self.meta['frame_ids'],
            'context_indices':self.context,'test_indices':self.test,
            'original_source_positions':self.original_positions[:48],
            'original_evaluation_positions':self.original_positions[48:],
            'original_frame_count':self.original_frame_count,'full_original_test_frames':self.full_heldout_count,
            'source_depth_scale_cm':self.scale_cm,'near':self.near,'far':self.far,
            'near_far_quantile_sample_size':self.depth_bound_quantile_sample_size,
            'image_wh':[self.w,self.h],'K':self.K.tolist(),
            'camera':'fixed rectified left-camera coordinates, identity c2w',
            'resize':('TMI-matched OpenCV INTER_AREA RGB, INTER_NEAREST mask and INTER_LINEAR depth'
                      if self.tmi_resize else 'PIL BOX RGB; nearest stereo prior/mask'),
            'clip_stereo_prior_per_frame':self.clip_prior_like_tmi,
            'calibration_sha256':hashlib.sha256(self.calibration_path.read_bytes()).hexdigest(),
            'context_target_disjoint':True,'heldout_RGB_in_conditioning':False,
            'heldout_stereo_prior_loaded':False,'independent_sensor_reference_loaded':False,
            'label':'development/interface check: Vid00 was used in previous project analysis, not untouched external test'}
