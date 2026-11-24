"""Shared point-query adapter with motion-aligned source appearance and NeRF.

The cached teacher is frozen. All trainable modules below are shared across
videos. Density receives no render direction or target camera identifier.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F
from .conditioned_nerf import fourier, volume_render


def interpolate_tracks(cache, time):
    ts = cache['context_times']
    t = torch.as_tensor(time,device=ts.device).clamp(ts[0],ts[-1]).contiguous()
    hi = torch.searchsorted(ts,t,right=True).clamp(1,len(ts)-1)
    lo = hi-1
    a = (t-ts[lo])/(ts[hi]-ts[lo]).clamp_min(1e-8)
    return (1-a)*cache['tracks'][lo]+a*cache['tracks'][hi], (1-a)*cache['reliability'][lo]+a*cache['reliability'][hi]


class SharedQueryNeRF(nn.Module):
    def __init__(self, width=128, neighbours=8, source_views=4, use_d4rt=True, motion=True):
        super().__init__()
        self.neighbours=neighbours; self.source_views=source_views
        self.use_d4rt=use_d4rt; self.motion=motion
        self.latent_proj=nn.Sequential(nn.LayerNorm(1280),nn.Linear(1280,256),nn.GELU(),nn.Linear(256,width))
        self.motion_residual=nn.Sequential(nn.Linear(width+7,64),nn.SiLU(),nn.Linear(64,3))
        nn.init.zeros_(self.motion_residual[-1].weight); nn.init.zeros_(self.motion_residual[-1].bias)
        self.geometry=nn.Sequential(nn.Linear(width+27+27+7+1,width),nn.SiLU(),
            nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.sigma=nn.Linear(width,1); nn.init.constant_(self.sigma.bias,1.)
        self.log_bandwidth=nn.Parameter(torch.tensor(math.log(.12)))
        self.texture_half=nn.Sequential(nn.Conv2d(3,16,3,stride=2,padding=1),nn.SiLU(),nn.Conv2d(16,16,3,padding=1),nn.SiLU())
        self.texture_quarter=nn.Sequential(nn.Conv2d(16,32,3,stride=2,padding=1),nn.SiLU(),nn.Conv2d(32,32,3,padding=1),nn.SiLU())
        self.view_score=nn.Sequential(nn.Linear(width+48+3+3+2,64),nn.SiLU(),nn.Linear(64,1))
        self.color_residual=nn.Sequential(nn.Linear(width+48+27,64),nn.SiLU(),nn.Linear(64,3))
        nn.init.zeros_(self.color_residual[-1].weight);nn.init.zeros_(self.color_residual[-1].bias)
        self.fallback_color=nn.Linear(width+27,3)

    def anchor_state(self,cache,time):
        features=self.latent_proj(cache['latent'].float())
        if not self.use_d4rt:
            features=features*0
        position, reliability=interpolate_tracks(cache,time)
        te=fourier(time.reshape(1,1),3).expand(len(features),-1)
        offset=.05*torch.tanh(self.motion_residual(torch.cat([features,te],-1)))
        return position+offset,features,reliability,offset

    def geometry_at(self,xyz,time,state):
        anchors,features,reliability,_=state
        # Neighbor identity is discrete. Recompute differentiable distances after
        # indexing so learned residual positions still receive a gradient.
        with torch.no_grad():
            distance=torch.cdist(xyz.detach()[None],anchors.detach()[None])[0]
            indices=distance.topk(self.neighbours,dim=-1,largest=False).indices
        relative=xyz[:,None]-anchors[indices]
        squared=relative.square().sum(-1)
        r=reliability[indices].clamp_min(.001)
        weights=torch.softmax(-squared/.12**2+torch.log(r),-1)
        aggregate=(features[indices]*weights[...,None]).sum(1)
        delta=(relative*weights[...,None]).sum(1)
        nearest=squared.min(-1).values.clamp_min(1e-12).sqrt()
        t=time.reshape(1,1).expand(len(xyz),1)
        g=self.geometry(torch.cat([aggregate,fourier(delta,4),fourier(xyz,4),fourier(t,3),nearest[:,None]],-1))
        band=self.log_bandwidth.exp().clamp(.03,.3)
        support=torch.exp(-nearest.square()/(band*band))
        density=F.softplus(self.sigma(g)).squeeze(-1)*15*support
        return density,g,indices,weights,nearest

    def source_features(self,source,selected):
        images=source['images'][selected]
        a=self.texture_half(images);b=self.texture_quarter(a)
        return images,a,b

    def appearance_at(self,xyz,time,direction,state,geometry,indices,weights,source,cache,selected,textures):
        anchors,features,reliability,offset=state
        images,half,quarter=textures
        target_tracks=anchors
        # The same shared residual is evaluated at each source time as at target.
        source_t=source['times'][selected]
        te=fourier(source_t[:,None],3)[:,None].expand(-1,len(features),-1)
        feat=features[None].expand(len(selected),-1,-1)
        source_offset=.05*torch.tanh(self.motion_residual(torch.cat([feat,te],-1)))
        source_tracks=cache['tracks'][selected]+source_offset
        movement=source_tracks-target_tracks[None]
        displacement=(movement[:,indices]*weights[None,:,:,None]).sum(2)
        if not self.motion:
            displacement=displacement*0
        warped=xyz[None]+displacement
        w2c=source['w2c'][selected]
        cam=torch.einsum('vij,vnj->vni',w2c[:,:3,:3],warped)+w2c[:,:3,3][:,None]
        z=cam[...,2]
        K=source['K']
        u=K[0,0]*cam[...,0]/z.clamp_min(.001)+K[0,2]
        v=K[1,1]*cam[...,1]/z.clamp_min(.001)+K[1,2]
        h,w=images.shape[-2:]
        grid=torch.stack([2*(u+.5)/w-1,2*(v+.5)/h-1],-1)[:,None]
        def sample(img):
            return F.grid_sample(img,grid,mode='bilinear',align_corners=False,padding_mode='zeros')[:,:,0].permute(0,2,1)
        rgb=sample(images);appearance=torch.cat([sample(half),sample(quarter)],-1)
        prior=sample(source['depths'][selected])[...,0]
        mask=sample(source['masks'][selected])[...,0]
        valid_prior=sample(source['depth_valid'][selected])[...,0]
        valid=(z>.01)&(grid[:,0].abs().amax(-1)<1)&(mask>.5)&(valid_prior>.5)
        agreement=torch.exp(-torch.abs(torch.log(z.clamp_min(.01)/prior.clamp_min(.01)))/.3)
        anchor_rel=(cache['reliability'][selected][:,indices]*weights[None]).sum(-1)
        reliability=valid.float()*agreement*anchor_rel.clamp_min(.01)
        viewdir=F.normalize(warped-source['c2w'][selected,:3,3][:,None],dim=-1)
        dt=(source_t-time).abs()[:,None,None].expand(-1,len(xyz),1)
        score=self.view_score(torch.cat([geometry[None].expand(len(selected),-1,-1),appearance,rgb,viewdir,
            dt,reliability[...,None]],-1))[...,0]
        unnormalized=(score-score.max(0,keepdim=True).values).exp()*reliability
        blend=unnormalized/unnormalized.sum(0,keepdim=True).clamp_min(1e-8)
        pooled=(appearance*blend[...,None]).sum(0)
        mixed=(rgb*blend[...,None]).sum(0)
        d=fourier(F.normalize(direction,dim=-1),4)
        residual=.1*torch.tanh(self.color_residual(torch.cat([geometry,pooled,d],-1)))
        fallback=torch.sigmoid(self.fallback_color(torch.cat([geometry,d],-1)))
        has_source=unnormalized.sum(0)>1e-7
        color=torch.where(has_source[:,None],mixed+residual,fallback).clamp(0,1)
        return color,has_source.float()


def render_shared(field,scene,cache,index,pixels,device,samples=48,stratified=False,chunk=16384,
                  frozen_texture=False,prepared=None):
    origin,direction=scene.rays(index,pixels,device)
    source=scene.context_tensors(device)
    time=scene.times[index].to(device)
    # These indices refer only to the declared source context, never targets.
    assert index not in scene.context
    if prepared is not None:
        selected,textures,state=prepared
    else:
        selected=(source['times']-time).abs().argsort()[:field.source_views]
        if frozen_texture:
            with torch.no_grad():textures=field.source_features(source,selected)
        else:
            textures=field.source_features(source,selected)
        state=field.anchor_state(cache,time)
    edges=torch.linspace(scene.near,scene.far,samples+1,device=device)
    z=(edges[1:]+edges[:-1])[None].expand(len(pixels),-1)/2
    if stratified:
        z=edges[:-1][None]+torch.rand_like(z)*(edges[1:]-edges[:-1])[None]
    xyz=(origin[:,None]+direction[:,None]*z[...,None]).reshape(-1,3)
    dirs=direction[:,None].expand(-1,samples,-1).reshape(-1,3)
    sigmas=[];colors=[];cover=[];dist=[]
    for low in range(0,len(xyz),chunk):
        x=xyz[low:low+chunk]
        sigma,g,ids,weights,nearest=field.geometry_at(x,time,state)
        c,coverage=field.appearance_at(x,time,dirs[low:low+chunk],state,g,ids,weights,source,cache,selected,textures)
        sigmas.append(sigma);colors.append(c);cover.append(coverage);dist.append(nearest)
    pred=volume_render(torch.cat(sigmas).reshape(len(pixels),samples),
        torch.cat(colors).reshape(len(pixels),samples,3),z,direction)
    pred['appearance_coverage']=(pred['weights']*torch.cat(cover).reshape(len(pixels),samples)).sum(-1)/pred['accumulation'].clamp_min(1e-8)
    pred['surface_anchor_distance']=(pred['weights']*torch.cat(dist).reshape(len(pixels),samples)).sum(-1)/pred['accumulation'].clamp_min(1e-8)
    pred['motion_regularizer']=state[3].square().mean()
    return pred
