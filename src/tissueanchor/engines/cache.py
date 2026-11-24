"""Cache frozen original OpenD4RT point-query latents and trajectories.

Only 48 declared source frames per scene enter the encoder, RGB patches, camera
alignment and reliability checks. Held-out RGB/depth never enter these caches.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tissueanchor.datasets.stereomis import SharedScene, TRAIN_SCENES, VAL_SCENES, TEST_SCENES, atomic_json
import hashlib
import sys

from tissueanchor.engines.inputs import DATASETS, open_scene


def load_full(device, checkpoint, config, upstream):
    if not (upstream/'src/model/d4rt.py').is_file():
        raise FileNotFoundError(f'OpenD4RT checkout missing: {upstream}')
    checksum = hashlib.sha256()
    with checkpoint.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    digest = checksum.hexdigest()
    if digest != '65936f72eb902f9a3629e25b2888e8cc2ebf87a89bee7103fe94313fb53dd179':
        raise ValueError('OpenD4RT weights do not match the recorded release SHA256')
    sys.path.insert(0, str(upstream.resolve()))
    from src.model.d4rt import D4RTModel

    proof = {'sha256': digest, 'size_bytes': checkpoint.stat().st_size}
    payload = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)
    cfg = yaml.safe_load(config.read_text())['model']
    cfg['encoder']['pretrained']['enabled'] = False
    with torch.device('meta'):
        model = D4RTModel(cfg)
    keys = model.load_state_dict({k.removeprefix('module.'):v for k,v in payload['model'].items()}, strict=True, assign=True)
    for key, b in list(model.named_buffers()):
        if b.is_meta:
            if not key.endswith('uv_encoder.frequencies'):
                raise RuntimeError(f'Unexpected meta buffer {key}')
            parent, attr = key.rsplit('.',1)
            setattr(model.get_submodule(parent), attr, 2.**torch.arange(b.numel(), dtype=torch.float32)*torch.pi)
    del payload
    model.requires_grad_(False).eval().to(device)
    return model, proof


def similarity(x, y):
    """Single robust similarity from source correspondences; never target labels."""
    keep = np.ones(len(x), dtype=bool)
    for _ in range(3):
        a, b = x[keep].astype(np.float64), y[keep].astype(np.float64)
        ma, mb = a.mean(0), b.mean(0)
        a, b = a-ma, b-mb
        U, S, Vh = np.linalg.svd((b.T @ a)/len(a))
        sign = np.eye(3); sign[2,2] = np.sign(np.linalg.det(U@Vh))
        R = U@sign@Vh
        scale = float((S*np.diag(sign)).sum() / max(float((a*a).sum()/len(a)), 1e-10))
        if scale <= 0:
            raise ValueError('Invalid positive-scale source alignment')
        t = mb-scale*(R@ma)
        error = np.linalg.norm(scale*x@R.T+t-y, axis=1)
        keep = error <= np.quantile(error,.8)
    return scale, R.astype(np.float32), t.astype(np.float32), float(np.median(error))


def prepare(model, scene, out, anchor_count=1024, batch=128):
    device = next(model.parameters()).device
    source = scene.context_tensors(device)
    video = F.interpolate(source['images'], (256,256), mode='bilinear', align_corners=False)[None]
    # Exact frozen checkpoint interface: low-resolution local patches, original aspect token.
    aspect = torch.tensor([[scene.w/scene.h]], device=device)
    start = time.monotonic()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        memory = model.encode_video(video, aspect)
    rng = np.random.default_rng(42)
    srcs, xs, ys = [], [], []
    source_locals = np.linspace(0,47,8).round().astype(int)
    for local in source_locals:
        idx = scene.context[local]
        valid = scene.masks[idx] & scene.depth_valid[idx]
        # Stratified image coverage, then seed-fixed fill from valid source tissue.
        yy, xx = np.meshgrid(np.linspace(8,scene.h-9,16), np.linspace(8,scene.w-9,24), indexing='ij')
        x, y = np.rint(xx).astype(int).ravel(), np.rint(yy).astype(int).ravel()
        ok = valid[y,x].numpy(); x, y = x[ok],y[ok]
        n = anchor_count//8 + (1 if len(srcs) < anchor_count%8 else 0)
        if len(x) < n:
            cy,cx = torch.nonzero(valid,as_tuple=True)
            sel = rng.choice(len(cx), n-len(x), replace=False)
            x = np.concatenate([x,cx.numpy()[sel]]); y = np.concatenate([y,cy.numpy()[sel]])
        chosen = np.linspace(0,len(x)-1,n).round().astype(int)
        xs.extend(x[chosen]); ys.extend(y[chosen]); srcs.extend([int(local)]*n)
    x = torch.tensor(xs, device=device, dtype=torch.float32)
    y = torch.tensor(ys, device=device, dtype=torch.float32)
    src = torch.tensor(srcs, device=device, dtype=torch.long)
    u,v = x/(scene.w-1), y/(scene.h-1)
    n = len(src)

    def decode(indices, target):
        q = {'u':u[indices][None], 'v':v[indices][None], 't_src':src[indices][None],
             't_tgt':target[None], 't_cam':torch.zeros((1,len(indices)),device=device,dtype=torch.long)}
        with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
            token = model.query_embedder(video=video, **q)
            latent = model.decoder(token,memory)
            pred = model.heads(latent)
        return latent[0].float(), {k:a[0].float() for k,a in pred.items()}

    latents = []
    for low in range(0,n,batch):
        indices = torch.arange(low,min(n,low+batch),device=device)
        latent, _ = decode(indices,src[indices])
        latents.append(latent.cpu().half())
    tracks = []; visibility = []; confidence = []
    for local in range(48):
        points, vis, conf = [],[],[]
        for low in range(0,n,batch):
            indices = torch.arange(low,min(n,low+batch),device=device)
            _, pred = decode(indices,torch.full_like(src[indices],local))
            points.append(pred['xyz_3d'].cpu()); vis.append(pred['visibility'].sigmoid().cpu())
            conf.append(pred['confidence'].sigmoid().cpu())
        tracks.append(torch.cat(points)); visibility.append(torch.cat(vis)); confidence.append(torch.cat(conf))
        if local%8==0:
            print(json.dumps({'scene':scene.name,'cache_context_time':local,'anchors':n,'seconds':time.monotonic()-start}),flush=True)
    tracks = torch.stack(tracks)
    if not torch.isfinite(tracks).all():
        raise FloatingPointError('Nonfinite original decoder tracks')
    src_cpu = src.cpu(); own = tracks[src_cpu,torch.arange(n)]
    ctx_indices = torch.tensor(scene.context)[src_cpu]
    own_depth = scene.depths[ctx_indices,torch.tensor(ys),torch.tensor(xs)]
    K = scene.K
    rays = torch.stack([(x.cpu()-K[0,2])/K[0,0], (y.cpu()-K[1,2])/K[1,1],torch.ones(n)],-1)
    poses = scene.c2w[ctx_indices]
    observed = torch.bmm(poses[:,:3,:3],(rays*own_depth[:,None])[:,:,None]).squeeze(-1)+poses[:,:3,3]
    scale,R,t,align_error = similarity(own.numpy(),observed.numpy())
    transformed = scale*tracks@torch.from_numpy(R).T+torch.from_numpy(t)
    own_world = transformed[src_cpu,torch.arange(n)]
    own_cam = torch.bmm(scene.w2c[ctx_indices,:3,:3],own_world[:,:,None]).squeeze(-1)+scene.w2c[ctx_indices,:3,3]
    z = own_cam[:,2].clamp(scene.near,scene.far)
    own_on_ray = torch.bmm(poses[:,:3,:3],(rays*z[:,None])[:,:,None]).squeeze(-1)+poses[:,:3,3]
    # Constant anchor correction enforces the observed source ray while retaining
    # the aligned pretrained trajectory's relative displacement.
    corrected = transformed+(own_on_ray-own_world)[None]
    reliabilities = []
    reprojection = []
    for local in range(48):
        xyz = corrected[local].to(device)
        C = source['w2c'][local]
        cam = xyz@C[:3,:3].T+C[:3,3]
        z = cam[:,2]
        uv = cam[:,:2]/z.clamp_min(.001)[:,None]
        px = uv*torch.stack([source['K'][0,0],source['K'][1,1]])+source['K'][:2,2]
        grid = (2*(px+.5)/torch.tensor([scene.w,scene.h],device=device)-1)[None,None]
        depth = F.grid_sample(source['depths'][local:local+1],grid,align_corners=False)[0,0,0]
        mask = F.grid_sample(source['masks'][local:local+1],grid,align_corners=False)[0,0,0]
        dvalid = F.grid_sample(source['depth_valid'][local:local+1],grid,align_corners=False)[0,0,0]
        agreement = torch.exp(-torch.abs(torch.log(z.clamp_min(.01)/depth.clamp_min(.01)))/.3)
        valid = (z>.01)&(grid[0,0].abs().amax(-1)<1)&(mask>.5)&(dvalid>.5)
        reliability = valid.float()*agreement*torch.stack(visibility)[local].to(device)
        reliabilities.append(reliability.cpu())
        if local in source_locals:
            ids = src_cpu==local
            reprojection.append(float(torch.linalg.vector_norm(px.cpu()[ids]-torch.stack([x.cpu(),y.cpu()],-1)[ids],dim=-1).median()))
    report = {
        'scene':scene.name, 'anchors':n, 'context_times':48,
        'cached_descriptor_shape':[n,1280], 'frozen_query_decoder_retained':True,
        'query_patch_resolution':[256,256], 'source_context_only':True,
        'single_source_only_similarity':{'scale':scale,'R':R.tolist(),'t':t.tolist(),'median_residual_normalized':align_error},
        'source_ray_correction':True, 'source_ray_reprojection_median_px_max':max(reprojection),
        'raw_confidence_mean':float(torch.stack(confidence).mean()),
        'reliability_mean':float(torch.stack(reliabilities).mean()),
        'reliability_nonzero_fraction':float((torch.stack(reliabilities)>0).float().mean()),
        'cache_seconds':time.monotonic()-start,
        'limitations':'Single source-context Sim3 plus per-anchor source-ray correction. Reliability is heuristic, not calibrated probability. No independent 3D GT or round-trip tracking validation.',
    }
    cache = {'tracks':corrected,'latent':torch.cat(latents),'reliability':torch.stack(reliabilities),
             'raw_visibility':torch.stack(visibility), 'raw_confidence':torch.stack(confidence),
             'source_local':src_cpu,'source_xy':torch.stack([x.cpu(),y.cpu()],-1),
             'context_times':scene.context_times,'manifest':scene.manifest(),'report':report,
             'schema':'shared_query_cache_v1'}
    out.mkdir(parents=True,exist_ok=True)
    torch.save(cache,out/'query_cache.tmp')
    (out/'query_cache.tmp').replace(out/'query_cache.pt')
    atomic_json(out/'cache_report.json',report)
    atomic_json(out/'input_manifest.json',scene.manifest())
    print(json.dumps(report),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--dataset', choices=DATASETS, default='stereomis')
    parser.add_argument('--scenes', nargs='+')
    parser.add_argument('--opend4rt', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--model-config', type=Path, required=True)
    parser.add_argument('--query-batch', type=int, default=128)
    args = parser.parse_args()
    scenes = args.scenes
    if scenes is None:
        if args.dataset != 'stereomis':
            parser.error('--scenes is required for this dataset')
        scenes = TRAIN_SCENES + VAL_SCENES + TEST_SCENES
    torch.set_num_threads(4)
    model, proof = load_full('cuda', args.checkpoint, args.model_config, args.opend4rt)
    atomic_json(args.output/'checkpoint_provenance.json', proof)
    for name in scenes:
        scene = open_scene(args.dataset, name, args.data_root)
        out = args.output/'cache'/scene.name
        if (out/'query_cache.pt').exists():
            raise FileExistsError(f'{out}: choose a fresh output or omit completed scenes')
        prepare(model, scene, out, 1024, args.query_batch)
        del scene
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
