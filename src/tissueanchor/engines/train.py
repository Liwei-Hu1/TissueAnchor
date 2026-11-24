"""Train one shared query/appearance NeRF across disjoint source/target videos."""
import argparse
import hashlib
import json
import math
import random
import time
import traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from tissueanchor.datasets.stereomis import SharedScene, TRAIN_SCENES, VAL_SCENES, TEST_SCENES, atomic_json
from tissueanchor.models.shared_adapter import SharedQueryNeRF,render_shared


def load_cache(root,name,device):
    cache=torch.load(root/'cache'/name/'query_cache.pt',map_location='cpu',weights_only=True)
    assert cache['schema']=='shared_query_cache_v1'
    for k,v in cache.items():
        if torch.is_tensor(v):cache[k]=v.to(device)
    return cache


def save_model(path,field,opt,step,config,extra=None):
    content={'field':field.state_dict(),'optimizer':opt.state_dict(),'step':step,'config':config,
        'method':'shared original-query-conditioned dynamic NeRF v1','torch_rng':torch.get_rng_state(),
        'cuda_rng':torch.cuda.get_rng_state(),'python_rng':random.getstate()}
    if extra:content.update(extra)
    temp=path.with_suffix('.tmp');torch.save(content,temp);temp.replace(path)


@torch.no_grad()
def validation(field,scenes,caches,device,rays=2048,samples=48):
    was_training=field.training;field.eval();rows=[]
    for scene in scenes:
        # Fixed original hold-out RGB, not source/adaptation pixels. These are
        # used only for validation scene checkpoint selection, never P3 testing.
        indices=np.asarray(scene.test)[np.linspace(0,len(scene.test)-1,4).round().astype(int)].tolist()
        for index in indices:
            generator=torch.Generator().manual_seed(20261031+index)
            available=scene.pixels[index]
            pixels=available[torch.randint(len(available),(rays,),generator=generator)]
            pred=render_shared(field,scene,caches[scene.name],index,pixels,device,samples)
            y,x=pixels[:,0],pixels[:,1]
            target=scene.images[index,y,x].to(device).float()/255
            mse=(pred['rgb']-target).square().mean().clamp_min(1e-12)
            rows.append({'scene':scene.name,'frame_index':index,'psnr_tissue_ray_probe':float(-10*torch.log10(mse)),
                'appearance_coverage':float(pred['appearance_coverage'].mean()),'accumulation':float(pred['accumulation'].mean())})
    field.train(was_training)
    return {'mean_psnr_tissue_ray_probe':sum(r['psnr_tissue_ray_probe'] for r in rows)/len(rows),'frames':rows,
        'note':'Fixed sparse validation pixel probe, not full-image test metrics.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--cache-root',type=Path,required=True,help='Directory containing cache/<scene>/query_cache.pt')
    p.add_argument('--config',type=Path,help='Optional JSON file of source hyperparameters')
    p.add_argument('--steps',type=int,default=3000);p.add_argument('--rays',type=int,default=512)
    p.add_argument('--samples',type=int,default=48);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--val-every',type=int,default=500);p.add_argument('--checkpoint-every',type=int,default=250)
    p.add_argument('--depth-weight',type=float,default=.01);p.add_argument('--lr',type=float,default=.0005)
    p.add_argument('--smoke',action='store_true');p.add_argument('--resume',type=Path)
    a = p.parse_args()
    if a.config is not None:
        defaults = json.loads(a.config.read_text())
        allowed = {'steps', 'rays', 'samples', 'seed', 'val_every', 'checkpoint_every', 'depth_weight', 'lr'}
        if defaults.keys() - allowed:
            p.error(f'Unsupported configuration keys: {sorted(defaults.keys() - allowed)}')
        p.set_defaults(**defaults)
        a = p.parse_args()
    out = a.output
    if out.exists() and not a.resume and any(out.iterdir()):
        p.error('A new source run requires an empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    device=torch.device('cuda')
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    train_names=TRAIN_SCENES[:1] if a.smoke else TRAIN_SCENES
    val_names=[] if a.smoke else VAL_SCENES
    config.update(train_scenes=train_names,validation_scenes=val_names,test_scenes=TEST_SCENES,
        source_context_frames=48,target_rgb_isolated_from_all_conditioning=True,
        shared_parameters_across_videos=True,pretrained_encoder_and_query_decoder_frozen=True,
        trainable_scene_codes=False,base_checkpoint='Unofficial OpenD4RT 48CLIP_9Mix_NoCropAUG',
        losses={'tissue_rgb_L1':1.,'normalized_inverse_pseudo_depth_L1':a.depth_weight,
                'opacity_regularizer':.001,'motion_residual_L2':.01},
        no_independent_geometry_GT=True,stage='connectivity_smoke' if a.smoke else 'shared_training')
    atomic_json(out/'training_config.json',config)
    status={'status':'loading_source_caches','started_utc':datetime.now(timezone.utc).isoformat(),'iteration':0,'total_steps':a.steps}
    atomic_json(out/'run_status.json',status)
    start=time.monotonic()
    try:
        scenes={n:SharedScene(n, root=a.data_root, load_test_rgb=n in val_names) for n in train_names+val_names}
        caches={n:load_cache(a.cache_root,n,device) for n in scenes}
        for name,scene in scenes.items():
            from tissueanchor.engines.inputs import validate_cache
            validate_cache(caches[name], scene)
            scene.context_tensors(device)
        manifest={'train_videos':train_names,'validation_videos':val_names,'test_videos':TEST_SCENES,
            'shared_train_animals':['P1','P2'],'test_animals':['P3'],
            'split_note':'Single predefined held-out-animal fold; not completed three-fold leave-one-animal-out.',
            'source_frames_per_video':48,'target_policy':'train/adapt on original training indices minus 48 source frames; validation/test on original every-eighth split',
            'scenes':{n:s.manifest() for n,s in scenes.items()}}
        atomic_json(out/'split_manifest.json',manifest)
        package = Path(__file__).resolve().parents[1]
        tracked = ('engines/train.py', 'engines/cache.py', 'datasets/stereomis.py', 'models/shared_adapter.py')
        atomic_json(out/'code_hashes.json', {x: hashlib.sha256((package/x).read_bytes()).hexdigest() for x in tracked})
        field=SharedQueryNeRF().to(device)
        optimizer=torch.optim.AdamW(field.parameters(),lr=a.lr,weight_decay=1e-6)
        initial_step=0;best=-float('inf')
        if a.resume:
            ck=torch.load(a.resume,map_location=device,weights_only=True)
            field.load_state_dict(ck['field'],strict=True);optimizer.load_state_dict(ck['optimizer'])
            initial_step=ck['step'];torch.set_rng_state(ck['torch_rng'].cpu())
            torch.cuda.set_rng_state(ck['cuda_rng'].cpu());random.setstate(ck['python_rng'])
            best=ck.get('best_validation_psnr',best)
        status.update(status='training',trainable_parameters=sum(p.numel() for p in field.parameters()),device_name=torch.cuda.get_device_name(),shared_train_videos=train_names)
        atomic_json(out/'run_status.json',status)
        if val_names and initial_step==0:
            v=validation(field,[scenes[n] for n in val_names],caches,device,samples=a.samples)
            atomic_json(out/'initial_validation.json',v)
        logfile=(out/'training_log.jsonl').open('a',buffering=1)
        grad_groups={}
        for step in range(initial_step,a.steps):
            name=train_names[step%len(train_names)]
            scene=scenes[name];index=random.choice(scene.targets)
            assert index not in scene.context and index not in scene.test
            available=scene.pixels[index]
            pixels=available[torch.randint(len(available),(a.rays,))]
            y,x=pixels[:,0],pixels[:,1]
            target=scene.images[index,y,x].to(device).float()/255
            prior=scene.depths[index,y,x].to(device)
            depth_valid=scene.depth_valid[index,y,x].to(device)
            optimizer.zero_grad(set_to_none=True)
            lr=a.lr*(.15+.85*.5*(1+math.cos(math.pi*step/a.steps)))
            for group in optimizer.param_groups:group['lr']=lr
            pred=render_shared(field,scene,caches[name],index,pixels,device,a.samples,True)
            rgb=(pred['rgb']-target).abs().mean()
            depth=((1/pred['depth'].clamp_min(.05)-1/prior.clamp_min(.05)).abs()*depth_valid).sum()/depth_valid.sum().clamp_min(1)
            opacity=(1-pred['accumulation']).abs().mean()
            motion=pred['motion_regularizer']
            loss=rgb+a.depth_weight*depth+.001*opacity+.01*motion
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss {step}')
            loss.backward()
            if step==initial_step:
                for label,prefix in [('latent','latent_proj'),('motion','motion_residual'),('geometry','geometry'),('density','sigma'),('texture','texture_'),('blend','view_score'),('color','color_residual')]:
                    grads=[p.grad for n,p in field.named_parameters() if n.startswith(prefix) and p.grad is not None]
                    grad_groups[label]=sum(float(g.abs().sum()) for g in grads)
                if not all(v>0 and math.isfinite(v) for v in grad_groups.values()):
                    raise RuntimeError(f'Broken gradient branches: {grad_groups}')
                atomic_json(out/'gradient_connectivity.json',grad_groups)
            norm=torch.nn.utils.clip_grad_norm_(field.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            if step%25==0 or step+1==a.steps:
                row={'iteration':step+1,'scene':name,'target_index':index,'loss':float(loss.detach()),
                    'rgb_L1':float(rgb.detach()),'inverse_pseudo_depth_L1':float(depth.detach()),
                    'train_ray_psnr':float(-10*torch.log10((pred['rgb'].detach()-target).square().mean().clamp_min(1e-12))),
                    'appearance_coverage':float(pred['appearance_coverage'].detach().mean()),
                    'accumulation':float(pred['accumulation'].detach().mean()),'grad_norm':float(norm),
                    'lr':lr,'wall_seconds':time.monotonic()-start}
                logfile.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
                status.update(iteration=step+1,last_training_row=row,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
                atomic_json(out/'run_status.json',status)
            if val_names and ((step+1)%a.val_every==0 or step+1==a.steps):
                val=validation(field,[scenes[n] for n in val_names],caches,device,samples=a.samples)
                atomic_json(out/f'validation/step_{step+1:06d}.json',val)
                score=val['mean_psnr_tissue_ray_probe']
                if score>best:
                    best=score
                    save_model(out/'best.pt',field,optimizer,step+1,config,{'best_validation_psnr':best})
                print(json.dumps({'validation_step':step+1,'validation_psnr_probe':score,'best':best}),flush=True)
            if (step+1)%a.checkpoint_every==0 or step+1==a.steps:
                save_model(out/'latest.pt',field,optimizer,step+1,config,{'best_validation_psnr':best if math.isfinite(best) else None})
        save_model(out/'model.pt',field,optimizer,a.steps,config,{'best_validation_psnr':best if math.isfinite(best) else None})
        logfile.close()
        status.update(status='smoke_passed' if a.smoke else 'trained',iteration=a.steps,
            train_wall_seconds=time.monotonic()-start,finished_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(out/'run_status.json',status)
    except Exception as e:
        status.update(status='failed',error=repr(e),traceback=traceback.format_exc())
        atomic_json(out/'run_status.json',status)
        raise


if __name__=='__main__':
    main()
