"""Scene-local target updates ported from the benchmark."""
import json
import time

import torch

from tissueanchor.models.shared_adapter import render_shared


def adapt(field,scene,cache,device,steps,rays,samples):
    """Only declared adaptation target frames; shared base file stays unchanged."""
    generator=torch.Generator().manual_seed(61)
    optimizer=torch.optim.AdamW(field.parameters(),lr=1e-4,weight_decay=1e-6)
    log=[];start=time.monotonic();field.train()
    for step in range(steps):
        index=scene.targets[step%len(scene.targets)]
        assert index not in scene.test and index not in scene.context
        available=scene.pixels[index]
        pixels=available[torch.randint(len(available),(rays,),generator=generator)]
        y,x=pixels[:,0],pixels[:,1]
        rgb=scene.images[index,y,x].to(device).float()/255
        prior=scene.depths[index,y,x].to(device);valid=scene.depth_valid[index,y,x].to(device)
        optimizer.zero_grad(set_to_none=True)
        pred=render_shared(field,scene,cache,index,pixels,device,samples,True)
        photo=(pred['rgb']-rgb).abs().mean()
        depth=((1/pred['depth'].clamp_min(.05)-1/prior.clamp_min(.05)).abs()*valid).sum()/valid.sum().clamp_min(1)
        loss=photo+.01*depth+.001*(1-pred['accumulation']).abs().mean()+.01*pred['motion_regularizer']
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite adaptation loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(field.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if step%25==0 or step+1==steps:
            row={'iteration':step+1,'target_index':index,'loss':float(loss.detach()),'wall_seconds':time.monotonic()-start}
            log.append(row);print(json.dumps({'adapt_scene':scene.name,**row}),flush=True)
    return {'steps':steps,'parameters_updated':'all shared adapter/texture/NeRF parameters in a scene-local copy; foundation remains frozen',
            'adaptation_target_indices':scene.targets,'wall_seconds':time.monotonic()-start,'log':log}
