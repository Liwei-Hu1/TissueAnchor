"""Fourier features and volume integration for the shared field."""
import math
import torch


def fourier(x, bands):
    frequencies=2.0**torch.arange(bands,device=x.device,dtype=x.dtype)*math.pi
    a=x[...,None]*frequencies
    return torch.cat([x,a.sin().flatten(-2),a.cos().flatten(-2)],dim=-1)


def volume_render(density,color,z_values,ray_directions,background=None):
    """z_values parameterize camera Z depth, in the same units as world points.

    Unnormalized ray directions have camera z=1. Multiplying by their length
    converts Z-step intervals to metric distances for alpha integration.
    """
    distance=z_values[:,1:]-z_values[:,:-1]
    distance=torch.cat([distance,distance[:,-1:]],dim=1)
    distance=distance*torch.linalg.vector_norm(ray_directions,dim=-1,keepdim=True)
    alpha=-torch.expm1(-density*distance)
    transmittance=torch.cumprod(torch.cat([torch.ones_like(alpha[:,:1]),1-alpha+1e-10],dim=1),dim=1)[:,:-1]
    weights=alpha*transmittance
    accumulation=weights.sum(-1)
    rgb=(weights[...,None]*color).sum(-2)
    if background is not None:rgb=rgb+(1-accumulation)[...,None]*background
    depth=(weights*z_values).sum(-1)/accumulation.clamp_min(1e-8)
    return {'rgb':rgb,'depth':depth,'accumulation':accumulation,'weights':weights}
