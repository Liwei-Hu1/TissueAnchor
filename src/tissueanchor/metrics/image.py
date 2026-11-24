"""Input-camera appearance metrics for one frame with an explicit tissue mask."""
from math import exp

import torch
from torch.nn.functional import conv2d


def masked_inputs(rgb, reference, tissue):
    if rgb.shape != reference.shape or rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError('Prediction and reference must be identically shaped H×W×3 RGB')
    if tissue.shape != rgb.shape[:2] or not tissue.any():
        raise ValueError('Expected a nonempty H×W tissue mask')
    x = rgb.float() / 255
    y = reference.float() / 255
    mse = ((x-y).square()*tissue[..., None]).sum() / (3*tissue.sum())
    psnr = -10 * torch.log10(mse.clamp_min(1e-12))
    return psnr, (x*tissue[..., None]).permute(2, 0, 1)[None], (y*tissue[..., None]).permute(2, 0, 1)[None]


def ssim(x, y):
    """Channel-wise SSIM with an 11×11 Gaussian and zero padding."""
    weights = torch.tensor([exp(-p*p/4.5) for p in range(-5, 6)], dtype=torch.float32)
    weights /= weights.sum()
    kernel = torch.outer(weights, weights).to(x).expand(x.shape[1], 1, 11, 11).contiguous()

    def average(v):
        return conv2d(v, kernel, padding=5, groups=x.shape[1])

    mx, my = average(x), average(y)
    vx, vy = average(x.square())-mx.square(), average(y.square())-my.square()
    cov = average(x*y)-mx*my
    return (((2*mx*my + 0.0001) * (2*cov + 0.0009)) /
            ((mx.square()+my.square()+0.0001) * (vx+vy+0.0009))).mean()
