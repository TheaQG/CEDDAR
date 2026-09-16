"""Indexed Gaussian streams for paired sampler experiments, independent of churn branches."""
import hashlib
import json

import torch

PROTOCOL = 'ceddar-indexed-noise-v1'


def indexed_seed(seed, label):
    payload = json.dumps([PROTOCOL, int(seed), str(label)], separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') % (2**63)


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def paired_normal(shape, *, seed, stream, device, dtype=torch.float32, audit=None):
    """Same seed/stream/shape/device/dtype gives the same standard normal tensor.

    No process RNG state is consumed. Different devices/PyTorch versions need not
    produce identical draws. The amplitude is applied separately by the sampler.
    """
    draw_seed = indexed_seed(seed, stream)
    generator = torch.Generator(device=device).manual_seed(draw_seed)
    noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    if audit is not None:
        audit[stream] = dict(seed=draw_seed, shape=list(shape), dtype=str(dtype),
                             standard_normal_sha256=tensor_sha256(noise))
    return noise


def sampling_noise_mode(cfg):
    mode = cfg.get('full_gen_eval', {}).get('sigma_control', {}).get('noise_mode', 'sequential')
    if mode not in ('sequential', 'paired'):
        raise ValueError('sigma_control.noise_mode must be sequential or paired')
    return mode
