"""Sampling utilities for AceStep SFT (Supervised Fine-Tuning) 1.5.
Ported from ComfyUI-AceStep_SFT for modular use.
"""

import math
import torch
import torch.nn.functional as F

class MomentumBuffer:
    def __init__(self, momentum: float = -0.75):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average


def _project(v0, v1, dims=[-1]):
    dtype = v0.dtype
    device_type = v0.device.type
    if device_type == "mps":
        v0, v1 = v0.cpu(), v1.cpu()
    v0, v1 = v0.double(), v1.double()
    v1 = F.normalize(v1, dim=dims)
    v0_parallel = (v0 * v1).sum(dim=dims, keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype).to(device_type), v0_orthogonal.to(dtype).to(device_type)


def apg_guidance(pred_cond, pred_uncond, guidance_scale, momentum_buffer=None,
                 eta=0.0, norm_threshold=2.5, dims=[-1]):
    """APG guidance as used by AceStep SFT's generate_audio."""
    diff = pred_cond - pred_uncond
    if momentum_buffer is not None:
        momentum_buffer.update(diff)
        diff = momentum_buffer.running_average
    if norm_threshold > 0:
        ones = torch.ones_like(diff)
        diff_norm = diff.norm(p=2, dim=dims, keepdim=True)
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
        diff = diff * scale_factor
    diff_parallel, diff_orthogonal = _project(diff, pred_cond, dims)
    normalized_update = diff_orthogonal + eta * diff_parallel
    return pred_cond + (guidance_scale - 1) * normalized_update


def _cos_sim(t1, t2):
    t1 = t1 / torch.linalg.norm(t1, dim=1, keepdim=True).clamp(min=1e-8)
    t2 = t2 / torch.linalg.norm(t2, dim=1, keepdim=True).clamp(min=1e-8)
    return torch.sum(t1 * t2, dim=1, keepdim=True).clamp(min=-1.0 + 1e-6, max=1.0 - 1e-6)


def _perpendicular(diff, base):
    n, t, c = diff.shape
    diff = diff.view(n * t, c).float()
    base = base.view(n * t, c).float()
    dot = torch.sum(diff * base, dim=1, keepdim=True)
    norm_sq = torch.sum(base * base, dim=1, keepdim=True)
    proj = (dot / (norm_sq + 1e-8)) * base
    perp = diff - proj
    return proj.view(n, t, c), perp.reshape(n, t, c)


def adg_guidance(latents, v_cond, v_uncond, sigma, guidance_scale,
                angle_clip=3.14159265 / 6, apply_norm=False, apply_clip=True):
    """ADG guidance (Angle-based Dynamic Guidance) for flow matching.

    Operates on velocity predictions in [B, T, C] layout.
    """
    n, t, c = v_cond.shape
    if isinstance(sigma, (int, float)):
        sigma = torch.tensor(sigma, device=latents.device, dtype=latents.dtype)
    sigma = sigma.view(-1, 1, 1).expand(n, 1, 1)

    weight = max(guidance_scale - 1, 0) + 1e-3

    x0_cond = latents - sigma * v_cond
    x0_uncond = latents - sigma * v_uncond
    x0_diff = x0_cond - x0_uncond

    theta = torch.acos(_cos_sim(
        x0_cond.view(-1, c).float(), x0_uncond.reshape(-1, c).contiguous().float()
    ))
    theta_new = torch.clip(weight * theta, -angle_clip, angle_clip) if apply_clip else weight * theta
    proj, perp = _perpendicular(x0_diff, x0_uncond)
    v_part = torch.cos(theta_new) * x0_cond
    mask = (torch.sin(theta) > 1e-3).float()
    p_part = perp * torch.sin(theta_new) / torch.sin(theta) * mask + perp * weight * (1 - mask)
    x0_new = v_part + p_part
    if apply_norm:
        x0_new = x0_new * (torch.linalg.norm(x0_cond, dim=1, keepdim=True)
                           / torch.linalg.norm(x0_new, dim=1, keepdim=True))

    v_out = (latents - x0_new) / sigma
    return v_out.reshape(n, t, c).to(latents.dtype)


def clone_conditioning(conditioning):
    def _clone_value(value):
        if torch.is_tensor(value):
            return value.clone()
        if isinstance(value, dict):
            return {k: _clone_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_clone_value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(_clone_value(v) for v in value)
        if hasattr(value, "_copy_with") and hasattr(value, "cond"):
            cond_value = _clone_value(value.cond)
            return value._copy_with(cond_value)
        return value

    return [
        (list(_clone_value(value) for value in cond_item)
         if isinstance(cond_item, (list, tuple)) else _clone_value(cond_item))
        for cond_item in conditioning
    ]

def clone_processed_conditioning(conditioning):
    """Deep clone for pre-processed internal conditioning structures."""
    if conditioning is None: return None
    def _clone_inner(v):
        if torch.is_tensor(v): return v.clone()
        if isinstance(v, dict): return {k: _clone_inner(i) for k, i in v.items()}
        if isinstance(v, list): return [_clone_inner(i) for i in v]
        if isinstance(v, tuple): return tuple(_clone_inner(i) for i in v)
        if hasattr(v, "_copy_with") and hasattr(v, "cond"):
            return v._copy_with(_clone_inner(v.cond))
        return v
    return [_clone_inner(item) for item in conditioning]


def zero_conditioning_value(value):
    if torch.is_tensor(value):
        return torch.zeros_like(value)
    if isinstance(value, list):
        return [zero_conditioning_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(zero_conditioning_value(v) for v in value)
    if isinstance(value, dict):
        return {k: zero_conditioning_value(v) for k, v in value.items()}
    return value


def reweight_conditioning_energy(tensor, erg_scale):
    if not torch.is_tensor(tensor) or abs(erg_scale) < 1e-8:
        return tensor
    mean = tensor.mean(dim=-1, keepdim=True)
    return mean + (tensor - mean) * (1.0 + erg_scale)


def apply_erg_to_conditioning(conditioning, erg_scale):
    if abs(erg_scale) < 1e-8:
        return conditioning

    conditioned = clone_conditioning(conditioning)
    for cond_item in conditioned:
        if cond_item and torch.is_tensor(cond_item[0]):
            cond_item[0] = reweight_conditioning_energy(cond_item[0], erg_scale)
        if len(cond_item) > 1 and isinstance(cond_item[1], dict):
            lyrics_cond = cond_item[1].get("conditioning_lyrics")
            if lyrics_cond is not None:
                cond_item[1]["conditioning_lyrics"] = reweight_conditioning_energy(
                    lyrics_cond, erg_scale
                )

    return conditioned


def apply_omega_scale(model_output, omega_scale):
    if abs(omega_scale) < 1e-8:
        return model_output

    omega = 0.9 + 0.2 / (1.0 + math.exp(-float(omega_scale)))
    reduce_dims = tuple(range(1, model_output.ndim))
    mean = model_output.mean(dim=reduce_dims, keepdim=True)
    return mean + (model_output - mean) * omega

def build_text_only_conditioning(positive):
    """Build a version of conditioning with lyrics zeroed out."""
    new_positive = clone_conditioning(positive)
    for p in new_positive:
        if len(p) > 1 and isinstance(p[1], dict):
            if "conditioning_lyrics" in p[1]:
                p[1]["conditioning_lyrics"] = torch.zeros_like(p[1]["conditioning_lyrics"])
    return new_positive

def build_processed_text_only_conditioning(processed_conditioning):
    """Variant for use during sampling loop (operates on list of dicts)."""
    if processed_conditioning is None:
        return None

    text_only = []
    has_lyric_branch = False
    
    # Processed conditioning in ComfyUI sampler is typically a list of dicts
    for cond_item in processed_conditioning:
        if not isinstance(cond_item, dict):
            text_only.append(cond_item)
            continue
            
        cloned = cond_item.copy()
        model_conds = cloned.get("model_conds")
        if model_conds is not None:
            cloned_model_conds = model_conds.copy()
            
            # These are the keys used in ACE-Step 1.5's internal model_conds
            for key in ("lyric_embed", "lyric_token_idx", "conditioning_lyrics"):
                lyric_val = cloned_model_conds.get(key)
                if lyric_val is not None:
                    # Case 1: Wrapped conditioning object (has .cond)
                    if hasattr(lyric_val, "_copy_with") and hasattr(lyric_val, "cond"):
                        cloned_model_conds[key] = lyric_val._copy_with(
                            torch.zeros_like(lyric_val.cond)
                        )
                        has_lyric_branch = True
                    # Case 2: Direct tensor
                    elif torch.is_tensor(lyric_val):
                        cloned_model_conds[key] = torch.zeros_like(lyric_val)
                        has_lyric_branch = True

            # Also zero out lyrics_strength if present
            strength = cloned_model_conds.get("lyrics_strength")
            if strength is not None:
                if hasattr(strength, "_copy_with"):
                    cloned_model_conds["lyrics_strength"] = strength._copy_with(0.0)
                else:
                    cloned_model_conds["lyrics_strength"] = 0.0

            cloned["model_conds"] = cloned_model_conds
        
        text_only.append(cloned)

    return text_only if has_lyric_branch else None
