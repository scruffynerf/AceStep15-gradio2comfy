"""FSQ implementation and utilities for ACE-Step audio codes."""
import math
import torch
import re

def fsq_decode_indices(indices, levels):
    """
    Composite integer codes -> 6d float vectors in [-1, 1]
    indices : [B, T] long tensor
    levels  : list of ints e.g. [8, 8, 8, 5, 5, 5]
    returns : [B, T, 6] float32
    """
    levels = [int(l) for l in levels]
    remainder = indices.clone()
    codes = []
    for l in levels:
        d = (remainder % l).float()
        remainder = remainder // l
        val = (2.0 * d / (l - 1)) - 1.0 if l > 1 else torch.zeros_like(d)
        codes.append(val)
    return torch.stack(codes, dim=-1)

def fsq_encode_to_indices(codes_6d, levels):
    """
    6d float vectors -> composite integer codes
    codes_6d : [B, T, 6] float, values in [-1, 1]
    levels   : list of ints
    returns  : [B, T] long tensor
    """
    levels = [int(l) for l in levels]
    B, T, _ = codes_6d.shape
    device = codes_6d.device
    codes_6d = codes_6d.float().clamp(-1.0, 1.0)
    composite = torch.zeros(B, T, dtype=torch.long, device=device)
    stride = 1
    for i, l in enumerate(levels):
        if l == 1:
            dim_idx = torch.zeros(B, T, dtype=torch.long, device=device)
        else:
            d = ((codes_6d[..., i] + 1.0) / 2.0 * (l - 1)).round().long().clamp(0, l - 1)
            dim_idx = d
        composite = composite + dim_idx * stride
        stride *= l
    return composite

def get_fsq_levels(q=None) -> list:
    """Return FSQ quantizer levels [8,8,8,5,5,5], dynamically read from quantizer if possible.

    Args:
        q: Optional quantizer object (tok.quantizer or the model quantizer layer).
           Accepts None for the hardcoded ACE-Step 1.5 default.
    """
    if q is not None:
        try:
            layer = q.layers[0]
            for attr in ("_levels", "levels"):
                levels = getattr(layer, attr, None)
                if levels is not None:
                    return levels.tolist() if hasattr(levels, "tolist") else list(levels)
        except Exception:
            pass
    return [8, 8, 8, 5, 5, 5]  # Default for ACE-Step 1.5


def get_fsq_vocab_size(q=None) -> int:
    """Return the FSQ codebook size (product of levels). Default: 8*8*8*5*5*5 = 64000."""
    return math.prod(get_fsq_levels(q))

def parse_audio_codes(audio_codes):
    """Normalise input to [[int, int, ...]] nested list"""
    if not isinstance(audio_codes, list):
        audio_codes = [audio_codes]
    if audio_codes and not isinstance(audio_codes[0], list):
        audio_codes = [audio_codes]
    result = []
    for batch_item in audio_codes:
        code_ids = []
        for x in batch_item:
            if isinstance(x, (int, float)):
                code_ids.append(int(x))
            elif isinstance(x, str):
                code_ids.extend([int(v) for v in re.findall(r"(\d+)", x)])
        result.append(code_ids)
    return result

def fsq_indices_to_quantized(q, code_ids, device, dtype):
    """codes -> [1, T, 2048] for feeding detokenizer"""
    levels = get_fsq_levels(q)
    indices = torch.tensor(code_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        codes_6d = fsq_decode_indices(indices, levels)
        quantized = torch.nn.functional.linear(
            codes_6d.to(dtype),
            q.project_out.weight.to(dtype),
            q.project_out.bias.to(dtype) if q.project_out.bias is not None else None,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  VAE encode  —  CORRECT API: vae.encode(waveform.movedim(1, -1))
#
#  ComfyUI VAEEncodeAudio source (nodes_audio.py):
#    t = vae.encode(waveform.movedim(1, -1))
#  where waveform is [B, C, N] → movedim(1,-1) → [B, N, C]  (channels last)
# ─────────────────────────────────────────────────────────────────────────────

_SR = 48_000  # ACE-Step VAE sample rate


def vae_encode_audio(vae, audio_np):
    """
    Encode float32 mono numpy audio → latent tensor [B, C, T] at 25 Hz.

    Uses the exact same call as ComfyUI's built-in VAEEncodeAudio node:
        vae.encode(waveform.movedim(1, -1))
    where waveform is [B, C, N] → [B, N, C] after movedim.
    """
    import numpy as np
    import torchaudio

    vae_sr = int(getattr(vae, "audio_sample_rate", _SR))

    audio_t = torch.from_numpy(audio_np).float()          # [N]
    # Build stereo [1, 2, N] then resample if needed
    waveform = audio_t.unsqueeze(0).unsqueeze(0).expand(1, 2, -1).contiguous()  # [1, 2, N]
    if vae_sr != _SR:
        waveform = torchaudio.functional.resample(waveform, _SR, vae_sr)
        print(f"  [VAE] resampled {_SR}→{vae_sr} Hz")

    print(f"  [VAE] waveform {list(waveform.shape)}  → movedim(1,-1) → "
          f"{list(waveform.movedim(1,-1).shape)}")

    # vae.encode handles device/dtype internally (ComfyUI model management)
    try:
        with torch.no_grad():
            latents = vae.encode(waveform.movedim(1, -1))   # [B, N, C] channels-last
        if isinstance(latents, dict):
            latents = latents.get("samples", next(iter(latents.values())))
        print(f"  [VAE] ✓ latents {list(latents.shape)}")
        return latents.float().cpu()
    except Exception as exc:
        print(f"  [VAE] ✗ vae.encode: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  FSQ tokenisation  —  hardcoded path: model.diffusion_model.tokenizer
# ─────────────────────────────────────────────────────────────────────────────

def get_tokenizer(model):
    """Walk the ComfyUI model wrapper to find the FSQ tokenizer and its quantizer."""
    inner = getattr(model, "model", model)
    dm    = getattr(inner, "diffusion_model", None)
    if dm is None:
        print("  [FSQ] diffusion_model not found"); return None, None
    tok   = getattr(dm, "tokenizer", None)
    if tok is None:
        print("  [FSQ] tokenizer not found"); return None, None
    quant = getattr(tok, "quantizer", None)
    return tok, quant


def unwrap_codes(out):
    """Extract integer code tensor from whatever the tokenizer returns."""
    if out is None: return None
    items = [out] if isinstance(out, torch.Tensor) else \
            list(out) if isinstance(out, (tuple, list)) else []
    for c in items:
        if isinstance(c, torch.Tensor) and c.dtype in (torch.int64, torch.int32, torch.long):
            return c
    for c in items:
        if isinstance(c, torch.Tensor) and c.numel() > 0:
            cf = c.float()
            if cf.min() >= 0 and cf.max() < 200_000:
                return c.long()
    return None


def extract_fsq_codes(model, latents):
    """
    Encode VAE latents → 5 Hz FSQ integer codes (as a flat Python list).

    From ace_step15.py source (confirmed):
    - prepare_condition calls:
        lm_hints_5Hz = tokenizer.quantizer.get_output_from_indices(audio_codes, ...)
    - get_output_from_indices expects shape [B, T, num_quantizers] = [1, T, 1]
    - FSQ vocab size = 8*8*8*5*5*5 = 64000, valid range [0, 63999]
    - Values ≥ 64000 cause CUDA device-side assert (out-of-range codebook lookup)
    """
    import comfy.model_management as mm

    gpu     = mm.get_torch_device()
    offload = mm.unet_offload_device()

    tok, quant = get_tokenizer(model)
    if quant is None:
        print("  [FSQ] could not reach tokenizer.quantizer")
        return None

    vocab_size = get_fsq_vocab_size(quant)
    max_code   = vocab_size - 1
    print(f"  [FSQ] tokenizer={type(tok).__name__}  quant={type(quant).__name__}")
    print(f"  [FSQ] vocab_size={vocab_size}  valid_range=[0,{max_code}]")

    tok.to(gpu)

    # latents [1, 64, T] → [1, T, 64] bfloat16
    inp = latents.transpose(1, 2).to(device=gpu, dtype=torch.bfloat16)
    print(f"  [FSQ] tokenizer input {list(inp.shape)} {inp.dtype}")

    result = None
    for method in ("encode", "tokenize", "forward", "__call__"):
        fn = getattr(tok, method, None) if method != "__call__" else tok
        if fn is None:
            continue
        try:
            with torch.no_grad():
                out = fn(inp)
            codes = unwrap_codes(out)
            if codes is not None:
                flat = codes.reshape(-1).long()
                n_before = flat.numel()
                n_oob    = (flat > max_code).sum().item()
                flat     = flat.clamp(0, max_code)
                print(f"  [FSQ] ✓ tok.{method} → {list(codes.shape)} "
                      f"min={flat.min().item()} max={flat.max().item()}")
                if n_oob:
                    print(f"  [FSQ]   clamped {n_oob}/{n_before} out-of-range tokens")
                result = flat.tolist()
                break
        except Exception as exc:
            print(f"  [FSQ] ✗ tok.{method}: {exc}")

    tok.to(offload)
    if result is None:
        print("  [FSQ] all attempts failed")
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Conditioning patcher  —  format [[int, int, ...]]
# ─────────────────────────────────────────────────────────────────────────────

import copy


def patch_conditioning(conditioning, codes_list):
    """
    Inject audio_codes in the format that get_output_from_indices expects:
      tensor shape [1, T, 1]  (B, time, num_quantizers)

    We store as [[[c0], [c1], ...]] so torch.tensor converts to [1, T, 1].
    Plain [[c0, c1, ...]] converts to [1, T] which unbinds incorrectly.
    """
    out = []
    for tensor, d in conditioning:
        nd = copy.copy(d)
        if codes_list is not None:
            # Wrap each token in a list: [[c]] per time step → [1, T, 1] tensor
            codes_3d = [[[c] for c in codes_list]]
            nd["audio_codes"] = codes_3d
            print(f"  [patch] ✓ audio_codes [[[c],...]] → tensor [1,{len(codes_list)},1]  "
                  f"first 8: {codes_list[:8]}")
        else:
            print("  [patch] ⚠ codes is None — conditioning unchanged")
        out.append([tensor, nd])
    return out
