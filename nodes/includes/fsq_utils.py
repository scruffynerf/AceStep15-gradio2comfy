"""FSQ implementation and utilities for ACE-Step audio codes."""
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

def get_fsq_levels(model):
    """Extract FSQ levels from model's tokenizer quantizer"""
    # if hasattr(model, "tokenizer") and hasattr(model.tokenizer, "quantizer"):
    #    q = model.tokenizer.quantizer
    #    if hasattr(q, "layers") and len(q.layers) > 0 and hasattr(q.layers[0], "_levels"):
    #        return q.layers[0]._levels.tolist()
    return [8, 8, 8, 5, 5, 5] # Default for ACE-Step 1.5

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
    levels = get_fsq_levels_from_q(q)
    indices = torch.tensor(code_ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        codes_6d = fsq_decode_indices(indices, levels)
        quantized = torch.nn.functional.linear(
            codes_6d.to(dtype),
            q.project_out.weight.to(dtype),
            q.project_out.bias.to(dtype) if q.project_out.bias is not None else None,
        )
    return quantized

def get_fsq_levels_from_q(q):
    #if hasattr(q, "layers") and len(q.layers) > 0 and hasattr(q.layers[0], "_levels"):
    #   return q.layers[0]._levels.tolist()
    return [8, 8, 8, 5, 5, 5]
