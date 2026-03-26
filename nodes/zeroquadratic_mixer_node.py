"""AceStepZeroQuadraticMixer node for ACE-Step"""
import torch
import torch.nn.functional as F
import json
import math
import hashlib
import logging

import comfy.model_management
from .includes.fsq_utils import (
    parse_audio_codes, fsq_decode_indices, fsq_encode_to_indices, get_fsq_levels
)
from .includes.zerobytes_utils import (
    pair_hash, asymmetric_pair_hash, hash_to_float,
    DIM_SALTS, parse_section_map, lookup_section,
)

logger = logging.getLogger(__name__)


class AceStepZeroQuadraticMixer:
    """Pair-is-seed deterministic audio code mixing via Zero-Quadratic hashing.

    Instead of manually setting blend weights, the relationship between
    track A and track B is computed from their coordinate pair hash.
    Each FSQ dimension gets its own blend weight. Each timestep gets
    its own variation. The pair IS the recipe.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_codes_A": ("LIST",),
                "audio_codes_B": ("LIST",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFF}),
                "relationship_mode": ([
                    "mutual", "call_response", "tension",
                    "echo", "complement", "interference"
                ], {"default": "mutual"}),
                "affinity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "asymmetry": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dim_coupling": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "scale_mode": (["scale_B_to_A", "scale_A_to_B", "pad_to_match",
                                "loop_match", "none"], {"default": "scale_B_to_A"}),
            },
            "optional": {
                "section_map": ("STRING", {"default": "", "multiline": True}),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("LIST", "STRING")
    RETURN_NAMES = ("audio_codes", "blend_map")
    FUNCTION = "mix"
    CATEGORY = "Scromfy/Ace-Step/Conditioning"

    @classmethod
    def IS_CHANGED(s, audio_codes_A, audio_codes_B, seed, relationship_mode,
                   affinity, asymmetry, dim_coupling, scale_mode,
                   section_map="", mask=None):
        try:
            sample_a = str(audio_codes_A[0][:5]) if audio_codes_A else "e"
            sample_b = str(audio_codes_B[0][:5]) if audio_codes_B else "e"
            info = (f"{sample_a}_{sample_b}_{seed}_{relationship_mode}_"
                    f"{affinity}_{asymmetry}_{dim_coupling}_{scale_mode}")
            if mask is not None:
                info += f"_{mask.abs().mean().item():.4f}"
            return hashlib.md5(info.encode()).hexdigest()
        except Exception:
            return "random"

    def mix(self, audio_codes_A, audio_codes_B, seed, relationship_mode,
            affinity, asymmetry, dim_coupling, scale_mode,
            section_map="", mask=None):
        device = comfy.model_management.get_torch_device()
        levels = get_fsq_levels(None)

        parsed_A = parse_audio_codes(audio_codes_A)
        parsed_B = parse_audio_codes(audio_codes_B)

        if not parsed_A or not parsed_A[0]:
            return (audio_codes_B, json.dumps({"error": "empty A"}))
        if not parsed_B or not parsed_B[0]:
            return (audio_codes_A, json.dumps({"error": "empty B"}))

        ids_A, ids_B = parsed_A[0], parsed_B[0]

        codes_A = fsq_decode_indices(
            torch.tensor(ids_A, dtype=torch.long, device=device).unsqueeze(0), levels)
        codes_B = fsq_decode_indices(
            torch.tensor(ids_B, dtype=torch.long, device=device).unsqueeze(0), levels)

        # Match lengths
        codes_A, codes_B = self._match_lengths(codes_A, codes_B, scale_mode)
        T = codes_A.shape[1]

        # Compute per-timestep, per-dimension blend weights
        blend_weights = torch.zeros(1, T, 6, device=device)
        blend_log = []

        for t in range(T):
            w = self._compute_pair_weights(
                t, seed, relationship_mode, affinity, asymmetry, dim_coupling)
            blend_weights[0, t, :] = torch.tensor(w, device=device)
            if t < 10 or t % 50 == 0:
                blend_log.append({"t": t, "weights": [round(x, 3) for x in w]})

        # Apply external mask
        if mask is not None:
            mask_t = mask.to(device)
            if mask_t.dim() == 2:
                mask_t = mask_t.unsqueeze(-1)
            elif mask_t.dim() == 3:
                mask_t = mask_t.mean(dim=1).unsqueeze(-1)
            if mask_t.shape[1] != T:
                mask_t = mask_t.transpose(1, 2)
                mask_t = F.interpolate(mask_t, size=T, mode='linear', align_corners=False)
                mask_t = mask_t.transpose(1, 2)
            blend_weights = blend_weights * mask_t

        # Section-aware modulation
        if section_map and section_map.strip():
            sections = parse_section_map(section_map)
            if sections:
                for t in range(T):
                    sec_type, sec_idx = lookup_section(t, sections)
                    sec_mod = hash_to_float(pair_hash(
                        sec_type, sec_idx, sec_type, sec_idx + 100,
                        seed ^ 0x5ECA1))
                    blend_weights[0, t, :] *= (0.5 + 0.5 * sec_mod)

        # Blend in 6D space
        out = codes_A * (1.0 - blend_weights) + codes_B * blend_weights
        out = out.clamp(-1.0, 1.0)

        result_ids = fsq_encode_to_indices(out, levels)[0].tolist()
        blend_map = json.dumps({"mode": relationship_mode, "samples": blend_log})

        return ([result_ids], blend_map)

    def _compute_pair_weights(self, t, seed, mode, affinity, asymmetry, dim_coupling):
        """Compute 6 blend weights for timestep t from pair hashes."""
        if mode in ("mutual", "tension", "complement", "interference"):
            base_h = pair_hash(t, 0, t, 1, seed)
        else:
            base_h = asymmetric_pair_hash(t, 0, t, 1, seed)

        base_weight = hash_to_float(base_h) * affinity
        weights = []

        for d in range(6):
            dim_h = pair_hash(t, d, t, d + 6, seed ^ DIM_SALTS[d])
            dim_weight = hash_to_float(dim_h) * affinity
            w = dim_weight * (1.0 - dim_coupling) + base_weight * dim_coupling
            w = w * (1.0 - asymmetry)

            if mode == "tension":
                w *= 0.2 if d >= 3 else 1.5
            elif mode == "complement":
                w = affinity - w
            elif mode == "echo":
                w *= 0.3
            elif mode == "interference":
                phase = hash_to_float(pair_hash(t, d, 0, 0, seed ^ 0x0A7E))
                w *= abs(math.sin(t * 0.3 + phase * 6.28))

            weights.append(max(0.0, min(1.0, w)))
        return weights

    def _match_lengths(self, A, B, scale_mode):
        """Match tensor sequence lengths."""
        len_A, len_B = A.shape[1], B.shape[1]
        if len_A == len_B:
            return A, B

        if scale_mode == "scale_B_to_A":
            B = self._interpolate(B, len_A)
        elif scale_mode == "scale_A_to_B":
            A = self._interpolate(A, len_B)
        elif scale_mode == "pad_to_match":
            target = max(len_A, len_B)
            A = self._pad(A, target)
            B = self._pad(B, target)
        elif scale_mode == "loop_match":
            target = max(len_A, len_B)
            A = self._loop(A, target)
            B = self._loop(B, target)
        else:
            B = self._interpolate(B, len_A)
        return A, B

    def _interpolate(self, t, target_len):
        t = t.transpose(1, 2)
        t = F.interpolate(t, size=target_len, mode='linear', align_corners=False)
        return t.transpose(1, 2)

    def _pad(self, t, target_len):
        if t.shape[1] >= target_len:
            return t
        pad_size = target_len - t.shape[1]
        padding = torch.zeros(t.shape[0], pad_size, t.shape[2],
                              device=t.device, dtype=t.dtype)
        return torch.cat([t, padding], dim=1)

    def _loop(self, t, target_len):
        if t.shape[1] >= target_len:
            return t[:, :target_len, :]
        reps = (target_len + t.shape[1] - 1) // t.shape[1]
        out = t.repeat(1, reps, 1)
        return out[:, :target_len, :]


NODE_CLASS_MAPPINGS = {
    "AceStepZeroQuadraticMixer": AceStepZeroQuadraticMixer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepZeroQuadraticMixer": "Zero Quadratic Audio Codes Mixer",
}
