"""AceStepAudioCodesMixer node for ACE-Step"""
import torch
import torch.nn.functional as F
import logging
import hashlib
import comfy.model_management
from .includes.fsq_utils import (
    parse_audio_codes, fsq_decode_indices, fsq_encode_to_indices, get_fsq_levels
)

logger = logging.getLogger(__name__)

class AceStepAudioCodesMixer:
    """
    Mix two sets of audio codes in 6d FSQ space.
    Provides functionality equivalent to AceStepTensorMixer but for discrete tokens.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_codes_A": ("LIST",),
                "audio_codes_B": ("LIST",),
                #"model": ("MODEL",),
                "mode": ([
                    "blend", "lerp", "inject", "average", "difference_injection", 
                    "dominant_recessive", "replace", "concatenate", "add", 
                    "multiply", "maximum", "minimum"
                ], {"default": "blend"}),
                "alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "weight": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "eps": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.001}),
                "scale_mode": (["scale_B_to_A", "scale_A_to_B", "pad_to_match", "loop_match", "none"], {"default": "scale_B_to_A"}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("audio_codes",)
    FUNCTION = "mix"
    CATEGORY = "Scromfy/Ace-Step/Conditioning"

    @classmethod
    def IS_CHANGED(s, audio_codes_A, audio_codes_B, mode, alpha, ratio, weight, eps, scale_mode, mask=None):
        try:
            # Hash some samples and parameters
            sample_A = str(audio_codes_A[0][:5]) if audio_codes_A else "e"
            sample_B = str(audio_codes_B[0][:5]) if audio_codes_B else "e"
            info = f"{sample_A}_{sample_B}_{mode}_{alpha}_{ratio}_{weight}_{eps}_{scale_mode}"
            if mask is not None:
                info += f"_{mask.abs().mean().item():.4f}"
            return hashlib.md5(info.encode()).hexdigest()
        except:
            return "random"

    def mix(self, audio_codes_A, audio_codes_B, mode, alpha, ratio, weight, eps, scale_mode, mask=None):
        #inner_model = model.model
        #if hasattr(inner_model, "diffusion_model"):
        #    inner_model = inner_model.diffusion_model

        #comfy.model_management.load_model_gpu(model)
        device = comfy.model_management.get_torch_device()
        levels = get_fsq_levels(None)

        parsed_A = parse_audio_codes(audio_codes_A)
        parsed_B = parse_audio_codes(audio_codes_B)

        if not parsed_A or not parsed_A[0]:
            logger.error("Empty audio_codes_A input")
            return (audio_codes_B,)
        if not parsed_B or not parsed_B[0]:
            return (audio_codes_A,)

        ids_A = parsed_A[0]
        ids_B = parsed_B[0]

        # Decode to 6D float space [1, T, 6]
        codes_6d_A = fsq_decode_indices(torch.tensor(ids_A, dtype=torch.long, device=device).unsqueeze(0), levels)
        codes_6d_B = fsq_decode_indices(torch.tensor(ids_B, dtype=torch.long, device=device).unsqueeze(0), levels)

        # Handle scaling / matching
        len_A = codes_6d_A.shape[1]
        len_B = codes_6d_B.shape[1]

        if len_A != len_B:
            if scale_mode == "scale_B_to_A":
                codes_6d_B = self.interpolate_6d(codes_6d_B, len_A)
            elif scale_mode == "scale_A_to_B":
                codes_6d_A = self.interpolate_6d(codes_6d_A, len_B)
            elif scale_mode == "pad_to_match":
                codes_6d_A, codes_6d_B = self.pad_6d(codes_6d_A, codes_6d_B)
            elif scale_mode == "loop_match":
                target_len = max(len_A, len_B)
                codes_6d_A = self.loop_6d(codes_6d_A, target_len)
                codes_6d_B = self.loop_6d(codes_6d_B, target_len)
            elif scale_mode == "none" and mode != "concatenate":
                # Default to scaling B to A for safety
                codes_6d_B = self.interpolate_6d(codes_6d_B, len_A)

        # Prepare mask
        target_len = codes_6d_A.shape[1]
        if mask is None:
            mask = torch.ones((1, target_len, 1), device=device)
        else:
            # Resize mask if needed [B, T] or [B, H, W] to [1, T, 1]
            mask = mask.to(device)
            if mask.dim() == 2: # [B, T]
                mask = mask.unsqueeze(-1)
            elif mask.dim() == 3: # [B, H, W]
                mask = mask.mean(dim=1).unsqueeze(-1) # Flatten spatial if accidentally passed junk
            
            if mask.shape[1] != target_len:
                # Interpolate mask
                mask = mask.transpose(1, 2)
                mask = F.interpolate(mask, size=target_len, mode='linear', align_corners=False)
                mask = mask.transpose(1, 2)

        # Perform mixing in 6D space
        A = codes_6d_A
        B = codes_6d_B

        if mode == "blend":
            out = mask * A + (1.0 - mask) * B
        elif mode == "lerp":
            blended = alpha * A + (1.0 - alpha) * B
            out = mask * blended + (1.0 - mask) * A
        elif mode == "inject":
            out = A + mask * B
        elif mode == "average":
            res = 0.5 * (A + B)
            out = mask * res + (1.0 - mask) * A
        elif mode == "difference_injection":
            delta = weight * (B - A)
            out = A + mask * delta
        elif mode == "dominant_recessive":
            out = A + mask * (eps * B)
        elif mode == "replace":
            out = mask * B + (1.0 - mask) * A
        elif mode == "concatenate":
            # Apply mask to B before concat
            out = torch.cat([A, B * mask], dim=1)
        elif mode == "add":
            res = A + B
            out = mask * res + (1.0 - mask) * A
        elif mode == "multiply":
            res = A * B
            out = mask * res + (1.0 - mask) * A
        elif mode == "maximum":
            res = torch.max(A, B)
            out = mask * res + (1.0 - mask) * A
        elif mode == "minimum":
            res = torch.min(A, B)
            out = mask * res + (1.0 - mask) * A
        else:
            out = A

        # Clamp to FSQ range [-1, 1] before encoding
        out = out.clamp(-1.0, 1.0)
        
        # Encode back to indices
        result_ids = fsq_encode_to_indices(out, levels)[0].tolist()

        return ([result_ids],)

    def interpolate_6d(self, t, target_len):
        # [1, T, 6] -> [1, 6, T]
        t = t.transpose(1, 2)
        t = F.interpolate(t, size=target_len, mode='linear', align_corners=False)
        return t.transpose(1, 2)

    def pad_6d(self, A, B):
        len_A = A.shape[1]
        len_B = B.shape[1]
        max_len = max(len_A, len_B)
        
        def pad_one(tensor, current_len, target_len):
            if current_len >= target_len:
                return tensor
            pad_size = target_len - current_len
            padding = torch.zeros((tensor.shape[0], pad_size, tensor.shape[2]), device=tensor.device, dtype=tensor.dtype)
            return torch.cat([tensor, padding], dim=1)

        return pad_one(A, len_A, max_len), pad_one(B, len_B, max_len)

    def loop_6d(self, tensor, target_len):
        current_len = tensor.shape[1]
        if current_len >= target_len:
            return tensor[:, :target_len, :]
        reps = (target_len + current_len - 1) // current_len
        out = tensor.repeat(1, reps, 1)
        return out[:, :target_len, :]

NODE_CLASS_MAPPINGS = {
    "AceStepAudioCodesMixer": AceStepAudioCodesMixer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepAudioCodesMixer": "Audio Codes Mixer",
}
