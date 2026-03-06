"""AceStepAudioCodesUnaryOp node for ACE-Step"""
import torch
import torch.nn.functional as F
import logging
import hashlib
import comfy.model_management
from .includes.fsq_utils import (
    parse_audio_codes, fsq_decode_indices, fsq_encode_to_indices, get_fsq_levels
)

logger = logging.getLogger(__name__)

class AceStepAudioCodesUnaryOp:
    """Operations that transform a single set of audio codes (A) in 6D FSQ space with optional masking"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_codes": ("LIST",),
                #"model": ("MODEL",),
                "mode": ([
                    "noop_visualize",
                    "gate",
                    "scale_masked",
                    "noise_masked",
                    "fade_out"
                ], {"default": "noop_visualize"}),
                "ssm_blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "length_pct": ("FLOAT", {"default": 100.0, "min": 0.1, "max": 1000.0, "step": 0.1}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "sigma": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("LIST", "IMAGE", "IMAGE")
    RETURN_NAMES = ("audio_codes", "visualization_ssm", "visualization_linear")
    FUNCTION = "process"
    CATEGORY = "Scromfy/Ace-Step/mixers"

    @classmethod
    def IS_CHANGED(s, audio_codes, mode, length_pct, strength, sigma, seed, ssm_blur, mask=None):
        try:
            # Hash some samples and parameters
            sample = str(audio_codes[0][:5]) if audio_codes else "e"
            info = f"{sample}_{mode}_{length_pct}_{strength}_{sigma}_{seed}_{ssm_blur}"
            if mask is not None:
                info += f"_{mask.abs().mean().item():.4f}"
            return hashlib.md5(info.encode()).hexdigest()
        except:
            return "random"

    def _linear_token_image(self, A, upscale=32):
        """
        Linear time visualization of tokens.
        A shape: [1, T, 6]
        """

        X = A[0]  # [T,6]

        # normalize [-1,1] -> [0,1]
        X = (X.clamp(-1,1) + 1.0) / 2.0

        # transpose so dims become rows
        img = X.T  # [6,T]

        # upscale vertically
        img = img.repeat_interleave(upscale, dim=0)

        # expand RGB
        img = img.unsqueeze(0).unsqueeze(-1).repeat(1,1,1,3)

    return img

    def _self_similarity_image(self, A, blur_sigma=0.0):
        """
        Create a Self-Similarity Matrix image from decoded tensor A.
        A shape: [1, T, 6]
        """
        X = A[0]  # [T,6]

        # normalize vectors
        X = torch.nn.functional.normalize(X, dim=1)

        # cosine similarity
        S = X @ X.T   # [T,T]

        # convert [-1,1] → [0,1]
        S = (S + 1.0) / 2.0

        # optional gaussian blur (the "cool trick")
        if blur_sigma > 0:
            k = int(blur_sigma * 4 + 1)
            if k % 2 == 0:
                k += 1

            coords = torch.arange(k, device=S.device) - k // 2
            g = torch.exp(-(coords**2) / (2 * blur_sigma**2))
            g = g / g.sum()

            kernel = torch.outer(g, g)
            kernel = kernel / kernel.sum()

            kernel = kernel.view(1,1,k,k)

            S = S.unsqueeze(0).unsqueeze(0)
            S = F.conv2d(S, kernel, padding=k//2)
            S = S[0,0]

        # convert to RGB image
        img = S.unsqueeze(0).unsqueeze(-1).repeat(1,1,1,3)
        return img

    def process(self, audio_codes, mode, length_pct, strength, sigma, seed, ssm_blur, mask=None):
        #inner_model = model.model
        #if hasattr(inner_model, "diffusion_model"):
        #    inner_model = inner_model.diffusion_model

        #comfy.model_management.load_model_gpu(model)
        device = comfy.model_management.get_torch_device()
        levels = get_fsq_levels(None)

        parsed = parse_audio_codes(audio_codes)

        if not parsed or not parsed[0]:
            logger.error("Empty audio_codes input")
            return (audio_codes,)

        ids = parsed[0]

        # Decode to 6D float space [1, T, 6]
        A = fsq_decode_indices(torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0), levels)

        # --- Visualization (time vs FSQ dimensions) ---
        vis_image1 = self._self_similarity_image(A, blur_sigma=ssm_blur)
        vis_image2 = self._linear_token_image(A, upscale=32)

        # Handle Length Scaling
        if length_pct != 100.0:
            curr_len = A.shape[1]
            new_len = max(1, int(curr_len * (length_pct / 100.0)))
            
            # [1, L, 6] -> [1, 6, L]
            A = A.transpose(1, 2)
            A = torch.nn.functional.interpolate(A, size=new_len, mode='linear', align_corners=False)
            A = A.transpose(1, 2)

        # Prepare mask
        target_len = A.shape[1]
        if mask is None:
            mask = torch.ones((1, target_len, 1), device=device)
        else:
            # Resize mask if needed [B, T] or [B, H, W] to [1, T, 1]
            mask = mask.to(device)
            if mask.dim() == 2: # [B, T]
                mask = mask.unsqueeze(-1)
            elif mask.dim() == 3: # [B, H, W]
                mask = mask.mean(dim=1).unsqueeze(-1) # Flatten spatial
            
            if mask.shape[1] != target_len:
                # Interpolate mask
                mask = mask.transpose(1, 2)
                mask = F.interpolate(mask, size=target_len, mode='linear', align_corners=False)
                mask = mask.transpose(1, 2)

        # Perform Operations in 6D space

        if mode == "noop_visualize":
            out = A

        elif mode == "gate":
            # Gating: A * mask (values move towards 0.0 midpoint)
            out = A * mask
            
        elif mode == "scale_masked":
            # Scale A only where mask > 0
            # formula: A * (1.0 + mask * (strength - 1.0))
            out = A * (1.0 + mask * (strength - 1.0))
            
        elif mode == "noise_masked":
            # Add noise only where mask allows
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            noise = torch.randn_like(A, generator=generator) * sigma
            out = A + mask * noise
            
        elif mode == "fade_out":
            # Special case: linear ramp multiply
            N = target_len
            fade = torch.linspace(1, 0, N, device=device)[None, :, None]
            # (1.0 - mask) * A + mask * (A * fade)
            out = A * (1.0 - mask) + (A * fade) * mask
            
        else:
            out = A

        # Clamp to FSQ range [-1, 1] before encoding
        out = out.clamp(-1.0, 1.0)
        
        # Encode back to indices
        result_ids = fsq_encode_to_indices(out, levels)[0].tolist()

        return ([result_ids], vis_image1, vis_image2)

NODE_CLASS_MAPPINGS = {
    "AceStepAudioCodesUnaryOp": AceStepAudioCodesUnaryOp,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepAudioCodesUnaryOp": "Audio Codes Unary Operations",
}
