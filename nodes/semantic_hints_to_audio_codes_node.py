"""AceStepSemanticHintsToAudioCodes node for ACE-Step"""
import torch
import logging
import comfy.model_management
from .includes.fsq_utils import fsq_encode_to_indices

logger = logging.getLogger(__name__)

class AceStepSemanticHintsToAudioCodes:
    """Convert 25Hz semantic hints back to 5Hz audio codes (lossy)."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "semantic_hints": ("SEMANTIC_HINTS",),
                "model": ("MODEL",),
                "latent_scaling": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("audio_codes",)
    FUNCTION = "convert"
    CATEGORY = "Scromfy/Ace-Step/Conditioning"

    @classmethod
    def IS_CHANGED(s, semantic_hints, latent_scaling):
        if semantic_hints is None: return "none"
        import hashlib
        try:
            info = f"{semantic_hints.shape}_{semantic_hints.abs().mean().item():.6f}_{latent_scaling}"
            return hashlib.md5(info.encode()).hexdigest()
        except:
            return f"fallback_{latent_scaling}"

    def convert(self, semantic_hints, model, latent_scaling):
        inner_model = model.model
        if hasattr(inner_model, "diffusion_model"):
            inner_model = inner_model.diffusion_model

        if not hasattr(inner_model, "tokenizer"):
            logger.error("Model does not have tokenizer.")
            return ([],)

        tokenizer = inner_model.tokenizer
        device = comfy.model_management.get_torch_device()
        dtype = model.model.get_dtype()

        samples = semantic_hints.to(device=device, dtype=dtype)
        if latent_scaling != 1.0:
            samples = samples / latent_scaling

        latent_25hz = samples.movedim(-1, -2)  # [B, T_25hz, 64]

        with torch.no_grad():
            B, T_25hz, D = latent_25hz.shape
            pool_window = 5
            if T_25hz % pool_window != 0:
                pad = pool_window - (T_25hz % pool_window)
                latent_25hz = torch.nn.functional.pad(latent_25hz, (0, 0, 0, pad))
                T_25hz = latent_25hz.shape[1]

            T_5hz = T_25hz // pool_window
            q = tokenizer.quantizer
            levels = q.layers[0]._levels.tolist()

            x = latent_25hz.contiguous().view(B, T_5hz, pool_window, D)
            x = tokenizer.audio_acoustic_proj(x)
            x = tokenizer.attention_pooler(x)
            
            # Map to 6D FSQ space manually to match the user's high-fidelity version
            codes_6d = torch.nn.functional.linear(
                x.to(dtype),
                q.project_in.weight.to(dtype),
                q.project_in.bias.to(dtype) if q.project_in.bias is not None else None,
            )
            
            # Encode to composite indices using utility
            indices = fsq_encode_to_indices(codes_6d, levels)
            audio_codes = [indices[b].tolist() for b in range(B)]

        return (audio_codes,)

NODE_CLASS_MAPPINGS = {
    "AceStepSemanticHintsToAudioCodes": AceStepSemanticHintsToAudioCodes,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepSemanticHintsToAudioCodes": "Semantic Hints -> Audio Codes (VERY Flawed)",
}
