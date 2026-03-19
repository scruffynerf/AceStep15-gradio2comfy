"""AceStepAudioCodesToSemanticHints node for ACE-Step"""
import torch
import logging
import comfy.model_management
from .includes.fsq_utils import parse_audio_codes, fsq_indices_to_quantized

logger = logging.getLogger(__name__)

class AceStepAudioCodesToSemanticHints:
    """Convert 5Hz audio codes to 25Hz semantic hints for DiT conditioning."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_codes": ("LIST",),
                "model": ("MODEL",),
                "latent_scaling": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("SEMANTIC_HINTS",)
    RETURN_NAMES = ("semantic_hints",)
    FUNCTION = "convert"
    CATEGORY = "Scromfy/Ace-Step/Conditioning"

    @classmethod
    def IS_CHANGED(s, audio_codes, latent_scaling):
        if not audio_codes: return "none"
        import hashlib
        try:
            L = len(audio_codes)
            info = f"{L}_{str(audio_codes[0][:10]) if L > 0 else 'e'}_{latent_scaling}"
            return hashlib.md5(info.encode()).hexdigest()
        except:
            return f"fallback_{latent_scaling}"

    def convert(self, audio_codes, model, latent_scaling):
        inner_model = model.model
        if hasattr(inner_model, "diffusion_model"):
            inner_model = inner_model.diffusion_model

        if not (hasattr(inner_model, "tokenizer") and hasattr(inner_model, "detokenizer")):
            logger.error("Model does not have tokenizer/detokenizer.")
            return (torch.zeros([1, 64, 1]),)

        tokenizer = inner_model.tokenizer
        detokenizer = inner_model.detokenizer

        comfy.model_management.load_model_gpu(model)
        device = comfy.model_management.get_torch_device()
        dtype = model.model.get_dtype()

        batch_samples = []
        for code_ids in parse_audio_codes(audio_codes):
            if not code_ids: continue
            
            # Use utility to get quantized embeddings
            quantized = fsq_indices_to_quantized(tokenizer.quantizer, code_ids, device, dtype)
            
            with torch.no_grad():
                # Detokenize to 25Hz
                lm_hints = detokenizer(quantized)
                # [1, T, 64] -> [1, 64, T]
                semantic_item = lm_hints.movedim(-1, -2)
                
                if latent_scaling != 1.0:
                    semantic_item = semantic_item * latent_scaling
                batch_samples.append(semantic_item)

        if not batch_samples:
            return (torch.zeros([1, 64, 1]),)

        return (torch.cat(batch_samples, dim=0).cpu(),)

NODE_CLASS_MAPPINGS = {
    "AceStepAudioCodesToSemanticHints": AceStepAudioCodesToSemanticHints,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepAudioCodesToSemanticHints": "Audio Codes -> Semantic Hints",
}
