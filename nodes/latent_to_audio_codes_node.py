"""AceStepLatentToAudioCodes node for ACE-Step"""
import torch
import logging

logger = logging.getLogger(__name__)

class AceStepLatentToAudioCodes:
    """
    Convert 25Hz latents back to 5Hz audio codes (re-tokenization).
    Completes the roundtrip for audio manipulation.
    """
    
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
    CATEGORY = "Scromfy/Ace-Step/audio"
    
    @classmethod
    def IS_CHANGED(s, semantic_hints, model, latent_scaling):
        # Tensor-aware change detection
        if semantic_hints is None: return "none"
        import hashlib
        try:
            # Hash shape and mean to detect content changes quickly
            mean_val = semantic_hints.abs().mean().item()
            info = f"{semantic_hints.shape}_{mean_val:.6f}_{latent_scaling}"
            return hashlib.md5(info.encode()).hexdigest()
        except:
            return f"fallback_{latent_scaling}"
    
    def convert(self, semantic_hints, model, latent_scaling):
        # 1. Access the model and components
        inner_model = model.model
        if hasattr(inner_model, "diffusion_model"):
            inner_model = inner_model.diffusion_model
            
        if not hasattr(inner_model, "tokenizer"):
            logger.error("Model does not have required tokenizer attribute.")
            return ([],)

        tokenizer = inner_model.tokenizer
        device = next(inner_model.parameters()).device
        model_dtype = next(inner_model.parameters()).dtype

        # 2. Prepare latents
        # Source is [B, 64, T]. Tokenizer expects [B, T, 64]
        device = comfy.model_management.get_torch_device()
        dtype = model.model.get_dtype()
        
        samples = semantic_hints.to(device=device, dtype=dtype)
        
        # Apply inverse scaling
        if latent_scaling != 1.0:
            samples = samples / latent_scaling
            
        # Parity with extract_semantic_hints: movedim(-1, -2)
        x = samples.movedim(-1, -2)

        # 3. Tokenize
        with torch.no_grad():
            # tokenize(x) handles padding and reshaping to [B, T_patch, 5, 64]
            # and returns (quantized, indices)
            # quantized: [B, T_5hz, 2048]
            # indices: [B, T_5hz, num_quantizers]
            _, indices = tokenizer.tokenize(x)
            
            # Return as nested list [Batch, Time*Quantizers]
            # This handles both combined (Q=1) and split (Q>1) codes consistently 
            # while matching the user's loader format.
            B = indices.shape[0]
            audio_codes = indices.reshape(B, -1).detach().cpu().tolist()

        return (audio_codes,)

NODE_CLASS_MAPPINGS = {
    "AceStepLatentToAudioCodes": AceStepLatentToAudioCodes,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepLatentToAudioCodes": "Latent to Audio Codes",
}
