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
                "latent": ("LATENT",),
                "model": ("MODEL",),
                "latent_scaling": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01}),
            }
        }
    
    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("audio_codes",)
    FUNCTION = "convert"
    CATEGORY = "Scromfy/Ace-Step/audio"

    def convert(self, latent, model, latent_scaling):
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
        # ComfyUI latent shape: [B, C, T] -> [1, 64, T_25hz]
        samples = latent["samples"].to(device)
        
        # Apply inverse scaling if needed (to get back to detokenizer-native space)
        if latent_scaling != 1.0:
            samples = samples / latent_scaling
            
        # Transpose back to [1, T_25hz, 64]
        x = samples.transpose(1, 2).to(model_dtype)

        # 3. Tokenize
        with torch.no_grad():
            # tokenize(x) handles padding and reshaping to [B, T_patch, 5, 64]
            # and returns (quantized, indices)
            # quantized: [B, T_5hz, 2048]
            # indices: [B, T_5hz, num_quantizers]
            _, indices = tokenizer.tokenize(x)
            
            # indices shape: [B, T_5hz, num_quantizers]
            # Flatten to a simple list of integers
            audio_codes = indices.flatten().cpu().tolist()

        return (audio_codes,)

NODE_CLASS_MAPPINGS = {
    "AceStepLatentToAudioCodes": AceStepLatentToAudioCodes,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepLatentToAudioCodes": "Latent to Audio Codes",
}
