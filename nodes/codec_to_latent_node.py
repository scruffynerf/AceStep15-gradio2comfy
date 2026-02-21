"""AceStepCodecToLatent node for ACE-Step"""
import torch
import re
import logging

logger = logging.getLogger(__name__)

class AceStepCodecToLatent:
    """Convert FSQ codes back to latent representation using the model's detokenizer"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_codes": ("STRING", {"multiline": True}),
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "decode"
    CATEGORY = "Scromfy/Ace-Step/audio"

    def decode(self, audio_codes, model):
        try:
            # parsing tokens from string
            tokens = re.findall(r'(\d+)', audio_codes)
            if not tokens:
                return ({"samples": torch.zeros([1, 64, 2])},)
            
            code_ids = [int(t) for t in tokens]
            indices = torch.tensor(code_ids, dtype=torch.long).unsqueeze(0).unsqueeze(-1) # [1, T, 1]
            
            # Use model's detokenizer
            inner_model = getattr(model.model, "diffusion_model", model.model)
            
            if hasattr(inner_model, "tokenizer") and hasattr(inner_model, "detokenizer"):
                quantizer = inner_model.tokenizer.quantizer
                detokenizer = inner_model.detokenizer
                
                device = next(inner_model.parameters()).device
                indices = indices.to(device)
                
                with torch.no_grad():
                    quantized = quantizer.get_output_from_indices(indices)
                    latent_25hz = detokenizer(quantized)
                    samples = latent_25hz.transpose(1, 2).unsqueeze(2) # [1, 64, 1, T_25]
                    
                return ({"samples": samples.cpu()},)
            else:
                logger.warning("Model does not have tokenizer/detokenizer - returning empty latent")
                return ({"samples": torch.zeros([1, 64, 2])},)
                
        except Exception as e:
            logger.error(f"Codec to latent conversion failed: {e}")
            return ({"samples": torch.zeros([1, 64, 2])},)


NODE_CLASS_MAPPINGS = {
    "AceStepCodecToLatent": AceStepCodecToLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepCodecToLatent": "Codec to Latent",
}
