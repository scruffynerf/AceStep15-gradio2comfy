"""AceStepAudioCodesToLatent node for ACE-Step"""
import torch
import re
import logging

logger = logging.getLogger(__name__)

class AceStepAudioCodesToLatent:
    """
    Convert ACE-Step audio codes (5Hz) to LM-hint latents (25Hz).
    These latents describe the structural acoustic features and are used 
    for conditioning the DiT, as well as for structural manipulation.
    """
    
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
    CATEGORY = "Scromfy/Ace-Step/audio"
    
    @classmethod
    def IS_CHANGED(s, audio_codes, model, latent_scaling):
        # Audio codes are nested lists; use string hashing to detect any change in the sequence
        import hashlib
        code_str = str(audio_codes)
        return hashlib.sha256(code_str.encode()).hexdigest() + f"_{latent_scaling}"

    def convert(self, audio_codes, model, latent_scaling):
        import comfy.model_management
        
        if not audio_codes:
            logger.warning("No audio codes provided")
            return ({"samples": torch.zeros([1, 64, 1])},)

        # 1. Access the model and components
        inner_model = model.model
        if hasattr(inner_model, "diffusion_model"):
            inner_model = inner_model.diffusion_model
            
        if not (hasattr(inner_model, "tokenizer") and hasattr(inner_model, "detokenizer")):
            logger.error("Model does not have required tokenizer/detokenizer attributes.")
            return ({"samples": torch.zeros([1, 64, 1])},)

        tokenizer = inner_model.tokenizer
        quantizer = tokenizer.quantizer
        detokenizer = inner_model.detokenizer
        
        # Load model to GPU
        comfy.model_management.load_model_gpu(model)
        device = comfy.model_management.get_torch_device()
        dtype = model.model.get_dtype()
        
        # 2. Determine quantizer structure
        num_quantizers = 1
        levels = getattr(quantizer, '_levels', getattr(quantizer, 'levels', None))
        if levels is not None:
            num_quantizers = len(levels)
        
        # 3. Process indices batch-wise
        batch_samples = []
        
        # If input is a flat list, wrap it in a batch dim
        if isinstance(audio_codes, list) and audio_codes and not isinstance(audio_codes[0], list):
            audio_codes = [audio_codes]
            
        for batch_item in audio_codes:
            # batch_item is a list of integers
            indices_tensor = torch.tensor(batch_item, device=device, dtype=torch.long)
            
            # Reshape to (T, Q)
            try:
                indices_tensor = indices_tensor.reshape(-1, num_quantizers)
            except Exception as e:
                logger.error(f"Failed to reshape audio codes to {num_quantizers} quantizers: {e}")
                # Fallback to whatever fits if possible, or skip
                continue
                
            # Add batch dim for quantizer
            indices_tensor = indices_tensor.unsqueeze(0) # (1, T, Q)
            
            with torch.no_grad():
                # get_output_from_indices returns (1, T, 2048)
                quantized = quantizer.get_output_from_indices(indices_tensor, dtype=dtype)
                
                # detokenizer: (1, T_5hz, 2048) -> (1, T_25hz, 64)
                lm_hints = detokenizer(quantized)
                
                # Convert to ComfyUI format: [1, T, 64] -> [1, 64, T]
                # Parity with extract_semantic_hints: lm_hints.movedim(-1, -2)
                semantic_item = lm_hints.movedim(-1, -2)
                
                if latent_scaling != 1.0:
                    semantic_item = semantic_item * latent_scaling
                    
                batch_samples.append(semantic_item)

        if not batch_samples:
            return ({"samples": torch.zeros([1, 64, 1])},)

        # Concatenate batch items back together
        samples = torch.cat(batch_samples, dim=0)

        # Return the raw tensor to match extract_semantic_hints parity
        # and satisfy nodes expecting a .shape attribute
        return (samples.cpu(),)

NODE_CLASS_MAPPINGS = {
    "AceStepAudioCodesToLatent": AceStepAudioCodesToLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepAudioCodesToLatent": "Audio Codes to Latent",
}
