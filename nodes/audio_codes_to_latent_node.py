"""AceStepAudioCodesToLatent node for ACE-Step"""
import torch
import re
import logging

logger = logging.getLogger(__name__)

class AceStepAudioCodesToSemanticHints:
    """
    Convert ACE-Step audio codes (5Hz) to Semantic Hints (25Hz).
    These hints describe the structural acoustic features (rhythm, melody, harmony)
    and are used for conditioning and structural manipulation.
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
        # Efficiently hash the input structure without full string conversion
        if not audio_codes: return "none"
        import hashlib
        # Hash based on: Outer list length, Inner list length (of first item), Item count (of first item)
        # and a slice of the first 10 values to detect selection changes
        try:
            sample = str(audio_codes[0][:10]) if audio_codes and audio_codes[0] else "empty"
            info = f"{len(audio_codes)}_{len(audio_codes[0]) if audio_codes[0] else 0}_{sample}_{latent_scaling}"
            return hashlib.md5(info.encode()).hexdigest()
        except:
            return f"fallback_{latent_scaling}"

    def convert(self, audio_codes, model, latent_scaling):
        import comfy.model_management
        
        if not audio_codes:
            logger.warning("No audio codes provided")
            return (torch.zeros([1, 64, 1]),)

        # 1. Access the model and components
        inner_model = model.model
        if hasattr(inner_model, "diffusion_model"):
            inner_model = inner_model.diffusion_model
            
        if not (hasattr(inner_model, "tokenizer") and hasattr(inner_model, "detokenizer")):
            logger.error("Model does not have required tokenizer/detokenizer attributes.")
            return (torch.zeros([1, 64, 1]),)

        tokenizer = inner_model.tokenizer
        quantizer = tokenizer.quantizer
        detokenizer = inner_model.detokenizer
        
        # Load model to GPU
        comfy.model_management.load_model_gpu(model)
        device = comfy.model_management.get_torch_device()
        dtype = model.model.get_dtype()
        
        # 2. Determine quantizer structure
        # In Ace-Step 1.5, the 'audio_codes' are stored as COMBINED indices (one per timestep).
        # Therefore, we always reshape to (-1, 1) when feeding back into the quantizer
        # if using combined indices.
        num_quantizers = 1
        
        # 3. Process indices batch-wise
        batch_samples = []
        
        # If input is a flat list, wrap it in a batch dim
        if isinstance(audio_codes, list) and audio_codes and not isinstance(audio_codes[0], list):
            audio_codes = [audio_codes]
            
        for batch_item in audio_codes:
            # 3. Parse batch item to integers
            code_ids = []
            if isinstance(batch_item, (int, float)):
                code_ids = [int(batch_item)]
            elif isinstance(batch_item, list):
                for sub in batch_item:
                    if isinstance(sub, (int, float)):
                        code_ids.append(int(sub))
                    elif isinstance(sub, str):
                        found = re.findall(r"(\d+)", sub)
                        code_ids.extend([int(x) for x in found])
            elif isinstance(batch_item, torch.Tensor):
                code_ids = batch_item.flatten().tolist()
                
            if not code_ids:
                continue

            indices_tensor = torch.tensor(code_ids, device=device, dtype=torch.long)
            
            # Reshape to (T, Q)
            try:
                indices_tensor = indices_tensor.reshape(-1, num_quantizers)
            except Exception as e:
                logger.error(f"Failed to reshape batch item of length {len(code_ids)} to {num_quantizers} quantizers: {e}")
                continue
                
            # Add batch dim for quantizer
            indices_tensor = indices_tensor.unsqueeze(0) # (1, T, 1)
            
            with torch.no_grad():
                # get_output_from_indices returns (1, T, 2048)
                # It handles the mapping from one integer to the multi-level FSQ vector
                quantized = quantizer.get_output_from_indices(indices_tensor, dtype=dtype)
                
                # detokenizer: (1, T_5hz, 2048) -> (1, T_25hz, 64)
                lm_hints = detokenizer(quantized)
                
                # Convert to ComfyUI format: [1, T, 64] -> [1, 64, T]
                semantic_item = lm_hints.movedim(-1, -2)
                
                if latent_scaling != 1.0:
                    semantic_item = semantic_item * latent_scaling
                    
                batch_samples.append(semantic_item)

        if not batch_samples:
            return (torch.zeros([1, 64, 1]),)

        # Concatenate batch items back together
        samples = torch.cat(batch_samples, dim=0)

        # Return the raw tensor to match extract_semantic_hints parity
        # and satisfy nodes expecting a .shape attribute
        return (samples.cpu(),)

NODE_CLASS_MAPPINGS = {
    "AceStepAudioCodesToSemanticHints": AceStepAudioCodesToSemanticHints,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepAudioCodesToSemanticHints": "Audio Codes to Semantic Hints",
}
