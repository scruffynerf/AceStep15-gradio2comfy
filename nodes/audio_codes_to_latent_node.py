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
        # Content-aware change detection
        if not audio_codes: return "none"
        import hashlib
        try:
            # Hash samples from the lists to detect value changes
            # Includes length and a few samples from start/mid/end
            if isinstance(audio_codes[0], list):
                item = audio_codes[0]
                L = len(item)
                # Sample values at start, middle and end
                s1 = str(item[0]) if L > 0 else "e"
                s2 = str(item[L//2]) if L > 1 else "m"
                s3 = str(item[-1]) if L > 2 else "l"
                info = f"{len(audio_codes)}_{L}_{s1}_{s2}_{s3}_{latent_scaling}"
            else:
                # Flat list
                info = f"{len(audio_codes)}_{str(audio_codes[:5])}_{latent_scaling}"
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
        num_quantizers = 1
        if hasattr(quantizer, "num_quantizers"):
            num_quantizers = quantizer.num_quantizers
        elif hasattr(quantizer, "layers"):
            num_quantizers = len(quantizer.layers)
        elif hasattr(quantizer, "_levels") and not hasattr(quantizer, "layers"):
            # Single-level FSQ
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
            
            # Reshape to (T, Q) if it was flattened, or just ensure [T, Q]
            try:
                indices_tensor = indices_tensor.reshape(-1, num_quantizers)
            except Exception as e:
                logger.error(f"Failed to reshape batch item of length {len(code_ids)} to {num_quantizers} quantizers: {e}")
                # Fallback: if it's a power of num_quantizers or just divisible
                if len(code_ids) % num_quantizers == 0:
                    indices_tensor = indices_tensor.reshape(-1, num_quantizers)
                else:
                    continue
                
            # Add batch dim for quantizer: (1, T, Q)
            indices_tensor = indices_tensor.unsqueeze(0) 
            
            with torch.no_grad():
                # Step 2 bridge: Convert integer IDs back to quantized embeddings (2048-dim)
                # This is a lookup, NOT requantization. Detokenizer needs these floats.
                quantized = quantizer.get_output_from_indices(indices_tensor, dtype=dtype)
                
                # Step 2: Detokenize - upsample from 5Hz to 25Hz
                lm_hints = detokenizer(quantized)
                
                # Transpose back to ComfyUI format: [B, T, D] -> [B, D, T]
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
