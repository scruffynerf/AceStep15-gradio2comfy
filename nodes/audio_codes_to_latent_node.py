"""AceStepAudioCodesToLatent node for ACE-Step"""
import torch
import re
import logging
import comfy.model_management

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
        
        # 3. Process indices
        # We need to distinguish between [batch_of_samples] and [list_of_indices_for_one_sample]
        # Common cases:
        # 1. [[idx1, idx2...]] -> List of lists, outer len 1. Inner is the sequence.
        # 2. [idx1, idx2...]   -> Flat list. Treat as one sequence.
        
        # Normalize to list of tensors [B, T, Q]
        sequences = []
        
        if not isinstance(audio_codes, list):
            # Fallback for tensor or other types
            audio_codes = [audio_codes]
            
        # Check if it's already a batch of lists or a single list
        if audio_codes and not isinstance(audio_codes[0], list):
            # Case 2: [idx1, idx2...] -> Wrap so we have [[idx1, idx2...]]
            audio_codes = [audio_codes]
            
        for batch_item in audio_codes:
            # Parse this batch item (sequence) into a flat list of integers
            code_ids = []
            if isinstance(batch_item, list):
                for x in batch_item:
                    if isinstance(x, (int, float)): code_ids.append(int(x))
                    elif isinstance(x, str):
                        found = re.findall(r"(\d+)", x)
                        code_ids.extend([int(v) for v in found])
            elif isinstance(batch_item, (int, float)):
                # Handle scalar if ComfyUI mapped over the list
                code_ids = [int(batch_item)]
            elif isinstance(batch_item, torch.Tensor):
                code_ids = batch_item.flatten().tolist()
                
            if not code_ids: continue
            
            # Convert to [T, Q]
            tensor = torch.tensor(code_ids, device=device, dtype=torch.long)
            try:
                # If they are packed, Q=1. 
                # If they are flattened split indices, Q=num_quantizers.
                # We try to reshape to [T, Q]. If it doesn't fit, we default to Q=1.
                if len(code_ids) % num_quantizers == 0:
                    tensor = tensor.reshape(-1, num_quantizers)
                else:
                    tensor = tensor.reshape(-1, 1)
            except:
                tensor = tensor.reshape(-1, 1)
                
            sequences.append(tensor.unsqueeze(0)) # [1, T, Q]

        if not sequences:
            return (torch.zeros([1, 64, 1]),)

        batch_samples = []
        for indices_tensor in sequences:
            # indices_tensor is [1, T, Q]
            # Match the quantizer's expected Q dim
            if indices_tensor.shape[-1] < getattr(quantizer, 'num_quantizers', 1):
                # If we have packed indices but the quantizer expects split...
                # This would need unpacking. For now we assume they match.
                pass
            
            with torch.no_grad():
                # Step 2 bridge: IDs -> embeddings
                quantized = quantizer.get_output_from_indices(indices_tensor, dtype=dtype)
                # Step 2: Detokenize -> 25Hz
                lm_hints = detokenizer(quantized)
                # [1, T, 64] -> [1, 64, T]
                semantic_item = lm_hints.movedim(-1, -2)
                
                if latent_scaling != 1.0:
                    semantic_item = semantic_item * latent_scaling
                    
                batch_samples.append(semantic_item)

        # Result is [B, 64, T]
        samples = torch.cat(batch_samples, dim=0)
        return (samples.cpu(),)

NODE_CLASS_MAPPINGS = {
    "AceStepAudioCodesToSemanticHints": AceStepAudioCodesToSemanticHints,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepAudioCodesToSemanticHints": "Audio Codes to Semantic Hints",
}
