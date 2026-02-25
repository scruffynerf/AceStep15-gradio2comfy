"""High-fidelity roundtrip test nodes for ACE-Step 1.5"""
import torch
import re
import logging
import hashlib
import comfy.model_management

logger = logging.getLogger(__name__)

class AceStepAudioCodesToSemanticHints_Test:
    """
    High-fidelity version of Audio Codes to Semantic Hints.
    Uses precise 3D index formatting [B, T, 1] for detokenization.
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
    CATEGORY = "Scromfy/Ace-Step/audio/test"
    
    @classmethod
    def IS_CHANGED(s, audio_codes, model, latent_scaling):
        if not audio_codes: return "none"
        try:
            L = len(audio_codes)
            info = f"{L}_{str(audio_codes[0][:10]) if L > 0 else 'e'}_{latent_scaling}"
            return hashlib.md5(info.encode()).hexdigest()
        except:
            return f"fallback_{latent_scaling}"

    def convert(self, audio_codes, model, latent_scaling):
        # 1. Access the model and components
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
        
        # 2. Process indices
        if not isinstance(audio_codes, list):
            audio_codes = [audio_codes]
        if audio_codes and not isinstance(audio_codes[0], list):
            audio_codes = [audio_codes]
            
        batch_samples = []
        for batch_item in audio_codes:
            # Parse integers
            code_ids = []
            for x in batch_item:
                if isinstance(x, (int, float)): code_ids.append(int(x))
                elif isinstance(x, str):
                    found = re.findall(r"(\d+)", x)
                    code_ids.extend([int(v) for v in found])
                
            if not code_ids: continue
            
            # Forward Logic: [B, T] -> [B, T, 1] -> [B, T, D] -> [B, 64, T*5]
            indices = torch.tensor(code_ids, dtype=torch.long, device=device).unsqueeze(0).unsqueeze(-1)
            
            with torch.no_grad():
                # Get embeddings from indices
                quantized = tokenizer.quantizer.get_output_from_indices(indices, dtype=dtype)
                # Detokenize to 25Hz
                lm_hints = detokenizer(quantized)
                # [1, T, 64] -> [1, 64, T]
                semantic_item = lm_hints.movedim(-1, -2)
                
                if latent_scaling != 1.0:
                    semantic_item = semantic_item * latent_scaling
                    
                batch_samples.append(semantic_item)

        if not batch_samples:
            return (torch.zeros([1, 64, 1]),)

        samples = torch.cat(batch_samples, dim=0)
        return (samples.cpu(),)


class AceStepSemanticHintsToAudioCodes_Test:
    """
    High-fidelity version of Semantic Hints to Audio Codes.
    Implements explicit 4D patching [B, T, 5, 64] for the AttentionPooler.
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
    CATEGORY = "Scromfy/Ace-Step/audio/test"
    
    @classmethod
    def IS_CHANGED(s, semantic_hints, model, latent_scaling):
        if semantic_hints is None: return "none"
        try:
            m = semantic_hints.abs().mean().item()
            info = f"{semantic_hints.shape}_{m:.6f}_{latent_scaling}"
            return hashlib.md5(info.encode()).hexdigest()
        except:
            return f"fallback_{latent_scaling}"

    def convert(self, semantic_hints, model, latent_scaling):
        # 1. Access components
        inner_model = model.model
        if hasattr(inner_model, "diffusion_model"):
            inner_model = inner_model.diffusion_model
            
        if not hasattr(inner_model, "tokenizer"):
            logger.error("Model does not have tokenizer.")
            return ([],)

        tokenizer = inner_model.tokenizer
        device = comfy.model_management.get_torch_device()
        dtype = model.model.get_dtype()
        
        # 2. Prepare latents
        samples = semantic_hints.to(device=device, dtype=dtype)
        if latent_scaling != 1.0:
            samples = samples / latent_scaling
            
        # [B, 64, T_25hz] -> [B, T_25hz, 64]
        latent_25hz = samples.movedim(-1, -2)
        
        # 3. Securely Tokenize step-by-step
        with torch.no_grad():
            B, T_25hz, D = latent_25hz.shape
            pool_window = 5
            
            # Ensure T_25hz is divisible by 5
            if T_25hz % pool_window != 0:
                logger.warning(f"Latent length {T_25hz} not divisible by 5, padding...")
                pad = pool_window - (T_25hz % pool_window)
                latent_25hz = torch.nn.functional.pad(latent_25hz, (0, 0, 0, pad))
                T_25hz = latent_25hz.shape[1]

            # Step 1: Reshape to patches [B, T_5hz, P, D]
            T_5hz = T_25hz // pool_window
            x = latent_25hz.view(B, T_5hz, pool_window, D)
            
            # Step 2: Project 64 -> hidden_size
            x = tokenizer.audio_acoustic_proj(x)
            
            # Step 3: Attention Pool [B, T_5hz, P, D] -> [B, T_5hz, D]
            x = tokenizer.attention_pooler(x)
            
            # Step 4: Quantize
            quantized, indices = tokenizer.quantizer(x)
            # indices: [B, T_5hz, num_quantizers]
            
            # Format output as [[idx1, idx2...]]
            audio_codes = indices.reshape(B, -1).detach().cpu().tolist()

        return (audio_codes,)


NODE_CLASS_MAPPINGS = {
    "AceStepAudioCodesToSemanticHints_Test": AceStepAudioCodesToSemanticHints_Test,
    "AceStepSemanticHintsToAudioCodes_Test": AceStepSemanticHintsToAudioCodes_Test,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepAudioCodesToSemanticHints_Test": "Audio Codes to Semantic Hints (Test)",
    "AceStepSemanticHintsToAudioCodes_Test": "Semantic Hints to Audio Codes (Test)",
}
