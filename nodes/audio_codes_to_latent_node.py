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
    
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "convert"
    CATEGORY = "Scromfy/Ace-Step/audio"

    def convert(self, audio_codes, model, latent_scaling):
        # 1. Parse input to flat int list
        code_ids = []
        for item in audio_codes:
            if isinstance(item, (int, float)):
                code_ids.append(int(item))
            elif torch.is_tensor(item):
                if item.numel() == 1:
                    code_ids.append(int(item.item()))
                else:
                    code_ids.extend(item.flatten().tolist())
            elif isinstance(item, list):
                code_ids.extend(item)
            elif isinstance(item, str):
                # Handle string format if passed inside a list
                found = re.findall(r"<\|audio_code_(\d+)\|>", item)
                code_ids.extend([int(x) for x in found])

        if not code_ids:
            logger.warning("No valid audio codes found in input")
            return ({"samples": torch.zeros([1, 64, 1, 100])},)

        # 2. Access the model and components
        # ComfyUI model object usually has model.model as the nn.Module
        inner_model = model.model
        if hasattr(inner_model, "diffusion_model"):
            inner_model = inner_model.diffusion_model
            
        if not (hasattr(inner_model, "tokenizer") and hasattr(inner_model, "detokenizer")):
            logger.error("Model does not have required tokenizer/detokenizer attributes.")
            return ({"samples": torch.zeros([1, 64, 1, len(code_ids) * 5])},)

        quantizer = inner_model.tokenizer.quantizer
        detokenizer = inner_model.detokenizer
        device = next(inner_model.parameters()).device
        
        # 3. Build indices tensor and clamp OOB
        codebook_size = getattr(quantizer, 'codebook_size', None)
        if codebook_size is None:
            levels = getattr(quantizer, '_levels', getattr(quantizer, 'levels', None))
            if levels is not None:
                codebook_size = 1
                for l in levels:
                    codebook_size *= int(l)

        indices = torch.tensor(code_ids, device=device, dtype=torch.long)

        if codebook_size is not None:
            oob = (indices < 0) | (indices >= codebook_size)
            if oob.any():
                logger.warning(f"Clamping {oob.sum().item()} OOB codes (valid 0-{codebook_size-1})")
                indices = indices.clamp(0, codebook_size - 1)

        # 4. Convert indices to latents
        # indices shape for ResidualFSQ.get_output_from_indices needs to be (B, T, L)
        # where L is number of quantizers.
        indices = indices.unsqueeze(0).unsqueeze(-1) # (1, T_5hz, 1)

        # Get model dtype to avoid float32 vs bfloat16 mismatch
        model_dtype = next(inner_model.parameters()).dtype

        with torch.no_grad():
            # get_output_from_indices returns (1, T_5hz, 2048) with project_out applied
            quantized = quantizer.get_output_from_indices(indices, dtype=model_dtype)
            
            # detokenizer: transformer expansion 5Hz -> 25Hz
            latents = detokenizer(quantized)
            # latents: (1, T_25hz, 64)

            # ComfyUI audio latent format: [B, C, T]
            samples = latents.transpose(1, 2) # [1, 64, T_25hz]
            
            # Apply scaling (user suggested ~0.5 relative to DiT space)
            if latent_scaling != 1.0:
                samples = samples * latent_scaling


            # Log stats
            logger.info(f"AudioCode Latents Stats - min: {samples.min().item():.4f}, max: {samples.max().item():.4f}, abs mean: {samples.abs().mean().item():.6f}, std: {samples.std().item():.6f}")

        return ({"samples": samples.cpu()},)

NODE_CLASS_MAPPINGS = {
    "AceStepAudioCodesToLatent": AceStepAudioCodesToLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepAudioCodesToLatent": "Audio Codes to Latent",
}
