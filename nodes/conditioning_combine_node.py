"""AceStepConditioningCombine node for ACE-Step"""
import torch

class AceStepConditioningCombine:
    """Assemble separate components into a full ACE-Step conditioning"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "tune_tensor": ("TENSOR",),
                "pooled_output": ("TENSOR",),
                "lyrics_tensor": ("TENSOR",),
                "audio_codes": ("LIST",),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "combine"
    CATEGORY = "Scromfy/Ace-Step/advanced"

    def combine(self, tune_tensor=None, pooled_output=None, lyrics_tensor=None, audio_codes=None):
        # If tune_tensor is not provided, create a default zero tensor
        if tune_tensor is None:
            # Try to infer batch size and device from other inputs
            batch_size = 1
            device = "cpu"
            if pooled_output is not None:
                batch_size = pooled_output.shape[0]
                device = pooled_output.device
            elif lyrics_tensor is not None:
                if lyrics_tensor.dim() == 3:
                    batch_size = lyrics_tensor.shape[0]
                device = lyrics_tensor.device
            
            # Default to sequence length 1, dimension 1024 (standard for ACE-Step 1.5)
            # We use [B, 1, 1024]
            tune_tensor = torch.zeros((batch_size, 1, 1024), device=device)
            
        metadata = {
            "pooled_output": pooled_output,
            "conditioning_lyrics": lyrics_tensor,
            "audio_codes": audio_codes
        }
        
        # Ensure tune_tensor has proper batch dim if missing
        if tune_tensor.dim() == 2:
            tune_tensor = tune_tensor.unsqueeze(0)
            
        return ([[tune_tensor, metadata]],)

NODE_CLASS_MAPPINGS = {
    "AceStepConditioningCombine": AceStepConditioningCombine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepConditioningCombine": "Conditioning Component Combiner",
}
