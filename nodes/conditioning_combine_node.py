"""AceStepConditioningCombine node for ACE-Step"""
import torch

class AceStepConditioningCombine:
    """Assemble separate components into a full ACE-Step conditioning"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tune_tensor": ("TENSOR",),
            },
            "optional": {
                "pooled_output": ("TENSOR",),
                "lyrics_tensor": ("TENSOR",),
                "audio_codes": ("LIST",),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "combine"
    CATEGORY = "Scromfy/Ace-Step/advanced"

    def combine(self, tune_tensor, pooled_output=None, lyrics_tensor=None, audio_codes=None):
        # Conditioning is a list of lists: [[cond, metadata_dict]]
        # We assume batch size 1 for simplicity of assembly here
        
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
