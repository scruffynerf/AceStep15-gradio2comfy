"""AceStepConditioningSplitter node for ACE-Step"""
import torch

class AceStepConditioningSplitter:
    """Split an ACE-Step conditioning object into its individual component tensors and lists"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
            }
        }
    
    RETURN_TYPES = ("TENSOR", "TENSOR", "TENSOR", "LIST")
    RETURN_NAMES = ("tune_tensor", "pooled_output", "lyrics_tensor", "audio_codes")
    FUNCTION = "split"
    CATEGORY = "Scromfy/Ace-Step/advanced"

    def split(self, conditioning):
        if not conditioning or len(conditioning) == 0:
            return (None, None, None, None)
            
        # We take the first item in the conditioning list (standard for ACE-Step)
        # Note: If there are multiple items, this node follows the pattern of only processing the first.
        item = conditioning[0]
        tune_tensor = item[0]
        metadata = item[1]
        
        pooled_output = metadata.get("pooled_output")
        lyrics_tensor = metadata.get("conditioning_lyrics")
        audio_codes = metadata.get("audio_codes")
        
        return (tune_tensor, pooled_output, lyrics_tensor, audio_codes)

NODE_CLASS_MAPPINGS = {
    "AceStepConditioningSplitter": AceStepConditioningSplitter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepConditioningSplitter": "Conditioning Component Splitter",
}
