"""
Text encoding nodes for ACE-Step
"""

from __future__ import annotations
import torch

class AceStepMetadataBuilder:
    """Format music metadata for ACE-Step conditioning"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bpm": ("INT", {"default": 0, "min": 0, "max": 300, "step": 1}),
                "duration": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1000.0, "step": 0.1}),
                "keyscale": ("STRING", {"default": ""}),
                "timesignature": ("INT", {"default": 4, "min": 2, "max": 4}),
                "language": (["en", "zh", "ja", "ko", "auto"], {"default": "en"}),
                "instrumental": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("metadata",)
    FUNCTION = "build"
    CATEGORY = "Scromfy/Ace-Step/text"

    def build(self, bpm, duration, keyscale, timesignature, language, instrumental):
        metadata = {
            "bpm": bpm if bpm > 0 else None,
            "duration": duration if duration > 0 else None,
            "keyscale": keyscale if keyscale.strip() else None,
            "timesignature": timesignature,
            "language": language,
            "instrumental": instrumental,
        }
        # Filter out None values
        metadata = {k: v for k, v in metadata.items() if v is not None}
        return (metadata,)


class AceStepCLIPTextEncode:
    """Specialized CLIP text encoding that accepts metadata for ACE-Step"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True}),
                "metadata": ("DICT",),
            },
            "optional": {
                "lyrics": ("STRING", {"multiline": True, "default": ""}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "Scromfy/Ace-Step/text"

    def encode(self, clip, text, metadata, lyrics=""):
        # Make a copy of metadata to avoid modifying the input dict
        meta_copy = metadata.copy()
        
        # Merge lyrics into metadata if provided
        if lyrics.strip():
            meta_copy["lyrics"] = lyrics
            
        # Logic from NODE_SPECS.md:
        # Call clip.tokenize_with_weights(text, **metadata) then clip.encode_from_tokens()
        tokens = clip.tokenize_with_weights(text, return_word_ids=False, **meta_copy)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]], )


NODE_CLASS_MAPPINGS = {
    "AceStepMetadataBuilder": AceStepMetadataBuilder,
    "AceStepCLIPTextEncode": AceStepCLIPTextEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepMetadataBuilder": "Metadata Builder",
    "AceStepCLIPTextEncode": "CLIP Text Encode (ACE-Step)",
}
