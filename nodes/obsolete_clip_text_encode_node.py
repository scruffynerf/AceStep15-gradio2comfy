"""AceStepCLIPTextEncode node for ACE-Step"""

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
    CATEGORY = "Scromfy/Ace-Step/obsolete"

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
    "AceStepCLIPTextEncode": AceStepCLIPTextEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepCLIPTextEncode": "CLIP Text Encode (ACE-Step)",
}
