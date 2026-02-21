"""AceStepConditioning node for ACE-Step"""

class AceStepConditioning:
    """Combine text, lyrics, and timbre conditioning"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_cond": ("CONDITIONING",),
            },
            "optional": {
                "lyrics": ("STRING", {"multiline": True}),
                "timbre_audio": ("AUDIO",),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "combine"
    CATEGORY = "Scromfy/Ace-Step/advanced"

    def combine(self, text_cond, lyrics=None, timbre_audio=None):
        # Conditioning is a list of lists: [[cond, {"pooled_output": ...}]]
        new_cond = []
        for t in text_cond:
            c = t[0]
            metadata = t[1].copy()
            if lyrics:
                metadata["lyrics"] = lyrics
            if timbre_audio:
                metadata["timbre_audio"] = timbre_audio
            new_cond.append([c, metadata])
            
        return (new_cond,)


NODE_CLASS_MAPPINGS = {
    "AceStepConditioning": AceStepConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepConditioning": "Combined Conditioning",
}
