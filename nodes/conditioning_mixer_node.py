"""AceStepConditioningMixer node for ACE-Step"""
import torch

class AceStepConditioningMixer:
    """Mix components (tune tensor, pooled output, lyrics, audio codes) from two conditioning inputs"""
    
    @classmethod
    def INPUT_TYPES(s):
        source_options = ["A", "B", "none"]
        return {
            "required": {
                "conditioning_A": ("CONDITIONING",),
                "conditioning_B": ("CONDITIONING",),
                "tune_tensor_source": (source_options, {"default": "A"}),
                "pooled_output_source": (source_options, {"default": "A"}),
                "lyrics_source": (source_options, {"default": "A"}),
                "audio_codes_source": (source_options, {"default": "A"}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "mix"
    CATEGORY = "Scromfy/Ace-Step/advanced"

    def mix(self, conditioning_A, conditioning_B, tune_tensor_source, pooled_output_source, lyrics_source, audio_codes_source):
        result = []
        max_len = max(len(conditioning_A), len(conditioning_B))
        
        for i in range(max_len):
            # Safe access (looping behavior like ComfyUI handles batch mismatches)
            item_A = conditioning_A[i] if i < len(conditioning_A) else conditioning_A[-1]
            item_B = conditioning_B[i] if i < len(conditioning_B) else conditioning_B[-1]
            
            # Determine base metadata and tune tensor
            if tune_tensor_source == "A":
                base_tensor = item_A[0]
                new_meta = item_A[1].copy()
            elif tune_tensor_source == "B":
                base_tensor = item_B[0]
                new_meta = item_B[1].copy()
            else: # none
                # Use a zero tensor of the same shape as A or B
                shape_ref = item_A[0] if item_A is not None else item_B[0]
                base_tensor = torch.zeros_like(shape_ref)
                new_meta = {} # Start fresh if no tune source
            
            # Override specific metadata items
            
            # Pooled Output
            if pooled_output_source == "A":
                new_meta["pooled_output"] = item_A[1].get("pooled_output")
            elif pooled_output_source == "B":
                new_meta["pooled_output"] = item_B[1].get("pooled_output")
            elif pooled_output_source == "none":
                new_meta["pooled_output"] = None
                
            # Lyrics
            if lyrics_source == "A":
                new_meta["conditioning_lyrics"] = item_A[1].get("conditioning_lyrics")
            elif lyrics_source == "B":
                new_meta["conditioning_lyrics"] = item_B[1].get("conditioning_lyrics")
            elif lyrics_source == "none":
                new_meta["conditioning_lyrics"] = None
                
            # Audio Codes
            if audio_codes_source == "A":
                new_meta["audio_codes"] = item_A[1].get("audio_codes")
            elif audio_codes_source == "B":
                new_meta["audio_codes"] = item_B[1].get("audio_codes")
            elif audio_codes_source == "none":
                new_meta["audio_codes"] = None
                
            result.append([base_tensor, new_meta])
            
        return (result,)

NODE_CLASS_MAPPINGS = {
    "AceStepConditioningMixer": AceStepConditioningMixer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepConditioningMixer": "Conditioning Component Mixer",
}
