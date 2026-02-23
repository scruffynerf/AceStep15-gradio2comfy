"""AceStepConditioningSave node for ACE-Step"""
import os
import json
import torch
from safetensors.torch import save_file

class AceStepConditioningSave:
    """Save conditioning components to separate files (safetensors for tensors, json for codes)"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "save_path": ("STRING", {"default": "output/conditioning"}),
                "filename_prefix": ("STRING", {"default": "ace_cond"}),
            }
        }
    
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "Scromfy/Ace-Step/advanced"

    def save(self, conditioning, save_path, filename_prefix):
        os.makedirs(save_path, exist_ok=True)

        for i, item in enumerate(conditioning):
            # Handle list of conditionings (batch)
            tune_tensor = item[0]
            metadata = item[1]
            
            # Use suffix for batch if more than 1
            suffix = f"_{i}" if len(conditioning) > 1 else ""
            base_name = f"{filename_prefix}{suffix}"
            
            # 1. Tune Tensor (safetensors)
            save_file({"tune": tune_tensor}, os.path.join(save_path, f"{base_name}_tune.safetensors"))
            
            # 2. Pooled Output (safetensors if tensor)
            pooled = metadata.get("pooled_output")
            if pooled is not None and isinstance(pooled, torch.Tensor):
                save_file({"pooled": pooled}, os.path.join(save_path, f"{base_name}_pooled.safetensors"))
            
            # 3. Conditioning Lyrics (safetensors)
            lyrics = metadata.get("conditioning_lyrics")
            if lyrics is not None and isinstance(lyrics, torch.Tensor):
                save_file({"lyrics": lyrics}, os.path.join(save_path, f"{base_name}_lyrics.safetensors"))
                
            # 4. Audio Codes (json)
            codes = metadata.get("audio_codes")
            if codes is not None:
                with open(os.path.join(save_path, f"{base_name}_codes.json"), "w") as f:
                    json.dump(codes, f)
                
        return {}

NODE_CLASS_MAPPINGS = {
    "AceStepConditioningSave": AceStepConditioningSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepConditioningSave": "Save Conditioning Elements",
}
