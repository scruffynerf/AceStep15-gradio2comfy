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

        # 1. Find the next available counter for the prefix to prevent overwriting
        counter = 1
        # We check for the existence of ANY of the potential files for this prefix
        # We assume if the 'timbre' file exists, the whole bundle exists or we shouldn't overwrite it.
        while True:
            candidate_base = f"{filename_prefix}_{counter:04d}"
            # Check if any of our expected files exist with this base
            exists = False
            for ext in ["_timbre.safetensors", "_pooled.safetensors", "_lyrics.safetensors", "_codes.json"]:
                if os.path.exists(os.path.join(save_path, f"{candidate_base}{ext}")):
                    exists = True
                    break
            if not exists:
                break
            counter += 1

        for i, item in enumerate(conditioning):
            # Handle list of conditionings (batch)
            timbre_tensor = item[0]
            metadata = item[1]
            
            # Use suffix for batch if more than 1 (inner batch index)
            # If batch > 1, we still use the same base counter prefix for the whole batch?
            # Or should each item in the batch get its own unique number?
            # Standard ComfyUI SaveImage handles batches by incrementing for each image.
            # But here we are saving a "conditioning" which might be a list of parts.
            
            # Let's keep it simple: the whole "conditioning" object gets the counter.
            # If it's a batch of 4, we might want _0, _1, _2, _3 or just increment the global counter.
            # Usually, SaveImage saves multiple images as separate files with incrementing numbers.
            
            suffix = f"_{i}" if len(conditioning) > 1 else ""
            base_name = f"{candidate_base}{suffix}"
            
            # 1. Timbre Tensor (safetensors)
            save_file({"timbre": timbre_tensor}, os.path.join(save_path, f"{base_name}_timbre.safetensors"))
            
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
