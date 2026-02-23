"""AceStepConditioningMixerLoader node for ACE-Step"""
import os
import json
import torch
import random
from safetensors.torch import load_file

def get_conditioning_files(suffix):
    base_path = "output/conditioning"
    if not os.path.exists(base_path):
        return ["none", "random"]
    
    files = [f for f in os.listdir(base_path) if f.endswith(suffix)]
    return sorted(files) + ["none", "random"]

class AceStepConditioningMixerLoader:
    """Load and mix specific conditioning components from saved files (safetensors/json)"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "main_tensor_file": (get_conditioning_files("_main.safetensors"),),
                "pooled_output_file": (get_conditioning_files("_pooled.safetensors"),),
                "lyrics_file": (get_conditioning_files("_lyrics.safetensors"),),
                "audio_codes_file": (get_conditioning_files("_codes.json"),),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "conditioning_info")
    FUNCTION = "load_and_mix"
    CATEGORY = "Scromfy/Ace-Step/advanced"

    @classmethod
    def IS_CHANGED(s, main_tensor_file, pooled_output_file, lyrics_file, audio_codes_file, seed):
        return f"{main_tensor_file}_{pooled_output_file}_{lyrics_file}_{audio_codes_file}_{seed}"

    def load_and_mix(self, main_tensor_file, pooled_output_file, lyrics_file, audio_codes_file, seed):
        base_path = "output/conditioning"
        rng = random.Random(seed)
        
        def pick_file(selected, suffix):
            if selected == "random":
                options = [f for f in os.listdir(base_path) if f.endswith(suffix)] if os.path.exists(base_path) else []
                if not options:
                    return "none"
                return rng.choice(options)
            return selected

        # Resolve randoms
        main_tensor_file = pick_file(main_tensor_file, "_main.safetensors")
        pooled_output_file = pick_file(pooled_output_file, "_pooled.safetensors")
        lyrics_file = pick_file(lyrics_file, "_lyrics.safetensors")
        audio_codes_file = pick_file(audio_codes_file, "_codes.json")

        # 1. Main Tensor (Required base)
        if main_tensor_file == "none":
            raise ValueError("Mixer Loader requires at least a Main Tensor file to establish the base conditioning.")
            
        main_path = os.path.join(base_path, main_tensor_file)
        main_tensor = load_file(main_path).get("main")
        
        metadata = {}
        
        # 2. Pooled Output
        if pooled_output_file != "none":
            metadata["pooled_output"] = load_file(os.path.join(base_path, pooled_output_file)).get("pooled")
        else:
            metadata["pooled_output"] = None
            
        # 3. Conditioning Lyrics
        if lyrics_file != "none":
            metadata["conditioning_lyrics"] = load_file(os.path.join(base_path, lyrics_file)).get("lyrics")
            
        # 4. Audio Codes
        if audio_codes_file != "none":
            with open(os.path.join(base_path, audio_codes_file), "r") as f:
                metadata["audio_codes"] = json.load(f)
            
        # Construct filename-safe info string: main_pool(if any)_lyrics_codes
        def get_base(filename, suffix):
            if filename == "none": return None
            return filename.replace(suffix, "")

        base_main = get_base(main_tensor_file, "_main.safetensors")
        base_pool = get_base(pooled_output_file, "_pooled.safetensors")
        base_lyrics = get_base(lyrics_file, "_lyrics.safetensors")
        base_codes = get_base(audio_codes_file, "_codes.json")
        
        parts = [base_main]
        if base_pool: parts.append(base_pool)
        parts.append(base_lyrics if base_lyrics else "nolyrics")
        parts.append(base_codes if base_codes else "noaudiocodes")
        
        filename_info = "_".join(parts)
            
        return ([[main_tensor, metadata]], filename_info)

NODE_CLASS_MAPPINGS = {
    "AceStepConditioningMixerLoader": AceStepConditioningMixerLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepConditioningMixerLoader": "Conditioning Mixer Loader",
}
