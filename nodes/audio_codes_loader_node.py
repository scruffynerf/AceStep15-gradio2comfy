"""AceStepAudioCodesLoader node for ACE-Step"""
import os
import json
import random

def get_codes_files():
    base_path = "output/conditioning"
    if not os.path.exists(base_path):
        return ["none", "random"]
    files = [f for f in os.listdir(base_path) if f.endswith("_codes.json")]
    return sorted(files) + ["none", "random"]

class AceStepAudioCodesLoader:
    """Load audio codes from disk"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_codes_file": (get_codes_files(),),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("LIST", "STRING")
    RETURN_NAMES = ("audio_codes", "filename")
    FUNCTION = "load"
    CATEGORY = "Scromfy/Ace-Step/loaders"

    @classmethod
    def IS_CHANGED(s, audio_codes_file, seed):
        if audio_codes_file == "none":
            return "none"
        
        base_path = "output/conditioning"
        if audio_codes_file == "random":
            # For random, the seed is the determinant
            return f"random_{seed}"
            
        path = os.path.join(base_path, audio_codes_file)
        if os.path.exists(path):
            return f"{audio_codes_file}_{os.path.getmtime(path)}"
        
        return f"{audio_codes_file}_{seed}"

    def load(self, audio_codes_file, seed):
        base_path = "output/conditioning"
        rng = random.Random(seed)
        
        if audio_codes_file == "random":
            options = [f for f in os.listdir(base_path) if f.endswith("_codes.json")] if os.path.exists(base_path) else []
            if not options:
                return (None, "none")
            audio_codes_file = rng.choice(options)
            
        if audio_codes_file == "none":
            return (None, "none")
            
        path = os.path.join(base_path, audio_codes_file)
        with open(path, "r") as f:
            codes = json.load(f)
        
        return (codes, audio_codes_file.replace("_codes.json", ""))

NODE_CLASS_MAPPINGS = {
    "AceStepAudioCodesLoader": AceStepAudioCodesLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepAudioCodesLoader": "Load Audio Codes",
}
