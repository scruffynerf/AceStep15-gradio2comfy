"""AceStepLyricsLoader node for ACE-Step"""
import os
import random
from safetensors.torch import load_file

def get_lyrics_files():
    base_path = "output/conditioning"
    if not os.path.exists(base_path):
        return ["none", "random"]
    files = [f for f in os.listdir(base_path) if f.endswith("_lyrics.safetensors")]
    return sorted(files) + ["none", "random"]

class AceStepLyricsTensorLoader:
    """Load a lyrics conditioning tensor from disk"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lyrics_file": (get_lyrics_files(),),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("TENSOR", "STRING")
    RETURN_NAMES = ("lyrics_tensor", "filename")
    FUNCTION = "load"
    CATEGORY = "Scromfy/Ace-Step/loaders"

    @classmethod
    def IS_CHANGED(s, lyrics_file, seed):
        return f"{lyrics_file}_{seed}"

    def load(self, lyrics_file, seed):
        base_path = "output/conditioning"
        rng = random.Random(seed)
        
        if lyrics_file == "random":
            options = [f for f in os.listdir(base_path) if f.endswith("_lyrics.safetensors")] if os.path.exists(base_path) else []
            if not options:
                return (None, "none")
            lyrics_file = rng.choice(options)
            
        if lyrics_file == "none":
            return (None, "none")
            
        path = os.path.join(base_path, lyrics_file)
        tensor = load_file(path).get("lyrics")
        
        return (tensor, lyrics_file.replace("_lyrics.safetensors", ""))

NODE_CLASS_MAPPINGS = {
    "AceStepLyricsTensorLoader": AceStepLyricsTensorLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepLyricsTensorLoader": "Load Lyrics Tensor",
}
