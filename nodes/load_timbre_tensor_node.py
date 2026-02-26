"""AceStepMainLoader node for ACE-Step"""
import os
import random
from safetensors.torch import load_file

def get_timbre_files():
    base_path = "output/conditioning"
    if not os.path.exists(base_path):
        return ["none", "random"]
    files = [f for f in os.listdir(base_path) if f.endswith("_timbre.safetensors")]
    return sorted(files) + ["none", "random"]

class AceStepTimbreTensorLoader:
    """Load a timbre conditioning tensor from disk"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "timbre_tensor_file": (get_timbre_files(),),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("TENSOR", "STRING")
    RETURN_NAMES = ("timbre_tensor", "filename")
    FUNCTION = "load"
    CATEGORY = "Scromfy/Ace-Step/loaders"

    @classmethod
    def IS_CHANGED(s, timbre_tensor_file, seed):
        if timbre_tensor_file == "none":
            return "none"
            
        base_path = "output/conditioning"
        if timbre_tensor_file == "random":
            return f"random_{seed}"
            
        path = os.path.join(base_path, timbre_tensor_file)
        if os.path.exists(path):
            return f"{timbre_tensor_file}_{os.path.getmtime(path)}"
            
        return f"{timbre_tensor_file}_{seed}"

    def load(self, timbre_tensor_file, seed):
        base_path = "output/conditioning"
        rng = random.Random(seed)
        
        if timbre_tensor_file == "random":
            options = [f for f in os.listdir(base_path) if f.endswith("_timbre.safetensors")] if os.path.exists(base_path) else []
            if not options:
                raise FileNotFoundError("No timbre conditioning files found for random selection.")
            timbre_tensor_file = rng.choice(options)
            
        if timbre_tensor_file == "none":
            return (None, "none")
            
        path = os.path.join(base_path, timbre_tensor_file)
        tensor = load_file(path).get("timbre")
        
        return (tensor, timbre_tensor_file.replace("_timbre.safetensors", ""))

NODE_CLASS_MAPPINGS = {
    "AceStepTimbreTensorLoader": AceStepTimbreTensorLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepTimbreTensorLoader": "Load Timbre Tensor",
}
