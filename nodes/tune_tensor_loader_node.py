"""AceStepMainLoader node for ACE-Step"""
import os
import random
from safetensors.torch import load_file

def get_tune_files():
    base_path = "output/conditioning"
    if not os.path.exists(base_path):
        return ["none", "random"]
    files = [f for f in os.listdir(base_path) if f.endswith("_tune.safetensors")]
    return sorted(files) + ["none", "random"]

class AceStepTuneTensorLoader:
    """Load a tune conditioning tensor from disk"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tune_tensor_file": (get_tune_files(),),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("TENSOR", "STRING")
    RETURN_NAMES = ("tune_tensor", "filename")
    FUNCTION = "load"
    CATEGORY = "Scromfy/Ace-Step/loaders"

    @classmethod
    def IS_CHANGED(s, tune_tensor_file, seed):
        if tune_tensor_file == "none":
            return "none"
            
        base_path = "output/conditioning"
        if tune_tensor_file == "random":
            return f"random_{seed}"
            
        path = os.path.join(base_path, tune_tensor_file)
        if os.path.exists(path):
            return f"{tune_tensor_file}_{os.path.getmtime(path)}"
            
        return f"{tune_tensor_file}_{seed}"

    def load(self, tune_tensor_file, seed):
        base_path = "output/conditioning"
        rng = random.Random(seed)
        
        if tune_tensor_file == "random":
            options = [f for f in os.listdir(base_path) if f.endswith("_tune.safetensors")] if os.path.exists(base_path) else []
            if not options:
                raise FileNotFoundError("No tune conditioning files found for random selection.")
            tune_tensor_file = rng.choice(options)
            
        if tune_tensor_file == "none":
            return (None, "none")
            
        path = os.path.join(base_path, tune_tensor_file)
        tensor = load_file(path).get("tune")
        
        return (tensor, tune_tensor_file.replace("_tune.safetensors", ""))

NODE_CLASS_MAPPINGS = {
    "AceStepTuneTensorLoader": AceStepTuneTensorLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepTuneTensorLoader": "Load Tune Tensor",
}
