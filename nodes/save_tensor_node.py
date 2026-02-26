"""AceStepTensorSave node for ACE-Step"""
import os
from safetensors.torch import save_file

class AceStepTensorSave:
    """Save an individual conditioning tensor (timbre or lyrics) to disk"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": ("TENSOR",),
                "save_type": (["timbre", "lyric"], {"default": "timbre"}),
                "save_path": ("STRING", {"default": "output/conditioning"}),
                "filename_prefix": ("STRING", {"default": "mixed_tensor"}),
            }
        }
    
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"
    CATEGORY = "Scromfy/Ace-Step/advanced"

    def save(self, tensor, save_type, save_path, filename_prefix):
        os.makedirs(save_path, exist_ok=True)

        if save_type == "timbre":
            suffix = "_timbre.safetensors"
            key = "timbre"
        else:
            suffix = "_lyrics.safetensors"
            key = "lyrics"

        full_filename = f"{filename_prefix}{suffix}"
        full_path = os.path.join(save_path, full_filename)
        
        save_file({key: tensor}, full_path)
            
        return {}

NODE_CLASS_MAPPINGS = {
    "AceStepTensorSave": AceStepTensorSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepTensorSave": "Save Conditioning Tensor",
}
