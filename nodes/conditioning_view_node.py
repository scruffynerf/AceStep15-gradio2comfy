"""AceStepConditioningExplore node for ACE-Step"""
import json
import torch
import lovely_tensors as lt

class AceStepConditioningExplore:
    """Show conditioning content summarized with lovely-tensors"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_cond": ("CONDITIONING",),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_text",)
    FUNCTION = "extract"
    CATEGORY = "Scromfy/Ace-Step/advanced"

    def extract(self, text_cond):
        # Conditioning is a list of lists: [[cond, {"pooled_output": ...}]]
        # Convert to JSON string with indentation and lovely-tensors summaries
        serializable_data = self.to_json_serializable(text_cond)
        json_string = json.dumps(serializable_data, indent=4)
        return (json_string,)

    def to_json_serializable(self, obj):
        if isinstance(obj, torch.Tensor):
            return str(lt.lovely(obj))
        elif isinstance(obj, dict):
            return {k: self.to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.to_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self.to_json_serializable(item) for item in obj]
        else:
            return obj

NODE_CLASS_MAPPINGS = {
    "AceStepConditioningExplore": AceStepConditioningExplore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepConditioningExplore": "Conditioning to Json Text",
}
