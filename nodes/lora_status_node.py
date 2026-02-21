"""AceStepLoRAStatus node for ACE-Step"""

class AceStepLoRAStatus:
    """Display information about loaded LoRA"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "lora_name": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "check_status"
    CATEGORY = "Scromfy/Ace-Step/advanced"

    def check_status(self, lora_name=""):
        if not lora_name:
            return ("No LoRA loaded",)
        return (f"LoRA Active: {lora_name}",)


NODE_CLASS_MAPPINGS = {
    "AceStepLoRAStatus": AceStepLoRAStatus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepLoRAStatus": "LoRA Status",
}
