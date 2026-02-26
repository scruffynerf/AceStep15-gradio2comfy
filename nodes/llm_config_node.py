"""AceStep5HzLMConfig node for ACE-Step"""

class AceStep5HzLMConfig:
    """Configures Language Model parameters for ACE-Step generation"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "temperature": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 2.0, "step": 0.01}),
                "cfg_scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 5.0, "step": 0.1}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 100}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "negative_prompt": ("STRING", {"default": "NO USER INPUT", "multiline": True}),
            }
        }
    
    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("lm_config",)
    FUNCTION = "build_config"
    CATEGORY = "Scromfy/Ace-Step/advanced"

    def build_config(self, temperature, cfg_scale, top_k, top_p, negative_prompt):
        return ({
            "temperature": temperature,
            "cfg_scale": cfg_scale,
            "top_k": top_k if top_k > 0 else None,
            "top_p": top_p,
            "negative_prompt": negative_prompt,
        },)


NODE_CLASS_MAPPINGS = {
    "AceStep5HzLMConfig": AceStep5HzLMConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStep5HzLMConfig": "5Hz LLM Config",
}
