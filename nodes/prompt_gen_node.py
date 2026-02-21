"""AceStepPromptGen node for ACE-Step"""
import random
from .includes.prompt_utils import STYLE_PRESETS

class AceStepPromptGen:
    """Generate music style prompts from 200+ presets"""
    
    @classmethod
    def INPUT_TYPES(cls):
        styles = sorted(list(STYLE_PRESETS.keys()))
        return {
            "required": {
                "style": (styles, {"default": "Synthwave"}),
                 "random_variation": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "generate"
    CATEGORY = "Scromfy/Ace-Step/prompts"

    def generate(self, style: str, random_variation: bool, seed: int):
        base_prompt = STYLE_PRESETS.get(style, "Generic electronic music")
        
        if random_variation:
            random.seed(seed)
            # Add slight variation to make each generation unique
            variations = [
                "energetic", "mellow", "atmospheric", "driving",
                "emotional", "uplifting", "dark", "bright"
            ]
            mood = random.choice(variations)
            prompt = f"{mood} {base_prompt}"
        else:
            prompt = base_prompt
        
        return (prompt,)


NODE_CLASS_MAPPINGS = {
    "AceStepPromptGen": AceStepPromptGen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepPromptGen": "Prompt Generator",
}
