"""AceStepPromptFreeform node for ACE-Step – allows freeform text with __WILDCARD__ resolution"""
import random
from .includes.prompt_utils import expand_wildcards

class AceStepPromptFreeform:
    """Resolve wildcards in freeform text using prompt components"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True, "placeholder": "Enter text with __WILDCARDS__"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "resolve"
    CATEGORY = "Scromfy/Ace-Step/Prompt"

    def resolve(self, text, seed):
        rng = random.Random(seed)
        resolved = expand_wildcards(text, rng)
        return (resolved,)


NODE_CLASS_MAPPINGS = {
    "AceStepPromptFreeform": AceStepPromptFreeform,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepPromptFreeform": "Prompt Freeform",
}
