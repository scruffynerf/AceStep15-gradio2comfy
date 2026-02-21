"""AceStepRandomPrompt node for ACE-Step"""
import random
from .includes.prompt_utils import GENRES, MOODS, INSTRUMENTS

class AceStepRandomPrompt:
    """Generate random music prompts from predefined templates"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "template": ([
                    "genre + mood",
                    "genre + instrument",
                    "mood + genre + instrument",
                    "full description"
                ], {"default": "genre + mood"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "generate"
    CATEGORY = "Scromfy/Ace-Step/prompts"

    def generate(self, seed, template):
        """Generate random music prompt"""
        random.seed(seed)
        
        genre = random.choice(GENRES)
        mood = random.choice(MOODS)
        instrument = random.choice(INSTRUMENTS)
        
        if template == "genre + mood":
            prompt = f"{mood} {genre}"
        elif template == "genre + instrument":
            prompt = f"{genre} with {instrument}"
        elif template == "mood + genre + instrument":
            prompt = f"{mood} {genre} featuring {instrument}"
        else:  # full description
            bpm = random.randint(80, 160)
            prompt = f"{mood} {genre} at {bpm} BPM with {instrument}"
        
        return (prompt,)


NODE_CLASS_MAPPINGS = {
    "AceStepRandomPrompt": AceStepRandomPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepRandomPrompt": "Random Prompt",
}
