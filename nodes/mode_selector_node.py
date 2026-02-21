"""AceStepModeSelector node for ACE-Step"""

class AceStepModeSelector:
    """Convenience node to route inputs based on generation mode"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": (["Simple", "Custom", "Cover", "Repaint"],),
                "description": ("STRING", {"multiline": True, "default": ""}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "lyrics": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "reference_audio": ("AUDIO",),
                "source_audio": ("AUDIO",),
                "repaint_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.1}),
                "repaint_end": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 1000.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "AUDIO", "FLOAT", "FLOAT")
    RETURN_NAMES = ("final_prompt", "final_lyrics", "active_audio", "start", "end")
    FUNCTION = "route"
    CATEGORY = "Scromfy/Ace-Step/advanced"

    def route(self, mode, description, prompt, lyrics, reference_audio=None, source_audio=None, repaint_start=0.0, repaint_end=-1.0):
        final_prompt = ""
        final_lyrics = lyrics
        active_audio = None
        start = 0.0
        end = -1.0
        
        if mode == "Simple":
            final_prompt = description
        elif mode == "Custom":
            final_prompt = prompt
            active_audio = reference_audio
        elif mode == "Cover":
            final_prompt = prompt
            active_audio = source_audio
        elif mode == "Repaint":
            final_prompt = prompt
            active_audio = source_audio
            start = repaint_start
            end = repaint_end
            
        return (final_prompt, final_lyrics, active_audio, start, end)


NODE_CLASS_MAPPINGS = {
    "AceStepModeSelector": AceStepModeSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepModeSelector": "Mode Selector",
}
