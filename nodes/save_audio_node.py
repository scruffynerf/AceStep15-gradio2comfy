from __future__ import annotations

import folder_paths
from .includes.audio_utils import scromfy_save_audio

class ScromfySaveAudio:
    """Save audio in FLAC/WAV format with embedded generation metadata.
    
    Inputs:
        audio (AUDIO): The waveform to save.
        filename_prefix (STRING): Save path prefix.
        prompt / extra_pnginfo: Hidden inputs containing ComfyUI generation metadata.
        
    Outputs:
        filepath (STRING): Absolute path to the saved file.
    """
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "audio": ("AUDIO", ),
                            "filename_prefix": ("STRING", {"default": "audio/ACE-Step"}),
                            },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Scromfy/Ace-Step/Audio"

    def save(self, audio, filename_prefix="audio/ACE-Step", prompt=None, extra_pnginfo=None):
        res = scromfy_save_audio(self, audio, filename_prefix, "flac", prompt, extra_pnginfo)
        return res


NODE_CLASS_MAPPINGS = {
    "ScromfySaveAudio": ScromfySaveAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScromfySaveAudio": "Scromfy Save Audio (FLAC/WAV)",
}
