from __future__ import annotations

import folder_paths
from .includes.audio_utils import scromfy_save_audio

class ScromfySaveAudioOpus:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "audio": ("AUDIO", ),
                            "filename_prefix": ("STRING", {"default": "audio/ACE-Step"}),
                            "quality": (["64k", "96k", "128k", "192k", "320k"], {"default": "128k"}),
                            },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Scromfy/Ace-Step/Audio"

    def save(self, audio, filename_prefix="audio/ACE-Step", quality="128k", prompt=None, extra_pnginfo=None):
        res = scromfy_save_audio(self, audio, filename_prefix, "opus", prompt, extra_pnginfo, quality)
        return res

NODE_CLASS_MAPPINGS = {
    "ScromfySaveAudioOpus": ScromfySaveAudioOpus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScromfySaveAudioOpus": "Scromfy Save Audio (Opus)",
}
