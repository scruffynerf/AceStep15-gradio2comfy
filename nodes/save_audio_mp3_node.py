from __future__ import annotations

import folder_paths
from .includes.audio_utils import scromfy_save_audio


class ScromfySaveAudioMP3:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "audio": ("AUDIO", ),
                            "filename_prefix": ("STRING", {"default": "audio/ACE-Step"}),
                            "quality": (["V0", "128k", "320k"], {"default": "V0"}),
                            },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Scromfy/Ace-Step/Audio"

    def save(self, audio, filename_prefix="audio/ACE-Step", quality="V0", prompt=None, extra_pnginfo=None):
        res = scromfy_save_audio(self, audio, filename_prefix, "mp3", prompt, extra_pnginfo, quality)
        return res

NODE_CLASS_MAPPINGS = {
    "ScromfySaveAudioMP3": ScromfySaveAudioMP3,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScromfySaveAudioMP3": "Scromfy Save Audio (MP3)",
}
