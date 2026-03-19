from __future__ import annotations

import os
import torch
import folder_paths
import json
from typing import Optional, Literal
from comfy.comfy_types import IO, FileLocator, ComfyNodeABC
from comfy_api.latest import Input, InputImpl, Types
from comfy.cli_args import args

class ScromfySaveVideo(ComfyNodeABC):
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type: Literal["output"] = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (IO.VIDEO, {"tooltip": "The video to save."}),
                "filename_prefix": ("STRING", {"default": "video/ACE-Step", "tooltip": "The prefix for the file to save."}),
                "format": (Types.VideoContainer.as_input(), {"default": "auto", "tooltip": "The format to save the video as."}),
                "codec": (Types.VideoCodec.as_input(), {"default": "auto", "tooltip": "The codec to use for the video."}),
                "remove_extension": ("BOOLEAN", {"default": False, "tooltip": "If True, the output filepath will not include the file extension."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save_video"

    OUTPUT_NODE = True

    CATEGORY = "Scromfy/Ace-Step/Video"
    DESCRIPTION = "Saves the input video to your ComfyUI output directory and returns the absolute path."

    def save_video(self, video: Input.Video, filename_prefix, format, codec, remove_extension=False, prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        width, height = video.get_dimensions()
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix,
            self.output_dir,
            width,
            height
        )
        
        saved_metadata = None
        if not args.disable_metadata:
            metadata = {}
            if extra_pnginfo is not None:
                metadata.update(extra_pnginfo)
            if prompt is not None:
                metadata["prompt"] = prompt
            if len(metadata) > 0:
                saved_metadata = metadata
        
        ext = Types.VideoContainer.get_extension(format)
        file = f"{filename}_{counter:05}_.{ext}"
        output_path = os.path.join(full_output_folder, file)
        
        video.save_to(
            output_path,
            format=format,
            codec=codec,
            metadata=saved_metadata
        )

        results = [{
            "filename": file,
            "subfolder": subfolder,
            "type": self.type
        }]

        # Determine the absolute path to return
        absolute_path = os.path.abspath(output_path)
        if remove_extension:
            return_path = os.path.splitext(absolute_path)[0]
        else:
            return_path = absolute_path

        return { "ui": { "images": results, "animated": (True,) }, "result": (return_path,) }

NODE_CLASS_MAPPINGS = {
    "ScromfySaveVideo": ScromfySaveVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScromfySaveVideo": "Scromfy Save Video",
}
