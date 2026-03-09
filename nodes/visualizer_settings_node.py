import numpy as np
import os

class ScromfyFlexVisualizerSettingsNode:
    @classmethod
    def _get_schema_names(cls):
        schemas_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "color_schemas")
        names = ["none"]
        if os.path.isdir(schemas_dir):
            names += sorted(
                os.path.splitext(f)[0]
                for f in os.listdir(schemas_dir)
                if f.lower().endswith(".json")
            )
        return names

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "randomize": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "visualization_method": (["bar", "line"], {"default": "bar"}),
                "visualization_feature": (["frequency", "waveform"], {"default": "frequency"}),
                "color_mode": (["white", "spectrum", "custom", "amplitude", "radial", "angular", "path", "screen"], {"default": "spectrum"}),
                "color_schema": (cls._get_schema_names(), {"default": "none"}),
                "custom_color": ("COLOR", {"default": "#00ffff"}),
                "smoothing": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fft_size": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 256}),
                "min_frequency": ("FLOAT", {"default": 20.0, "min": 20.0, "max": 20000.0, "step": 10.0}),
                "max_frequency": ("FLOAT", {"default": 8000.0, "min": 20.0, "max": 20000.0, "step": 10.0}),
                "color_shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "rotation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "line_width": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "direction": (["outward", "inward", "both", "centroid", "starburst"], {"default": "outward"}),
                "sequence_direction": (["left", "right", "centered", "both ends"], {"default": "right"}),
                "centroid_offset_x": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "centroid_offset_y": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "direction_skew": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5,
                                             "tooltip": "Rotate direction vectors by this many degrees. Positive=clockwise, negative=counter-clockwise. Works on all direction modes."}),
            }
        }

    RETURN_TYPES = ("VISUALIZER_SETTINGS",)
    RETURN_NAMES = ("settings",)
    FUNCTION = "get_settings"
    CATEGORY = "Scromfy/Ace-Step/Settings"

    def get_settings(self, **kwargs):
        return (kwargs,)

NODE_CLASS_MAPPINGS = {
    "ScromfyFlexVisualizerSettings": ScromfyFlexVisualizerSettingsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScromfyFlexVisualizerSettings": "Flex Visualizer Settings (Scromfy)",
}
