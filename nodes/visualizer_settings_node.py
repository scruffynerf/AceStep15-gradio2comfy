import numpy as np

class ScromfyFlexVisualizerSettingsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "randomize": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "visualization_method": (["bar", "line"], {"default": "bar"}),
                "visualization_feature": (["frequency", "waveform"], {"default": "frequency"}),
                "color_mode": (["white", "spectrum", "custom", "amplitude", "radial", "angular", "path", "screen"], {"default": "spectrum"}),
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
