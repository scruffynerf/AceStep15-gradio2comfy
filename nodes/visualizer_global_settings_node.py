import numpy as np
import os

class ScromfyFlexVisualizerSettingsNode:
    """Global configuration settings for the Flex Visualizer system.
    
    This node defines the audio analysis parameters (FFT vs waveform), color assignment logic,
    global motion properties, and how the visualizer responds to audio frequencies. It outputs
    a VISUALIZER_SETTINGS dictionary that drives the rendering of downstream visualizer nodes
    (Circular, Line, Contour).
    
    Inputs:
        randomize (BOOLEAN): Auto-vary parameters between batches.
        seed (INT): RNG seed for randomization.
        loop_background (BOOLEAN): Loop the background video (True) or clamp (False).
        use_mask_as_visibility_filter (BOOLEAN): If True, restrict rendering strictly to the alpha of the reference mask.
        visualization_feature (STRING): 'frequency' (FFT) or 'waveform' (raw amplitude).
        num_points (INT): Resolution of the generated geometry.
        smoothing (FLOAT): Temporal smoothing applied to the audio output.
        fft_size (INT): Window size for frequency analysis.
        min_frequency (FLOAT): Lower bound for bass response.
        max_frequency (FLOAT): Upper bound for treble response.
        visualization_method (STRING): Rendering style ('bar' or 'line').
        color_mode (STRING): Determines how colors are assigned (e.g., 'spectrum', 'radial', 'schema').
        color_schema (STRING): Base palette selection from local color_schemas folder.
        custom_color (COLOR): Hex fallback color.
        color_shift (FLOAT): Global hue offset sequence cycle.
        saturation (FLOAT): Global intensity multiplier.
        brightness (FLOAT): Global brightness multiplier.
        line_width (INT): Thickness of rendered strokes in pixels.
        direction (STRING): Vector flow (e.g., 'outward', 'inward', 'centroid').
        sequence_direction (STRING): Data distribution around the geometry.
        direction_skew (FLOAT): Global angular rotation offset for strokes (degrees).
        centroid_offset_x (FLOAT): Focal offset along the X-axis for 'centroid' direction mode (percent).
        centroid_offset_y (FLOAT): Focal offset along the Y-axis for 'centroid' direction mode (percent).
        
    Outputs:
        settings (VISUALIZER_SETTINGS): A dictionary encapsulating the global configuration state.
    """
    
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
                # --- System & Setup ---
                "randomize": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "loop_background": ("BOOLEAN", {"default": True}),
                "use_mask_as_visibility_filter": ("BOOLEAN", {"default": False}),
                
                # --- Audio Analysis ---
                "visualization_feature": (["frequency", "waveform"], {"default": "frequency"}),
                "num_points": ("INT", {"default": 64, "min": 4, "max": 1024, "step": 1}),
                "smoothing": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fft_size": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 256}),
                "min_frequency": ("FLOAT", {"default": 20.0, "min": 20.0, "max": 20000.0, "step": 10.0}),
                "max_frequency": ("FLOAT", {"default": 8000.0, "min": 20.0, "max": 20000.0, "step": 10.0}),

                # --- Color & Style ---
                "visualization_method": (["bar", "line"], {"default": "bar"}),
                "color_mode": (["white", "spectrum", "custom", "schema", "amplitude", "radial", "angular", "path", "screen"], {"default": "spectrum"}),
                "color_schema": (cls._get_schema_names(), {"default": "none"}),
                "custom_color": ("COLOR", {"default": "#00ffff"}),
                "color_shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "line_width": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),

                # --- Motion & Direction ---
                "direction": (["outward", "inward", "both", "centroid", "starburst"], {"default": "outward"}),
                "sequence_direction": (["left", "right", "centered", "both ends"], {"default": "right"}),
                "direction_skew": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.5}),
                "centroid_offset_x": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "centroid_offset_y": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("VISUALIZER_SETTINGS",)
    RETURN_NAMES = ("settings",)
    FUNCTION = "get_settings"
    CATEGORY = "Scromfy/Ace-Step/Visualizers"

    def get_settings(self, **kwargs):
        return (kwargs,)

NODE_CLASS_MAPPINGS = {
    "ScromfyFlexVisualizerSettings": ScromfyFlexVisualizerSettingsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScromfyFlexVisualizerSettings": "Visualizer Settings (Scromfy)",
}
