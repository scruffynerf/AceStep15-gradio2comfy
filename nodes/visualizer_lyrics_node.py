import numpy as np
import cv2
from .includes.visualizer_utils import FlexAudioVisualizerBase

class ScromfyFlexLyricsNode(FlexAudioVisualizerBase):
    """A passthrough node that overlays rendered lyrics onto a background video or image.
    
    This node serves as the final compositing step for the Flex Lyrics system. It accepts
    structured lyric settings and timing data, and passes them to the BaseAudioProcessor
    which handles the actual OpenCV text rendering.
    
    Inputs:
        audio (AUDIO): Base waveform for synchronization and analysis.
        frame_rate (FLOAT): Target FPS for the output sequence.
        screen_width (INT): Output pixel width.
        screen_height (INT): Output pixel height.
        strength (FLOAT): Opacity scaling.
        feature_param (STRING): Dynamic parameter mapped to the feature_mode logic.
    
    Optional Inputs:
        opt_video (IMAGE): The underlying background sequence to overlay text onto.
        lyric_settings (LYRIC_SETTINGS): Configuration dictionary defining font, color, and positioning.
        
    Outputs:
        IMAGE: The fully composed lyrics sequence frames.
        MASK: An alpha channel sequence (usually empty for pure lyrics text).
        SETTINGS (STRING): A JSON-safe debug record of active parameters.
        SOURCE_MASK: Not used.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Keep audio, video, frame_rate, dimensions, and ALL lyric settings
        base = super().INPUT_TYPES()
        required = base["required"]
        
        # Keep audio, video, frame_rate, dimensions, and audio sync logic
        # explicitly remove any leaked visualizer globals from base
        exclude = ["color_mode", "randomize", "seed", "visualization_method",
                   "visualization_feature", "smoothing", "fft_size",
                   "min_frequency", "max_frequency", "line_width",
                   "direction", "sequence_direction", "direction_skew",
                   "centroid_offset_x", "centroid_offset_y", "num_points",
                   "color_shift", "saturation", "brightness", "custom_color",
                   "position_x", "position_y", "rotation"]
        
        cleaned_required = {
            k: v for k, v in required.items() if k not in exclude
        }
        
        # Add modifiable feature param
        cleaned_required["feature_param"] = (cls.get_modifiable_params(), {"default": "None"})
        
        # Use simplified base inputs from optional
        optional = base["optional"]
        cleaned_optional = {
            "opt_video": optional.get("opt_video"),
            "lyric_settings": optional.get("lyric_settings"),
        }
        
        return {
            "required": cleaned_required,
            "optional": cleaned_optional
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK", "SETTINGS", "SOURCE_MASK")
    FUNCTION = "apply_effect"
    CATEGORY = "Scromfy/Ace-Step/Visualizers"

    @classmethod
    def get_modifiable_params(cls):
        return []

    def apply_effect_internal(self, processor, frame_index, screen_width, screen_height, 
                               background=None, **kwargs):
        # This node does NO visualizer drawing, just returns the background
        # The base apply_effect will then render lyrics on top of this.
        if background is not None:
            if background.shape[0] != screen_height or background.shape[1] != screen_width:
                return cv2.resize(background, (screen_width, screen_height))
            return background.copy()
        else:
            return np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

NODE_CLASS_MAPPINGS = {
    "ScromfyFlexLyricsNode": ScromfyFlexLyricsNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScromfyFlexLyricsNode": "Lyrics Overlay (Scromfy)"
}
