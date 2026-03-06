import numpy as np
import cv2
from .includes.visualizer_utils import FlexAudioVisualizerBase

class ScromfyFlexLyricsNode(FlexAudioVisualizerBase):
    @classmethod
    def INPUT_TYPES(cls):
        # We want to prune the visualizer-specific parameters to keep it clean
        base = super().INPUT_TYPES()
        required = base["required"]
        
        # Remove visualizer specific color/shape params
        cleaned_required = {
            "audio": required["audio"],
            "frame_rate": required["frame_rate"],
            "screen_width": required["screen_width"],
            "screen_height": required["screen_height"],
            "strength": required["strength"],
            "feature_threshold": required["feature_threshold"],
            "feature_param": (["None"],),
            "feature_mode": required["feature_mode"],
        }
        
        return {
            "required": cleaned_required,
            "optional": base["optional"]
        }

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
    "ScromfyFlexLyricsNode": "Flex Lyrics (Overlay)"
}
