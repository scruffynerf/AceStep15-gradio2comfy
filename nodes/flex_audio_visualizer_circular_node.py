import numpy as np
import cv2
import torch
from .includes.visualizer_utils import FlexAudioVisualizerBase, BaseAudioProcessor, get_color_for_frequency, parse_color

class ScromfyFlexAudioVisualizerCircularNode(FlexAudioVisualizerBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_required = base_inputs.get("required", {})
        base_optional = base_inputs.get("optional", {})
        
        base_required["feature_param"] = (cls.get_modifiable_params(), {"default": "None"})
        
        new_inputs = {
            "required": {
                "visualization_method": (["bar", "line"], {"default": "bar"}),
                "visualization_feature": (["frequency", "waveform"], {"default": "frequency"}),
                "smoothing": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "rotation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "num_points": ("INT", {"default": 360, "min": 3, "max": 1000, "step": 1}),
                "fft_size": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 256}),
                "min_frequency": ("FLOAT", {"default": 20.0, "min": 20.0, "max": 20000.0, "step": 10.0}),
                "max_frequency": ("FLOAT", {"default": 8000.0, "min": 20.0, "max": 20000.0, "step": 10.0}),
                "radius": ("FLOAT", {"default": 200.0, "min": 10.0, "max": 1000.0, "step": 10.0}),
                "line_width": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "amplitude_scale": ("FLOAT", {"default": 100.0, "min": 1.0, "max": 1000.0, "step": 10.0}),
                "base_radius": ("FLOAT", {"default": 200.0, "min": 10.0, "max": 1000.0, "step": 10.0}),
            }
        }

        all_required = {**base_required, **new_inputs["required"]}
        
        return {
            "required": all_required,
            "optional": base_optional
        }

    @classmethod
    def get_modifiable_params(cls):
        return ["smoothing", "rotation", "num_points", "fft_size", 
                "min_frequency", "max_frequency", "radius", "line_width",
                "amplitude_scale", "base_radius", "position_x", "position_y", 
                "color_shift", "saturation", "brightness", "None"]

    def get_audio_data(self, processor: BaseAudioProcessor, frame_index, **kwargs):
        visualization_feature = kwargs.get('visualization_feature', 'frequency')
        smoothing = kwargs.get('smoothing', 0.5)
        num_points = kwargs.get('num_points', 360)
        fft_size = kwargs.get('fft_size', 2048)
        min_frequency = kwargs.get('min_frequency', 20.0)
        max_frequency = kwargs.get('max_frequency', 8000.0)

        _, feature_value, _ = self.process_audio_data(
            processor, frame_index, visualization_feature,
            num_points, smoothing, fft_size, min_frequency, max_frequency
        )
        return feature_value

    def apply_effect_internal(self, processor: BaseAudioProcessor, **kwargs):
        visualization_method = kwargs.get('visualization_method', 'bar')
        rotation = kwargs.get('rotation', 0.0) % 360
        num_points = kwargs.get('num_points', 360)
        screen_width = processor.width
        screen_height = processor.height
        position_x = kwargs.get('position_x', 0.5)
        position_y = kwargs.get('position_y', 0.5)
        base_radius = kwargs.get('base_radius', 200.0)
        amplitude_scale = kwargs.get('amplitude_scale', 100.0)
        line_width = kwargs.get('line_width', 2)
        
        color_mode = kwargs.get('color_mode', 'white')
        color_shift = kwargs.get('color_shift', 0.0)
        saturation = kwargs.get('saturation', 1.0)
        brightness = kwargs.get('brightness', 1.0)
        item_freqs = kwargs.get('item_freqs', None)
        
        image = np.zeros((screen_height, screen_width, 3), dtype=np.float32)
        center_x = screen_width * position_x
        center_y = screen_height * position_y

        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        rotation_rad = np.deg2rad(rotation)
        angles += rotation_rad

        data = processor.spectrum

        if visualization_method == 'bar':
            for i, (angle, amplitude) in enumerate(zip(angles, data)):
                x_start = center_x + base_radius * np.cos(angle)
                y_start = center_y + base_radius * np.sin(angle)
                x_end = center_x + (base_radius + amplitude * amplitude_scale) * np.cos(angle)
                y_end = center_y + (base_radius + amplitude * amplitude_scale) * np.sin(angle)
                
                # Determine color
                if color_mode == "spectrum" and item_freqs is not None:
                    color = get_color_for_frequency(item_freqs[i], color_shift, saturation, brightness)
                elif color_mode == "custom":
                    color = parse_color(kwargs.get("custom_color", "#00ffff"))
                else:
                    color = (1.0, 1.0, 1.0)
                
                cv2.line(image, (int(x_start), int(y_start)), (int(x_end), int(y_end)),
                         color, thickness=line_width)
        elif visualization_method == 'line':
            radii = base_radius + data * amplitude_scale
            x_values = center_x + radii * np.cos(angles)
            y_values = center_y + radii * np.sin(angles)
            points = np.array([x_values, y_values]).T.astype(np.int32)
            
            if len(points) > 2:
                if color_mode == "spectrum" and item_freqs is not None:
                    # Draw segments with colors
                    num_pts = len(points)
                    for i in range(num_pts):
                        p1 = points[i]
                        p2 = points[(i+1) % num_pts]
                        color = get_color_for_frequency(item_freqs[i], color_shift, saturation, brightness)
                        cv2.line(image, tuple(p1), tuple(p2), color, line_width)
                elif color_mode == "custom":
                    color = parse_color(kwargs.get("custom_color", "#00ffff"))
                    cv2.polylines(image, [points], isClosed=True, color=color, thickness=line_width)
                else:
                    cv2.polylines(image, [points], isClosed=True, color=(1.0, 1.0, 1.0), thickness=line_width)

        return image.copy()

NODE_CLASS_MAPPINGS = {
    "ScromfyFlexAudioVisualizerCircular": ScromfyFlexAudioVisualizerCircularNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScromfyFlexAudioVisualizerCircular": "Flex Audio Visualizer Circular (Scromfy)",
}
