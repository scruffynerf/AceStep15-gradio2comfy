import numpy as np
import cv2
import torch
from .includes.visualizer_utils import FlexAudioVisualizerBase, BaseAudioProcessor, get_color_for_frequency

class ScromfyFlexAudioVisualizerLineNode(FlexAudioVisualizerBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_required = base_inputs.get("required", {})
        base_optional = base_inputs.get("optional", {})
        
        # Override feature_param with valid options
        base_required["feature_param"] = (cls.get_modifiable_params(), {"default": "None"})
        
        new_inputs = {
            "required": {
                "visualization_method": (["bar", "line"], {"default": "bar"}),
                "visualization_feature": (["frequency", "waveform"], {"default": "frequency"}),
                "smoothing": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "rotation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "length": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 4000.0, "step": 10.0}),
                "num_bars": ("INT", {"default": 64, "min": 1, "max": 1024, "step": 1}),
                "max_height": ("FLOAT", {"default": 200.0, "min": 10.0, "max": 2000.0, "step": 10.0}),
                "min_height": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 500.0, "step": 5.0}),
                "separation": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "curvature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 1.0}),
                "curve_smoothing": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fft_size": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 256}),
                "min_frequency": ("FLOAT", {"default": 20.0, "min": 20.0, "max": 20000.0, "step": 10.0}),
                "max_frequency": ("FLOAT", {"default": 8000.0, "min": 20.0, "max": 20000.0, "step": 10.0}),
                "reflect": ("BOOLEAN", {"default": False}),
            }
        }

        # Combine, putting new specific inputs first
        all_required = {**new_inputs["required"], **base_required}
        
        return {
            "required": all_required,
            "optional": base_optional
        }

    @classmethod
    def get_modifiable_params(cls):
        return ["smoothing", "rotation", "position_y", "position_x",
                "num_bars", "max_height", "min_height", "separation", "curvature", 
                "curve_smoothing", "fft_size", "min_frequency", "max_frequency", 
                "color_shift", "saturation", "brightness", "None"]

    def get_audio_data(self, processor: BaseAudioProcessor, frame_index, **kwargs):
        visualization_feature = kwargs.get('visualization_feature', 'frequency')
        smoothing = kwargs.get('smoothing', 0.5)
        num_bars = kwargs.get('num_bars', 64)
        fft_size = kwargs.get('fft_size', 2048)
        min_frequency = kwargs.get('min_frequency', 20.0)
        max_frequency = kwargs.get('max_frequency', 8000.0)

        _, feature_value, _ = self.process_audio_data(
            processor, frame_index, visualization_feature,
            num_bars, smoothing, fft_size, min_frequency, max_frequency
        )
        return feature_value

    def apply_effect_internal(self, processor: BaseAudioProcessor, **kwargs):
        visualization_method = kwargs.get('visualization_method', 'bar')
        screen_width = processor.width
        screen_height = processor.height
        rotation = kwargs.get('rotation', 0.0) % 360
        position_y = kwargs.get('position_y', 0.5)
        position_x = kwargs.get('position_x', 0.5)
        reflect = kwargs.get('reflect', False)
        num_bars = kwargs.get('num_bars', 64)
        length = kwargs.get('length', 0.0)
        max_height = kwargs.get('max_height', 200.0)
        min_height = kwargs.get('min_height', 10.0)
        
        color_mode = kwargs.get('color_mode', 'white')
        color_shift = kwargs.get('color_shift', 0.0)
        saturation = kwargs.get('saturation', 1.0)
        brightness = kwargs.get('brightness', 1.0)
        item_freqs = kwargs.get('item_freqs', None)

        if length == 0:
            rotation_rad = np.deg2rad(rotation)
            cos_theta = abs(np.cos(rotation_rad))
            sin_theta = abs(np.sin(rotation_rad))
            if cos_theta > sin_theta:
                visualization_length = screen_width / cos_theta
            else:
                visualization_length = screen_height / sin_theta
        else:
            visualization_length = length

        padding = int(max(visualization_length, max_height) * 0.5)
        padded_width = int(visualization_length + 2 * padding)
        padded_height = int(screen_height + 2 * padding)
        padded_image = np.zeros((padded_height, padded_width, 3), dtype=np.float32)

        data = processor.spectrum

        if visualization_method == 'bar':
            curvature = kwargs.get('curvature', 0.0)
            separation = kwargs.get('separation', 5.0)
            total_separation = separation * (num_bars - 1)
            total_bar_width = visualization_length - total_separation
            bar_width = total_bar_width / num_bars
            baseline_y = padded_height // 2
            x_offset = (padded_width - visualization_length) // 2

            for i, bar_value in enumerate(data):
                x = int(x_offset + i * (bar_width + separation))
                bar_h = min_height + (max_height - min_height) * bar_value
                if reflect:
                    y_start = int(baseline_y)
                    y_end = int(baseline_y + bar_h)
                else:
                    y_start = int(baseline_y - bar_h)
                    y_end = int(baseline_y)
                
                y_start = max(0, y_start)
                y_end = min(padded_height - 1, y_end)
                if y_start > y_end: y_start, y_end = y_end, y_start
                x_end = int(x + bar_width)
                
                # Determine color
                if color_mode == "spectrum" and item_freqs is not None:
                    color = get_color_for_frequency(item_freqs[i], color_shift, saturation, brightness)
                else:
                    color = (1.0, 1.0, 1.0)
                
                rect_width = max(1, int(bar_width))
                rect_height = max(1, int(y_end - y_start))

                if curvature > 0 and rect_width > 1 and rect_height > 1:
                    radius = max(1, min(int(curvature), rect_width // 2, rect_height // 2))
                    mask = np.full((rect_height, rect_width), 0, dtype=np.uint8)
                    cv2.rectangle(mask, (0, 0), (rect_width - 1, rect_height - 1), 255, -1)
                    if radius > 1:
                        mask = cv2.GaussianBlur(mask, (radius * 2 + 1, radius * 2 + 1), 0)
                    padded_image[y_start:y_start+rect_height, x:x+rect_width][mask > 0] = color
                else:
                    cv2.rectangle(padded_image, (x, y_start), (x_end, y_end), color, thickness=-1)

        elif visualization_method == 'line':
            curve_smoothing = kwargs.get('curve_smoothing', 0.0)
            baseline_y = padded_height // 2
            x_offset = (padded_width - visualization_length) // 2
            if curve_smoothing > 0:
                window_size = int(len(data) * curve_smoothing)
                if window_size % 2 == 0: window_size += 1
                data_smooth = self.smooth_curve(data, window_size) if window_size > 2 else data
            else:
                data_smooth = data
            amplitude = min_height + data_smooth * (max_height - min_height)
            num_pts = len(amplitude)
            x_values = np.linspace(x_offset, x_offset + visualization_length, num_pts)
            y_values = (baseline_y + amplitude) if reflect else (baseline_y - amplitude)
            points = np.array([x_values, y_values]).T.astype(np.int32)
            
            if len(points) > 1:
                if color_mode == "spectrum" and item_freqs is not None:
                    # For line mode, we draw segments with different colors
                    for i in range(len(points) - 1):
                        color = get_color_for_frequency(item_freqs[i], color_shift, saturation, brightness)
                        cv2.line(padded_image, tuple(points[i]), tuple(points[i+1]), color, 1)
                else:
                    cv2.polylines(padded_image, [points], False, (1.0, 1.0, 1.0))

        if rotation != 0:
            M = cv2.getRotationMatrix2D((padded_width // 2, padded_height // 2), rotation, 1.0)
            padded_image = cv2.warpAffine(padded_image, M, (padded_width, padded_height))

        target_x = int(screen_width * position_x)
        target_y = int(screen_height * position_y)
        start_x = padded_width // 2 - target_x
        start_y = padded_height // 2 - target_y
        
        # CRITICAL: Use .copy() to return a new array instead of a view of the huge padded_image.
        # This allows the padded_image to be garbage collected for each frame.
        image = padded_image[start_y:start_y + screen_height, start_x:start_x + screen_width].copy()
        return image

    def smooth_curve(self, y, window_size):
        if window_size < 3: return y
        box = np.ones(window_size) / window_size
        return np.convolve(y, box, mode='same')

NODE_CLASS_MAPPINGS = {
    "ScromfyFlexAudioVisualizerLine": ScromfyFlexAudioVisualizerLineNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScromfyFlexAudioVisualizerLine": "Flex Audio Visualizer Line (Scromfy)",
}
