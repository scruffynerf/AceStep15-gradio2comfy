import numpy as np
import cv2
import torch
from .includes.visualizer_utils import FlexAudioVisualizerBase, BaseAudioProcessor, get_color_for_frequency, parse_color

class ScromfyFlexAudioVisualizerLineNode(FlexAudioVisualizerBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_required = base_inputs.get("required", {})
        base_optional = base_inputs.get("optional", {})
        
        # Override feature_param with valid options
        base_required["feature_param"] = (cls.get_modifiable_params(), {"default": "None"})

        # Remove parameters handled by base/settings
        for param in ["position_x", "position_y",
                      "color_mode", "randomize", "seed", "visualization_method",
                      "visualization_feature", "smoothing", "line_width", "rotation",
                      "fft_size", "min_frequency", "max_frequency"]:
            if param in base_required:
                del base_required[param]
        
        new_inputs = {
            "required": {
                "num_bars": ("INT", {"default": 64, "min": 1, "max": 1024, "step": 1}),
                "max_height": ("FLOAT", {"default": 200.0, "min": 10.0, "max": 2000.0, "step": 10.0}),
                "min_height": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 500.0, "step": 5.0}),
                "bar_length_mode": (["absolute", "relative"], {"default": "absolute"}),
                "length": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 4000.0, "step": 10.0}),
                "separation": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "curvature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 50.0, "step": 1.0}),
                "curve_smoothing": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

        # Combine, putting new specific inputs first
        all_required = {**base_required, **new_inputs["required"]}
        all_optional = {**base_optional, "mask": ("MASK",)}
        
        return {
            "required": all_required,
            "optional": all_optional
        }

    @classmethod
    def get_modifiable_params(cls):
        return ["smoothing", "rotation", "position_y", "position_x",
                "num_bars", "max_height", "min_height", "separation", "curvature", 
                "curve_smoothing", "fft_size", "min_frequency", "max_frequency", 
                "color_shift", "saturation", "brightness", "bar_length_mode", "None"]

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("IMAGE", "MASK", "SETTINGS")

    def apply_effect(self, audio, frame_rate, screen_width, screen_height, strength, feature_param,
                     feature_mode, feature_threshold, mask=None, opt_feature=None, **kwargs):
        
        seed = kwargs.get("seed", 0)
        import random
        s_rng = random.Random(seed)

        if kwargs.get("randomize", False):
            kwargs["bar_length_mode"] = "relative"
            # max_height is the "bar" height equivalent here
            if kwargs.get("visualization_feature", "frequency") == "waveform":
                kwargs["max_height"] = s_rng.uniform(10.0, 30.0)
            else:
                kwargs["max_height"] = s_rng.uniform(30.0, 70.0)
            
            kwargs["num_bars"] = s_rng.choice([16, 32, 64, 128])

        # Get screen dimensions
        screen_width = kwargs.get("screen_width", 512)
        screen_height = kwargs.get("screen_height", 512)
        
        # images, masks, settings, _ (source_mask ignored)
        images, masks, settings, _ = super().apply_effect(
            audio, frame_rate, screen_width, screen_height,
            strength, feature_param, feature_mode, feature_threshold,
            opt_feature, source_mask=mask, **kwargs
        )
        
        return (images, masks, settings)

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
        num_bars = self.get_point_count(kwargs)
        length = kwargs.get('length', 0.0)
        direction = kwargs.get('direction', 'outward')
        sequence_direction = kwargs.get('sequence_direction', 'right')
        
        max_height = kwargs.get('max_height', 200.0)
        min_height = kwargs.get('min_height', 10.0)
        bar_length_mode = kwargs.get('bar_length_mode', 'absolute')
        
        if bar_length_mode == "relative":
            # Treat max_height as a percentage of screen_height
            effective_max_height = (max_height / 100.0) * screen_height
            effective_min_height = (min_height / 100.0) * screen_height
        else:
            effective_max_height = max_height
            effective_min_height = min_height

        color_mode = kwargs.get('color_mode', 'white')
        color_shift = kwargs.get('color_shift', 0.0)
        saturation = kwargs.get('saturation', 1.0)
        brightness = kwargs.get('brightness', 1.0)
        item_freqs = kwargs.get('item_freqs', None)
        line_width = kwargs.get('line_width', 2)

        baseline_y = screen_height * position_y
        
        # Determine effective length
        if length == 0.0:
            rotation_rad = np.deg2rad(rotation)
            cos_theta = abs(np.cos(rotation_rad))
            sin_theta = abs(np.sin(rotation_rad))
            if cos_theta > sin_theta:
                visualization_length = screen_width / cos_theta
            else:
                visualization_length = screen_height / sin_theta
        else:
            visualization_length = length

        padding = int(max(visualization_length, effective_max_height, screen_width, screen_height) * 0.5)
        padded_width = int(max(visualization_length, screen_width) + 2 * padding)
        padded_height = int(max(screen_height, effective_max_height) + 2 * padding)
        padded_image = np.zeros((padded_height, padded_width, 3), dtype=np.float32)

        # Baseline point calculation (unpadded)
        center_x = screen_width * position_x
        center_y = screen_height * position_y

        # For geometric color/direction logic, find the center of the mask if provided
        mask = kwargs.get("source_mask")
        if mask is not None:
            # mask is likely a torch tensor [batch, height, width]
            frame_idx = min(kwargs.get("frame_index", 0), mask.shape[0] - 1)
            mask_np = (mask[frame_idx].cpu().numpy() * 255).astype(np.uint8)
            M = cv2.moments(mask_np)
            if M["m00"] > 0:
                cx_rel, cy_rel = M["m10"] / M["m00"], M["m01"] / M["m00"]
            else:
                cx_rel, cy_rel = center_x, center_y
        else:
            cx_rel, cy_rel = center_x, center_y

        # Apply user-specified CoM offset
        cx_rel += kwargs.get("centroid_offset_x", 0.0) * screen_width
        cy_rel += kwargs.get("centroid_offset_y", 0.0) * screen_height

        # Final cx/cy in the PADDED coordinate space for drawing
        cx = cx_rel + (padded_width - screen_width) // 2
        cy = cy_rel + (padded_height - screen_height) // 2

        data = self.transform_sequence(processor.spectrum, sequence_direction)
        if item_freqs is not None:
            item_freqs = self.transform_sequence(item_freqs, sequence_direction)

        if visualization_method == 'bar':
            curvature = kwargs.get('curvature', 0.0)
            separation = kwargs.get('separation', 5.0)
            total_separation = separation * (num_bars - 1)
            total_bar_width = visualization_length - total_separation
            bar_width = total_bar_width / num_bars
            baseline_y_padded = baseline_y + padding
            x_offset = (padded_width - visualization_length) // 2

            direction_skew = kwargs.get("direction_skew", 0.0)

            for i, bar_value in enumerate(data):
                x_base = x_offset + i * (bar_width + separation) + bar_width / 2
                y_base = baseline_y_padded
                
                bar_h = effective_min_height + (effective_max_height - effective_min_height) * bar_value
                
                # Determine direction vector (vx, vy)
                if direction in ('centroid', 'starburst'):
                    dx_com, dy_com = cx - x_base, cy - y_base
                    dist_com = np.sqrt(dx_com**2 + dy_com**2)
                    if dist_com > 0:
                        vx, vy = dx_com / dist_com, dy_com / dist_com
                    else:
                        vx, vy = 0, -1
                    if direction == 'starburst':
                        vx, vy = -vx, -vy
                else:
                    # upward/downward
                    vx, vy = 0, -1
                    if direction == 'inward':
                        vx, vy = 0, 1

                # Apply skew
                if direction_skew != 0:
                    skew_rad = np.deg2rad(direction_skew)
                    s_cos, s_sin = np.cos(skew_rad), np.sin(skew_rad)
                    vx, vy = vx * s_cos - vy * s_sin, vx * s_sin + vy * s_cos

                # Draw point calculation
                if direction == 'both':
                    half = bar_h / 2.0
                    # For 'both', we use the vertical radial-like expansion on the baseline
                    x1, y1 = int(x_base), int(y_base - half)
                    x2, y2 = int(x_base), int(y_base + half)
                    # We override vx, vy for color mapping below if needed, but for 'both' 
                    # we stick to standard rectangle logic if no skew
                else:
                    x1, y1 = int(x_base), int(y_base)
                    x2, y2 = int(x_base + vx * bar_h), int(y_base + vy * bar_h)

                # Determine color using shared helper
                # Pass cx, cy (potentially offset) to get_draw_color
                color = self.get_draw_color(i, num_bars, bar_value,
                                            x1, y1, cx, cy, 
                                            max(screen_width, screen_height), **kwargs)
                
                # Draw logic: if we have skew or centroid, we MUST use cv2.line.
                # If we have curvature and it's vertical, we use the original mask logic.
                is_vertical = (vx == 0 and direction_skew == 0 and direction != 'both') or (direction == 'both' and direction_skew == 0)
                
                if not is_vertical:
                    thickness = max(1, int(bar_width))
                    cv2.line(padded_image, (x1, y1), (x2, y2), color, thickness)
                else:
                    # Original rectangle/curvature logic
                    rect_x_start = int(x_base - bar_width / 2)
                    rect_x_end = int(rect_x_start + bar_width)
                    rect_y_start, rect_y_end = y1, y2
                    if rect_y_start > rect_y_end: rect_y_start, rect_y_end = rect_y_end, rect_y_start
                    
                    rect_w = max(1, int(rect_x_end - rect_x_start))
                    rect_h = max(1, int(rect_y_end - rect_y_start))
                    
                    if curvature > 0 and rect_w > 1 and rect_h > 1:
                        radius = max(1, min(int(curvature), rect_w // 2, rect_h // 2))
                        mask = np.zeros((rect_h, rect_w), dtype=np.uint8)
                        r = radius
                        cv2.circle(mask, (r, r), r, 255, -1)
                        cv2.circle(mask, (rect_w - r, r), r, 255, -1)
                        cv2.circle(mask, (r, rect_h - r), r, 255, -1)
                        cv2.circle(mask, (rect_w - r, rect_h - r), r, 255, -1)
                        cv2.rectangle(mask, (r, 0), (rect_w - r, rect_h), 255, -1)
                        cv2.rectangle(mask, (0, r), (rect_w, rect_h - r), 255, -1)
                        padded_image[rect_y_start:rect_y_start+rect_h, rect_x_start:rect_x_start+rect_w][mask > 0] = color
                    else:
                        cv2.rectangle(padded_image, (rect_x_start, rect_y_start), (rect_x_end, rect_y_end), color, thickness=-1)

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
            y_values = (baseline_y + amplitude) if direction == 'inward' else (baseline_y - amplitude)
            points = np.array([x_values, y_values]).T.astype(np.int32)
            
            if len(points) > 1:
                # Draw segments to support multi-color modes
                for i in range(len(points) - 1):
                    p1 = points[i]
                    p2 = points[i+1]
                    color = self.get_draw_color(i, num_pts, data[i],
                                                p1[0], p1[1], cx, cy,
                                                max(visualization_length, effective_max_height), **kwargs)
                    cv2.line(padded_image, tuple(p1), tuple(p2), color, line_width)

        if rotation != 0:
            M = cv2.getRotationMatrix2D((padded_width // 2, padded_height // 2), rotation, 1.0)
            padded_image = cv2.warpAffine(padded_image, M, (padded_width, padded_height))

        target_x = int(screen_width * position_x)
        target_y = int(screen_height * position_y)
        start_x = max(0, padded_width // 2 - target_x)
        start_y = max(0, padded_height // 2 - target_y)
        
        # CRITICAL: Use .copy() to return a new array instead of a view of the huge padded_image.
        # This allows the padded_image to be garbage collected for each frame.
        # We also ensure the slice stays within bounds of the padded_image.
        end_y = min(start_y + screen_height, padded_height)
        end_x = min(start_x + screen_width, padded_width)
        image = padded_image[start_y:end_y, start_x:end_x].copy()
        
        # If the slice was smaller than expected, pad it back to the target size
        if image.shape[0] != screen_height or image.shape[1] != screen_width:
            final_img = np.zeros((screen_height, screen_width, 3), dtype=np.float32)
            h, w = image.shape[:2]
            final_img[:h, :w] = image
            image = final_img

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
