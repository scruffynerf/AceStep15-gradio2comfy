import numpy as np
import cv2
import torch
from .includes.visualizer_utils import FlexAudioVisualizerBase, BaseAudioProcessor, get_color_for_frequency, parse_color

class ScromfyFlexAudioVisualizerLineNode(FlexAudioVisualizerBase):
    """Generates an audio-reactive linear visualization driven by the Flex System.
    
    Reacts to audio parameters established by a visualizer settings node to render
    straight, responsive waveform or frequency spectrum lines with variable spacing.
    
    Inputs:
        audio (AUDIO): Base waveform for synchronization and analysis.
        frame_rate (FLOAT): Target FPS for the output sequence.
        screen_width (INT): Output pixel width.
        screen_height (INT): Output pixel height.
        strength (FLOAT): Opacity of the rendered visualizer overlay.
        feature_param (STRING): Dynamic parameter mapped to the feature_mode logic.
        feature_mode (STRING): Mathematical operation applied to feature_param.
        feature_threshold (FLOAT): Activation gate for feature modulation.
        max_height (FLOAT): Peak bar height.
        min_height (FLOAT): Minimum resting bar height.
        bar_length_mode (STRING): Evaluate height as 'absolute' pixels or 'relative' screen percentages.
        length (FLOAT): Total base length of the line.
        separation (FLOAT): Spacing in pixels between individual bars.
        curvature (FLOAT): End-cap rounding intensity for discrete bars.
        curve_smoothing (FLOAT): Smoothing factor applying Gaussian blur to the output points.
    
    Optional Inputs:
        mask (MASK): Reference geometry mask passed downwards or used as a rendering bounds clipper.
        opt_feature (FLOAT): Injection array for specific feature parameters.
        background (IMAGE): Optional underlying background overlay for blending.
        settings (VISUALIZER_SETTINGS): The master configuration dictionary controlling colors and frequencies.
        
    Outputs:
        IMAGE: The fully composed visualizer sequence frames.
        MASK: An alpha channel sequence of the generated solid pixels.
        SETTINGS (STRING): A JSON-safe debug record of all active parameters on the node.
        SOURCE_MASK: The passed-through or auto-generated reference mask.
    """

    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_required = base_inputs.get("required", {})
        base_optional = base_inputs.get("optional", {})
        
        # Override feature_param with valid options
        base_required["feature_param"] = (cls.get_modifiable_params(), {"default": "None"})

        # Remove ALL global parameters handled by Settings node
        for param in [
                      "color_mode", "randomize", "seed", "visualization_method",
                      "visualization_feature", "smoothing", "fft_size",
                      "min_frequency", "max_frequency", "line_width",
                      "direction", "sequence_direction", "direction_skew",
                      "centroid_offset_x", "centroid_offset_y", "num_points",
                      "color_shift", "saturation", "brightness", "custom_color"]:
            if param in base_required:
                del base_required[param]
        
        new_inputs = {
            "required": {
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

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK", "SETTINGS", "SOURCE_MASK")
    FUNCTION = "apply_effect"
    CATEGORY = "Scromfy/Ace-Step/Visualizers"

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
            
            kwargs["num_bars"] = s_rng.choice([64, 128, 256])

        # Get screen dimensions
        screen_width = kwargs.get("screen_width", 512)
        screen_height = kwargs.get("screen_height", 512)
        
        # images, masks, settings, source_mask
        images, masks, settings, source_mask = super().apply_effect(
            audio, frame_rate, screen_width, screen_height,
            strength, feature_param, feature_mode, feature_threshold,
            opt_feature, source_mask=mask, **kwargs
        )
        
        return (images, masks, settings, source_mask)

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

        center_x = screen_width * position_x
        center_y = screen_height * position_y

        # Initialize image with background or black
        background = kwargs.get("background")
        if background is not None:
            image = background.copy().astype(np.float32) / 255.0
            if image.shape[0] != screen_height or image.shape[1] != screen_width:
                image = cv2.resize(image, (screen_width, screen_height))
        else:
            image = np.zeros((screen_height, screen_width, 3), dtype=np.float32)

        # For geometric color/direction logic, find the center of the mask if provided
        mask = kwargs.get("source_mask")
        if mask is not None:
            frame_idx = min(kwargs.get("frame_index", 0), mask.shape[0] - 1)
            mask_np = (mask[frame_idx].cpu().numpy() * 255).astype(np.uint8)
            M = cv2.moments(mask_np)
            if M["m00"] > 0:
                cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
            else:
                cx, cy = center_x, center_y
        else:
            cx, cy = center_x, center_y

        # Apply user-specified CoM offset
        cx += kwargs.get("centroid_offset_x", 0.0) * screen_width
        cy += kwargs.get("centroid_offset_y", 0.0) * screen_height

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

        data = self.transform_sequence(processor.spectrum, sequence_direction)
        if item_freqs is not None:
            item_freqs = self.transform_sequence(item_freqs, sequence_direction)

        # Rotation matrix for coordinate transformation
        rot_rad = np.deg2rad(-rotation) # negative because screen-Y is down
        cos_rot = np.cos(rot_rad)
        sin_rot = np.sin(rot_rad)

        def rotate_point(px, py, pivot_x, pivot_y):
            # px, py are relative to pivot
            rx = px * cos_rot - py * sin_rot
            ry = px * sin_rot + py * cos_rot
            return rx + pivot_x, ry + pivot_y

        direction_skew = kwargs.get("direction_skew", 0.0)

        if visualization_method == 'bar':
            curvature = kwargs.get('curvature', 0.0)
            separation = kwargs.get('separation', 5.0)
            total_separation = separation * (num_bars - 1)
            total_bar_width = visualization_length - total_separation
            bar_width = total_bar_width / num_bars
            
            # Local space: line goes from -visualization_length/2 to +visualization_length/2
            start_x_local = -visualization_length / 2.0

            for i, bar_value in enumerate(data):
                # Calculate bar center in local space (horizontal line along X axis)
                x_local = start_x_local + i * (bar_width + separation) + bar_width / 2
                y_local = 0
                
                bar_h = effective_min_height + (effective_max_height - effective_min_height) * bar_value
                
                # Global space base point (after rotation)
                x_base, y_base = rotate_point(x_local, y_local, center_x, center_y)

                # Determine direction vector (vx, vy)
                if direction in ('centroid', 'starburst'):
                    dx_com, dy_com = cx - x_base, cy - y_base
                    dist_com = np.sqrt(dx_com**2 + dy_com**2)
                    if dist_com > 0:
                        vx, vy = dx_com / dist_com, dy_com / dist_com
                    else:
                        # Fallback to local vertical rotated
                        _, vy_rot = rotate_point(0, -1, 0, 0)
                        vx, vy = 0, vy_rot
                    if direction == 'starburst':
                        vx, vy = -vx, -vy
                else:
                    # upward/downward in rotated local space
                    # Unrotated vertical vector is (0, -1)
                    v_local_y = -1.0 if direction != 'inward' else 1.0
                    vx, vy = rotate_point(0, v_local_y, 0, 0)
                    # Note: rotate_point already adds pivot, but here we want a pure vector
                    vx -= 0
                    vy -= 0

                # Apply skew
                if direction_skew != 0:
                    skew_rad = np.deg2rad(direction_skew)
                    s_cos, s_sin = np.cos(skew_rad), np.sin(skew_rad)
                    vx_new = vx * s_cos - vy * s_sin
                    vy_new = vx * s_sin + vy * s_cos
                    vx, vy = vx_new, vy_new

                # Draw point calculation
                if direction == 'both':
                    half = bar_h / 2.0
                    # For both, we use the local vertical rotated
                    vx_v, vy_v = rotate_point(0, -1, 0, 0)
                    vx_v -= 0; vy_v -= 0
                    x1, y1 = x_base - half * vx_v, y_base - half * vy_v
                    x2, y2 = x_base + half * vx_v, y_base + half * vy_v
                else:
                    x1, y1 = x_base, y_base
                    x2, y2 = x_base + vx * bar_h, y_base + vy * bar_h

                color = self.get_draw_color(i, num_bars, bar_value,
                                            x1, y1, cx, cy, 
                                            max(screen_width, screen_height), **kwargs)
                
                # Check if we can use simple rectangle (no rotation/skew/centroid)
                is_simple = (rotation == 0 and direction_skew == 0 and direction in ('outward', 'inward', 'both'))
                
                if not is_simple or curvature > 0:
                    thickness = max(1, int(bar_width))
                    # Curvature on rotated lines is much harder without padding.
                    # For now, we use rounded lines if curvature > 0
                    if curvature > 0:
                        # Draw a line with rounded caps
                        p1 = (int(x1), int(y1))
                        p2 = (int(x2), int(y2))
                        cv2.line(image, p1, p2, color, thickness, lineType=cv2.LINE_AA)
                        # Draw circles at ends for "curvature" look
                        if curvature > 2:
                            cv2.circle(image, p1, thickness // 2, color, -1, lineType=cv2.LINE_AA)
                            cv2.circle(image, p2, thickness // 2, color, -1, lineType=cv2.LINE_AA)
                    else:
                        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                else:
                    # Original fast rectangle logic for zero rotation
                    rect_x_start = int(x_base - bar_width / 2)
                    rect_x_end = int(rect_x_start + bar_width)
                    ry1, ry2 = int(y1), int(y2)
                    if ry1 > ry2: ry1, ry2 = ry2, ry1
                    cv2.rectangle(image, (rect_x_start, ry1), (rect_x_end, ry2), color, thickness=-1)

        elif visualization_method == 'line':
            curve_smoothing = kwargs.get('curve_smoothing', 0.0)
            if curve_smoothing > 0:
                window_size = int(len(data) * curve_smoothing)
                if window_size % 2 == 0: window_size += 1
                data_smooth = self.smooth_curve(data, window_size) if window_size > 2 else data
            else:
                data_smooth = data
                
            amplitudes = effective_min_height + data_smooth * (effective_max_height - effective_min_height)
            num_pts = len(amplitudes)
            
            # Local space: horizontal line from -vis_len/2 to +vis_len/2
            x_bases_local = np.linspace(-visualization_length/2.0, visualization_length/2.0, num_pts)
            
            pts = []
            for i in range(num_pts):
                xb_local = x_bases_local[i]
                amp = amplitudes[i]
                
                # Base point in global space
                xb, yb = rotate_point(xb_local, 0, center_x, center_y)
                
                # Determine direction vector (vx, vy)
                if direction in ('centroid', 'starburst'):
                    dx_com, dy_com = cx - xb, cy - yb
                    dist_com = np.sqrt(dx_com**2 + dy_com**2)
                    if dist_com > 0:
                        vx, vy = dx_com / dist_com, dy_com / dist_com
                    else:
                        _, vy_rot = rotate_point(0, -1, 0, 0)
                        vx, vy = 0, vy_rot
                    if direction == 'starburst':
                        vx, vy = -vx, -vy
                else:
                    # upward/downward
                    v_local_y = -1.0 if direction != 'inward' else 1.0
                    vx, vy = rotate_point(0, v_local_y, 0, 0)
                    vx -= 0; vy -= 0
                
                # Apply skew
                if direction_skew != 0:
                    skew_rad = np.deg2rad(direction_skew)
                    s_cos, s_sin = np.cos(skew_rad), np.sin(skew_rad)
                    vx_new = vx * s_cos - vy * s_sin
                    vy_new = vx * s_sin + vy * s_cos
                    vx, vy = vx_new, vy_new
                
                pts.append([xb + vx * amp, yb + vy * amp])
                
            points = np.array(pts).astype(np.int32)
            
            if len(points) > 1:
                # Draw segments to support multi-color modes
                for i in range(len(points) - 1):
                    p1 = points[i]
                    p2 = points[i+1]
                    color = self.get_draw_color(i, num_pts, data[i],
                                                p1[0], p1[1], cx, cy,
                                                max(screen_width, screen_height), **kwargs)
                    cv2.line(image, tuple(p1), tuple(p2), color, line_width)

        return image

    def smooth_curve(self, y, window_size):
        if window_size < 3: return y
        box = np.ones(window_size) / window_size
        return np.convolve(y, box, mode='same')

NODE_CLASS_MAPPINGS = {
    "ScromfyFlexAudioVisualizerLine": ScromfyFlexAudioVisualizerLineNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScromfyFlexAudioVisualizerLine": "Line Audio Visualizer (Scromfy)",
}
