import numpy as np
import cv2
import torch
from .includes.visualizer_utils import FlexAudioVisualizerBase, BaseAudioProcessor, get_color_for_frequency, parse_color

class ScromfyFlexAudioVisualizerCircularNode(FlexAudioVisualizerBase):
    """Generates an audio-reactive circular visualization driven by the Flex System.
    
    Reacts to audio parameters established by a visualizer settings node to render
    expanding rings, sunbursts, or inward-collapsing circles of audio amplitude data.
    
    Inputs:
        audio (AUDIO): Base waveform for synchronization and analysis.
        frame_rate (FLOAT): Target FPS for the output sequence.
        screen_width (INT): Output pixel width.
        screen_height (INT): Output pixel height.
        strength (FLOAT): Opacity of the rendered visualizer overlay.
        feature_param (STRING): Dynamic parameter mapped to the feature_mode logic.
        feature_mode (STRING): Mathematical operation applied to feature_param.
        feature_threshold (FLOAT): Activation gate for feature modulation.
        radius (FLOAT): Max absolute boundary limit in pixels for the circle.
        base_radius (FLOAT): Minimum static resting size in pixels.
        amplitude_scale (FLOAT): Multiplier determining how violently boundaries react to volume.
        bar_length_mode (STRING): Controls whether amplitude maps to 'absolute' pixels or 'relative' scaling.
    
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
                "radius": ("FLOAT", {"default": 200.0, "min": 10.0, "max": 1000.0, "step": 10.0}),
                "base_radius": ("FLOAT", {"default": 200.0, "min": 10.0, "max": 1000.0, "step": 10.0}),
                "amplitude_scale": ("FLOAT", {"default": 100.0, "min": 1.0, "max": 1000.0, "step": 10.0}),
                "bar_length_mode": (["absolute", "relative"], {"default": "absolute"}),
            }
        }

        all_required = {**base_required, **new_inputs["required"]}
        all_optional = {**base_optional, "mask": ("MASK",)}
        
        return {
            "required": all_required,
            "optional": all_optional
        }

    @classmethod
    def get_modifiable_params(cls):
        return ["smoothing", "rotation", "num_points", "fft_size", 
                "min_frequency", "max_frequency", "radius", "line_width",
                "amplitude_scale", "base_radius", "position_x", "position_y", 
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

        # Get screen dimensions from base or defaults
        screen_width = kwargs.get("screen_width", 512)
        screen_height = kwargs.get("screen_height", 512)
        min_dim = min(screen_width, screen_height)

        if kwargs.get("randomize", False):
            kwargs["bar_length_mode"] = "relative"
            # amplitude_scale is the "bar" height equivalent here
            if kwargs.get("visualization_feature", "frequency") == "waveform":
                kwargs["amplitude_scale"] = s_rng.uniform(5.0, 15.0)
            else:
                kwargs["amplitude_scale"] = s_rng.uniform(15.0, 40.0)
            
            # Base radius relative to screen size (10% to 40% of min dimension)
            kwargs["base_radius"] = s_rng.uniform(min_dim * 0.1, min_dim * 0.4)
            kwargs["radius"] = kwargs["base_radius"]

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
        num_points = self.get_point_count(kwargs)
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
        num_points = self.get_point_count(kwargs)
        screen_width = processor.width
        screen_height = processor.height
        position_x = kwargs.get('position_x', 0.5)
        position_y = kwargs.get('position_y', 0.5)
        base_radius = kwargs.get('base_radius', 200.0)
        amplitude_scale = kwargs.get('amplitude_scale', 100.0)
        bar_length_mode = kwargs.get('bar_length_mode', 'absolute')
        direction = kwargs.get('direction', 'outward')
        sequence_direction = kwargs.get('sequence_direction', 'right')
        
        if bar_length_mode == "relative":
            # Treat amplitude_scale as a percentage of base_radius (consistent with contour scaling)
            effective_amplitude_scale = (amplitude_scale / 100.0) * base_radius
        else:
            effective_amplitude_scale = amplitude_scale

        line_width = kwargs.get('line_width', 2)
        
        color_mode = kwargs.get('color_mode', 'white')
        color_shift = kwargs.get('color_shift', 0.0)
        saturation = kwargs.get('saturation', 1.0)
        brightness = kwargs.get('brightness', 1.0)
        item_freqs = kwargs.get('item_freqs', None)
        
        # Use background if provided, else black
        background = kwargs.get("background")
        if background is not None:
            image = background.copy().astype(np.float32) / 255.0
            if image.shape[0] != screen_height or image.shape[1] != screen_width:
                image = cv2.resize(image, (screen_width, screen_height))
        else:
            image = np.zeros((screen_height, screen_width, 3), dtype=np.float32)
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
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            else:
                cx, cy = center_x, center_y
        else:
            cx, cy = center_x, center_y

        # Apply user-specified CoM offset
        cx = int(cx + kwargs.get("centroid_offset_x", 0.0) * screen_width)
        cy = int(cy + kwargs.get("centroid_offset_y", 0.0) * screen_height)

        # Angles logic based on sequence_direction. 
        if sequence_direction == "left":
            # Anticlockwise: start 0, end -2PI
            angles = np.linspace(0, -2 * np.pi, num_points, endpoint=False)
        else:
            # Clockwise (right, centered, both ends): standard 0 to 2PI
            angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
            
        # Offset by -90 degrees (-PI/2) to start at top
        angles -= np.pi / 2.0
        
        rotation_rad = np.deg2rad(rotation)
        angles += rotation_rad

        data = self.transform_sequence(processor.spectrum, sequence_direction)
        if item_freqs is not None:
             item_freqs = self.transform_sequence(item_freqs, sequence_direction)
        max_dist = base_radius + effective_amplitude_scale

        if visualization_method == 'bar':
            for i, (angle, amplitude) in enumerate(zip(angles, data)):
                # Base radial vector
                rx, ry = np.cos(angle), np.sin(angle)
                x_base = center_x + base_radius * rx
                y_base = center_y + base_radius * ry

                # Determine direction vector (vx, vy)
                if direction in ('centroid', 'starburst'):
                    dx_com, dy_com = cx - x_base, cy - y_base
                    dist_com = np.sqrt(dx_com**2 + dy_com**2)
                    if dist_com > 0:
                        vx, vy = dx_com / dist_com, dy_com / dist_com
                    else:
                        vx, vy = rx, ry
                    if direction == 'starburst':
                        vx, vy = -vx, -vy
                else:
                    # inward/outward/both use radial vector
                    vx, vy = rx, ry
                    if direction == 'inward':
                        vx, vy = -vx, -vy

                # Apply skew to the chosen vector
                skew = kwargs.get("direction_skew", 0.0)
                if skew != 0:
                    skew_rad = np.deg2rad(skew)
                    s_cos, s_sin = np.cos(skew_rad), np.sin(skew_rad)
                    vx_new = vx * s_cos - vy * s_sin
                    vy_new = vx * s_sin + vy * s_cos
                    vx, vy = vx_new, vy_new

                bar_len = amplitude * effective_amplitude_scale
                
                if direction == 'both':
                    half = bar_len / 2.0
                    x_start, y_start = x_base - half * rx, y_base - half * ry
                    x_end, y_end = x_base + half * rx, y_base + half * ry
                else:
                    x_start, y_start = x_base, y_base
                    x_end, y_end = x_base + bar_len * vx, y_base + bar_len * vy
                
                # Determine color using shared helper
                # Pass cx, cy (potentially offset) to get_draw_color
                color = self.get_draw_color(i, num_points, amplitude,
                                            x_start, y_start, cx, cy, max_dist, **kwargs)
                
                cv2.line(image, (int(x_start), int(y_start)), (int(x_end), int(y_end)),
                         color, thickness=line_width)
        elif visualization_method == 'line':
            pts = []
            skew = kwargs.get("direction_skew", 0.0)
            
            for i, (angle, amplitude) in enumerate(zip(angles, data)):
                rx, ry = np.cos(angle), np.sin(angle)
                x_base = center_x + base_radius * rx
                y_base = center_y + base_radius * ry
                
                # Determine direction vector (vx, vy)
                if direction in ('centroid', 'starburst'):
                    dx_com, dy_com = cx - x_base, cy - y_base
                    dist_com = np.sqrt(dx_com**2 + dy_com**2)
                    if dist_com > 0:
                        vx, vy = dx_com / dist_com, dy_com / dist_com
                    else:
                        vx, vy = rx, ry
                    if direction == 'starburst':
                        vx, vy = -vx, -vy
                else:
                    # inward/outward use radial vector
                    vx, vy = rx, ry
                    if direction == 'inward':
                        vx, vy = -vx, -vy
                
                # Apply skew
                if skew != 0:
                    skew_rad = np.deg2rad(skew)
                    s_cos, s_sin = np.cos(skew_rad), np.sin(skew_rad)
                    vx, vy = vx * s_cos - vy * s_sin, vx * s_sin + vy * s_cos
                
                bar_len = amplitude * effective_amplitude_scale
                pts.append([x_base + bar_len * vx, y_base + bar_len * vy])
                
            points = np.array(pts).astype(np.int32)
            num_pts = len(points)
            for i in range(num_pts):
                p1 = points[i]
                p2 = points[(i+1) % num_pts]
                    
                # Determine color for this segment
                color = self.get_draw_color(i, num_points, data[i],
                                            p1[0], p1[1], cx, cy, max_dist, **kwargs)
                
                cv2.line(image, tuple(p1), tuple(p2), color, line_width)

        return image.copy()

NODE_CLASS_MAPPINGS = {
    "ScromfyFlexAudioVisualizerCircular": ScromfyFlexAudioVisualizerCircularNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScromfyFlexAudioVisualizerCircular": "Circular Audio Visualizer (Scromfy)",
}
