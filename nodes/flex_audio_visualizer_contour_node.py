import numpy as np
import cv2
import torch
import os
import random
from PIL import Image
from .includes.visualizer_utils import FlexAudioVisualizerBase, BaseAudioProcessor, get_color_for_frequency, parse_color

class ScromfyFlexAudioVisualizerContourNode(FlexAudioVisualizerBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_required = base_inputs.get("required", {})
        base_optional = base_inputs.get("optional", {})
        
        base_required["feature_param"] = (cls.get_modifiable_params(), {"default": "None"})
        
        # Remove parameters not used by contour or handled by base
        for param in ["position_x", "position_y",
                      "color_mode", "randomize", "seed", "visualization_method",
                      "visualization_feature", "smoothing", "num_points", "fft_size",
                      "min_frequency", "max_frequency", "line_width", "rotation"]:
            if param in base_required:
                del base_required[param]

        # Get list of masks
        masks_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "masks")
        installed_masks = ["random"]
        if os.path.exists(masks_dir):
            masks_list = sorted([f for f in os.listdir(masks_dir) if f.lower().endswith(".png")])
            installed_masks.extend(masks_list)

        new_inputs = {
            "required": {
                "installed_mask": (installed_masks, {"default": "random"}),
                "mask_scale": ("FLOAT", {"default": 0.60, "min": 0.01, "max": 1.0, "step": 0.01}),
                "mask_top_margin": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01}),
                "bar_length": ("FLOAT", {"default": 20.0, "min": 0.01, "max": 1000.0, "step": 0.1}),
                "bar_length_mode": (["absolute", "relative"], {"default": "absolute"}),
                "contour_smoothing": ("INT", {"default": 0, "min": 0, "max": 50, "step": 1}),
                "ghost_mask_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ghost_use_custom_color": ("BOOLEAN", {"default": True}),
                "adaptive_point_density": ("BOOLEAN", {"default": False}),
                "min_contour_area": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 10000.0, "step": 10.0}),
                "max_contours": ("INT", {"default": 5, "min": 1, "max": 50, "step": 1}),
                "distribute_by": (["area", "perimeter", "equal"], {"default": "perimeter"}),
                "contour_color_shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
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
                "min_frequency", "max_frequency", "bar_length", "line_width",
                "contour_smoothing", "min_contour_area", "max_contours", 
                "color_shift", "saturation", "brightness", "ghost_mask_strength", 
                "ghost_use_custom_color", "adaptive_point_density", "bar_length_mode", "None"]

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK", "SETTINGS", "SOURCE_MASK")

    def apply_effect(self, audio, frame_rate, screen_width, screen_height, strength, feature_param,
                     feature_mode, feature_threshold, mask=None, opt_feature=None, **kwargs):
        
        # Use seed from kwargs (might be from visualizer_settings or direct)
        seed = kwargs.get("seed", 0)
        s_rng = random.Random(seed)

        # Base class now handles universal randomization and vibrant color picking.
        # We do node-specific randomization here.
        if kwargs.get("randomize", False):
            kwargs["bar_length_mode"] = "relative"
            if kwargs.get("visualization_feature", "frequency") == "waveform":
                kwargs["bar_length"] = s_rng.uniform(5.0, 10.0)
            else:
                # Frequency bars on contour look better slightly longer than waveform
                kwargs["bar_length"] = s_rng.uniform(10.0, 25.0)
                
            kwargs["distribute_by"] = "perimeter"
            kwargs["max_contours"] = 50
            kwargs["min_contour_area"] = 0
            kwargs["contour_smoothing"] = 0
            kwargs["ghost_mask_strength"] = 0.15
            kwargs["ghost_use_custom_color"] = True
            kwargs["contour_color_shift"] = s_rng.uniform(0.0, 0.75)

        # Handle optional/missing mask
        if mask is None:
            masks_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "masks")
            installed_mask = kwargs.get("installed_mask", "random")
            
            if installed_mask == "random":
                masks_list = [f for f in os.listdir(masks_dir) if f.lower().endswith(".png")] if os.path.exists(masks_dir) else []
                installed_mask = s_rng.choice(masks_list) if masks_list else None
            
            mask_path = os.path.join(masks_dir, installed_mask) if installed_mask else ""
            if mask_path and os.path.exists(mask_path):
                pil_img = Image.open(mask_path).convert('L')
                mask = torch.from_numpy(np.array(pil_img).astype(np.float32) / 255.0)
            else:
                mask = torch.zeros((1, 512, 512), dtype=torch.float32)
                cv2.circle(mask[0].numpy(), (256, 256), 200, (1.0,), -1)

        # Capture the "Source Mask" before we do any resizing/processing
        if len(mask.shape) == 2:
            source_mask = mask.unsqueeze(0)
        else:
            source_mask = mask

        # Resizing and Positioning logic
        mask_scale = kwargs.get("mask_scale", 0.60)
        mask_top_margin = kwargs.get("mask_top_margin", 0.05)
        
        m_batch, m_height, m_width = mask.shape if len(mask.shape) == 3 else (1, *mask.shape)
        if len(mask.shape) == 2: mask = mask.unsqueeze(0)

        # Actually resize the mask content
        new_w = int(m_width * mask_scale)
        new_h = int(m_height * mask_scale)
        
        if new_w > 0 and new_h > 0:
            resized_masks = []
            for b in range(m_batch):
                m_np = mask[b].cpu().numpy()
                m_resized = cv2.resize(m_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
                canvas = np.zeros((m_height, m_width), dtype=np.float32)
                x_offset = (m_width - new_w) // 2
                y_offset = int(m_height * mask_top_margin)
                y_end = min(y_offset + new_h, m_height)
                x_end = min(x_offset + new_w, m_width)
                h_to_copy = y_end - y_offset
                w_to_copy = x_end - x_offset
                canvas[y_offset:y_end, x_offset:x_end] = m_resized[:h_to_copy, :w_to_copy]
                resized_masks.append(torch.from_numpy(canvas))
            mask = torch.stack(resized_masks) if m_batch > 1 else resized_masks[0].unsqueeze(0)

        # Get final dimensions for processing
        batch_size, screen_height, screen_width = mask.shape
        kwargs['mask'] = mask

        # Find contours here so we can use them for adaptive density
        mask_uint8 = (mask[0].cpu().numpy() * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        min_contour_area = kwargs.get('min_contour_area', 100.0)
        max_contours = kwargs.get('max_contours', 5)
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_contour_area]
        valid_contours.sort(key=cv2.contourArea, reverse=True)
        valid_contours = valid_contours[:max_contours]

        if sequence_direction == "left":
            # Reverse point order for all selected contours
            valid_contours = [c[::-1] for c in valid_contours]

        # Scale num_points if adaptive density is enabled
        if kwargs.get("adaptive_point_density", False) and valid_contours:
            # Calculate total perimeter of selected contours
            total_perimeter = 0
            for c in valid_contours:
                total_perimeter += cv2.arcLength(c, True)
            
            if total_perimeter > 0:
                # Reference: num_points units per 2000 perimeter units
                kwargs["num_points"] = int(total_perimeter * (kwargs.get("num_points", 100) / 2000.0))
                kwargs["num_points"] = max(10, min(kwargs["num_points"], 4000))
        
        # Calculate mask scale for relative bar lengths
        if valid_contours:
            # Get combined bounding box of all valid contours
            all_pts = np.vstack([c.reshape(-1, 2) for c in valid_contours])
            x, y, w, h = cv2.boundingRect(all_pts)
            mask_scale = (w + h) / 2.0
            kwargs["_mask_scale"] = mask_scale
        else:
            kwargs["_mask_scale"] = 100.0 # Fallback

        images, masks, settings, source_mask_out = super().apply_effect(
            audio, frame_rate, screen_width, screen_height,
            strength, feature_param, feature_mode, feature_threshold,
            opt_feature, source_mask=source_mask, **kwargs
        )
        
        return (images, masks, settings, source_mask_out)

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
        mask = kwargs.get('mask')
        visualization_method = kwargs.get('visualization_method', 'bar')
        batch_size, screen_height, screen_width = mask.shape
        line_width = kwargs.get('line_width', 2)
        bar_length = kwargs.get('bar_length', 20.0)
        bar_length_mode = kwargs.get('bar_length_mode', 'absolute')
        mask_scale = kwargs.get('_mask_scale', 100.0)
        direction = kwargs.get('direction', 'outward')
        sequence_direction = kwargs.get('sequence_direction', 'right')
        
        if bar_length_mode == "relative":
            # Treat bar_length as a percentage (e.g. 5.0 means 5% of mask scale)
            effective_bar_length = (bar_length / 100.0) * mask_scale
        else:
            effective_bar_length = bar_length

        contour_smoothing = kwargs.get('contour_smoothing', 0)
        rotation = kwargs.get('rotation', 0.0) % 360.0
        direction = kwargs.get('direction', 'outward')
        min_contour_area = kwargs.get('min_contour_area', 100.0)
        max_contours = kwargs.get('max_contours', 5)
        distribute_by = kwargs.get('distribute_by', 'perimeter')
        
        color_mode = kwargs.get('color_mode', 'white')
        color_shift = kwargs.get('color_shift', 0.0)
        saturation = kwargs.get('saturation', 1.0)
        brightness = kwargs.get('brightness', 1.0)
        item_freqs = kwargs.get('item_freqs', None)

        frame_index = processor.current_frame
        image = np.zeros((screen_height, screen_width, 3), dtype=np.float32)
        
        frame_idx = min(frame_index, batch_size - 1)
        mask_uint8 = (mask[frame_idx].cpu().numpy() * 255).astype(np.uint8)
        
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return image

        # For geometric color modes, find the center of the mask
        M = cv2.moments(mask_uint8)
        if M["m00"] > 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        else:
            cx, cy = screen_width // 2, screen_height // 2
        max_dist = np.sqrt(cx**2 + cy**2) # Max possible distance from center

        # Prioritize pre-calculated contours from apply_effect
        valid_contours = kwargs.get("_valid_contours")
        if valid_contours is None:
            valid_contours = [c for c in contours if cv2.contourArea(c) >= min_contour_area]
            valid_contours.sort(key=cv2.contourArea, reverse=True)
            valid_contours = valid_contours[:max_contours]
        
        if not valid_contours: return image
        
        # Option 1: Draw ghost mask (filled dimmed area) if enabled
        ghost_mask_strength = kwargs.get("ghost_mask_strength", 0.0)
        if ghost_mask_strength > 0:
            # Determine ghost color
            if kwargs.get("ghost_use_custom_color", True):
                base_ghost_color = parse_color(kwargs.get("custom_color", "#00ffff"))
            elif color_mode == "white":
                base_ghost_color = (1.0, 1.0, 1.0)
            else:
                base_ghost_color = (1.0, 1.0, 1.0) # Fallback to white/gray
                
            ghost_color = tuple(val * ghost_mask_strength for val in base_ghost_color)
            
            # Fill the mask area in the image
            image[mask_uint8 > 0] = ghost_color

        if distribute_by == 'area':
            weights = [cv2.contourArea(c) for c in valid_contours]
        elif distribute_by == 'perimeter':
            weights = [cv2.arcLength(c, True) for c in valid_contours]
        else:
            weights = [1] * len(valid_contours)

        total_weight = sum(weights)
        weights = [w / total_weight for w in weights] if total_weight > 0 else [1/len(weights)]*len(weights)

        data = self.transform_sequence(processor.spectrum, sequence_direction)
        if item_freqs is not None:
            item_freqs = self.transform_sequence(item_freqs, sequence_direction)
        def process_contour(contour, start_idx, end_idx, direction_multiplier=1.0, contour_idx=0, total_contours=1):
            if contour_smoothing > 0:
                epsilon = contour_smoothing * cv2.arcLength(contour, True) * 0.01
                contour = cv2.approxPolyDP(contour, epsilon, True)
            
            contour = contour.squeeze()
            if len(contour.shape) < 2: return
            contour_length = len(contour)
            if not np.array_equal(contour[0], contour[-1]):
                contour = np.vstack([contour, contour[0]])
                contour_length += 1
            
            rotation_offset = int((rotation / 360.0) * contour_length)
            contour_data = data[start_idx:end_idx]
            num_pts = len(contour_data)
            if num_pts == 0: return

            indices = (np.linspace(0, contour_length - 1, num_pts) + rotation_offset) % (contour_length - 1)
            x_coords = np.interp(indices, range(contour_length), contour[:, 0])
            y_coords = np.interp(indices, range(contour_length), contour[:, 1])
            
            dx = np.gradient(x_coords)
            dy = np.gradient(y_coords)
            lengths = np.sqrt(dx**2 + dy**2)
            lengths = np.where(lengths > 0, lengths, 1.0)
            normals_x = -dy / lengths
            normals_y = dx / lengths

            if visualization_method == 'bar':
                for i, amplitude in enumerate(contour_data):
                    x1, y1 = int(x_coords[i]), int(y_coords[i])
                    bar_h = amplitude * effective_bar_length * direction_multiplier
                    x2, y2 = int(x1 + normals_x[i] * bar_h), int(y1 + normals_y[i] * bar_h)
                    
                    # Determine color
                    color = self.get_draw_color(start_idx + i, total_pts, amplitude,
                                                x1, y1, cx, cy, max_dist, **kwargs)
                    
                    # Apply contour-specific color shift if in custom/spectrum mode
                    if color_mode in ["custom", "spectrum"] and total_contours > 1:
                        color_shift_val = kwargs.get("contour_color_shift", 0.0)
                        if color_shift_val > 0:
                            import colorsys
                            h, l, s = colorsys.rgb_to_hls(*color)
                            color = colorsys.hls_to_rgb((h + (contour_idx / total_contours) * color_shift_val) % 1.0, l, s)
                        
                    cv2.line(image, (x1, y1), (x2, y2), color, thickness=line_width)
            else:
                pts = np.column_stack([
                    x_coords + normals_x * contour_data * effective_bar_length * direction_multiplier,
                    y_coords + normals_y * contour_data * effective_bar_length * direction_multiplier
                ]).astype(np.int16)
                
                # Draw segments to support multi-color modes
                for i in range(len(pts) - 1):
                    p1 = pts[i]
                    p2 = pts[i+1]
                    color = self.get_draw_color(start_idx + i, total_pts, contour_data[i],
                                                p1[0], p1[1], cx, cy, max_dist, **kwargs)
                    
                    # Apply contour-specific color shift
                    if color_mode in ["custom", "spectrum"] and total_contours > 1:
                        color_shift_val = kwargs.get("contour_color_shift", 0.0)
                        if color_shift_val > 0:
                            import colorsys
                            h, l, s = colorsys.rgb_to_hls(*color)
                            color = colorsys.hls_to_rgb((h + (contour_idx / total_contours) * color_shift_val) % 1.0, l, s)
                            
                    cv2.line(image, tuple(p1), tuple(p2), color, line_width)
                
                # Close loop
                color = self.get_draw_color(end_idx - 1, total_pts, contour_data[-1],
                                            pts[-1][0], pts[-1][1], cx, cy, max_dist, **kwargs)
                cv2.line(image, tuple(pts[-1]), tuple(pts[0]), color, line_width)

        start_idx = 0
        total_pts = len(data)
        total_contours = len(valid_contours)
        for i, (cnt, w) in enumerate(zip(valid_contours, weights)):
            num_pts = int(round(total_pts * w)) if i < len(valid_contours)-1 else total_pts - start_idx
            end_idx = start_idx + num_pts
            if direction == "both":
                process_contour(cnt, start_idx, end_idx, 0.5, i, total_contours)
                process_contour(cnt, start_idx, end_idx, -0.5, i, total_contours)
            else:
                mul = -1.0 if direction == "inward" else 1.0
                process_contour(cnt, start_idx, end_idx, mul, i, total_contours)
            start_idx = end_idx

        return image.copy()

NODE_CLASS_MAPPINGS = {
    "ScromfyFlexAudioVisualizerContour": ScromfyFlexAudioVisualizerContourNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScromfyFlexAudioVisualizerContour": "Flex Audio Visualizer Contour (Scromfy)",
}
