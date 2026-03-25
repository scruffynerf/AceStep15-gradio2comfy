import numpy as np
import cv2
import torch
import os
import random
from PIL import Image
from .includes.visualizer_utils import FlexAudioVisualizerBase, BaseAudioProcessor, get_color_for_frequency, parse_color

class ScromfyFlexAudioVisualizerContourNode(FlexAudioVisualizerBase):
    """Generates an audio-reactive contour visualization driven by the Flex System.
    
    Reacts to audio parameters established by a visualizer settings node to render
    bars or lines protruding from the edges of a reference mask or random preset mask.
    Supports complex shape hierarchies and intelligent density scaling.
    
    Inputs:
        audio (AUDIO): Base waveform for synchronization and analysis.
        frame_rate (FLOAT): Target FPS for the output sequence.
        screen_width (INT): Output pixel width.
        screen_height (INT): Output pixel height.
        strength (FLOAT): Opacity of the rendered visualizer overlay.
        feature_param (STRING): Dynamic parameter mapped to the feature_mode logic.
        feature_mode (STRING): Mathematical operation applied to feature_param.
        feature_threshold (FLOAT): Activation gate for feature modulation.
        installed_mask (STRING): Selection of internal mask presets, or 'random'.
        mask_scale (FLOAT): Multiplier controlling the on-screen size of the mask.
        mask_top_margin (FLOAT): Vertical offset for the mask (percentage of screen).
        bar_length (FLOAT): Output length of the reacting segments.
        bar_length_mode (STRING): Evaluate 'bar_length' as 'absolute' pixels or 'relative' to mask scale.
        contour_smoothing (INT): Amount of geometric simplification applied to the mask edges.
        ghost_mask_strength (FLOAT): Opacity of the underlying mask preview.
        ghost_mode (STRING): Rendering style for the underlying mask preview.
        adaptive_point_density (BOOLEAN): Auto-scale 'num_points' based on total perimeter length.
        min_contour_area (FLOAT): Ignore mask shapes with an area smaller than this (noise reduction).
        max_contours (INT): Hard limit on the number of shapes to trace.
        distribute_by (STRING): How to spread rendering points across multiple contours ('area', 'perimeter', 'equal', 'angular').
        contour_layers (STRING): Comma-separated list of target hierarchy depths, or 'all'.
        contour_color_shift (FLOAT): Base hue offset stepping applied sequentially across multiple contours.
        num_points (INT): Maximum density resolution for the geometry generation.
    
    Optional Inputs:
        mask (MASK): Custom external reference mask passed downwards.
        opt_feature (FLOAT): Injection array for specific feature parameters.
        background (IMAGE): Optional underlying background overlay for blending.
        settings (VISUALIZER_SETTINGS): The master configuration dictionary controlling colors and frequencies.
        
    Outputs:
        IMAGE: The fully composed visualizer sequence frames.
        MASK: An alpha channel sequence of the generated solid pixels.
        SETTINGS (STRING): A JSON-safe debug record of all active parameters on the node.
        SOURCE_MASK: The successfully captured and sized reference mask used for rendering.
        LAYER_MAP: A diagrammatic representation showing hierarchical depths of the found contours.
    """

    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_required = base_inputs.get("required", {})
        base_optional = base_inputs.get("optional", {})
        
        base_required["feature_param"] = (cls.get_modifiable_params(), {"default": "None"})
        base_required["num_points"] = ("INT", {"default": 128, "min": 4, "max": 4000, "step": 1})
        
        # Remove ALL global parameters handled by Settings node
        for param in [
                      "color_mode", "randomize", "seed", "visualization_method",
                      "visualization_feature", "smoothing", "fft_size",
                      "min_frequency", "max_frequency", "line_width",
                      "direction", "sequence_direction", "direction_skew",
                      "centroid_offset_x", "centroid_offset_y", "num_points",
                      "color_shift", "saturation", "brightness", "custom_color",
                      "position_x", "position_y", "rotation"]:
            if param in base_required:
                del base_required[param]

        # Get list of masks
        masks_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "masks")
        installed_masks = ["random"]
        if os.path.exists(masks_dir):
            import glob
            pattern = os.path.join(masks_dir, "**", "*.png")
            full_paths = glob.glob(pattern, recursive=True)
            relative_paths = sorted([os.path.relpath(p, masks_dir) for p in full_paths])
            installed_masks.extend(relative_paths)

        new_inputs = {
            "required": {
                "installed_mask": (installed_masks, {"default": "random"}),
                "mask_scale": ("FLOAT", {"default": 0.60, "min": 0.01, "max": 1.0, "step": 0.01}),
                "mask_top_margin": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01}),
                "bar_length": ("FLOAT", {"default": 20.0, "min": 0.01, "max": 1000.0, "step": 0.1}),
                "bar_length_mode": (["absolute", "relative"], {"default": "absolute"}),
                "contour_smoothing": ("INT", {"default": 0, "min": 0, "max": 50, "step": 1}),
                "ghost_mask_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "ghost_mode": (["None", "White", "Color", "Outline White", "Outline Color", "Outline White (Thin)", "Outline Color (Thin)"], {"default": "White"}),
                "adaptive_point_density": ("BOOLEAN", {"default": False}),
                "min_contour_area": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 10000.0, "step": 10.0}),
                "max_contours": ("INT", {"default": 5, "min": 1, "max": 50, "step": 1}),
                "distribute_by": (["area", "perimeter", "equal", "angular"], {"default": "angular"}),
                "contour_layers": ("STRING", {"default": "0"}),
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

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "MASK", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK", "SETTINGS", "SOURCE_MASK", "LAYER_MAP")
    FUNCTION = "apply_effect"
    CATEGORY = "Scromfy/Ace-Step/Visualizers"

    @staticmethod
    def filter_contours_by_hierarchy(contours, hierarchy, target_layers="0"):
        if hierarchy is None:
            return contours
        
        if target_layers.lower() == "all":
            return contours

        hierarchy = hierarchy[0] # [Next, Previous, First_Child, Parent]
        depths = np.zeros(len(contours), dtype=int)
        
        # Calculate depths
        for i in range(len(contours)):
            parent = hierarchy[i][3]
            depth = 0
            while parent != -1:
                depth += 1
                parent = hierarchy[parent][3]
            depths[i] = depth
        
        try:
            allowed_depths = [int(x.strip()) for x in target_layers.split(",")]
        except (ValueError, AttributeError):
            allowed_depths = [0]
            
        filtered_indices = [i for i in range(len(contours)) if depths[i] in allowed_depths]
        return [contours[i] for i in filtered_indices]

    def apply_effect(self, audio, frame_rate, screen_width, screen_height, strength, feature_param,
                     feature_mode, feature_threshold, opt_feature=None, opt_video=None, source_mask=None, **kwargs):
        
        # Unpack visualizer settings if provided (to get the seed for mask selection)
        # We ensure local settings already in kwargs are NOT overwritten.
        ext_settings = kwargs.get("visualizer_settings", {})
        if isinstance(ext_settings, dict):
            for k, v in ext_settings.items():
                if k not in kwargs:
                    kwargs[k] = v

        # Use seed from kwargs (might be from visualizer_settings or direct)
        seed = kwargs.get("seed", 0)
        s_rng = random.Random(seed)

        # Handle randomization (moved up so it can affect point density calc)
        if kwargs.get("randomize", False):
            kwargs["bar_length_mode"] = "relative"
            if kwargs.get("visualization_feature", "frequency") == "waveform":
                kwargs["bar_length"] = s_rng.uniform(5.0, 10.0)
            else:
                kwargs["bar_length"] = s_rng.uniform(10.0, 25.0)
            kwargs["distribute_by"] = "perimeter"
            kwargs["max_contours"] = 50
            kwargs["min_contour_area"] = 100
            kwargs["contour_smoothing"] = 0
            kwargs["ghost_mask_strength"] = 0.25
            kwargs["ghost_mode"] = "Outline Color (Thin)"
            kwargs["contour_color_shift"] = s_rng.uniform(0.0, 0.75)
            kwargs["contour_layers"] = "all"
            kwargs["adaptive_point_density"] = True
            kwargs["num_points"] = 512

        # Handle optional/missing mask
        mask = source_mask
        if mask is None:
            mask = kwargs.get("mask")
            
        if mask is None:
            masks_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "masks")
            installed_mask = kwargs.get("installed_mask", "random")
            
            if installed_mask == "random":
                import glob
                pattern = os.path.join(masks_dir, "**", "*.png")
                full_paths = glob.glob(pattern, recursive=True)
                masks_list = [os.path.relpath(p, masks_dir) for p in full_paths]
                installed_mask = s_rng.choice(masks_list) if masks_list else None
            
            mask_path = os.path.join(masks_dir, installed_mask) if installed_mask else ""
            if mask_path and os.path.exists(mask_path):
                pil_img = Image.open(mask_path).convert('L')
                mask = torch.from_numpy(np.array(pil_img).astype(np.float32) / 255.0)
            else:
                mask = torch.zeros((1, 512, 512), dtype=torch.float32)
                cv2.circle(mask[0].numpy(), (256, 256), 200, (1.0,), -1)

        # Final output dimensions
        out_w = screen_width
        out_h = screen_height
        if opt_video is not None:
            out_h, out_w = opt_video.shape[1], opt_video.shape[2]

        m_batch, m_height, m_width = mask.shape if len(mask.shape) == 3 else (1, *mask.shape)
        if len(mask.shape) == 2: mask = mask.unsqueeze(0)

        # Proportional scaling logic
        # Instead of stretching the mask to fill the screen, we fit it proportionally
        mask_scale = kwargs.get("mask_scale", 0.60)
        mask_top_margin = kwargs.get("mask_top_margin", 0.05)

        aspect_ratio = m_width / m_height
        screen_aspect = out_w / out_h
        
        if aspect_ratio > screen_aspect:
            # Mask is wider than screen aspect
            fit_w = out_w
            fit_h = int(out_w / aspect_ratio)
        else:
            # Mask is taller than screen aspect
            fit_h = out_h
            fit_w = int(out_h * aspect_ratio)
            
        # Apply user scaling to the proportional fit dimensions
        new_w = int(fit_w * mask_scale)
        new_h = int(fit_h * mask_scale)
        
        transformed_mask = mask # Fallback
        if new_w > 0 and new_h > 0:
            resized_masks = []
            for b in range(m_batch):
                m_np = mask[b].cpu().numpy()
                m_resized = cv2.resize(m_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Create canvas at final output dimensions
                canvas = np.zeros((out_h, out_w), dtype=np.float32)
                
                # Center horizontally
                x_offset = (out_w - new_w) // 2
                # Use top margin vertically
                y_offset = int(out_h * mask_top_margin)
                
                # Safety clip
                y_end = min(y_offset + new_h, out_h)
                x_end = min(x_offset + new_w, out_w)
                h_to_copy = y_end - y_offset
                w_to_copy = x_end - x_offset
                canvas[y_offset:y_end, x_offset:x_end] = m_resized[:h_to_copy, :w_to_copy]
                
                resized_masks.append(torch.from_numpy(canvas))
            transformed_mask = torch.stack(resized_masks) if m_batch > 1 else resized_masks[0].unsqueeze(0)

        # Get final dimensions for processing (Using distinct names to avoid shadowing)
        proc_batch, proc_h, proc_w = transformed_mask.shape
        kwargs['_transformed_mask'] = transformed_mask

        # Find contours here so we can use them for adaptive density
        # CRITICAL: We MUST use the transformed_mask here, not the original 'mask'
        mask_uint8 = (transformed_mask[0].cpu().numpy() * 255).astype(np.uint8)
        # Switch to RETR_TREE for hierarchy support
        contours, hierarchy = cv2.findContours(mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        contour_layers = kwargs.get("contour_layers", "all")
        filtered_contours = self.filter_contours_by_hierarchy(contours, hierarchy, contour_layers)
        
        min_contour_area = kwargs.get('min_contour_area', 100.0)
        max_contours = kwargs.get('max_contours', 5)
        valid_contours = [c for c in filtered_contours if cv2.contourArea(c) >= min_contour_area]
        valid_contours.sort(key=cv2.contourArea, reverse=True)
        valid_contours = valid_contours[:max_contours]
        
        sequence_direction = kwargs.get("sequence_direction", "right")
        if sequence_direction == "left":
            # Reverse point order for all selected contours
            valid_contours = [c[::-1] for c in valid_contours]

        # Scale num_points if adaptive density is enabled
        if kwargs.get("adaptive_point_density", False) and valid_contours:
            # Calculate total perimeter of selected contours
            total_perimeter = sum(cv2.arcLength(c, True) for c in valid_contours)
            
            if total_perimeter > 0:
                # Reference: num_points units per 2000 perimeter units
                kwargs["num_points"] = int(total_perimeter * (kwargs.get("num_points", 100) / 2000.0))
                kwargs["num_points"] = max(10, min(kwargs["num_points"], 4000))
        
        # Calculate mask scale for relative bar lengths
        if valid_contours:
            # Get combined bounding box of all valid contours
            all_pts = np.vstack([c.reshape(-1, 2) for c in valid_contours])
            x, y, w, h = cv2.boundingRect(all_pts)
            kwargs["_mask_scale"] = (w + h) / 2.0
            # PASS THE VALID CONTOURS TO THE ENGINE to prevent re-calculation from unresized mask
            kwargs["_valid_contours"] = valid_contours
        else:
            kwargs["_mask_scale"] = 100.0 # Fallback
            kwargs["_valid_contours"] = []

        # Generate the Layer Map Visualization
        layer_maps = []
        # Define a helpful palette for layers (B, G, R)
        layer_colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100), (255, 100, 255), (100, 255, 255), (255, 255, 255)]

        for b in range(proc_batch):
            # Use the mask for this frame
            m_idx = min(b, transformed_mask.shape[0] - 1)
            f_mask = (transformed_mask[m_idx].cpu().numpy() * 255).astype(np.uint8)
            f_contours, f_hierarchy = cv2.findContours(f_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create colored canvas
            l_map = np.zeros((proc_h, proc_w, 3), dtype=np.uint8)
            
            if f_contours and f_hierarchy is not None:
                f_hierarchy = f_hierarchy[0]
                # Calculate depths for all contours in this frame
                depths = np.zeros(len(f_contours), dtype=int)
                for i in range(len(f_contours)):
                    parent = f_hierarchy[i][3]
                    depth = 0
                    while parent != -1:
                        depth += 1
                        parent = f_hierarchy[parent][3]
                    depths[i] = depth

                for i, cnt in enumerate(f_contours):
                    depth = depths[i]
                    color = layer_colors[min(depth, len(layer_colors)-1)]
                    # Draw contour outline and filled with low alpha (or just outline + label)
                    cv2.drawContours(l_map, [cnt], -1, color, 2)
                    
                    # Label with "L{depth}" at centroid
                    M = cv2.moments(cnt)
                    if M["m00"] > 0:
                        lcx, lcy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                        # Small drop shadow for text
                        txt = f"L{depth}"
                        cv2.putText(l_map, txt, (lcx+1, lcy+1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                        cv2.putText(l_map, txt, (lcx, lcy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Overlay Total Counts
                max_depth = max(depths) if len(depths) > 0 else 0
                
                stats_text = f"Contours: {len(f_contours)} | Layers: {max_depth + 1}"
                cv2.putText(l_map, stats_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4) # Shadow
                cv2.putText(l_map, stats_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2) # Text

            layer_maps.append(torch.from_numpy(l_map.astype(np.float32) / 255.0).unsqueeze(0))
        
        layer_map_out = torch.cat(layer_maps, dim=0)

        images, masks, settings, _ = super().apply_effect(
            audio, frame_rate, screen_width, screen_height,
            strength, feature_param, feature_mode, feature_threshold,
            opt_feature, opt_video, transformed_mask, **kwargs
        )
        
        # We return the transformed_mask as the SOURCE_MASK output
        return (images, masks, settings, transformed_mask, layer_map_out)

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
        # Use background if provided, else black
        background = kwargs.get("background")
        if background is not None:
            image = background.copy().astype(np.float32) / 255.0
            if image.shape[0] != screen_height or image.shape[1] != screen_width:
                image = cv2.resize(image, (screen_width, screen_height))
        else:
            image = np.zeros((screen_height, screen_width, 3), dtype=np.float32)
        
        frame_idx = min(frame_index, batch_size - 1)
        mask_uint8 = (mask[frame_idx].cpu().numpy() * 255).astype(np.uint8)
        
        contours, hierarchy = cv2.findContours(mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return image
        
        contour_layers = kwargs.get("contour_layers", "0")
        filtered_contours = self.filter_contours_by_hierarchy(contours, hierarchy, contour_layers)

        # For geometric color modes, find the center of the mask
        M = cv2.moments(mask_uint8)
        if M["m00"] > 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        else:
            cx, cy = screen_width // 2, screen_height // 2

        # Apply user-specified CoM offset (as fraction of screen size)
        cx = int(cx + kwargs.get("centroid_offset_x", 0.0) * screen_width)
        cy = int(cy + kwargs.get("centroid_offset_y", 0.0) * screen_height)
        max_dist = np.sqrt(cx**2 + cy**2) # Max possible distance from center

        # Prioritize pre-calculated contours from apply_effect
        valid_contours = kwargs.get("_valid_contours")
        if valid_contours is None:
            valid_contours = [c for c in filtered_contours if cv2.contourArea(c) >= min_contour_area]
            valid_contours.sort(key=cv2.contourArea, reverse=True)
            valid_contours = valid_contours[:max_contours]
        
        if not valid_contours: return image

        # Sort final valid contours by angle from center to ensure spatial symmetry 
        # in distribution modes like 'perimeter'. 
        if len(valid_contours) > 1:
            def get_contour_angle(cnt):
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    mx = M["m10"] / M["m00"]
                    my = M["m01"] / M["m00"]
                else:
                    # Fallback to mean of all points in contour
                    mx, my = np.mean(cnt.squeeze(), axis=0)
                # Angle from 12 o'clock (0 to 2PI clockwise)
                return (np.arctan2(mx - cx, -(my - cy)) + np.pi * 2) % (np.pi * 2)
            
            valid_contours.sort(key=get_contour_angle)
        
        # Option 1: Draw ghost mask if enabled
        ghost_mode = kwargs.get("ghost_mode", "White")
        ghost_mask_strength = kwargs.get("ghost_mask_strength", 0.0)
        
        if ghost_mode != "None" and ghost_mask_strength > 0:
            # 1. Determine base color
            if "Color" in ghost_mode:
                base_ghost_color = parse_color(kwargs.get("custom_color", "#00ffff"))
            else:
                base_ghost_color = (1.0, 1.0, 1.0)
                
            # 2. Scale by strength
            ghost_color = tuple(val * ghost_mask_strength for val in base_ghost_color)
            
            # 3. Render (Solid vs Outline)
            if "Outline" in ghost_mode:
                # Draw outlines of ALL mask contours (not just filtered ones)
                # Check for "Thin" variant (fixed 1px)
                if "(Thin)" in ghost_mode:
                    thick = 1
                else:
                    thick = kwargs.get("line_width", 2)
                cv2.drawContours(image, contours, -1, ghost_color, thick)
            else:
                # Solid fill
                image[mask_uint8 > 0] = ghost_color

        if distribute_by == 'area':
            weights = [cv2.contourArea(c) for c in valid_contours]
        elif distribute_by == 'perimeter' or distribute_by == 'angular':
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
            
            # num_pts for this contour. If angular, it's the total number of points in the data sequence.
            # If not angular, it's the number of points assigned to this contour based on weights.
            current_contour_num_pts = num_pts if distribute_by == 'angular' else len(data[start_idx:end_idx])
            if current_contour_num_pts == 0: return

            indices = (np.linspace(0, contour_length - 1, current_contour_num_pts, endpoint=False) + rotation_offset) % (contour_length - 1)
            x_coords = np.interp(indices, range(contour_length), contour[:, 0])
            y_coords = np.interp(indices, range(contour_length), contour[:, 1])
            
            # Logic for data sampling
            if distribute_by == 'angular':
                # Map each point to its angle relative to center (consistent for all contours)
                p_angles = (np.arctan2(x_coords - cx, -(y_coords - cy)) + np.pi * 2) % (np.pi * 2)
                # Map [0, 2PI] to [0, total_pts-1]
                # We don't apply rotation here because it's already handled globally or via indices
                # Actually, the user might want frequencies to rotate too.
                # If we apply rotation, the "High" frequency point moves.
                indices_data = np.clip((p_angles / (np.pi * 2) * (total_pts - 1)).astype(np.int32), 0, total_pts - 1)
                contour_data = data[indices_data]
            else:
                # Standard sequence distribution (linear along perimeters)
                contour_data = data[start_idx:end_idx]
            
            num_pts = len(contour_data) # Update num_pts to reflect actual data points used
            if num_pts == 0: return

            if len(x_coords) < 2:
                # Degenerate case: single point, use a default normal
                dx = np.zeros_like(x_coords)
                dy = np.ones_like(y_coords)
            else:
                dx = np.gradient(x_coords)
                dy = np.gradient(y_coords)
            lengths = np.sqrt(dx**2 + dy**2)
            lengths = np.where(lengths > 0, lengths, 1.0)
            normals_x = -dy / lengths
            normals_y = dx / lengths

            # Override normals for centroid/starburst: vector between point and mask center
            if direction in ("centroid", "starburst"):
                cdx = cx - x_coords
                cdy = cy - y_coords
                clens = np.sqrt(cdx**2 + cdy**2)
                clens = np.where(clens > 0, clens, 1.0)
                # centroid = toward center, starburst = away from center
                sign = 1.0 if direction == "centroid" else -1.0
                normals_x = sign * cdx / clens
                normals_y = sign * cdy / clens

            # Apply angular skew (rotate direction vectors by N degrees)
            direction_skew = kwargs.get("direction_skew", 0.0)
            if direction_skew != 0.0:
                skew_rad = np.deg2rad(direction_skew)
                cos_s = np.cos(skew_rad)
                sin_s = np.sin(skew_rad)
                nx_rot = normals_x * cos_s - normals_y * sin_s
                ny_rot = normals_x * sin_s + normals_y * cos_s
                normals_x = nx_rot
                normals_y = ny_rot

            if visualization_method == 'bar':
                for i, amplitude in enumerate(contour_data):
                    x1, y1 = int(x_coords[i]), int(y_coords[i])
                    bar_h = amplitude * effective_bar_length * direction_multiplier
                    x2, y2 = int(x1 + normals_x[i] * bar_h), int(y1 + normals_y[i] * bar_h)
                    
                    # Determine color
                    current_idx = indices_data[i] if distribute_by == 'angular' else (start_idx + i)
                    color = self.get_draw_color(current_idx, total_pts, amplitude,
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
                    current_idx = indices_data[i] if distribute_by == 'angular' else (start_idx + i)
                    color = self.get_draw_color(current_idx, total_pts, contour_data[i],
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
                current_idx = indices_data[-1] if distribute_by == 'angular' else (end_idx - 1)
                color = self.get_draw_color(current_idx, total_pts, contour_data[-1],
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
                # centroid/starburst encode direction in normals; inward flips normals
                mul = -1.0 if direction == "inward" else 1.0
                process_contour(cnt, start_idx, end_idx, mul, i, total_contours)
            start_idx = end_idx

        return image.copy()

NODE_CLASS_MAPPINGS = {
    "ScromfyFlexAudioVisualizerContour": ScromfyFlexAudioVisualizerContourNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScromfyFlexAudioVisualizerContour": "Contour Audio Visualizer (Scromfy)",
}
