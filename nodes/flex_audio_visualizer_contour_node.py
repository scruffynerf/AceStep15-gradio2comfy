import numpy as np
import cv2
import torch
import os
import random
from PIL import Image
from .includes.visualizer_utils import FlexAudioVisualizerBase, BaseAudioProcessor, get_color_for_frequency

class ScromfyFlexAudioVisualizerContourNode(FlexAudioVisualizerBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_required = base_inputs.get("required", {})
        base_optional = base_inputs.get("optional", {})
        
        base_required["feature_param"] = (cls.get_modifiable_params(), {"default": "None"})
        
        # Remove parameters not used by contour
        for param in ["screen_width", "screen_height", "position_x", "position_y"]:
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
                "randomize": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "visualization_method": (["bar", "line"], {"default": "bar"}),
                "visualization_feature": (["frequency", "waveform"], {"default": "frequency"}),
                "smoothing": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "num_points": ("INT", {"default": 360, "min": 3, "max": 1000, "step": 1}),
                "fft_size": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 256}),
                "min_frequency": ("FLOAT", {"default": 20.0, "min": 20.0, "max": 20000.0, "step": 10.0}),
                "max_frequency": ("FLOAT", {"default": 8000.0, "min": 20.0, "max": 20000.0, "step": 10.0}),
                "bar_length": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 100.0, "step": 1.0}),
                "line_width": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "contour_smoothing": ("INT", {"default": 0, "min": 0, "max": 50, "step": 1}),
                "rotation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "direction": (["outward", "inward", "both"], {"default": "outward"}),
                "min_contour_area": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 10000.0, "step": 10.0}),
                "max_contours": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1}),
                "distribute_by": (["area", "perimeter", "equal"], {"default": "perimeter"}),
            }
        }

        all_required = {**new_inputs["required"], **base_required}
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
                "color_shift", "saturation", "brightness", "None"]

    def apply_effect(self, audio, frame_rate, strength, feature_param, feature_mode,
                     feature_threshold, mask=None, opt_feature=None, **kwargs):
        
        # Randomization logic
        if kwargs.get("randomize", False):
            s_rng = random.Random(kwargs.get("seed", 0))
            kwargs["visualization_method"] = s_rng.choice(["bar", "line"])
            kwargs["visualization_feature"] = s_rng.choice(["frequency", "waveform"])
            kwargs["color_mode"] = s_rng.choice(["white", "spectrum", "custom"])
            kwargs["bar_length"] = s_rng.uniform(5.0, 100.0)
            kwargs["line_width"] = s_rng.randint(1, 10)
            kwargs["distribute_by"] = s_rng.choice(["area", "perimeter", "equal"])
            kwargs["direction"] = s_rng.choice(["outward", "inward", "both"])
            kwargs["max_contours"] = s_rng.randint(1, 20)
            kwargs["contour_smoothing"] = s_rng.randint(0, 10) # subtle smoothing
            kwargs["smoothing"] = s_rng.uniform(0.1, 0.9)

        # Handle optional/missing mask
        if mask is None:
            masks_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "masks")
            installed_mask = kwargs.get("installed_mask", "random")
            
            if installed_mask == "random":
                masks_list = [f for f in os.listdir(masks_dir) if f.lower().endswith(".png")]
                if masks_list:
                    installed_mask = random.choice(masks_list)
                else:
                    # Fallback to a simple circle if no masks found
                    mask = torch.zeros((1, 512, 512), dtype=torch.float32)
                    cv2.circle(mask[0].numpy(), (256, 256), 200, (1.0,), -1)
            
            if mask is None: # Means we have a filename to load
                mask_path = os.path.join(masks_dir, installed_mask)
                if os.path.exists(mask_path):
                    pil_img = Image.open(mask_path).convert('L')
                    mask_np = np.array(pil_img).astype(np.float32) / 255.0
                    mask = torch.from_numpy(mask_np)
                else:
                    mask = torch.zeros((1, 512, 512), dtype=torch.float32)

        # Resizing and Positioning logic
        mask_scale = kwargs.get("mask_scale", 0.40)
        mask_top_margin = kwargs.get("mask_top_margin", 0.05)
        
        # We need the original/target dimensions to pad correctly
        # If mask was loaded, it might have its own size, but we usually want to fit it to a screen
        if len(mask.shape) == 3:
            m_batch, m_height, m_width = mask.shape
        else:
            m_height, m_width = mask.shape
            m_batch = 1
            mask = mask.unsqueeze(0)

        # Actually resize the mask content
        new_w = int(m_width * mask_scale)
        new_h = int(m_height * mask_scale)
        
        if new_w > 0 and new_h > 0:
            resized_masks = []
            for b in range(m_batch):
                m_np = mask[b].cpu().numpy()
                m_resized = cv2.resize(m_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Create a blank canvas of the original size
                canvas = np.zeros((m_height, m_width), dtype=np.float32)
                
                # Calculate positions
                x_offset = (m_width - new_w) // 2
                y_offset = int(m_height * mask_top_margin)
                
                # Ensure it fits
                y_end = min(y_offset + new_h, m_height)
                x_end = min(x_offset + new_w, m_width)
                h_to_copy = y_end - y_offset
                w_to_copy = x_end - x_offset
                
                canvas[y_offset:y_end, x_offset:x_end] = m_resized[:h_to_copy, :w_to_copy]
                resized_masks.append(torch.from_numpy(canvas))
            
            mask = torch.stack(resized_masks) if m_batch > 1 else resized_masks[0].unsqueeze(0)

        # Get final dimensions
        batch_size, screen_height, screen_width = mask.shape
            
        kwargs['mask'] = mask
        return super().apply_effect(
            audio, frame_rate, screen_width, screen_height,
            strength, feature_param, feature_mode, feature_threshold,
            opt_feature, **kwargs
        )

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

        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_contour_area]
        valid_contours.sort(key=cv2.contourArea, reverse=True)
        valid_contours = valid_contours[:max_contours]
        if not valid_contours: return image

        if distribute_by == 'area':
            weights = [cv2.contourArea(c) for c in valid_contours]
        elif distribute_by == 'perimeter':
            weights = [cv2.arcLength(c, True) for c in valid_contours]
        else:
            weights = [1] * len(valid_contours)

        total_weight = sum(weights)
        weights = [w / total_weight for w in weights] if total_weight > 0 else [1/len(weights)]*len(weights)

        data = processor.spectrum
        
        def process_contour(contour, start_idx, end_idx, direction_multiplier=1.0):
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
                    bar_h = amplitude * bar_length * direction_multiplier
                    x2, y2 = int(x1 + normals_x[i] * bar_h), int(y1 + normals_y[i] * bar_h)
                    
                    # Determine color
                    if color_mode == "spectrum" and item_freqs is not None:
                        color = get_color_for_frequency(item_freqs[start_idx + i], color_shift, saturation, brightness)
                    elif color_mode == "custom":
                        custom_hex = kwargs.get("custom_color", "#FFFFFF").lstrip('#')
                        color = tuple(int(custom_hex[i:i+2], 16)/255.0 for i in (0, 2, 4))
                    else:
                        color = (1.0, 1.0, 1.0)
                        
                    cv2.line(image, (x1, y1), (x2, y2), color, thickness=line_width)
            else:
                pts = np.column_stack([
                    x_coords + normals_x * contour_data * bar_length * direction_multiplier,
                    y_coords + normals_y * contour_data * bar_length * direction_multiplier
                ]).astype(np.int16) # Use int16 for coordinates
                
                if color_mode == "spectrum" and item_freqs is not None:
                    # Draw segments with colors
                    for i in range(len(pts) - 1):
                        color = get_color_for_frequency(item_freqs[start_idx + i], color_shift, saturation, brightness)
                        cv2.line(image, tuple(pts[i]), tuple(pts[i+1]), color, line_width)
                    # Close the loop if needed (contour is closed)
                    color = get_color_for_frequency(item_freqs[end_idx - 1], color_shift, saturation, brightness)
                    cv2.line(image, tuple(pts[-1]), tuple(pts[0]), color, line_width)
                elif color_mode == "custom":
                    custom_hex = kwargs.get("custom_color", "#FFFFFF").lstrip('#')
                    color = tuple(int(custom_hex[i:i+2], 16)/255.0 for i in (0, 2, 4))
                    cv2.polylines(image, [pts.astype(np.int32)], True, color, thickness=line_width)
                else:
                    cv2.polylines(image, [pts.astype(np.int32)], True, (1.0, 1.0, 1.0), thickness=line_width)

        start_idx = 0
        total_pts = len(data)
        for i, (cnt, w) in enumerate(zip(valid_contours, weights)):
            num_pts = int(round(total_pts * w)) if i < len(valid_contours)-1 else total_pts - start_idx
            end_idx = start_idx + num_pts
            if direction == "both":
                process_contour(cnt, start_idx, end_idx, 0.5)
                process_contour(cnt, start_idx, end_idx, -0.5)
            else:
                mul = -1.0 if direction == "inward" else 1.0
                process_contour(cnt, start_idx, end_idx, mul)
            start_idx = end_idx

        return image.copy()

NODE_CLASS_MAPPINGS = {
    "ScromfyFlexAudioVisualizerContour": ScromfyFlexAudioVisualizerContourNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScromfyFlexAudioVisualizerContour": "Flex Audio Visualizer Contour (Scromfy)",
}
