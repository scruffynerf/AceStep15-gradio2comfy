import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import random
import math
import pyconify
from .includes.emoji_utils import get_emoji_icon_names, load_icon_as_image, pil_to_tensor, tensor_to_pil

class ScromfyEmojiSpinnerNode:
    @classmethod
    def INPUT_TYPES(cls):
        collections = pyconify.collections()
        # Find emoji sets again for the dropdown
        emoji_sets = sorted([p for p in collections.keys() if 'emoji' in p or p in ['noto', 'twemoji', 'fluent']])
        if not emoji_sets:
            emoji_sets = ["fluent-emoji-flat"]
            
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "icon_set": (emoji_sets, {"default": "fluent-emoji-flat"}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 512, "min": 128, "max": 1024, "step": 64}),
                "fps": ("INT", {"default": 30, "min": 10, "max": 60}),
                "spin_duration": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.5}),
                "stop_stagger": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.1}),
                "render_size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "slot_icon_size": ("INT", {"default": 128, "min": 64, "max": 512, "step": 8}),
                "reel_padding": ("INT", {"default": 10, "min": 0, "max": 100, "step": 2}),
                "render_mode": (["color", "white_solid", "white_outline", "white_solid_black_outline"], {"default": "white_outline"}),
                "bw_stroke_width": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 2.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING", "STRING", "STRING", "MASK", "MASK", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("IMAGE", "emoji_1_image", "emoji_2_image", "emoji_3_image", "emoji_1_name", "emoji_2_name", "emoji_3_name", "emoji_1_mask", "emoji_2_mask", "emoji_3_mask", "combined_mask", "combined_description")
    FUNCTION = "spin"
    CATEGORY = "Scromfy/Ace-Step/Visualizers"

    def spin(self, seed, icon_set, width, height, fps, spin_duration, stop_stagger, render_size, slot_icon_size, reel_padding, render_mode="white_outline", bw_stroke_width=0.3):
        rng = random.Random(seed)
        
        # 1. Fetch icons
        all_names = get_emoji_icon_names(icon_set, count=25, seed=seed)
        if not all_names:
            return (torch.zeros((1, height, width, 3)), torch.zeros((1, 64, 64, 3)), torch.zeros((1, 64, 64, 3)), torch.zeros((1, 64, 64, 3)), "", "", "", torch.zeros((1, 64, 64)), torch.zeros((1, 64, 64)), torch.zeros((1, 64, 64)), torch.zeros((1, height, width)), "Error fetching icons")

        # 2. Assign to 3 wheels
        wheel_count = 3
        wheels_icons = []
        icons_per_wheel = 25
        for i in range(wheel_count):
            # Shuffle for each wheel to have different sequences but potentially same icons
            wheel_icons = list(all_names)
            rng.shuffle(wheel_icons)
            wheels_icons.append(wheel_icons)

        # 3. Calculate animation parameters
        num_frames = int(fps * (spin_duration + (wheel_count - 1) * stop_stagger))
        wheel_width = (width - (wheel_count - 1) * reel_padding) // wheel_count
        wheel_centers_x = [i * (wheel_width + reel_padding) + wheel_width // 2 for i in range(wheel_count)]
        
        target_indices = [rng.randint(0, icons_per_wheel - 1) for i in range(wheel_count)]
        target_names = [wheels_icons[i][target_indices[i]] for i in range(wheel_count)]
        
        frames = []
        
        # Prepare target images and masks for output (High quality)
        target_pil_images = [load_icon_as_image(f"{icon_set}:{name}", size=render_size, render_mode=render_mode, stroke_width=bw_stroke_width) for name in target_names]
        e1_img, m1 = pil_to_tensor(target_pil_images[0])
        e2_img, m2 = pil_to_tensor(target_pil_images[1])
        e3_img, m3 = pil_to_tensor(target_pil_images[2])
        
        # Create combined mask (result frame)
        # For non-color modes, background is black. For color, we use black but in RGBA.
        combined_result_pil = Image.new("RGBA", (width, height), (0, 0, 0, 255))
        for i, img in enumerate(target_pil_images):
            fit_img = img.copy()
            # Constrain to wheel_width or slot_icon_size
            target_fit_size = min(wheel_width, height, slot_icon_size)
            if fit_img.width > target_fit_size or fit_img.height > target_fit_size:
                fit_img.thumbnail((target_fit_size, target_fit_size), Image.LANCZOS)
                
            x = wheel_centers_x[i] - fit_img.width // 2
            y = height // 2 - fit_img.height // 2
            combined_result_pil.paste(fit_img, (x, y), fit_img)
        
        _, combined_mask = pil_to_tensor(combined_result_pil)

        # Cache for slot icons (resized ones)
        slot_icons_cache = {}

        # Rendering frames
        for f in range(num_frames):
            time_at_f = f / fps
            frame_img = Image.new("RGB", (width, height), (20, 20, 20)) # Dark background
            draw = ImageDraw.Draw(frame_img)
            
            # Draw dividers
            for i in range(1, wheel_count):
                x_start = i * (wheel_width + reel_padding) - reel_padding
                x_end = x_start + reel_padding
                draw.rectangle([x_start, 0, x_end, height], fill=(40, 40, 40))

            for i in range(wheel_count):
                wheel_duration = spin_duration + i * stop_stagger
                t = min(time_at_f / wheel_duration, 1.0)
                
                total_spins = 4 + i
                total_items_to_move = total_spins * icons_per_wheel + target_indices[i]
                eased_t = 1 - math.pow(1 - t, 3)
                current_pos_items = eased_t * total_items_to_move
                
                center_item_idx_float = current_pos_items
                show_range = height // (slot_icon_size if slot_icon_size > 0 else 1) + 2
                
                for offset in range(-show_range, show_range + 1):
                    item_idx = int(math.floor(center_item_idx_float)) + offset
                    y_offset = (item_idx - center_item_idx_float) * slot_icon_size
                    
                    actual_idx = item_idx % icons_per_wheel
                    icon_name = wheels_icons[i][actual_idx]
                    
                    icon_key = f"{icon_name}_{slot_icon_size}_{render_mode}_{bw_stroke_width}"
                    if icon_key not in slot_icons_cache:
                        full_img = load_icon_as_image(f"{icon_set}:{icon_name}", size=render_size, render_mode=render_mode, stroke_width=bw_stroke_width)
                        if slot_icon_size != render_size:
                            slot_icons_cache[icon_key] = full_img.resize((slot_icon_size, slot_icon_size), Image.LANCZOS)
                        else:
                            slot_icons_cache[icon_key] = full_img
                    
                    icon_img = slot_icons_cache[icon_key]
                    x = wheel_centers_x[i] - icon_img.width // 2
                    y = height // 2 + y_offset - icon_img.height // 2
                    
                    if y + icon_img.height > 0 and y < height:
                        frame_img.paste(icon_img, (int(x), int(y)), icon_img)

            img_tensor, _ = pil_to_tensor(frame_img)
            frames.append(img_tensor)

        final_images = torch.cat(frames, dim=0)
        desc = f"{target_names[0]}, {target_names[1]}, {target_names[2]}"
        
        return (final_images, e1_img, e2_img, e3_img, target_names[0], target_names[1], target_names[2], m1, m2, m3, combined_mask, desc)

NODE_CLASS_MAPPINGS = {
    "ScromfyEmojiSpinner": ScromfyEmojiSpinnerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScromfyEmojiSpinner": "Emoji Spinner (Scromfy)",
}
