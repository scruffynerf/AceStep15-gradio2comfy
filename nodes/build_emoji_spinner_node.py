import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import random
import math
import pyconify
from .includes.emoji_utils import get_emoji_icon_names, load_icon_as_image, pil_to_tensor

# Try to import the full list of collections from the user-editable file
try:
    from .includes.icon_collections import ICON_COLLECTIONS
except ImportError:
    ICON_COLLECTIONS = []

class ScromfyEmojiSpinnerNode:
    @classmethod
    def INPUT_TYPES(cls):
        # Curated fallback list
        fallback_collections = [
            "fluent-emoji-flat", "fluent-emoji", "fluent-color", "fluent-emoji-high-contrast",
            "noto", "noto-v1", "twemoji", "openmoji", "emojione", "emojione-monotone",
            "streamline-emojis", "fxemoji", "blob-emoji", "material-symbols", "material-symbols-light"
        ]
        
        # Merge external list if available, prioritizing the user's list but ensuring it's not empty
        collection_list = ICON_COLLECTIONS if ICON_COLLECTIONS else fallback_collections
        
        # Ensure "local" is in the list and unique
        if "local" not in collection_list:
            collection_list = ["local"] + list(collection_list) 
        
        return {
            "required": {
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xffffffffffffffff}),
                "icon_set": (collection_list, {"default": "fluent-emoji-flat"}),
                "width": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 256, "min": 64, "max": 2048, "step": 64}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
                "spin_duration": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 10.0, "step": 0.1}),
                "stop_stagger": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.05}),
                "render_size": ("INT", {"default": 256, "min": 256, "max": 2048, "step": 64}),
                "slot_icon_size": ("INT", {"default": 128, "min": 64, "max": 512, "step": 8}),
                "reel_padding": ("INT", {"default": 10, "min": 0, "max": 100, "step": 2}),
                "reel_inner_padding": ("INT", {"default": 15, "min": 0, "max": 100, "step": 2}),
                "reel_top_padding": ("INT", {"default": 15, "min": 0, "max": 100, "step": 2}),
                "render_mode": (["color", "white_solid", "white_outline", "white_solid_black_outline"], {"default": "white_outline"}),
                "bw_stroke_width": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 2.0, "step": 0.1}),
                "sample_rate": ("INT", {"default": 44100, "min": 8000, "max": 192000, "step": 100}),
                "sound": (["silence"], {"default": "silence"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING", "STRING", "STRING", "MASK", "MASK", "MASK", "MASK", "STRING", "AUDIO", "FLOAT")
    RETURN_NAMES = ("IMAGE", "emoji_1_image", "emoji_2_image", "emoji_3_image", "emoji_1_name", "emoji_2_name", "emoji_3_name", "emoji_1_mask", "emoji_2_mask", "emoji_3_mask", "combined_mask", "combined_description", "AUDIO", "fps")
    FUNCTION = "spin"
    CATEGORY = "Scromfy/Ace-Step/Misc"

    def spin(self, seed, icon_set, width, height, fps, spin_duration, stop_stagger, render_size, slot_icon_size, reel_padding, reel_inner_padding=15, reel_top_padding=15, render_mode="white_outline", bw_stroke_width=0.3, sample_rate=44100, sound="silence"):
        rng = random.Random(seed)
        
        # 1. Fetch icons
        icon_refs = get_emoji_icon_names(icon_set, count=25, seed=seed)
        if not icon_refs:
            return (torch.zeros((1, 128, width, 3)), torch.zeros((1, 64, 64, 3)), torch.zeros((1, 64, 64, 3)), torch.zeros((1, 64, 64, 3)), "", "", "", torch.zeros((1, 64, 64)), torch.zeros((1, 64, 64)), torch.zeros((1, 64, 64)), torch.zeros((1, 128, width)), "Error fetching icons")

        # 2. Assign to 3 wheels
        wheel_count = 3
        wheels_icons = []
        icons_per_wheel = len(icon_refs)
        for i in range(wheel_count):
            wheel_icons = list(icon_refs)
            rng.shuffle(wheel_icons)
            wheels_icons.append(wheel_icons)

        # 3. Layout
        reel_width = slot_icon_size + 2 * reel_inner_padding
        viewport_height = slot_icon_size + 2 * reel_top_padding
        total_content_width = wheel_count * reel_width + (wheel_count - 1) * reel_padding
        start_x = (width - total_content_width) // 2
        wheel_centers_x = [start_x + i * (reel_width + reel_padding) + reel_width // 2 for i in range(wheel_count)]
        
        target_indices = [rng.randint(0, icons_per_wheel - 1) for i in range(wheel_count)]
        target_names = [wheels_icons[i][target_indices[i]] for i in range(wheel_count)]
        
        # Randomize spin direction (1 = moves UP, -1 = moves DOWN)
        spin_directions = [rng.choice([-1, 1]) for _ in range(wheel_count)]
        
        frames = []
        
        # Prepare target images and masks for output (High quality)
        target_pil_images = [load_icon_as_image(ref, size=render_size, render_mode=render_mode, stroke_width=bw_stroke_width) for ref in target_names]
        e1_img, m1 = pil_to_tensor(target_pil_images[0], extract_luminance_mask=(render_mode != "color"))
        e2_img, m2 = pil_to_tensor(target_pil_images[1], extract_luminance_mask=(render_mode != "color"))
        e3_img, m3 = pil_to_tensor(target_pil_images[2], extract_luminance_mask=(render_mode != "color"))
        
        # Create combined mask (pure black background)
        combined_result_pil = Image.new("RGBA", (width, viewport_height), (0, 0, 0, 0))
        for i, img in enumerate(target_pil_images):
            fit_img = img.copy()
            if fit_img.width != slot_icon_size or fit_img.height != slot_icon_size:
                fit_img.thumbnail((slot_icon_size, slot_icon_size), Image.LANCZOS)
            
            # Ensure data is realized before pasting to prevent race conditions or blank areas
            fit_img.load()
                
            x = wheel_centers_x[i] - fit_img.width // 2
            y = viewport_height // 2 - fit_img.height // 2
            combined_result_pil.paste(fit_img, (int(x), int(y)), fit_img)
        
        _, combined_mask = pil_to_tensor(combined_result_pil, extract_luminance_mask=(render_mode != "color"))

        # Cache for slot icons (resized ones)
        slot_icons_cache = {}

        # Rendering frames
        num_frames = int(fps * (spin_duration + (wheel_count - 1) * stop_stagger))
        # Total vertical step between icons in the reel
        total_item_stride = slot_icon_size + 2 * reel_top_padding
        
        for f in range(num_frames):
            time_at_f = f / fps
            # Pure black background (0,0,0,255) to avoid gray lines
            frame_img = Image.new("RGBA", (width, viewport_height), (0, 0, 0, 255)) 
            draw = ImageDraw.Draw(frame_img)
            
            # Draw dividers only if reel_padding > 0
            if reel_padding > 0:
                for i in range(1, wheel_count):
                    x_mid = start_x + i * (reel_width + reel_padding) - reel_padding // 2
                    # Pure black divider to match background
                    draw.line([x_mid, 0, x_mid, viewport_height], fill=(0, 0, 0, 255), width=reel_padding)

            for i in range(wheel_count):
                wheel_duration = spin_duration + i * stop_stagger
                t = min(time_at_f / wheel_duration, 1.0)
                
                # Blank Start Logic:
                # We want to start from total silence/blank. 
                # We'll use a virtual 'starting offset'. 
                # If we start current_pos_items at a very large negative or positive number
                # outside the icon range, we get a blank start.
                # However, it's simpler to just make early frames blank if t is near 0 
                # OR start the item index at a value that isn't icon-aligned.
                
                total_spins = 4 + i
                total_items_to_move = total_spins * icons_per_wheel + target_indices[i]
                eased_t = 1 - math.pow(1 - t, 3)
                
                # We start from a 'virtual' position that is empty.
                # Offset by 2 spins so no icon is centered at t=0
                start_offset = 2.0 
                current_pos_items = (eased_t * (total_items_to_move + start_offset)) - start_offset
                
                center_item_idx_float = current_pos_items
                show_range = 2 
                direction = spin_directions[i]
                
                for offset in range(-show_range, show_range + 1):
                    item_idx = int(math.floor(center_item_idx_float)) + offset
                    y_offset = direction * (item_idx - center_item_idx_float) * total_item_stride
                    
                    # If item_idx is negative (before our spin starts), we show nothing
                    if item_idx < 0:
                        continue
                        
                    actual_idx = item_idx % icons_per_wheel
                    icon_ref = wheels_icons[i][actual_idx]
                    
                    icon_key = f"{icon_ref}_{slot_icon_size}_{render_mode}_{bw_stroke_width}"
                    if icon_key not in slot_icons_cache:
                        full_img = load_icon_as_image(icon_ref, size=render_size, render_mode=render_mode, stroke_width=bw_stroke_width)
                        if slot_icon_size != render_size:
                            slot_icons_cache[icon_key] = full_img.resize((slot_icon_size, slot_icon_size), Image.LANCZOS)
                        else:
                            slot_icons_cache[icon_key] = full_img
                        
                        # Realize the image data
                        slot_icons_cache[icon_key].load()
                    
                    icon_img = slot_icons_cache[icon_key]
                    x = wheel_centers_x[i] - icon_img.width // 2
                    y = viewport_height // 2 + y_offset - icon_img.height // 2
                    
                    if y + icon_img.height > 0 and y < viewport_height:
                        frame_img.paste(icon_img, (int(x), int(y)), icon_img)

            img_tensor, _ = pil_to_tensor(frame_img)
            frames.append(img_tensor)

        final_images = torch.cat(frames, dim=0)
        
        # Strip prefixes for output names
        clean_names = [name.split(":")[-1] if ":" in name else name for name in target_names]
        desc = f"{clean_names[0]}, {clean_names[1]}, {clean_names[2]}"
        
        # Generate audio based on sound selection
        # (Currently only "silence" is supported, creating placeholder for future expansion)
        num_samples = int((len(frames) / fps) * sample_rate)
        audio_waveform = torch.zeros((1, 2, num_samples))
        audio = {"waveform": audio_waveform, "sample_rate": sample_rate}
        
        return (final_images, e1_img, e2_img, e3_img, clean_names[0], clean_names[1], clean_names[2], m1, m2, m3, combined_mask, desc, audio, float(fps))

NODE_CLASS_MAPPINGS = {
    "ScromfyEmojiSpinner": ScromfyEmojiSpinnerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScromfyEmojiSpinner": "Emoji Spinner (Scromfy)",
}
