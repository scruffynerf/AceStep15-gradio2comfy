import torch
import numpy as np
import os
import re
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2
from typing import List, Dict

class ScromfyLyricsOverlay:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "lrc_text": ("STRING", {"multiline": True, "default": ""}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
                "font_size": ("INT", {"default": 24, "min": 10, "max": 200}),
                "highlight_color": ("STRING", {"default": "#34d399"}), # Vibrant green
                "normal_color": ("STRING", {"default": "#9ca3af"}),    # Dimmed gray
                "background_alpha": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blur_radius": ("INT", {"default": 10, "min": 0, "max": 50}),
                "y_position": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_lines": ("INT", {"default": 5, "min": 1, "max": 20}),
                "line_spacing": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1}),
            },
            "optional": {
                "font_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "overlay_lyrics"
    CATEGORY = "Scromfy/Ace-Step/lyrics"

    def parse_lrc(self, lrc_text: str) -> List[Dict]:
        lyrics = []
        # Support both [mm:ss.xx] and [mm:ss:xx]
        pattern = r"\[(\d+):(\d+\.?\d*)\](.*)"
        
        for line in lrc_text.splitlines():
            line = line.strip()
            if not line: continue
            
            match = re.search(pattern, line)
            if match:
                minutes = int(match.group(1))
                seconds = float(match.group(2))
                text = match.group(3).strip()
                timestamp = minutes * 60 + seconds
                lyrics.append({"time": timestamp, "text": text})
        
        # Sort by time
        lyrics.sort(key=lambda x: x["time"])
        return lyrics

    def parse_srt(self, srt_text: str) -> List[Dict]:
        lyrics = []
        blocks = re.split(r'\n\s*\n', srt_text.strip())
        for block in blocks:
            lines = block.splitlines()
            if len(lines) >= 3:
                time_match = re.search(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', lines[1])
                if time_match:
                    start_str = time_match.group(1).replace(',', '.')
                    h, m, s = start_str.split(':')
                    timestamp = int(h) * 3600 + int(m) * 60 + float(s)
                    text = " ".join(lines[2:]).strip()
                    lyrics.append({"time": timestamp, "text": text})
        lyrics.sort(key=lambda x: x["time"])
        return lyrics

    def overlay_lyrics(self, images, lrc_text, fps, font_size, highlight_color, normal_color, 
                       background_alpha, blur_radius, y_position, max_lines, line_spacing, font_path=""):
        
        import gc
        import psutil
        from comfy.utils import ProgressBar
        
        process = psutil.Process(os.getpid())
        def log_mem(frame):
            mem = process.memory_info().rss / 1024 / 1024
            print(f"[LyricOverlay] Frame {frame}/{batch_size}, Mem: {mem:.2f}MB")

        # Parse lyrics (detect mode)
        if "-->" in lrc_text:
            lyrics = self.parse_srt(lrc_text)
        else:
            lyrics = self.parse_lrc(lrc_text)
            
        if not lyrics:
            return (images,)

        # Load font once
        try:
            if font_path and os.path.exists(font_path):
                f_reg = ImageFont.truetype(font_path, font_size)
                f_bold = ImageFont.truetype(font_path, int(font_size * 1.3))
            else:
                # Use local Roboto fonts
                base_font_dir = os.path.join(os.path.dirname(__file__), "includes", "fonts")
                roboto_reg = os.path.join(base_font_dir, "Roboto-Regular.ttf")
                roboto_bold = os.path.join(base_font_dir, "Roboto-Bold.ttf")
                
                if os.path.exists(roboto_reg):
                    f_reg = ImageFont.truetype(roboto_reg, font_size)
                    f_bold = ImageFont.truetype(roboto_bold, int(font_size * 1.3))
                else:
                    f_reg = ImageFont.load_default()
                    f_bold = f_reg
        except:
            f_reg = ImageFont.load_default()
            f_bold = f_reg

        batch_size, height, width, channels = images.shape
        
        # MEMORY: Pre-allocate output tensor. 
        # If this line alone crashes, we can't do anything without in-place modification.
        try:
            out_tensor = torch.zeros_like(images)
        except RuntimeError:
            print("[LyricOverlay] OOM triggered at allocation. Falling back to in-place (dangerous) or smaller chunks.")
            raise

        pbar = ProgressBar(batch_size)

        # Pre-calculate colors
        def hex_to_rgb(hex_str):
            hex_str = hex_str.lstrip('#')
            return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
        
        try:
            high_rgb = hex_to_rgb(highlight_color)
            norm_rgb = hex_to_rgb(normal_color)
        except:
            high_rgb = (52, 211, 153); norm_rgb = (156, 163, 175)

        # REUSE: Pre-allocate a scratch PIL image to avoid allocations in loop
        # We only need it as big as the box, but let's just make it the box's max size
        max_box_w, max_box_h = int(width * 0.8), int(font_size * line_spacing * (max_lines + 2))
        scratch_overlay = Image.new("RGBA", (max_box_w, max_box_h), (0, 0, 0, 0))
        scratch_draw = ImageDraw.Draw(scratch_overlay)

        for i in range(batch_size):
            if i % 50 == 0:
                gc.collect() # Trigger GC every 50 frames
                # log_mem(i)

            time = i / fps
            current_idx = -1
            for j, lyric in enumerate(lyrics):
                if time >= lyric["time"]: current_idx = j
                else: break
            
            # 1. Access input and prepare frame
            # Use .clone() to decouple from the batch immediately, allowing input frames to be swapped out if needed
            frame_np = (images[i].cpu().numpy() * 255).astype(np.uint8)
            
            if current_idx != -1:
                start_l = max(0, current_idx - max_lines // 2)
                end_l = min(len(lyrics), start_l + max_lines)
                if end_l - start_l < max_lines: start_l = max(0, end_l - max_lines)
                
                lines_to_draw = []
                for j in range(start_l, end_l):
                    lines_to_draw.append({"txt": lyrics[j]["text"], "active": (j == current_idx), "off": j-current_idx})

                if lines_to_draw:
                    center_y, line_h = int(height * y_position), int(font_size * line_spacing)
                    total_h = line_h * len(lines_to_draw)
                    
                    b_top, b_left = max(0, center_y - total_h // 2 - 20), int(width*0.1)
                    b_bot, b_right = min(height, center_y + total_h // 2 + 20), int(width*0.9)
                    b_w, b_h = b_right - b_left, b_bot - b_top

                    if b_w > 0 and b_h > 0:
                        # 2. Fast In-Place Blur/Dim via CV2
                        sub = frame_np[b_top:b_bot, b_left:b_right]
                        if blur_radius > 0:
                            k = blur_radius if blur_radius % 2 == 1 else blur_radius + 1
                            cv2.GaussianBlur(sub, (k, k), 0, dst=sub)
                        if background_alpha > 0:
                            cv2.addWeighted(sub, 1.0 - background_alpha, np.zeros_like(sub), background_alpha, 0, dst=sub)
                        
                        # 3. Clean and reuse text overlay
                        scratch_draw.rectangle([0, 0, max_box_w, max_box_h], fill=(0,0,0,0))
                        
                        for item in lines_to_draw:
                            f = f_bold if item["active"] else f_reg
                            c = (*(high_rgb if item["active"] else norm_rgb), 255 if item["active"] else 180)
                            
                            bbox = scratch_draw.textbbox((0, 0), item["txt"], font=f)
                            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
                            tx = (b_w - tw) // 2
                            ty = (b_h // 2) + (item["off"] * line_h) - th // 2
                            if ty + th < b_h and ty >= 0:
                                scratch_draw.text((tx, ty), item["txt"], font=f, fill=c)
                        
                        # 4. Composite small mask back using numpy (avoid PIL full-frame convert)
                        text_np = np.array(scratch_overlay.crop((0, 0, b_w, b_h)))
                        t_rgb = text_np[:, :, :3]
                        t_alpha = text_np[:, :, 3:].astype(np.float32) / 255.0
                        
                        # In-place blend
                        frame_np[b_top:b_bot, b_left:b_right] = (t_rgb * t_alpha + sub * (1.0 - t_alpha)).astype(np.uint8)

            # 5. Save and Cleanup
            out_tensor[i] = torch.from_numpy(frame_np.astype(np.float32) / 255.0)
            pbar.update(1)
            
            # Explicitly free frame-specific memory
            del frame_np
            
        return (out_tensor,)

NODE_CLASS_MAPPINGS = {
    "ScromfyLyricsOverlay": ScromfyLyricsOverlay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScromfyLyricsOverlay": "Scrolling Lyrics Overlay",
}
