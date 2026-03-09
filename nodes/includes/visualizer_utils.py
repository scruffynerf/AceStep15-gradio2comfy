import numpy as np
import cv2
import torch
import os
import re
import gc
import json
from abc import abstractmethod
from PIL import Image, ImageDraw, ImageFont
from .flex_utils import FlexBase

class BaseAudioProcessor:
    def __init__(self, audio, num_frames, height, width, frame_rate):
        """
        Base class to process audio data.
        """
        # Convert waveform tensor to mono numpy array
        self.audio = audio['waveform'].squeeze(0).mean(axis=0).cpu().numpy()
        self.sample_rate = audio['sample_rate']
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.frame_rate = frame_rate

        self.audio_duration = len(self.audio) / self.sample_rate
        self.frame_duration = 1 / self.frame_rate if self.frame_rate > 0 else self.audio_duration / self.num_frames

        self.spectrum = None  # Initialize spectrum
        self.current_frame = 0

    def _normalize(self, data):
        return (data - data.min()) / (data.max() - data.min())

    def _enhance_contrast(self, data, power=0.3):
        return np.power(data, power)

    def _resize(self, data, new_width, new_height):
        return cv2.resize(data, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    def _get_audio_frame(self, frame_index):
        start_time = frame_index * self.frame_duration
        end_time = (frame_index + 1) * self.frame_duration
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        return self.audio[start_sample:end_sample]

    def compute_spectrum(self, frame_index, fft_size, min_frequency, max_frequency):
        audio_frame = self._get_audio_frame(frame_index)
        if len(audio_frame) < fft_size:
            audio_frame = np.pad(audio_frame, (0, fft_size - len(audio_frame)), mode='constant')

        # Apply window function
        window = np.hanning(len(audio_frame))
        audio_frame = audio_frame * window

        # Compute FFT
        spectrum = np.abs(np.fft.rfft(audio_frame, n=fft_size))

        # Extract desired frequency range
        freqs = np.fft.rfftfreq(fft_size, d=1.0 / self.sample_rate)
        freq_indices = np.where((freqs >= min_frequency) & (freqs <= max_frequency))[0]
        
        # Store active frequencies for color mapping
        self.active_frequencies = freqs[freq_indices]
        spectrum = spectrum[freq_indices]

        # Check if spectrum is not empty
        if spectrum.size > 0:
            # Apply logarithmic scaling
            spectrum = np.log1p(spectrum)

            # Apply low-frequency roll-off (fade in first 3 bins to mitigate noise/DC offset)
            if len(spectrum) > 3:
                spectrum[0] *= 0.1
                spectrum[1] *= 0.4
                spectrum[2] *= 0.7

            # Normalize with noise floor threshold
            max_spectrum = np.max(spectrum)
            noise_floor = 0.05 # Prevent tiny noise from being blown up
            if max_spectrum > noise_floor:
                spectrum = spectrum / max_spectrum
            else:
                spectrum = np.zeros_like(spectrum)
        else:
            # Return zeros if spectrum is empty
            spectrum = np.zeros(1)
            self.active_frequencies = np.array([min_frequency])

        return spectrum

    def update_spectrum(self, new_spectrum, smoothing):
        if self.spectrum is None or len(self.spectrum) != len(new_spectrum):
            self.spectrum = np.zeros(len(new_spectrum))

        # Apply smoothing
        self.spectrum = smoothing * self.spectrum + (1 - smoothing) * new_spectrum

import colorsys

def parse_color(color_input, fallback=(255, 255, 255), to_float=True):
    """
    Robustly parse various color input formats (hex string, list/tuple of ints/floats)
    returns (r, g, b) in [0.0, 1.0] if to_float=True, or [0, 255] if to_float=False.
    """
    r, g, b = 255, 255, 255
    try:
        if isinstance(color_input, str):
            c = color_input.lstrip('#')
            if len(c) == 3:
                c = ''.join([char*2 for char in c])
            if len(c) == 6:
                r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
        elif isinstance(color_input, (list, tuple)):
            if len(color_input) >= 3:
                # Check if floats [0,1] or ints [0,255]
                if all(isinstance(val, (float, int)) for val in color_input[:3]) and any(isinstance(val, float) and val <= 1.0 for val in color_input[:3]):
                    r, g, b = [int(val * 255) for val in color_input[:3]]
                else:
                    r, g, b = [int(val) for val in color_input[:3]]
    except Exception:
        r, g, b = fallback

    if to_float:
        return (r/255.0, g/255.0, b/255.0)
    return (r, g, b)

def get_color_for_frequency(freq, shift=0.0, saturation=1.0, brightness=1.0):
    """
    Maps a frequency to a color in the HSL spectrum.
    Uses log scale so octaves match.
    """
    if freq <= 0:
        # Default to neutral white/light gray if frequency is zero (common in waveform mode)
        return (1.0, 1.0, 1.0)
    
    # log2(freq) gives us a linear scale where +1 is one octave
    hue = (np.log2(freq) + shift) % 1.0
    
    # Convert HLS (Hue, Lightness, Saturation) to RGB
    r, g, b = colorsys.hls_to_rgb(hue, brightness * 0.5, saturation)
    return (r, g, b)

class LyricRenderer:
    def __init__(self, lrc_text, width, height, font_size, highlight_color, normal_color, 
                 background_alpha, blur_radius, active_blur_radius, y_position, max_lines, line_spacing, font_name="NotoSans-Regular.ttf"):
        self.width = width
        self.height = height
        self.font_size = font_size
        self.background_alpha = background_alpha
        self.blur_radius = blur_radius
        self.active_blur_radius = active_blur_radius
        self.y_position = y_position
        self.max_lines = max_lines
        self.line_spacing = line_spacing
        self.font_name = font_name

        # Parse lyrics
        if "-->" in lrc_text:
            self.lyrics = self._parse_srt(lrc_text)
        else:
            self.lyrics = self._parse_lrc(lrc_text)

        # Pre-calculate colors
        self.high_rgb = parse_color(highlight_color, fallback=(52, 211, 153), to_float=False)
        self.norm_rgb = parse_color(normal_color, fallback=(156, 163, 175), to_float=False)

        # Load font from root /fonts directory
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        base_font_dir = os.path.join(base_dir, "fonts")
        
        def try_load_font(name, size):
            # Try absolute path first, then relative to base_font_dir
            paths = [name]
            if base_font_dir:
                paths.append(os.path.join(base_font_dir, name))
                # Fallback versions
                paths.append(os.path.join(base_font_dir, "NotoSans-Regular.ttf"))
                paths.append(os.path.join(base_font_dir, "Roboto-Regular.ttf"))
            
            for p in paths:
                if p and os.path.exists(p):
                    try:
                        return ImageFont.truetype(p, size)
                    except:
                        continue
            return ImageFont.load_default()

        self.f_reg = try_load_font(font_name, font_size)
        self.font_path = self.f_reg.path if hasattr(self.f_reg, 'path') else None
        
        # Determine bold version name
        bold_name = font_name.replace("-Regular", "-Bold").replace("Regular", "Bold")
        if bold_name == font_name: # Didn't find pattern
             bold_name = "NotoSans-Bold.ttf" # Hard fallback for bold
        
        self.f_bold = try_load_font(bold_name, int(font_size * 1.3))
        if self.f_bold == self.f_reg: # If failed to get distinct bold, use reg
             self.f_bold = self.f_reg

        # Pre-allocate scratch buffer
        self.max_box_w = int(width * 0.8)
        self.max_box_h = int(font_size * line_spacing * (max_lines + 2))
        self.scratch_overlay = Image.new("RGBA", (self.max_box_w, self.max_box_h), (0, 0, 0, 0))
        self.scratch_draw = ImageDraw.Draw(self.scratch_overlay)

    def _parse_lrc(self, text):
        lyrics = []
        pattern = r"\[(\d+):(\d+\.?\d*)\](.*)"
        for line in text.splitlines():
            match = re.search(pattern, line.strip())
            if match:
                timestamp = int(match.group(1)) * 60 + float(match.group(2))
                lyrics.append({"time": timestamp, "text": match.group(3).strip()})
        lyrics.sort(key=lambda x: x["time"])
        return lyrics

    def _parse_srt(self, text):
        lyrics = []
        blocks = re.split(r'\n\s*\n', text.strip())
        for block in blocks:
            lines = block.splitlines()
            if len(lines) >= 3:
                time_match = re.search(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', lines[1])
                if time_match:
                    start_str = time_match.group(1).replace(',', '.')
                    h, m, s = start_str.split(':')
                    timestamp = int(h) * 3600 + int(m) * 60 + float(s)
                    lyrics.append({"time": timestamp, "text": " ".join(lines[2:]).strip()})
        lyrics.sort(key=lambda x: x["time"])
        return lyrics

    def render(self, frame_np, time):
        if not self.lyrics:
            return frame_np

        current_idx = -1
        for j, lyric in enumerate(self.lyrics):
            if time >= lyric["time"]: current_idx = j
            else: break
        
        if current_idx != -1:
            start_l = max(0, current_idx - self.max_lines // 2)
            end_l = min(len(self.lyrics), start_l + self.max_lines)
            if end_l - start_l < self.max_lines: start_l = max(0, end_l - self.max_lines)
            
            lines_to_draw = []
            for j in range(start_l, end_l):
                lines_to_draw.append({"txt": self.lyrics[j]["text"], "active": (j == current_idx), "off": j-current_idx})

            if lines_to_draw:
                center_y = int(self.height * self.y_position)
                line_h = int(self.font_size * self.line_spacing)
                total_h = line_h * len(lines_to_draw)
                
                b_top = max(0, center_y - total_h // 2 - 20)
                b_left = int(self.width * 0.1)
                b_bot = min(self.height, center_y + total_h // 2 + 20)
                b_right = int(self.width * 0.9)
                b_w, b_h = b_right - b_left, b_bot - b_top

                if b_w > 0 and b_h > 0:
                    sub = frame_np[b_top:b_bot, b_left:b_right]
                    if self.blur_radius > 0:
                        k = self.blur_radius if self.blur_radius % 2 == 1 else self.blur_radius + 1
                        cv2.GaussianBlur(sub, (k, k), 0, dst=sub)
                    if self.background_alpha > 0:
                        cv2.addWeighted(sub, 1.0 - self.background_alpha, np.zeros_like(sub), self.background_alpha, 0, dst=sub)
                    
                    self.scratch_draw.rectangle([0, 0, self.max_box_w, self.max_box_h], fill=(0,0,0,0))
                    for item in lines_to_draw:
                        f = self.f_bold if item["active"] else self.f_reg
                        c = (*(self.high_rgb if item["active"] else self.norm_rgb), 255 if item["active"] else 180)
                        
                        # Auto-shrink font size if too wide
                        current_f = f
                        bbox = self.scratch_draw.textbbox((0, 0), item["txt"], font=current_f)
                        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
                        
                        if tw > b_w - 40: # 20px padding on each side
                            scale = (b_w - 40) / tw
                            new_size = int(current_f.size * scale)
                            try:
                                if self.font_path:
                                    current_f = ImageFont.truetype(self.font_path, new_size)
                                elif hasattr(current_f, 'path'):
                                    current_f = ImageFont.truetype(current_f.path, new_size)
                            except:
                                pass # Keep original font size / object if resizing fails
                            
                            bbox = self.scratch_draw.textbbox((0, 0), item["txt"], font=current_f)
                            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]

                        tx = (b_w - tw) // 2
                        ty = (b_h // 2) + (item["off"] * line_h) - th // 2
                        if ty + th < b_h and ty >= 0:
                            self.scratch_draw.text((tx, ty), item["txt"], font=current_f, fill=c)
                    
                    text_np = np.array(self.scratch_overlay.crop((0, 0, b_w, b_h)))
                    t_rgb = text_np[:, :, :3]
                    t_alpha = text_np[:, :, 3:].astype(np.float32) / 255.0

                    # Apply extra blur behind ONLY the active lyric line if requested
                    if self.active_blur_radius > 0:
                        active_ty = (b_h // 2) - line_h // 2
                        active_y_start = max(0, active_ty - 10)
                        active_y_end = min(b_h, active_ty + line_h + 10)
                        
                        active_sub = sub[active_y_start:active_y_end, :]
                        k_active = self.active_blur_radius if self.active_blur_radius % 2 == 1 else self.active_blur_radius + 1
                        cv2.GaussianBlur(active_sub, (k_active, k_active), 0, dst=active_sub)

                    # Final blend: Use np.clip to prevent any wrap-around during conversion
                    blended = (t_rgb.astype(np.float32) * t_alpha + sub.astype(np.float32) * (1.0 - t_alpha))
                    frame_np[b_top:b_bot, b_left:b_right] = np.clip(blended, 0, 255).astype(np.uint8)

        return frame_np

VIBRANT_COLORS = [
    # Reds & Oranges
    "#ff0000", # Pure Red
    "#ff1744", # Bright Red-Pink
    "#ff0033", # Crimson
    "#ff0066", # Raspberry
    "#ff3300", # Red-Orange
    "#ff3d00", # Deep Orange
    "#ff6600", # Vibrant Orange
    "#ff9100", # Bright Orange
    "#ffab40", # Light Peach-Orange
    
    # Yellows
    "#ffff00", # Pure Yellow
    "#ffea00", # Bright Yellow
    "#ffee58", # Soft Neon Yellow
    "#afff00", # Lime Yellow
    
    # Greens
    "#99ff00", # Spring Green
    "#7fff00", # Chartreuse
    "#76ff03", # Bright Lime
    "#66ff00", # Bright Green
    "#39ff14", # Neon Green
    "#33ff00", # High-Voltage Green
    "#00ff00", # Pure Green
    "#00ff33", # Mint Green
    "#00ff66", # Seafoam Green
    "#00ff99", # Jade Green
    "#00e676", # Spring Green 2
    "#1de9b6", # Teal-Green
    "#64ffda", # Turquoise
    "#00ffbf", # Electric Aquamarine
    
    # Cyans & Blues
    "#00ffff", # Pure Cyan
    "#00f5ff", # Neon Cyan
    "#00e5ff", # Sky Blue
    "#00ccff", # Vivid Blue
    "#00b0ff", # Azure Blue
    "#0099ff", # Deep Sky Blue
    "#0066ff", # Royal Blue
    "#2979ff", # Electric Blue
    "#0033ff", # Vivid Blue 2
    "#3300ff", # Indigo Blue
    "#0000ff", # Pure Blue
    
    # Purples & Magentas
    "#6600ff", # Purple Heart
    "#7c4dff", # Deep Purple
    "#bf00ff", # Electric Purple
    "#cc00ff", # Vivid Purple
    "#d500f9", # Magenta-Purple
    "#ff00ff", # Pure Magenta/Fuchsia
    "#ff00cc", # Deep Pink
    "#ff007f", # Rose
    "#ff1493", # Deep Pink 2
    "#f50057", # Pink-Red
    "#ff4081", # Hot Pink
]

class FlexAudioVisualizerBase(FlexBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_required = base_inputs.get("required", {})
        base_optional = base_inputs.get("optional", {})

        new_inputs = {
            "required": {
                "audio": ("AUDIO",),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 1.0}),
                "screen_width": ("INT", {"default": 768, "min": 100, "max": 1920, "step": 1}),
                "screen_height": ("INT", {"default": 464, "min": 100, "max": 1080, "step": 1}),
                "position_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "position_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "opt_video": ("IMAGE",),
                "lyric_settings": ("LYRIC_SETTINGS",),
                "visualizer_settings": ("VISUALIZER_SETTINGS",),
            }
        }

        required = {**new_inputs["required"], **base_required}
        optional = {**new_inputs["optional"], **base_optional}

        return {
            "required": required,
            "optional": optional
        }

    CATEGORY = "Scromfy/Ace-Step/Visualizers"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "MASK")
    RETURN_NAMES = ("IMAGE", "MASK", "SETTINGS", "SOURCE_MASK")
    FUNCTION = "apply_effect"

    @classmethod
    @abstractmethod
    def get_modifiable_params(cls):
        """Return a list of parameter names that can be modulated."""
        pass

    def get_point_count(self, kwargs):
        """Determine the number of points/bars to calculate for the spectrum."""
        # Check both common names, default to 64
        return kwargs.get('num_points', kwargs.get('num_bars', 64))

    def validate_param(self, param_name, param_value):
        valid_params = {
            'fft_size': lambda x: max(256, int(2 ** np.round(np.log2(x)))) if x > 0 else 256,
            'min_frequency': lambda x: max(20.0, min(x, 20000.0)),
            'max_frequency': lambda x: max(20.0, min(x, 20000.0)),
            'num_bars': lambda x: max(1, int(x)),
            'num_points': lambda x: max(3, int(x)),
            'smoothing': lambda x: np.clip(x, 0.0, 1.0),
            'rotation': lambda x: x % 360.0,
            'curvature': lambda x: max(0.0, x),
            'separation': lambda x: max(0.0, x),
            'max_height': lambda x: max(10.0, x),
            'min_height': lambda x: max(0.0, x),
            'position_x': lambda x: np.clip(x, 0.0, 1.0),
            'position_y': lambda x: np.clip(x, 0.0, 1.0),
            'reflect': lambda x: bool(x),
            'line_width': lambda x: max(1, int(x)),
            'radius': lambda x: max(1.0, x),
            'base_radius': lambda x: max(1.0, x),
            'amplitude_scale': lambda x: max(0.0, x),
            'color_shift': lambda x: x % 1.0,
            'saturation': lambda x: np.clip(x, 0.0, 1.0),
            'brightness': lambda x: np.clip(x, 0.0, 1.0),
        }
        if param_name in valid_params:
            return valid_params[param_name](param_value)
        else:
            return param_value

    def get_draw_color(self, i, num_pts, amplitude, x, y, cx, cy, max_dist, **kwargs):
        color_mode = kwargs.get('color_mode', 'white')
        color_shift = kwargs.get('color_shift', 0.0)
        saturation = kwargs.get('saturation', 1.0)
        brightness = kwargs.get('brightness', 1.0)
        item_freqs = kwargs.get('item_freqs')
        
        if color_mode == "white":
            return (1.0, 1.0, 1.0)
        elif color_mode == "spectrum" and item_freqs is not None:
            return get_color_for_frequency(item_freqs[i], color_shift, saturation, brightness)
        elif color_mode == "custom" or (color_mode == "spectrum" and item_freqs is None):
            return parse_color(kwargs.get("custom_color", "#00ffff"))
        else:
            import colorsys
            val = 0.0
            if color_mode == "amplitude":
                val = amplitude
            elif color_mode == "radial":
                val = np.sqrt((x - cx)**2 + (y - cy)**2) / max(1.0, max_dist)
            elif color_mode == "angular":
                val = (np.arctan2(y - cy, x - cx) / (2 * np.pi)) + 0.5
            elif color_mode == "path":
                val = i / max(1, num_pts)
            elif color_mode == "screen":
                sw = kwargs.get("screen_width", 512)
                sh = kwargs.get("screen_height", 512)
                val = (x / max(1, sw) + y / max(1, sh)) / 2.0
            
            hue = (val + color_shift) % 1.0
            # Normalize to 0-1 for colorsys
            return colorsys.hls_to_rgb(hue, brightness * 0.5, saturation)

    def rotate_image(self, image, angle):
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, M, (w, h))
        return rotated_image

    @abstractmethod
    def get_audio_data(self, processor: BaseAudioProcessor, frame_index, **kwargs):
        pass

    @abstractmethod
    def apply_effect_internal(self, processor: BaseAudioProcessor, **kwargs) -> np.ndarray:
        pass

    def process_audio_data(self, processor: BaseAudioProcessor, frame_index, visualization_feature, num_points, smoothing, fft_size, min_frequency, max_frequency):
        if visualization_feature == 'frequency':
            spectrum = processor.compute_spectrum(frame_index, fft_size, min_frequency, max_frequency)
            data = np.interp(
                np.linspace(0, len(spectrum), num_points, endpoint=False),
                np.arange(len(spectrum)),
                spectrum,
            )
        elif visualization_feature == 'waveform':
            audio_frame = processor._get_audio_frame(frame_index)
            if len(audio_frame) < 1:
                data = np.zeros(num_points)
            else:
                data = np.interp(
                    np.linspace(0, len(audio_frame), num_points, endpoint=False),
                    np.arange(len(audio_frame)),
                    audio_frame,
                )
                max_abs_value = np.max(np.abs(data))
                if max_abs_value != 0:
                    data = data / max_abs_value
                else:
                    data = np.zeros_like(data)
        else:
            data = np.zeros(num_points)

        if processor.spectrum is None or len(processor.spectrum) != len(data):
            processor.spectrum = np.zeros(len(data))
        processor.update_spectrum(data, smoothing)

        feature_value = np.mean(np.abs(processor.spectrum))
        if visualization_feature == 'frequency' and hasattr(processor, 'active_frequencies'):
            item_freqs = np.interp(
                np.linspace(0, len(processor.active_frequencies), num_points, endpoint=False),
                np.arange(len(processor.active_frequencies)),
                processor.active_frequencies,
            )
        else:
            item_freqs = np.zeros(num_points)
        return processor.spectrum.copy(), feature_value, item_freqs

    def apply_effect(self, audio, frame_rate, screen_width, screen_height, 
                     strength, feature_param, feature_mode, feature_threshold,
                     opt_feature=None, opt_video=None, source_mask=None, **kwargs):
        
        # Unpack visualizer settings if provided
        ext_settings = kwargs.get("visualizer_settings", {})
        if isinstance(ext_settings, dict):
            for k, v in ext_settings.items():
                kwargs[k] = v

        # Get random generator if randomize is active
        s_rng = None
        if kwargs.get("randomize", False):
            import random
            s_rng = random.Random(kwargs.get("seed", 0))

        # Base Randomization Logic (shared across nodes)
        if s_rng:
            # Randomize core features
            kwargs["visualization_method"] = s_rng.choice(["bar", "line"])
            kwargs["visualization_feature"] = s_rng.choice(["frequency", "waveform"])
            kwargs["color_mode"] = s_rng.choice(["spectrum", "custom", "amplitude", "path"])
            
            kwargs["rotation"] = s_rng.uniform(0.0, 360.0)
            kwargs["line_width"] = s_rng.randint(1, 10)
            kwargs["smoothing"] = 0.0 # Force low smoothing for responsive randoms
            
            kwargs["direction"] = s_rng.choice(["outward", "inward", "both"])
            kwargs["sequence_direction"] = s_rng.choice(["left", "right"])

        # Waveform Color Fix: Waveforms have no frequency data, so "spectrum" mode 
        # ends up white. We force "custom" mode for waveforms if spectrum was selected.
        if kwargs.get("visualization_feature") == "waveform" and kwargs.get("color_mode") == "spectrum":
            kwargs["color_mode"] = "custom"

        # Vibrant Color Randomization: If randomize is on and we are in custom mode, 
        # pick a guaranteed vibrant color.
        if s_rng and kwargs.get("color_mode") == "custom":
            kwargs["custom_color"] = s_rng.choice(VIBRANT_COLORS)

        # Unpack lyric settings if provided
        lyric_settings = kwargs.get("lyric_settings", {})
        if isinstance(lyric_settings, dict):
            for k, v in lyric_settings.items():
                if k.startswith("lyric_"):
                    kwargs[k] = v

        # Construct settings string for debugging
        settings_dict = {
            "strength": strength,
            "feature_param": feature_param,
            "feature_mode": feature_mode,
            "feature_threshold": feature_threshold,
        }
        # Add all relevant kwargs (excluding large objects and non-serializable types)
        for k, v in kwargs.items():
            if k in ["opt_video", "opt_feature", "lyric_settings", "background", "mask", "item_freqs"] or k.startswith("_"):
                continue
            
            if isinstance(v, (np.ndarray, torch.Tensor)):
                settings_dict[k] = f"<{v.__class__.__name__} {v.shape}>"
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], (np.ndarray, torch.Tensor)):
                settings_dict[k] = f"<List of {len(v)} {v[0].__class__.__name__}s>"
            else:
                settings_dict[k] = v
        
        try:
            settings_str = json.dumps(settings_dict, indent=4)
        except Exception as e:
            settings_str = f"Error serializing settings: {str(e)}\nKeys: {list(settings_dict.keys())}"

        audio_duration = len(audio['waveform'].squeeze(0).mean(axis=0)) / audio['sample_rate']
        
        # Determine number of frames and resolution from video if provided
        if opt_video is not None:
            num_frames = opt_video.shape[0]
            v_height, v_width = opt_video.shape[1], opt_video.shape[2]
            # Use video resolution as override
            actual_width, actual_height = v_width, v_height
        else:
            num_frames = int(audio_duration * frame_rate)
            actual_width, actual_height = screen_width, screen_height

        processor = BaseAudioProcessor(audio, num_frames, actual_height, actual_width, frame_rate)
        
        # Initialize Lyric Renderer if text provided
        lrc_text = kwargs.get("lyric_lrc_text", "")
        lyric_renderer = None
        if lrc_text:
            lyric_renderer = LyricRenderer(
                lrc_text, actual_width, actual_height, 
                kwargs.get("lyric_font_size", 24),
                kwargs.get("lyric_highlight_color", "#34d399"),
                kwargs.get("lyric_normal_color", "#9ca3af"),
                kwargs.get("lyric_background_alpha", 0.4),
                kwargs.get("lyric_blur_radius", 10),
                kwargs.get("lyric_active_blur", 20),
                kwargs.get("lyric_y_position", 0.5),
                kwargs.get("lyric_max_lines", 5),
                kwargs.get("lyric_line_spacing", 1.5),
                kwargs.get("lyric_font_name", "NotoSans-Regular.ttf")
            )

        result = []
        self.start_progress(num_frames, desc=f"Applying {self.__class__.__name__}")

        for i in range(num_frames):
            if i % 100 == 0: gc.collect()
            
            # Get video frame for background if available
            background_np = None
            if opt_video is not None:
                # Get i-th frame, wrap around or clamp if video shorter than audio (though we clamped num_frames above)
                v_idx = min(i, opt_video.shape[0] - 1)
                background_np = (opt_video[v_idx].cpu().numpy() * 255).astype(np.uint8)

            processor.current_frame = i
            processed_kwargs = self.process_parameters(
                frame_index=i,
                feature_value=self.get_feature_value(i, opt_feature) if opt_feature is not None else None,
                feature_param=feature_param,
                feature_mode=feature_mode,
                strength=strength,
                feature_threshold=feature_threshold,
                **kwargs
            )
            processed_kwargs.update({
                "frame_index": i, "screen_width": actual_width, "screen_height": actual_height,
                "background": background_np
            })
            
            num_points = self.get_point_count(processed_kwargs)
            spectrum, _, item_freqs = self.process_audio_data(
                processor, i,
                processed_kwargs.get('visualization_feature', 'frequency'),
                num_points,
                processed_kwargs.get('smoothing', 0.5),
                processed_kwargs.get('fft_size', 2048),
                processed_kwargs.get('min_frequency', 20.0),
                processed_kwargs.get('max_frequency', 8000.0)
            )
            processed_kwargs["item_freqs"] = item_freqs

            image = self.apply_effect_internal(processor, **processed_kwargs)
            
            # Ensure image is uint8 [0, 255] for lyric renderer and efficient processing
            if image.dtype != np.uint8:
                # Force to [0, 1] range then to [0, 255] for consistency
                if np.max(image) <= 1.05 and np.min(image) >= -0.05:
                    image = np.clip(image, 0, 1)
                    image = (image * 255.0).astype(np.uint8)
                else:
                    # Already seems to be in 0-255 range but float?
                    image = np.clip(image, 0, 255).astype(np.uint8)
            
            # Apply lyrics if enabled
            if lyric_renderer:
                # Calculate the exact time for this frame
                frame_time = i / frame_rate
                image = lyric_renderer.render(image, frame_time)
                
            result.append(torch.from_numpy(image.astype(np.float32) / 255.0))
            self.update_progress()

        self.end_progress()
        if result:
            result_tensor = torch.stack(result)
            mask = result_tensor[:, :, :, 0]
            
            # Use provided source_mask or create a fallback black mask
            if source_mask is None:
                source_mask = torch.zeros((1, actual_height, actual_width), dtype=torch.float32)
            
            return (result_tensor, mask, settings_str, source_mask)
        else:
            empty_tensor = torch.zeros((1, actual_height, actual_width, 3), dtype=torch.float32)
            empty_mask = torch.zeros((1, actual_height, actual_width), dtype=torch.float32)
            if source_mask is None:
                source_mask = empty_mask
            return (empty_tensor, empty_mask, settings_str, source_mask)
