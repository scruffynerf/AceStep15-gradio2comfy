import json

class ScromfyLyricSettingsNode:
    @classmethod
    def INPUT_TYPES(cls):
        import os
        # Base directory is the workspace root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fonts_dir = os.path.join(base_dir, "fonts")
        
        font_files = ["default"]
        if os.path.exists(fonts_dir):
            file_list = [f for f in os.listdir(fonts_dir) if f.lower().endswith(('.ttf', '.otf', '.ttc'))]
            if file_list:
                font_files = sorted(file_list)
        
        # Prefer CJK Pan-Regional font as default for broader support
        if "NotoSansCJK-Regular.ttc" in font_files:
            default_font = "NotoSansCJK-Regular.ttc"
        elif "NotoSans-Regular.ttf" in font_files:
            default_font = "NotoSans-Regular.ttf"
        else:
            default_font = font_files[0] if font_files else "default"

        return {
            "required": {
                "lrc_text": ("STRING", {"multiline": True, "default": ""}),
                "font_name": (font_files, {"default": default_font}),
                "font_size": ("INT", {"default": 24, "min": 10, "max": 200}),
                "highlight_color": ("COLOR", {"default": "#34d399"}),
                "normal_color": ("COLOR", {"default": "#f3f4f6"}),
                "background_alpha": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blur_radius": ("INT", {"default": 1, "min": 0, "max": 50}),
                "active_blur": ("INT", {"default": 10, "min": 0, "max": 100}),
                "y_position": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_lines": ("INT", {"default": 5, "min": 1, "max": 20}),
                "line_spacing": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("LYRIC_SETTINGS",)
    RETURN_NAMES = ("lyric_settings",)
    FUNCTION = "create_settings"
    CATEGORY = "Scromfy/Ace-Step/Visualizers"

    def create_settings(self, **kwargs):
        # We prefix keys with "lyric_" to match the expected keys in the visualizer backend
        return ({"lyric_" + k: v for k, v in kwargs.items()},)

NODE_CLASS_MAPPINGS = {
    "ScromfyLyricSettings": ScromfyLyricSettingsNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScromfyLyricSettings": "Lyric Visualizer Settings (Scromfy)",
}
