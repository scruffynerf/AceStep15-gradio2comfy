"""SaveText node for ACE-Step"""
import os
import re
import folder_paths

class SaveText:
    """Save text content to file with automatic numbering"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Text content to save"
                }),
                "filename_prefix": ("STRING", {
                    "multiline": False,
                    "default": "text/lyrics",
                    "placeholder": "folder/path/filename"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Scromfy/Ace-Step/io"

    def _sanitize_prefix(self, prefix: str) -> str:
        """Clean up path/filename, preserve folder structure"""
        parts = prefix.split("/")
        sanitized_parts = []
        for part in parts:
            cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", part).strip("._-")
            if cleaned:
                sanitized_parts.append(cleaned)
        return "/".join(sanitized_parts) if sanitized_parts else "text/default"

    def _next_available_path(self, base_output: str, prefix: str):
        """Find next available filename with increment"""
        prefix_parts = prefix.split("/")
        if len(prefix_parts) < 2:
            folder_path = os.path.join(base_output, "text")
            filename_base = prefix_parts[0] or "default"
        else:
            filename_base = prefix_parts[-1]
            folder_rel = "/".join(prefix_parts[:-1])
            folder_path = os.path.join(base_output, folder_rel)
        
        os.makedirs(folder_path, exist_ok=True)
        
        # ComfyUI style: filename_0001, filename_0002, etc.
        index = 1
        while True:
            filename = f"{filename_base}_{index:04d}.txt"
            candidate = os.path.join(folder_path, filename)
            if not os.path.exists(candidate):
                return candidate
            index += 1

    def save(self, text: str, filename_prefix: str = "text/lyrics"):
        base_output = folder_paths.get_output_directory()
        
        # Sanitize the path
        prefix = self._sanitize_prefix(filename_prefix)
        
        # Get path and increment suffix if file exists
        target_path = self._next_available_path(base_output, prefix)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(text)

        return {"ui": {"text": [target_path]}, "result": (target_path,)}


NODE_CLASS_MAPPINGS = {
    "SaveText": SaveText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveText": "Save Text",
}
