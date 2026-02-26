"""AceStepLyricsFormatter node for ACE-Step"""

class AceStepLyricsFormatter:
    """Auto-format lyrics with ACE-Step required tags and line length limits"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyrics": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("formatted_lyrics",)
    FUNCTION = "format"
    CATEGORY = "Scromfy/Ace-Step/lyrics"

    def format(self, lyrics):
        lines = lyrics.strip().split('\n')
        formatted_lines = []
        
        has_intro = False
        has_outro = False
        
        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append("")
                continue
            
            # Check for tags
            if line.startswith('[') and line.endswith(']'):
                tag = line[1:-1].lower()
                if 'intro' in tag: has_intro = True
                if 'outro' in tag: has_outro = True
            
            # Limit line length to 80 chars
            if len(line) > 80:
                words = line.split()
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) + 1 > 80:
                        formatted_lines.append(current_line)
                        current_line = word
                    else:
                        current_line = f"{current_line} {word}" if current_line else word
                if current_line:
                    formatted_lines.append(current_line)
            else:
                formatted_lines.append(line)
        
        if not has_intro:
            formatted_lines.insert(0, "[Intro]")
        if not has_outro:
            if formatted_lines and formatted_lines[-1] != "":
                formatted_lines.append("")
            formatted_lines.append("[Outro]")
            
        return ("\n".join(formatted_lines),)


NODE_CLASS_MAPPINGS = {
    "AceStepLyricsFormatter": AceStepLyricsFormatter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepLyricsFormatter": "Lyrics Formatter",
}
