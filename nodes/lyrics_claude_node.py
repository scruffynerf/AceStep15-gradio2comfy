"""AceStepClaudeLyrics node for ACE-Step"""
import json
import urllib.error
import urllib.request
from .includes.lyrics_utils import get_lyrics_messages, load_system_prompt, build_simple_prompt, clean_markdown_formatting, load_api_key

class AceStepClaudeLyrics:
    """Generate lyrics using Anthropic Claude API"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Music style (e.g., Synthwave with female vocals)"
                }),
                "theme": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Music subject/theme (e.g., Love Ballad)"
                }),
                "model": ([
                    "claude-3-5-sonnet-20241022",
                    "claude-3-5-haiku-20241022",
                    "claude-3-opus-20240229",
                ], {"default": "claude-3-5-haiku-20241022"}),
                "max_tokens": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 128}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "generate"
    CATEGORY = "Scromfy/Ace-Step/Lyrics/AI"

    def generate(self, style: str, theme: str, model: str, max_tokens: int, seed: int):
        api_key = load_api_key("claude")
        if not api_key:
            return ("[Claude] API key is missing. Please add it to keys/claude_api_key.txt",)
        
        url = "https://api.anthropic.com/v1/messages"

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0.9,
            "system": load_system_prompt(),
            "messages": [{"role": "user", "content": build_simple_prompt(style, seed, theme)}]
        }
        
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01"
            }
        )
        
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                response_body = resp.read()
        except urllib.error.HTTPError as e:
            error_detail = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
            return (f"[Claude] HTTPError: {e.code} {error_detail}",)
        except Exception as e:
            return (f"[Claude] Error: {e}",)

        try:
            parsed = json.loads(response_body)
            content = parsed.get("content", [])
            if content and isinstance(content, list):
                text = content[0].get("text", "").strip()
                text = clean_markdown_formatting(text)
                if text:
                    return (text,)
        except:
            pass
        
        return ("[Claude] Empty or invalid response.",)


NODE_CLASS_MAPPINGS = {
    "AceStepClaudeLyrics": AceStepClaudeLyrics,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepClaudeLyrics": "Claude Lyrics",
}
