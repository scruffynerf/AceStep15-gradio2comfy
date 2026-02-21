"""AceStepClaudeLyrics node for ACE-Step"""
import json
import urllib.error
import urllib.request
from .includes.lyrics_utils import build_simple_prompt, clean_markdown_formatting

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
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "password": True,
                    "placeholder": "Anthropic API Key"
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
    CATEGORY = "Scromfy/Ace-Step/lyrics"

    def generate(self, style: str, api_key: str, model: str, max_tokens: int, seed: int):
        api_key = api_key.strip()
        if not api_key:
            return ("[Claude] API key is missing.",)

        prompt = build_simple_prompt(style, seed)
        
        url = "https://api.anthropic.com/v1/messages"
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0.9,
            "messages": [{"role": "user", "content": prompt}]
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
