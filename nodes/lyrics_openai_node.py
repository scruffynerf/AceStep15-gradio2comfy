"""AceStepOpenAILyrics node for ACE-Step"""
import json
import urllib.error
import urllib.request
from .includes.lyrics_utils import get_lyrics_messages, clean_markdown_formatting, load_api_key

class AceStepOpenAILyrics:
    """Generate lyrics using OpenAI API"""
    
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
                    "gpt-4o",
                    "gpt-4o-mini",
                    "gpt-4-turbo",
                    "gpt-3.5-turbo",
                ], {"default": "gpt-4o-mini"}),
                "max_tokens": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 128}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "generate"
    CATEGORY = "Scromfy/Ace-Step/Lyrics/AI"

    def generate(self, style: str, theme: str, model: str, max_tokens: int, seed: int):
        api_key = load_api_key("openai")
        if not api_key:
            return ("[OpenAI] API key is missing. Please add it to keys/openai_api_key.txt",)
        url = "https://api.openai.com/v1/chat/completions"

        payload = {
            "model": model,
            "messages": get_lyrics_messages(style, seed, theme),
            "temperature": 0.9,
            "max_tokens": max_tokens,
            "top_p": 0.95,
        }
        
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
        )
        
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                response_body = resp.read()
        except urllib.error.HTTPError as e:
            error_detail = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
            return (f"[OpenAI] HTTPError: {e.code} {error_detail}",)
        except Exception as e:
            return (f"[OpenAI] Error: {e}",)

        try:
            parsed = json.loads(response_body)
            choices = parsed.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                text = message.get("content", "").strip()
                text = clean_markdown_formatting(text)
                if text:
                    return (text,)
        except:
            pass
        
        return ("[OpenAI] Empty or invalid response.",)


NODE_CLASS_MAPPINGS = {
    "AceStepOpenAILyrics": AceStepOpenAILyrics,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepOpenAILyrics": "OpenAI Lyrics",
}
