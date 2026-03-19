"""AceStepGeminiLyrics node for ACE-Step"""
import json
import urllib.error
import urllib.request
from .includes.lyrics_utils import load_system_prompt, build_simple_prompt, clean_markdown_formatting, load_api_key

class AceStepGeminiLyrics:
    """Generate lyrics using Google Gemini API"""
    
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
                    "gemini-2.5-flash",
                    "gemini-2.5-flash-latest",
                    "gemini-2.5-flash-lite",
                    "gemini-2.5-flash-lite-latest",
                    "gemini-2.5-pro",
                    "gemini-2.5-pro-latest",
                    "gemini-2.0-flash",
                    "gemini-1.5-pro",
                    "gemini-1.5-flash",
                ], {"default": "gemini-2.5-flash"}),
                "max_tokens": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 128}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "generate"
    CATEGORY = "Scromfy/Ace-Step/Lyrics/AI"

    def generate(self, style: str, theme: str, model: str, max_tokens: int, seed: int):
        api_key = load_api_key("gemini")
        if not api_key:
            return ("[Gemini] API key is missing. Please add it to keys/gemini_api_key.txt",)

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": build_simple_prompt(style, seed, theme)}]}],
            "systemInstruction": {"parts": [{"text": load_system_prompt()}]},
            "generationConfig": {
                "temperature": 0.9,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": max_tokens,
            },
        }
        
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                response_body = resp.read()
        except urllib.error.HTTPError as e:
            error_detail = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else str(e)
            return (f"[Gemini] HTTPError: {e.code} {error_detail}",)
        except Exception as e:
            return (f"[Gemini] Error: {e}",)

        try:
            parsed = json.loads(response_body)
            candidates = parsed.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    text = parts[0].get("text", "").strip()
                    text = clean_markdown_formatting(text)
                    if text:
                        return (text,)
        except:
            pass
        
        return ("[Gemini] Empty or invalid response.",)


NODE_CLASS_MAPPINGS = {
    "AceStepGeminiLyrics": AceStepGeminiLyrics,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepGeminiLyrics": "Gemini Lyrics",
}
