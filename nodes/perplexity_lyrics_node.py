"""AceStepPerplexityLyrics node for ACE-Step"""
import json
import urllib.error
import urllib.request
from .includes.lyrics_utils import build_simple_prompt, clean_markdown_formatting

class AceStepPerplexityLyrics:
    """Generate lyrics using Perplexity API"""
    
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
                    "placeholder": "Perplexity API Key"
                }),
                "model": ([
                    "llama-3.1-sonar-large-128k-online",
                    "llama-3.1-sonar-small-128k-online",
                    "llama-3.1-sonar-large-128k-chat",
                    "llama-3.1-sonar-small-128k-chat",
                ], {"default": "llama-3.1-sonar-small-128k-chat"}),
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
            return ("[Perplexity] API key is missing.",)

        prompt = build_simple_prompt(style, seed)
        
        url = "https://api.perplexity.ai/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
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
            return (f"[Perplexity] HTTPError: {e.code} {error_detail}",)
        except Exception as e:
            return (f"[Perplexity] Error: {e}",)

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
        
        return ("[Perplexity] Empty or invalid response.",)


NODE_CLASS_MAPPINGS = {
    "AceStepPerplexityLyrics": AceStepPerplexityLyrics,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepPerplexityLyrics": "Perplexity Lyrics",
}
