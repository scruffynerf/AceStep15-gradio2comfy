"""AceStepGroqLyrics node for ACE-Step"""
from groq import Groq
from .includes.lyrics_utils import get_lyrics_messages, clean_markdown_formatting, load_api_key

class AceStepGroqLyrics:
    """Generate lyrics using Groq API (fast inference)"""
    
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
                    "allam-2-7b",
                    "canopylabs/orpheus-arabic-saudi",
                    "canopylabs/orpheus-v1-english",
                    "groq/compound-mini",
                    "groq/compound",
                    "llama-3.1-8b-instant",
                    "llama-3.3-70b-versatile",
                    "meta-llama/llama-4-maverick-17b-128e-instruct",
                    "meta-llama/llama-4-scout-17b-16e-instruct",
                    "meta-llama/llama-guard-4-12b",
                    "meta-llama/llama-prompt-guard-2-22m",
                    "meta-llama/llama-prompt-guard-2-86m",
                    "moonshotai/kimi-k2-instruct-0905",
                    "moonshotai/kimi-k2-instruct",
                    "openai/gpt-oss-120b",
                    "openai/gpt-oss-20b",
                    "openai/gpt-oss-safeguard-20b",
                    "qwen/qwen3-32b",
                    "whisper-large-v3-turbo",
                    "whisper-large-v3",
                ], {"default": "llama-3.3-70b-versatile"}),
                "max_tokens": ("INT", {"default": 1024, "min": 256, "max": 8192, "step": 128}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "generate"
    CATEGORY = "Scromfy/Ace-Step/Lyrics/AI"

    def generate(self, style: str, theme: str, model: str, max_tokens: int, seed: int):
        api_key = load_api_key("groq")
        if not api_key:
            return ("[Groq] API key is missing. Please add it to keys/groq_api_key.txt",)

        try:
            client = Groq(api_key=api_key)
            chat_completion = client.chat.completions.create(
                messages=get_lyrics_messages(style, seed, theme),
                model=model,
                temperature=0.9,
                max_tokens=max_tokens,
                top_p=0.95,
                seed=seed,
            )

            text = chat_completion.choices[0].message.content.strip()
            text = clean_markdown_formatting(text)
            if text:
                return (text,)

            return ("[Groq] Empty response.",)

        except Exception as e:
            return (f"[Groq] Error: {e}",)


NODE_CLASS_MAPPINGS = {
    "AceStepGroqLyrics": AceStepGroqLyrics,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepGroqLyrics": "Groq Lyrics",
}
