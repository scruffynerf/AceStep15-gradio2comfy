"""
Lyrics Generation Nodes for ACE-Step - Multi-API Support
Ported from JK-AceStep-Nodes (gemini_nodes.py, groq_nodes.py, lyrics_nodes.py)

Supports: Gemini, Groq, OpenAI, Claude, Perplexity APIs
All nodes use standardized prompt building for ACE-Step production tags
"""

import json
import re
import urllib.error
import urllib.request


# ==================== SHARED PROMPT BUILDER ====================

ALLOWED_TAGS_INFO = (
    "Use ONLY these section tags in square brackets (no numbers): [Intro], [Verse], [Pre-Chorus], [Chorus], "
    "[Post-Chorus], [Bridge], [Breakdown], [Drop], [Hook], [Refrain], [Instrumental], [Solo], [Rap], [Outro]. "
    "Do NOT add numbers to tags (e.g., use [Verse], not [Verse 1])."
)

BASE_INSTRUCTIONS = (
    "You are a music lyricist. Generate song lyrics in the requested style. "
    "Return ONLY the lyrics as plain text. Do not add titles, prefaces, markdown, code fences, or quotes. "
    f"{ALLOWED_TAGS_INFO} Never use parentheses for section labels. "
    "Keep it concise and coherent."
)


def build_simple_prompt(style: str, seed: int) -> str:
    """Simple prompt for basic lyrics generation"""
    base_style = style.strip() or "Generic song"
    return f"Style: {base_style}. {BASE_INSTRUCTIONS} [Generation seed: {seed}]"


def clean_markdown_formatting(text: str) -> str:
    """Remove markdown formatting and normalize section tags"""
    cleaned = text.strip()
    
    # Remove code fences
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned.strip("`").strip()
    
    # Normalize section labels
    normalized_lines = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        
        # Convert (Verse 1) style to [Verse]
        if stripped.startswith("(") and stripped.endswith(")") and len(stripped) <= 48:
            inner = stripped[1:-1].strip()
            if inner:
                parts = inner.split()
                if len(parts) >= 2 and parts[-1].isdigit():
                    inner = " ".join(parts[:-1])
                line = f"[{inner}]"
        
        # Clean [Verse 1] style to [Verse]
        if stripped.startswith("[") and stripped.endswith("]") and len(stripped) <= 64:
            inner = stripped[1:-1].strip()
            if inner:
                parts = inner.split()
                if len(parts) >= 2 and parts[-1].isdigit():
                    inner = " ".join(parts[:-1])
                line = f"[{inner}]"
        
        normalized_lines.append(line)
    
    return "\n".join(normalized_lines).strip()


# ==================== GEMINI ====================

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
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "password": True,
                    "placeholder": "Gemini API Key"
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
    CATEGORY = "Scromfy/Ace-Step/lyrics"

    def generate(self, style: str, api_key: str, model: str, max_tokens: int, seed: int):
        api_key = api_key.strip()
        if not api_key:
            return ("[Gemini] API key is missing.",)

        prompt = build_simple_prompt(style, seed)
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
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


# ==================== GROQ ====================

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
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "password": True,
                    "placeholder": "Groq API Key"
                }),
                "model": ([
                    "llama-3.3-70b-versatile",
                    "llama-3.1-8b-instant",
                    "mixtral-8x7b-32768",
                ], {"default": "llama-3.3-70b-versatile"}),
                "max_tokens": ("INT", {"default": 1024, "min": 256, "max": 8192, "step": 128}),
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
            return ("[Groq] API key is missing.",)

        prompt = build_simple_prompt(style, seed)
        
        url = "https://api.groq.com/openai/v1/chat/completions"
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
            return (f"[Groq] HTTPError: {e.code} {error_detail}",)
        except Exception as e:
            return (f"[Groq] Error: {e}",)

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
        
        return ("[Groq] Empty or invalid response.",)


# ==================== OPENAI ====================

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
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "password": True,
                    "placeholder": "OpenAI API Key"
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
    CATEGORY = "Scromfy/Ace-Step/lyrics"

    def generate(self, style: str, api_key: str, model: str, max_tokens: int, seed: int):
        api_key = api_key.strip()
        if not api_key:
            return ("[OpenAI] API key is missing.",)

        prompt = build_simple_prompt(style, seed)
        
        url = "https://api.openai.com/v1/chat/completions"
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


# ==================== CLAUDE ====================

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


# ==================== PERPLEXITY ====================

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


# Node registration
NODE_CLASS_MAPPINGS = {
    "AceStepGeminiLyrics": AceStepGeminiLyrics,
    "AceStepGroqLyrics": AceStepGroqLyrics,
    "AceStepOpenAILyrics": AceStepOpenAILyrics,
    "AceStepClaudeLyrics": AceStepClaudeLyrics,
    "AceStepPerplexityLyrics": AceStepPerplexityLyrics,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepGeminiLyrics": "Gemini Lyrics",
    "AceStepGroqLyrics": "Groq Lyrics",
    "AceStepOpenAILyrics": "OpenAI Lyrics",
    "AceStepClaudeLyrics": "Claude Lyrics",
    "AceStepPerplexityLyrics": "Perplexity Lyrics",
}
