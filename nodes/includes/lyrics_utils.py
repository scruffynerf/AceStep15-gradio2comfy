import os
import sys
import re

def load_api_key(service_name: str) -> str:
    """Load API key from the keys directory for a specific service"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    key_file = f"{service_name.lower()}_api_key.txt"
    key_path = os.path.join(base_dir, "keys", key_file)
    
    if not os.path.exists(key_path):
        print(f"Error: API key for {service_name} not found at {key_path}", file=sys.stderr)
        return ""
        
    try:
        with open(key_path, "r") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading API key for {service_name}: {e}", file=sys.stderr)
        return ""

_SYSTEM_PROMPT_CACHE = None

def load_system_prompt() -> str:
    """Load the system prompt from AI_instructions/systemprompt.txt (user) or systemprompt.default.txt (fallback)"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    ai_instr_dir = os.path.join(base_dir, "AI_instructions")
    
    user_prompt_path = os.path.join(ai_instr_dir, "systemprompt.txt")
    default_prompt_path = os.path.join(ai_instr_dir, "systemprompt.default.txt")
    
    # Prioritize user prompt, then default prompt
    if os.path.exists(user_prompt_path):
        target_path = user_prompt_path
    elif os.path.exists(default_prompt_path):
        target_path = default_prompt_path
    else:
        # Fallback to a very minimal version if both files are missing
        return "You are a music lyricist. Generate lyrics in the requested style and theme."
        
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading system prompt from {target_path}: {e}", file=sys.stderr)
        return "You are a music lyricist. Generate lyrics in the requested style and theme."


def get_lyrics_messages(style: str, seed: int, theme: str = "Love Song") -> list:
    """Return a list of messages for OpenAI-compatible Chat APIs"""
    return [
        {"role": "system", "content": load_system_prompt()},
        {"role": "user", "content": build_simple_prompt(style, seed, theme)}
    ]

def build_simple_prompt(style: str, seed: int, theme: str = "Love Song") -> str:
    """User-focused prompt for basic lyrics generation (no system instructions)"""
    base_style = style.strip() or "Generic song"
    return f"Style: {base_style}. Song Theme: {theme}."


def clean_markdown_formatting(text: str) -> str:
    """Remove markdown formatting, <think> blocks, and normalize section tags"""
    cleaned = text.strip()
    
    # Remove <think>...</think> blocks (often leaked by R1/DeepSeek)
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL | re.IGNORECASE).strip()

    # Remove code fences
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned.strip("`").strip()
    
    # Normalize section labels
    normalized_lines = []
    for line in cleaned.splitlines():
        stripped = line.strip()
        
        ## Convert (Verse 1) style to [Verse]
        #if stripped.startswith("(") and stripped.endswith(")") and len(stripped) <= 48:
        #    inner = stripped[1:-1].strip()
        #    if inner:
        #        parts = inner.split()
        #        if len(parts) >= 2 and parts[-1].isdigit():
        #            inner = " ".join(parts[:-1])
        #        line = f"[{inner}]"
        
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

def get_lyrics_dir() -> str:
    """Get the path to the lyrics directory, creating it if necessary"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    lyrics_dir = os.path.join(base_dir, "lyrics")
    if not os.path.exists(lyrics_dir):
        os.makedirs(lyrics_dir, exist_ok=True)
    return lyrics_dir

def safe_filename(text: str) -> str:
    """Sanitize a string for use as a filename"""
    # Remove characters that aren't alphanumeric, spaces, dots, or hyphens
    # Then replace spaces with underscores
    s = re.sub(r'[^\w\s\.\-]', '', text)
    s = re.sub(r'[-\s]+', '_', s).strip('-_')
    return s

def save_lyrics_to_disk(artist: str, title: str, lyrics: str) -> str:
    """Save lyrics to a text file in the lyrics directory"""
    if not lyrics or not lyrics.strip():
        return ""
        
    lyrics_dir = get_lyrics_dir()
    filename = f"{safe_filename(artist)}-{safe_filename(title)}.txt"
    filepath = os.path.join(lyrics_dir, filename)
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(lyrics.strip())
        return filepath
    except Exception as e:
        print(f"Error saving lyrics to disk: {e}", file=sys.stderr)
        return ""

def load_lyrics_from_disk(artist: str, title: str) -> str:
    """Load lyrics from a text file in the lyrics directory if it exists"""
    lyrics_dir = get_lyrics_dir()
    filename = f"{safe_filename(artist)}-{safe_filename(title)}.txt"
    filepath = os.path.join(lyrics_dir, filename)
    
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error loading lyrics from disk: {e}", file=sys.stderr)
    return ""

def get_random_cached_lyric(seed: int = None) -> tuple:
    """Pick a random lyric file from the lyrics directory and return (lyrics, title, artist)"""
    lyrics_dir = get_lyrics_dir()
    if not os.path.exists(lyrics_dir):
        return None, None, None
        
    files = sorted([f for f in os.listdir(lyrics_dir) if f.endswith(".txt")])
    
    if not files:
        return None, None, None
        
    import random
    if seed is not None:
        rng = random.Random(seed)
        filename = rng.choice(files)
    else:
        filename = random.choice(files)
        
    filepath = os.path.join(lyrics_dir, filename)
    
    # Try to parse artist and title from filename (artist-title.txt)
    name_part = filename.rsplit(".", 1)[0]
    if "-" in name_part:
        artist, title = name_part.split("-", 1)
        # Replace underscores back with spaces for better display
        artist = artist.replace("_", " ")
        title = title.replace("_", " ")
    else:
        artist, title = "Unknown", name_part.replace("_", " ")
        
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lyrics = f.read().strip()
        return lyrics, title, artist
    except Exception as e:
        print(f"Error reading random cached lyric: {e}", file=sys.stderr)
        return None, None, None
