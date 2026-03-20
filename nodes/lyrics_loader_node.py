"""AceStepLyricsLoader node – pick a saved lyric file from the /lyrics folder via dropdown"""
import os
from .includes.lyrics_utils import get_lyrics_dir


def _list_lyric_files():
    """Return sorted list of .txt files in the lyrics dir (excluding README)"""
    try:
        lyrics_dir = get_lyrics_dir()
        files = sorted([
            f for f in os.listdir(lyrics_dir)
            if f.endswith(".txt") and f.lower() != "readme.md"
        ])
        return files if files else ["(no lyrics saved yet)"]
    except Exception:
        return ["(no lyrics saved yet)"]


class AceStepLyricsLoader:
    """Load a previously-saved lyric file from the /lyrics folder via a dropdown list."""

    @classmethod
    def INPUT_TYPES(cls):
        files = _list_lyric_files()
        return {
            "required": {
                "lyric_file": (files,),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("lyrics", "title", "artist",)
    FUNCTION = "load"
    CATEGORY = "Scromfy/Ace-Step/Lyrics"

    @classmethod
    def IS_CHANGED(cls, lyric_file):
        # Re-read the file contents (if file exists) so stale cache is busted
        lyrics_dir = get_lyrics_dir()
        filepath = os.path.join(lyrics_dir, lyric_file)
        try:
            return os.path.getmtime(filepath)
        except Exception:
            return float("nan")

    def load(self, lyric_file):
        if lyric_file == "(no lyrics saved yet)":
            return ("No lyrics available. Use the Genius nodes with 'save' mode first.", "", "")

        lyrics_dir = get_lyrics_dir()
        filepath = os.path.join(lyrics_dir, lyric_file)

        if not os.path.exists(filepath):
            return (f"File not found: {filepath}", "", "")

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lyrics = f.read().strip()
        except Exception as e:
            return (f"Error reading lyrics file: {e}", "", "")

        # Parse artist and title from filename (artist-title.txt)
        name_part = lyric_file.rsplit(".", 1)[0]
        if "-" in name_part:
            artist_raw, title_raw = name_part.split("-", 1)
            artist = artist_raw.replace("_", " ").strip()
            title = title_raw.replace("_", " ").strip()
        else:
            artist = ""
            title = name_part.replace("_", " ").strip()

        print(f"[LyricsLoader] Loaded '{title}' by '{artist}' from {lyric_file}")
        return (lyrics, title, artist)


NODE_CLASS_MAPPINGS = {
    "AceStepLyricsLoader": AceStepLyricsLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepLyricsLoader": "Lyrics Loader (from /lyrics)",
}
