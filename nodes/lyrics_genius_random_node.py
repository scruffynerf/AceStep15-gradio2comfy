"""AceStepRandomLyrics node for ACE-Step – fetch lyrics from a random Genius song"""
import os
import sys
import random
from lyricsgenius import Genius


class AceStepRandomLyrics:
    """Pick a random song ID on Genius and return its lyrics. Retries until lyrics are found."""

    MAX_RETRIES = 20
    MAX_SONG_ID = 1_000_000

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "max_retries": ("INT", {"default": 20, "min": 1, "max": 100}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("lyrics", "title", "artist",)
    FUNCTION = "fetch_random"
    CATEGORY = "Scromfy/Ace-Step/lyrics"

    @classmethod
    def IS_CHANGED(cls, seed, max_retries=20):
        return seed

    def fetch_random(self, seed, max_retries=20):
        # Load API key
        base_dir = os.path.dirname(os.path.dirname(__file__))
        key_path = os.path.join(base_dir, "keys", "genius_api_key.txt")

        if not os.path.exists(key_path):
            msg = f"Genius API key not found at {key_path}. See keys/README.md"
            print(msg, file=sys.stderr)
            return (f"Error: {msg}", "", "")

        try:
            with open(key_path, "r") as f:
                token = f.read().strip()
            if not token:
                return ("Error: Genius API token is empty.", "", "")

            genius = Genius(token, timeout=20, retries=3,
                            remove_section_headers=False, verbose=False)

            rng = random.Random(seed)
            attempts = 0

            while attempts < max_retries:
                song_id = rng.randint(1, self.MAX_SONG_ID)
                attempts += 1
                try:
                    song = genius.song(song_id)
                except Exception:
                    continue

                if song is None:
                    continue

                # song() returns a dict with a "song" key
                song_data = song.get("song", {}) if isinstance(song, dict) else {}
                title = song_data.get("title", "Unknown")
                artist = song_data.get("primary_artist", {}).get("name", "Unknown")

                # Now fetch the actual lyrics
                try:
                    lyrics = genius.lyrics(song_id)
                except Exception:
                    lyrics = None

                if lyrics and lyrics.strip():
                    lowerlyrics = lyrics.lower()
                    if ("[chorus]" in lowerlyrics or "[verse " in lowerlyrics or "[bridge]" in lowerlyrics or "[outro]" in lowerlyrics or "[intro]" in lowerlyrics or "[Hook]" in lowerlyrics):
                        print(f"[RandomLyrics] Found: '{title}' by {artist} (id={song_id}, attempt {attempts})")
                        return (lyrics, title, artist)

            return (f"No lyrics found after {max_retries} attempts.", "", "")

        except Exception as e:
            msg = f"Error fetching random lyrics: {str(e)}"
            print(msg, file=sys.stderr)
            return (f"Error: {msg}", "", "")


NODE_CLASS_MAPPINGS = {
    "AceStepRandomLyrics": AceStepRandomLyrics,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepRandomLyrics": "Random Lyrics (Genius)",
}
