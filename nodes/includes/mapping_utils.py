"""Shared mapping dictionaries and UI formatting helpers for ACE-Step."""

TIMESIG_MAP = {
    "Auto-decide": "0",
    "2/4": "2",
    "3/4": "3",
    "4/4": "4",
    "6/8": "6",
    "0": "0",
    "2": "2",
    "3": "3",
    "4": "4",
    "6": "6",
}
VALID_TIME_SIGNATURES = list(TIMESIG_MAP.keys())

LANGUAGE_MAP = {
    "English": "en",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Arabic": "ar",
    "Hindi": "hi",
    "Vietnamese": "vi",
    "Thai": "th",
    "Indonesian": "id",
    "Malay": "ms",
    "Tagalog": "tl",
    "Dutch": "nl",
    "Polish": "pl",
    "Turkish": "tr",
    "Swedish": "sv",
    "Danish": "da",
    "Norwegian": "no",
    "Finnish": "fi",
    "Czech": "cs",
    "Slovak": "sk",
    "Hungarian": "hu",
    "Romanian": "ro",
    "Bulgarian": "bg",
    "Croatian": "hr",
    "Serbian": "sr",
    "Ukrainian": "uk",
    "Greek": "el",
    "Hebrew": "he",
    "Persian": "fa",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Punjabi": "pa",
    "Urdu": "ur",
    "Nepali": "ne",
    "Swahili": "sw",
    "Haitian Creole": "ht",
    "Icelandic": "is",
    "Lithuanian": "lt",
    "Latin": "la",
    "Azerbaijani": "az",
    "Catalan": "ca",
    "Sanskrit": "sa",
    "Cantonese": "yue",
    "Unknown": "unknown",
}
VALID_LANGUAGES = list(LANGUAGE_MAP.keys())

# ACE-Step 1.5 Base track types for separation/stem extraction/completion
TRACK_NAMES = [
    "woodwinds", "brass", "fx", "synth", "strings", "percussion",
    "keyboard", "guitar", "bass", "drums", "backing_vocals", "vocals"
]

def get_choices_for(items):
    """Build the dropdown list: none, random, random2, then all items (deduplicated and friendly formatting)."""
    if not items:
        return ["none", "random", "random2"]
        
    if isinstance(items, dict):
        items = items.keys()
    
    raw_set = set(items)
    # Natural sort based on clean name
    sorted_raw = sorted(list(raw_set), key=lambda x: str(x).strip("_").lower())
    
    choices = []
    for item in sorted_raw:
        s = str(item)
        if s.startswith("__") and s.endswith("__") and len(s) > 4:
            # Transform __WILDCARD__ to (wildcard) for UI (lowercase to match others)
            choices.append(f"({s[2:-2].lower()})")
        else:
            choices.append(s)
            
    return ["none", "random", "random2"] + choices
