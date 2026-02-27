"""AceStep Text Encode node for ACE-Step 1.5"""

class ACEStep15TaskTextEncodeNode:
    """
    Revised text encoder for ACE-Step 1.5 aka Scromfy
    """

    # Valid keyscales from ACE-Step 1.5 reference (constants.py):
    # 7 notes × 5 accidentals ('', '#', 'b', '♯', '♭') × 2 modes = 70 combinations
    # We use ASCII only ('#', 'b') plus 'Db/Eb/Gb/Ab/Bb' enharmonic spellings = 56 unique
    KEYSCALE_NOTES = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    KEYSCALE_ACCIDENTALS = ['', '#', 'b']
    KEYSCALE_MODES = ['major', 'minor']

    VALID_KEYSCALES = [""]
    for note in KEYSCALE_NOTES:
        for acc in KEYSCALE_ACCIDENTALS:
            for mode in KEYSCALE_MODES:
                if note == 'C' and mode == 'major' and acc != '':
                    continue
                if note == 'A' and mode == 'minor' and acc != '':
                    continue
                VALID_KEYSCALES.append(f"{note}{acc} {mode}")

    # Time signature display name → code mapping
    TIMESIG_MAP = {
        "Auto-detect": "",
        "2/4": "2",
        "3/4": "3",
        "4/4": "4",
        "6/8": "6",
    }
    VALID_TIME_SIGNATURES = list(TIMESIG_MAP.keys())


    # Language display name → ISO code mapping
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

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "A melodic electronic track with soft synths"}),
            },
            "optional": {
                "lyrics": ("STRING", {"multiline": True, "default": ""}),
                "bpm": ("INT", {"default": 0, "min": 0, "max": 300, "step": 1.0}),
                "duration": ("FLOAT", {"default": 0, "min": 0, "max": 600.0, "step": 0.1}),
                "keyscale": ("STRING", {"default": "", "options": s.VALID_KEYSCALES}),
                "timesignature": ("STRING", {"default": "", "options": s.VALID_TIME_SIGNATURES}),
                "language": ("STRING", {"default": "English", "options": s.VALID_LANGUAGES}),
                "llm_audio_codes": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
                "cfg_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1,
                              "tooltip": "Controls how closely the generated audio follows your text prompt. Higher values produce output that matches your description more literally, lower values allow more freedom. No effect on cover/extract/lego tasks, which use semantic hints from source audio instead of generating new audio codes."}),
                "temperature": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 2.0, "step": 0.01,
                                 "tooltip": "Controls randomness and creativity in the generated audio. Lower values (0.7-0.85) produce more consistent, predictable results. Higher values (0.9-1.1) produce more varied, surprising output. No effect on cover/extract/lego tasks, which use semantic hints from source audio instead of generating new audio codes."}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01,
                           "tooltip": "Limits how many possible audio choices are considered at each step. Lower values (e.g. 0.8) produce safer, more predictable output. Higher values allow more diversity. 1.0 disables this filter. No effect on cover/extract/lego tasks, which use semantic hints from source audio instead of generating new audio codes."}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 100,
                           "tooltip": "Restricts each generation step to only the top K most likely choices. 0 disables this filter. Lower values (e.g. 40) reduce unlikely outputs while keeping variety. No effect on cover/extract/lego tasks, which use semantic hints from source audio instead of generating new audio codes."}),
                "min_p": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                           "tooltip": "Minimum probability threshold for token sampling. Filters out tokens with probability below min_p × max_probability. 0.0 disables this filter. No effect on cover/extract/lego tasks."}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "conditioning"

    def encode(self, clip, text, task_type, track_name="", lyrics="", bpm=00,
               duration=0, keyscale="C major", timesignature="4/4", language="English", seed=0,
               cfg_scale=2.0, temperature=0.85, top_p=0.9, top_k=0, min_p=0.0, llm_audio_codes=True):
        # Convert display name to ISO code
        language_code = self.LANGUAGE_MAP.get(language, language)

        # Convert display name to numeric code
        timesig_code = self.TIMESIG_MAP.get(timesignature, timesignature)

        if bpm < 30: bpm = -1

        tokens = clip.tokenize(text, 
                                lyrics=lyrics, 
                                bpm=bpm, 
                                duration=duration, 
                                timesignature=int(timesig_code), 
                                language=language_code, 
                                keyscale=keyscale, 
                                seed=seed, 
                                generate_audio_codes=llm_audio_codes, 
                                cfg_scale=cfg_scale, 
                                temperature=temperature, 
                                top_p=top_p, 
                                top_k=top_k, 
                                min_p=min_p)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        # For negative conditioning, use ComfyUI's ConditioningZeroOut node
        # This matches the standard ACE-Step workflow pattern

        return (conditioning,)



# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    # ACE-Step 1.5 text encoder
    "ScromfyACEStep15TaskTextEncode": ScromfyACEStep15TaskTextEncodeNode,
}

# Display name mappings for ComfyUI
NODE_DISPLAY_NAME_MAPPINGS = {
    # ACE-Step 1.5 text encoder
    "ScromfyACEStep15TaskTextEncode": "ACE-Step 1.5 Task Text Encode (Scromfy)",
} 