import yaml
import math
import torch
from .includes.prompt_utils import get_keyscales

# Credit goes to https://github.com/jeankassio/ComfyUI-AceStep_SFT
# for his all-in-one SFT node implementation, I've split it into pieces.
# This is the text encode node.

class ScromfySFTTextEncode:
    """ScromfySFT Text Encode node.
    Implements SFT-specific text and lyric encoding with 'Enriched CoT' (Chain-of-Thought) 
    YAML formatting for the LLM prompt, matching the Gradio SFT pipeline.
    """

    VALID_KEYSCALES = get_keyscales()

    # Time signature mapping
    TIMESIG_MAP = {
        "auto": "0",
        "2": "2",
        "3": "3",
        "4": "4",
        "6": "6",
        "2/4": "2",
        "3/4": "3",
        "4/4": "4",
        "6/8": "6",

    }
    VALID_TIME_SIGNATURES = list(TIMESIG_MAP.keys())

    # Language display name → ISO code mapping
    LANGUAGE_MAP = {
        "English": "en", "Chinese": "zh", "Japanese": "ja", "Korean": "ko",
        "Spanish": "es", "French": "fr", "German": "de", "Italian": "it",
        "Portuguese": "pt", "Russian": "ru", "Arabic": "ar", "Hindi": "hi",
        "Vietnamese": "vi", "Thai": "th", "Indonesian": "id", "Malay": "ms",
        "Tagalog": "tl", "Dutch": "nl", "Polish": "pl", "Turkish": "tr",
        "Swedish": "sv", "Danish": "da", "Norwegian": "no", "Finnish": "fi",
        "Czech": "cs", "Slovak": "sk", "Hungarian": "hu", "Romanian": "ro",
        "Bulgarian": "bg", "Croatian": "hr", "Serbian": "sr", "Ukrainian": "uk",
        "Greek": "el", "Hebrew": "he", "Persian": "fa", "Bengali": "bn",
        "Tamil": "ta", "Telugu": "te", "Punjabi": "pa", "Urdu": "ur",
        "Nepali": "ne", "Swahili": "sw", "Haitian Creole": "ht", "Icelandic": "is",
        "Lithuanian": "lt", "Latin": "la", "Azerbaijani": "az", "Catalan": "ca",
        "Sanskrit": "sa", "Cantonese": "yue", "Unknown": "unknown",
    }
    VALID_LANGUAGES = list(LANGUAGE_MAP.keys())

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "caption": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "instrumental": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "lyrics": ("STRING", {"multiline": True, "default": "[Instrumental]"}),
                "bpm": ("INT", {"default": 0, "min": 0, "max": 300}),
                "duration": ("FLOAT", {"default": 60.0, "min": 0.0, "max": 600.0, "step": 0.1}),
                "keyscale": (["auto"] + cls.VALID_KEYSCALES[1:], {"default": "auto"}),
                "timesignature": (cls.VALID_TIME_SIGNATURES, {"default": "auto"}),
                "language": (cls.VALID_LANGUAGES, {"default": "English"}),
                "generate_audio_codes": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "lm_cfg_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "lm_temperature": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 2.0, "step": 0.01}),
                "lm_top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 2000.0, "step": 0.01}),
                "lm_top_k": ("INT", {"default": 0, "min": 0, "max": 100}),
                "lm_min_p": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "lm_negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "style_tags": ("STRING", {"default": "", "forceInput": True}),
                "style_bpm": ("INT", {"default": 0, "min": 0, "max": 300, "forceInput": True}),
                "style_keyscale": ("STRING", {"default": "", "forceInput": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "Scromfy/SFT"

    def encode(self, clip, caption, instrumental=True, lyrics="[Instrumental]", bpm=0, duration=60.0, 
               keyscale="C major", timesignature="4/4", language="English", 
               generate_audio_codes=True, seed=0, lm_cfg_scale=2.0, 
               lm_temperature=0.85, lm_top_p=0.9, lm_top_k=0, lm_min_p=0.0, 
               lm_negative_prompt="", style_tags="", style_bpm=0, style_keyscale=""):
        
        # --- Style overrides from Music Analyzer ---
        if style_tags and style_tags.strip():
            caption = f"{caption}, {style_tags}" if caption.strip() else style_tags
        if style_bpm > 0:
            if duration > 0 and bpm > 0 and bpm != style_bpm:
                duration = round(duration * bpm / style_bpm, 1)
            bpm = style_bpm
        if style_keyscale and style_keyscale.strip():
            keyscale = style_keyscale
        
        actual_lyrics = "[Instrumental]" if instrumental else lyrics
        language_iso = self.LANGUAGE_MAP.get(language, "en")
        timesig_code = self.TIMESIG_MAP.get(timesignature, "4")
        
        bpm_is_auto = (bpm == 0)
        ts_is_auto = (timesignature == "Auto-detect")
        ks_is_auto = (keyscale == "Auto-detect")
        
        tok_bpm = 120 if bpm_is_auto else bpm
        tok_ts = 4 if ts_is_auto else int(timesig_code)
        tok_ks = "C major" if ks_is_auto else keyscale

        # --- Base tokenization ---
        tokenize_kwargs = dict(
            lyrics=actual_lyrics,
            bpm=tok_bpm,
            duration=duration,
            timesignature=tok_ts,
            language=language_iso,
            keyscale=tok_ks,
            seed=seed,
            generate_audio_codes=generate_audio_codes,
            cfg_scale=lm_cfg_scale,
            temperature=lm_temperature,
            top_p=lm_top_p,
            top_k=lm_top_k,
            min_p=lm_min_p,
            caption_negative=lm_negative_prompt if lm_negative_prompt else ""
        )
        
        tokens = clip.tokenize(caption, **tokenize_kwargs)

        # --- Phase 1: SFT-specific Prompt Enrichment (CoT YAML) ---
        # This matches the Gradio SFT pipeline's exact prompt structure
        inner_tok = getattr(clip.tokenizer, "qwen3_06b", None)
        if inner_tok is not None:
            dur_ceil = int(math.ceil(duration))
            cot_items = {}
            if not bpm_is_auto:
                cot_items["bpm"] = bpm
            cot_items["caption"] = caption
            cot_items["duration"] = dur_ceil
            if not ks_is_auto:
                cot_items["keyscale"] = keyscale
            cot_items["language"] = language_iso
            if not ts_is_auto:
                cot_items["timesignature"] = tok_ts
                
            cot_yaml = yaml.dump(cot_items, allow_unicode=True, sort_keys=True).strip()
            enriched_cot = f"<think>\n{cot_yaml}\n</think>"

            lm_tpl = (
                "<|im_start|>system\n# Instruction\n"
                "Generate audio semantic tokens based on the given conditions:\n\n"
                "<|im_end|>\n<|im_start|>user\n# Caption\n{}\n\n# Lyric\n{}\n"
                "<|im_end|>\n<|im_start|>assistant\n{}\n\n<|im_end|>\n"
            )
            
            # Repopulate the lm_prompt within tokens
            tokens["lm_prompt"] = inner_tok.tokenize_with_weights(
                lm_tpl.format(caption, actual_lyrics.strip(), enriched_cot),
                False,
                disable_weights=True,
            )
            
            if lm_negative_prompt:
                tokens["lm_prompt_negative"] = inner_tok.tokenize_with_weights(
                    lm_tpl.format(lm_negative_prompt, "", ""),
                    False,
                    disable_weights=True,
                )

        # --- Phase 2: Encoding ---
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        
        return (conditioning,)

NODE_CLASS_MAPPINGS = {
    "ScromfySFTTextEncode": ScromfySFTTextEncode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScromfySFTTextEncode": "ScromfySFT Text Encode"
}
