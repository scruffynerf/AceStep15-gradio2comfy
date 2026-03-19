import yaml
import math
import torch
from .includes.prompt_utils import get_keyscales

class ScromfyAceStepTextEncoderPlusPlus:
    """Merged Text Encoder for ACE-Step 1.5.
    Combines SFT 'Enriched CoT' (Chain-of-Thought) formatting with granular base controls.
    Supports a toggle for enhanced prompting, trigger words/tags, and full LLM parameters.
    """

    VALID_KEYSCALES = get_keyscales()

    # Time signature mapping
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

    # Language display name -> ISO code mapping
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
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "caption": ("STRING", {
                    "multiline": True, 
                    "dynamicPrompts": True, 
                    "default": "A melodic electronic track with soft synths",
                    "placeholder": "Describe the music: genre, mood, instruments, style...",
                }),
                "enhanced_prompt": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If True: Uses SFT-style 'Enriched CoT' (YAML) formatting for better fine-tuned model performance. If False: Uses native ComfyUI encoding logic.",
                }),
                "instrumental": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Force [Instrumental] in lyrics.",
                }),
            },
            "optional": {
                "lyrics": ("STRING", {
                    "multiline": True, 
                    "default": "[Instrumental]",
                    "placeholder": "Song lyrics or [Instrumental]",
                }),
                "bpm": ("INT", {
                    "default": 0, "min": 0, "max": 300,
                    "tooltip": "Beats per minute. 0 = auto decide.",
                }),
                "duration": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 600.0, "step": 0.1,
                    "tooltip": "Duration in seconds. -1 = auto decide.",
                }),
                "keyscale": (cls.VALID_KEYSCALES, {"default": "Auto-decide"}),
                "timesignature": (cls.VALID_TIME_SIGNATURES, {"default": "Auto-decide"}),
                "language": (cls.VALID_LANGUAGES, {"default": "English"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "cfg_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "temperature": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 100}),
                "min_p": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "repetition_penalty": ("FLOAT", {"default": 1.3, "min": 0.0, "max": 2.0, "step": 0.01}),
                "negative_prompt": ("STRING", {
                    "multiline": True, 
                    "default": "",
                    "placeholder": "Negative prompt for audio code generation",
                }),
                "style_tags": ("STRING", {
                    "default": "", 
                    "forceInput": True,
                    "tooltip": "Tags to append to the end of the caption.",
                }),
                "trigger_word": ("STRING", {
                    "default": "",
                    "tooltip": "Word to prepend to the start of the caption.",
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "Scromfy/Ace-Step/Prompt"

    def encode(self, clip, caption, enhanced_prompt=True, instrumental=True, lyrics="[Instrumental]", 
               bpm=120, duration=60.0, keyscale="Auto-detect", timesignature="4/4", language="English", 
               seed=0, cfg_scale=2.0, temperature=0.85, top_p=0.9, top_k=0, min_p=0.0, 
               repetition_penalty=1.3, negative_prompt="", style_tags="", trigger_word=""):
        
        # 1. Assemble Full Caption
        full_caption = caption
        if trigger_word and trigger_word.strip():
            full_caption = f"{trigger_word.strip()} {full_caption}"
        if style_tags and style_tags.strip():
            full_caption = f"{full_caption}, {style_tags.strip()}"
        
        # 2. Handle Auto/Defaults
        actual_lyrics = "[Instrumental]" if instrumental else lyrics
        language_iso = self.LANGUAGE_MAP.get(language, "en")
        timesig_code = self.TIMESIG_MAP.get(timesignature, "4")
        
        bpm_val = -1 if bpm <= 20 else bpm # Align with base detection
        ks_val = "" if keyscale == "Auto-detect" else keyscale
        
        tok_bpm = 120 if bpm_val == -1 else bpm_val
        tok_ts = int(timesig_code)
        tok_ks = "C major" if ks_val == "" else ks_val

        # 3. Base Tokenization
        tokens = clip.tokenize(full_caption, 
                                lyrics=actual_lyrics, 
                                bpm=bpm_val, 
                                duration=duration, 
                                timesignature=timesig_code, 
                                language=language_iso, 
                                keyscale=ks_val, 
                                generate_audio_codes=True, 
                                seed=seed, 
                                cfg_scale=cfg_scale, 
                                temperature=temperature, 
                                top_p=top_p, 
                                top_k=top_k, 
                                min_p=min_p,
                                repetition_penalty=repetition_penalty,
                                caption_negative=negative_prompt)

        # 4. Enhanced Prompting (SFT Chain-of-Thought YAML)
        if enhanced_prompt:
            inner_tok = getattr(clip.tokenizer, "qwen3_06b", None)
            if inner_tok is not None:
                dur_ceil = int(math.ceil(duration)) if duration > 0 else -1
                cot_items = {}
                if bpm_val != -1:
                    cot_items["bpm"] = bpm_val
                cot_items["caption"] = full_caption
                cot_items["duration"] = dur_ceil
                if ks_val != "":
                    cot_items["keyscale"] = ks_val
                cot_items["language"] = language_iso
                if timesig_code != "0":
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
                    lm_tpl.format(full_caption, actual_lyrics.strip(), enriched_cot),
                    False,
                    disable_weights=True,
                )
                
                if negative_prompt:
                    tokens["lm_prompt_negative"] = inner_tok.tokenize_with_weights(
                        lm_tpl.format(negative_prompt, "", ""),
                        False,
                        disable_weights=True,
                    )

        # 5. Final Encoding
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        
        return (conditioning,)

NODE_CLASS_MAPPINGS = {"ScromfyAceStepTextEncoderPlusPlus": ScromfyAceStepTextEncoderPlusPlus}
NODE_DISPLAY_NAME_MAPPINGS = {"ScromfyAceStepTextEncoderPlusPlus": "ACE-Step Text Encoder PLUSPLUS"}
