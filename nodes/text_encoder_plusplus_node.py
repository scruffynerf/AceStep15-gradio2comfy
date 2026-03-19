import yaml
import math
import torch
from .includes.prompt_utils import get_keyscales
from .includes.sampling_utils import zero_out

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
                    "tooltip": "If True: Uses SFT-style 'Enriched CoT' (YAML) formatting for better fine-tuned model performance. If False: Uses native ComfyUI encoding logic",
                }),
                "generate_audio_codes": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable LLM audio code generation for semantic structure. Recommended to keep enabled even with reference_audio.",
                }),
                "instrumental": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Force [Instrumental] as lyrics",
                }),
            },
            "optional": {
                "lyrics": ("STRING", {
                    "multiline": True, 
                    "default": "[Instrumental]",
                    "placeholder": "Song lyrics or [Instrumental]",
                }),
                "trigger_word": ("STRING", {
                    "default": "",
                    "tooltip": "Model trigger word(s) to prepend to the start of the caption",
                }),
                "style_tags": ("STRING", {
                    "default": "", 
                    "forceInput": True,
                    "tooltip": "Tags to append to the end of the caption",
                }),
                "bpm": ("INT", {
                    "default": -1, "min": -1, "max": 300,
                    "tooltip": "Beats per minute. -1 = auto decide",
                }),
                "duration": ("FLOAT", {
                    "default": 120.0, "min": 10.0, "max": 600.0, "step": 0.1,
                    "tooltip": "Duration in seconds.",
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
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("conditioning", "zero_conditioning")
    FUNCTION = "encode"
    CATEGORY = "Scromfy/Ace-Step/Prompt"
    
    @classmethod
    def IS_CHANGED(cls, clip, caption="", enhanced_prompt=True, instrumental=True, lyrics="[Instrumental]", 
                bpm=-1, duration=120.0, keyscale="Auto-decide", timesignature="Auto-decide", language="English", 
                seed=0, cfg_scale=2.0, temperature=0.85, top_p=0.9, top_k=0, min_p=0.0, 
                repetition_penalty=1.3, negative_prompt="", style_tags="", trigger_word="", 
                generate_audio_codes=True):
        import hashlib
        m = hashlib.sha256()
        for v in [caption, enhanced_prompt, instrumental, lyrics, bpm, duration, keyscale, 
                timesignature, language, seed, cfg_scale, temperature, top_p, top_k, 
                min_p, repetition_penalty, negative_prompt, style_tags, trigger_word, 
                generate_audio_codes]:
            m.update(str(v).encode('utf-8'))
        return m.hexdigest()

    def encode(self, clip, caption="", enhanced_prompt=True, instrumental=True, lyrics="[Instrumental]", 
               bpm=-1, duration=120.0, keyscale="Auto-decide", timesignature="Auto-decide", language="English", 
               seed=0, cfg_scale=2.0, temperature=0.85, top_p=0.9, top_k=0, min_p=0.0, 
               repetition_penalty=1.3, negative_prompt="", style_tags="", trigger_word="", 
               generate_audio_codes=True):
        
        # 1. Assemble Full Caption
        full_caption = caption.strip()
        if trigger_word and trigger_word.strip():
            full_caption = f"{trigger_word.strip()} {full_caption}"
        if style_tags and style_tags.strip():
            full_caption = f"{full_caption}, {style_tags.strip()}"
        
        actual_lyrics = "[Instrumental]" if instrumental else lyrics
        language_iso = self.LANGUAGE_MAP.get(language, "en")
        timesig_code = self.TIMESIG_MAP.get(timesignature, "0")
        
        ks_val = "" if keyscale == "Auto-decide" else keyscale
        
        # --- PATH A: ENHANCED PROMPT (SFT LOGIC) ---
        if enhanced_prompt:
            # SFT-specific auto-detection flags
            bpm_is_auto = (bpm == -1 or bpm == 0)
            ts_is_auto = (timesignature == "Auto-decide" or timesignature == "0")
            ks_is_auto = (keyscale == "Auto-decide" or ks_val == "")
            
            tok_bpm = 120 if bpm_is_auto else bpm
            tok_ts = 4 if ts_is_auto else int(timesig_code)
            tok_ks = "C major" if ks_is_auto else ks_val

            # 1. Initial Tokenization (Generates audio codes if requested)
            tokens = clip.tokenize(full_caption, 
                                    lyrics=actual_lyrics, 
                                    bpm=tok_bpm, 
                                    duration=duration, 
                                    timesignature=tok_ts, 
                                    language=language_iso, 
                                    keyscale=tok_ks, 
                                    seed=seed, 
                                    generate_audio_codes=generate_audio_codes,
                                    cfg_scale=cfg_scale, 
                                    temperature=temperature, 
                                    top_p=top_p, 
                                    top_k=top_k, 
                                    min_p=min_p,
                                    caption_negative=negative_prompt)
            
            # 2. SFT enrichment
            inner_tok = getattr(clip.tokenizer, "qwen3_06b", None)
            if inner_tok is not None:
                dur_ceil = int(math.ceil(duration)) if duration > 0 else 0
                cot_items = {}
                if not bpm_is_auto: cot_items["bpm"] = bpm
                cot_items["caption"] = full_caption
                cot_items["duration"] = dur_ceil
                if not ks_is_auto: cot_items["keyscale"] = ks_val
                cot_items["language"] = language_iso
                if not ts_is_auto: cot_items["timesignature"] = tok_ts
                    
                cot_yaml = yaml.dump(cot_items, allow_unicode=True, sort_keys=True).strip()
                enriched_cot = f"<think>\n{cot_yaml}\n</think>"

                lm_tpl = (
                    "<|im_start|>system\n# Instruction\n"
                    "Generate audio semantic tokens based on the given conditions:\n\n"
                    "<|im_end|>\n<|im_start|>user\n# Caption\n{}\n\n# Lyric\n{}\n"
                    "<|im_end|>\n<|im_start|>assistant\n{}\n\n<|im_end|>\n"
                )
                
                # Overwrite lm_prompt with enriched CoT
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
            
            conditioning = clip.encode_from_tokens_scheduled(tokens)

        # --- PATH B: STANDARD PROMPT (ACESTEP 1.5 LOGIC) ---
        else:
            tokens = clip.tokenize(full_caption, 
                                    lyrics=actual_lyrics, 
                                    bpm=bpm, 
                                    duration=duration, 
                                    timesignature=timesig_code, 
                                    language=language_iso, 
                                    keyscale=ks_val, 
                                    generate_audio_codes=generate_audio_codes, 
                                    seed=seed, 
                                    cfg_scale=cfg_scale, 
                                    temperature=temperature, 
                                    top_p=top_p, 
                                    top_k=top_k, 
                                    min_p=min_p,
                                    repetition_penalty=repetition_penalty,
                                    caption_negative=negative_prompt)
            
            conditioning = clip.encode_from_tokens_scheduled(tokens)
        
        # 6. Shared: Zero-ed out conditioning
        zero_conditioning = zero_out(conditioning)
        
        return (conditioning, zero_conditioning)

NODE_CLASS_MAPPINGS = {"ScromfyAceStepTextEncoderPlusPlus": ScromfyAceStepTextEncoderPlusPlus}
NODE_DISPLAY_NAME_MAPPINGS = {"ScromfyAceStepTextEncoderPlusPlus": "ACE-Step Text Encoder PLUSPLUS"}
