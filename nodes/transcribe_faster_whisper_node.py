import os
import torch
import torchaudio
import numpy as np
import faster_whisper
from typing import Union, BinaryIO, Dict, List, Tuple
from comfy.utils import ProgressBar
import folder_paths

from .includes.whisper_utils import (
    collect_model_paths, 
    format_subtitles, 
    AVAILABLE_SUBTITLE_FORMATS,
    FULL_LANG_MAPPING
)

class AceStepLoadFasterWhisperModel:
    """Loads a Faster-Whisper model for local audio transcription.
    
    Inputs:
        model (STRING): Selected model size/name.
        device (STRING): CPU/CUDA target.
        compute_type (STRING): Quantization/precision.
        
    Outputs:
        model (FASTER_WHISPER_MODEL): Loaded transcription model.
    """
    @classmethod
    def INPUT_TYPES(s):
        models = list(collect_model_paths().keys())
        return {
            "required": {
                "model": (models,),
                "device": (['cuda', 'cpu', 'auto'],),
                "compute_type": (['float16', 'float32', 'int8_float16', 'int8'], {"default": "float16"}),
            },
        }

    RETURN_TYPES = ("FASTER_WHISPER_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "Scromfy/Ace-Step/Audio"

    def load_model(self, model: str, device: str, compute_type: str):
        from .includes.whisper_utils import faster_whisper_model_dir
        
        model_name_or_path = collect_model_paths()[model]
        
        # Load model
        whisper_model = faster_whisper.WhisperModel(
            model_size_or_path=model_name_or_path,
            device=device,
            compute_type=compute_type,
            download_root=faster_whisper_model_dir,
            local_files_only=False
        )
        return (whisper_model,)

class AceStepFasterWhisperTranscription:
    """Transcribes audio using a loaded Faster-Whisper model.
    Supports subtitle generation, VAD filtering, and multi-language.
    
    Inputs:
        model (FASTER_WHISPER_MODEL): Loader output.
        audio (AUDIO): Target audio waveform.
        (Optional variables: language, task, beam_size, vad_filter, etc.)
        
    Outputs:
        transcriptions (TRANSCRIPTIONS): Raw segment dictionary array.
        srt_text (STRING): SubRip formatted string.
        vtt_text (STRING): WebVTT formatted string.
        lrc_text (STRING): Lyric-sync formatted string.
    """
    @classmethod
    def INPUT_TYPES(s):
        langs = ["auto"] + sorted(list(FULL_LANG_MAPPING.keys()))
        return {
            "required": {
                "model": ("FASTER_WHISPER_MODEL",),
                "audio": ("AUDIO",),
            },
            "optional": {
                "language": (langs, {"default": "auto"}),
                "task": (["transcribe", "translate"],),
                "beam_size": ("INT", {"default": 5, "min": 1, "max": 20}),
                "best_of": ("INT", {"default": 5, "min": 1, "max": 20}),
                "patience": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "condition_on_previous_text": ("BOOLEAN", {"default": True}),
                "initial_prompt": ("STRING", {"default": ""}),
                "prefix": ("STRING", {"default": ""}),
                "hotwords": ("STRING", {"default": ""}),
                "word_timestamps": ("BOOLEAN", {"default": False}),
                "vad_filter": ("BOOLEAN", {"default": True}),
                "vad_parameters": ("STRING", {"default": ""}),
                "log_prob_threshold": ("FLOAT", {"default": -1.0, "min": -20.0, "max": 0.0, "step": 0.1}),
                "no_speech_threshold": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.1}),
                "compression_ratio_threshold": ("FLOAT", {"default": 2.4, "min": 0.0, "max": 10.0, "step": 0.1}),
                "length_penalty": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "repetition_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "no_repeat_ngram_size": ("INT", {"default": 0, "min": 0, "max": 10}),
                "suppress_blank": ("BOOLEAN", {"default": True}),
                "suppress_tokens": ("STRING", {"default": "[-1]"}),
                "max_initial_timestamp": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "prepend_punctuations": ("STRING", {"default": "\"'“¿([{-"}),
                "append_punctuations": ("STRING", {"default": "\"'.。,，!！?？:：”)]}、"}),
                "max_new_tokens": ("INT", {"default": 0, "min": 0, "max": 4096}), # 0 = None
                "chunk_length": ("INT", {"default": 0, "min": 0, "max": 30}),   # 0 = None
                "hallucination_silence_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}), # 0 = None
                "language_detection_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}), # 0 = None
                "language_detection_segments": ("INT", {"default": 1, "min": 1, "max": 100}),
                "prompt_reset_on_temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                "without_timestamps": ("BOOLEAN", {"default": False}),
                "clip_timestamps": ("STRING", {"default": "0"}),
            }
        }

    RETURN_TYPES = ("TRANSCRIPTIONS", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("transcriptions", "srt_text", "vtt_text", "lrc_text")
    FUNCTION = "transcribe"
    CATEGORY = "Scromfy/Ace-Step/Lyrics"

    def transcribe(self, 
                  model, 
                  audio, 
                  language="auto", 
                  task="transcribe", 
                  beam_size=5, 
                  best_of=5, 
                  patience=1.0, 
                  temperature=0.0, 
                  condition_on_previous_text=True,
                  initial_prompt="", 
                  prefix="", 
                  hotwords="", 
                  word_timestamps=False, 
                  vad_filter=True, 
                  vad_parameters="", 
                  log_prob_threshold=-1.0, 
                  no_speech_threshold=0.6, 
                  compression_ratio_threshold=2.4, 
                  length_penalty=1.0, 
                  repetition_penalty=1.0, 
                  no_repeat_ngram_size=0,
                  suppress_blank=True, 
                  suppress_tokens="[-1]", 
                  max_initial_timestamp=1.0,
                  prepend_punctuations="\"'“¿([{-", 
                  append_punctuations="\"'.。,，!！?？:：”)]}、",
                  max_new_tokens=0, 
                  chunk_length=0, 
                  hallucination_silence_threshold=0.0,
                  language_detection_threshold=0.5, 
                  language_detection_segments=1,
                  prompt_reset_on_temperature=0.5, 
                  without_timestamps=False,
                  clip_timestamps="0"):
        
        # Determine source: Direct audio input (required)
        if audio is not None:
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            if waveform.shape[1] > 1:
                waveform = torch.mean(waveform, dim=1)
            else:
                waveform = waveform.squeeze(1)
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            source = waveform[0].cpu().numpy()
        else:
            raise ValueError("No valid audio source provided. Please connect an AUDIO input.")

        # Map language name to code
        lang_code = FULL_LANG_MAPPING.get(language) if language != "auto" else None

        # Clean "None" values and complex types
        p = {
            "language": lang_code,
            "task": task,
            "beam_size": beam_size,
            "best_of": best_of,
            "patience": patience,
            "temperature": temperature,
            "condition_on_previous_text": condition_on_previous_text,
            "initial_prompt": initial_prompt if initial_prompt else None,
            "prefix": prefix if prefix else None,
            "hotwords": hotwords if hotwords else None,
            "word_timestamps": word_timestamps,
            "vad_filter": vad_filter,
            "log_prob_threshold": log_prob_threshold,
            "no_speech_threshold": no_speech_threshold,
            "compression_ratio_threshold": compression_ratio_threshold,
            "length_penalty": length_penalty,
            "repetition_penalty": repetition_penalty,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "suppress_blank": suppress_blank,
            "max_initial_timestamp": max_initial_timestamp,
            "prepend_punctuations": prepend_punctuations,
            "append_punctuations": append_punctuations,
            "language_detection_segments": language_detection_segments,
            "prompt_reset_on_temperature": prompt_reset_on_temperature,
            "without_timestamps": without_timestamps,
            "clip_timestamps": clip_timestamps,
        }

        # Handling "None" overrides for 0/empty
        if max_new_tokens > 0: p["max_new_tokens"] = max_new_tokens
        if chunk_length > 0: p["chunk_length"] = chunk_length
        if hallucination_silence_threshold > 0: p["hallucination_silence_threshold"] = hallucination_silence_threshold
        if language_detection_threshold > 0: p["language_detection_threshold"] = language_detection_threshold

        # Parse complex strings
        if suppress_tokens:
            try:
                p["suppress_tokens"] = eval(suppress_tokens)
            except:
                p["suppress_tokens"] = [-1]
        
        if vad_parameters:
            try:
                import json
                p["vad_parameters"] = json.loads(vad_parameters)
            except:
                pass

        # Perform Transcription
        segments, info = model.transcribe(audio=source, **p)

        pbar = ProgressBar(100)
        results = []
        for segment in segments:
            results.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })
            if hasattr(info, 'duration') and info.duration > 0:
                pbar.update_absolute(int((segment.end / info.duration) * 100))

        # Format outputs directly
        srt = format_subtitles(results, ".srt")
        vtt = format_subtitles(results, ".vtt")
        lrc = format_subtitles(results, ".lrc")

        return (results, srt, vtt, lrc)

class AceStepSaveSubtitleLyrics:
    """Utility to save generated subtitle/lyrics strings to disk.
    
    Inputs:
        text (STRING): The subtitle text content.
        filepath_base (STRING): Absolute path without extension.
        extension (STRING): Target format (.srt, .vtt, .lrc).
        
    Outputs:
        filepath (STRING): Absolute path to the saved file.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "filepath_base": ("STRING", {"forceInput": True}),
                "extension": (AVAILABLE_SUBTITLE_FORMATS, {"default": ".lrc"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save"
    CATEGORY = "Scromfy/Ace-Step/Lyrics"
    OUTPUT_NODE = True

    def save(self, text: str, filepath_base: str, extension: str):
        # Combine base path with chosen extension
        full_path = filepath_base + extension
        
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(text)
            
        return (full_path,)

NODE_CLASS_MAPPINGS = {
    "AceStepLoadFasterWhisperModel": AceStepLoadFasterWhisperModel,
    "AceStepFasterWhisperTranscription": AceStepFasterWhisperTranscription,
    "AceStepSaveSubtitleLyrics": AceStepSaveSubtitleLyrics,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepLoadFasterWhisperModel": "Faster Whisper Loader",
    "AceStepFasterWhisperTranscription": "Faster Whisper Transcribe",
    "AceStepSaveSubtitleLyrics": "Save Subtitle/Lyrics (Matched)",
}
