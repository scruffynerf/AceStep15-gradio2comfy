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
    CATEGORY = "Scromfy/Ace-Step/Whisper"

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
                "word_timestamps": ("BOOLEAN", {"default": False}),
                "vad_filter": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("TRANSCRIPTIONS", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("transcriptions", "srt_text", "vtt_text", "lrc_text")
    FUNCTION = "transcribe"
    CATEGORY = "Scromfy/Ace-Step/Whisper"

    def transcribe(self, model, audio, language="auto", task="transcribe", beam_size=5, word_timestamps=False, vad_filter=True):
        source = None
        
        # Determine source: Direct audio input (required)
        if audio is not None:
            # ComfyUI audio format: {"waveform": torch.Tensor [batch, channels, samples], "sample_rate": int}
            waveform = audio["waveform"]
            sample_rate = audio["sample_rate"]
            
            # Convert to mono if needed
            if waveform.shape[1] > 1:
                waveform = torch.mean(waveform, dim=1)
            else:
                waveform = waveform.squeeze(1)
                
            # Resample to 16000 as required by Faster Whisper numpy array input
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
                
            # Take first item in batch and convert to numpy
            source = waveform[0].cpu().numpy()
        else:
            raise ValueError("No valid audio source provided. Please connect an AUDIO input.")

        # Map language name to code
        lang_code = FULL_LANG_MAPPING.get(language) if language != "auto" else None

        segments, info = model.transcribe(
            audio=source,
            language=lang_code,
            task=task,
            beam_size=beam_size,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter
        )

        # Progress bar setup (if possible to estimate)
        # Segments is a generator, so we don't know the count yet
        pbar = ProgressBar(100)
        
        results = []
        for segment in segments:
            results.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })
            # Crude progress update based on time relative to info.duration if available
            if hasattr(info, 'duration') and info.duration > 0:
                pbar.update_absolute(int((segment.end / info.duration) * 100))

        # Format outputs directly
        srt = format_subtitles(results, ".srt")
        vtt = format_subtitles(results, ".vtt")
        lrc = format_subtitles(results, ".lrc")

        return (results, srt, vtt, lrc)

class AceStepSaveSubtitleLyrics:
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
    CATEGORY = "Scromfy/Ace-Step/Whisper"
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
