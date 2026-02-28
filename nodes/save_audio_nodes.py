from __future__ import annotations

import av
import torchaudio
import torch
import comfy.model_management
import folder_paths
import os
import io
import json
import random
import hashlib
import node_helpers
import logging
from comfy.cli_args import args

def f32_pcm(wav: torch.Tensor) -> torch.Tensor:
    """Convert audio to float 32 bits PCM format."""
    if wav.dtype.is_floating_point:
        return wav
    elif wav.dtype == torch.int16:
        return wav.float() / (2 ** 15)
    elif wav.dtype == torch.int32:
        return wav.float() / (2 ** 31)
    raise ValueError(f"Unsupported wav dtype: {wav.dtype}")

def scromfy_save_audio(self, audio, filename_prefix="ComfyUI", format="flac", prompt=None, extra_pnginfo=None, quality="128k"):
    filename_prefix += self.prefix_append
    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
    results = []
    saved_filepaths = []

    # Prepare metadata dictionary
    metadata = {}
    if not args.disable_metadata:
        if prompt is not None:
            metadata["prompt"] = json.dumps(prompt)
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata[x] = json.dumps(extra_pnginfo[x])

    # Opus supported sample rates
    OPUS_RATES = [8000, 12000, 16000, 24000, 48000]

    for (batch_number, waveform) in enumerate(audio["waveform"].cpu()):
        filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
        file = f"{filename_with_batch_num}_{counter:05}_.{format}"
        output_path = os.path.join(full_output_folder, file)

        # Use original sample rate initially
        sample_rate = audio["sample_rate"]

        # Handle Opus sample rate requirements
        if format == "opus":
            if sample_rate > 48000:
                sample_rate = 48000
            elif sample_rate not in OPUS_RATES:
                for rate in sorted(OPUS_RATES):
                    if rate > sample_rate:
                        sample_rate = rate
                        break
                if sample_rate not in OPUS_RATES:
                    sample_rate = 48000

            if sample_rate != audio["sample_rate"]:
                waveform = torchaudio.functional.resample(waveform, audio["sample_rate"], sample_rate)

        # Create output with specified format
        output_buffer = io.BytesIO()
        output_container = av.open(output_buffer, mode='w', format=format)

        for key, value in metadata.items():
            output_container.metadata[key] = value

        layout = 'mono' if waveform.shape[0] == 1 else 'stereo'
        if format == "opus":
            out_stream = output_container.add_stream("libopus", rate=sample_rate, layout=layout)
            bitrates = {"64k": 64000, "96k": 96000, "128k": 128000, "192k": 192000, "320k": 320000}
            out_stream.bit_rate = bitrates.get(quality, 128000)
        elif format == "mp3":
            out_stream = output_container.add_stream("libmp3lame", rate=sample_rate, layout=layout)
            if quality == "V0":
                out_stream.codec_context.qscale = 1
            else:
                out_stream.bit_rate = 128000 if quality == "128k" else 320000
        else:
            out_stream = output_container.add_stream("flac", rate=sample_rate, layout=layout)

        frame = av.AudioFrame.from_ndarray(waveform.movedim(0, 1).reshape(1, -1).float().numpy(), format='flt', layout=layout)
        frame.sample_rate = sample_rate
        frame.pts = 0
        output_container.mux(out_stream.encode(frame))
        output_container.mux(out_stream.encode(None))
        output_container.close()

        output_buffer.seek(0)
        with open(output_path, 'wb') as f:
            f.write(output_buffer.getbuffer())

        results.append({
            "filename": file,
            "subfolder": subfolder,
            "type": self.type
        })
        
        # Save the absolute path without extension
        path_no_ext = os.path.splitext(os.path.abspath(output_path))[0]
        saved_filepaths.append(path_no_ext)
        
        counter += 1

    # Return UI results and the first filepath (or comma-separated if batch? 
    # Usually ACE-Step is 1:1, so we'll return the first one for best compatibility with Whisper nodes)
    primary_path = saved_filepaths[0] if saved_filepaths else ""
    
    return { "ui": { "audio": results }, "result": (primary_path,) }


class ScromfySaveAudio:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "audio": ("AUDIO", ),
                            "filename_prefix": ("STRING", {"default": "audio/ACE-Step"}),
                            },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Scromfy/Ace-Step/save"

    def save(self, audio, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        res = scromfy_save_audio(self, audio, filename_prefix, "flac", prompt, extra_pnginfo)
        return {"ui": res["ui"], "result": res["result"]}

class ScromfySaveAudioMP3:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "audio": ("AUDIO", ),
                            "filename_prefix": ("STRING", {"default": "audio/ACE-Step"}),
                            "quality": (["V0", "128k", "320k"], {"default": "V0"}),
                            },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Scromfy/Ace-Step/save"

    def save(self, audio, filename_prefix="ComfyUI", quality="V0", prompt=None, extra_pnginfo=None):
        res = scromfy_save_audio(self, audio, filename_prefix, "mp3", prompt, extra_pnginfo, quality)
        return {"ui": res["ui"], "result": res["result"]}

class ScromfySaveAudioOpus:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "audio": ("AUDIO", ),
                            "filename_prefix": ("STRING", {"default": "audio/ACE-Step"}),
                            "quality": (["64k", "96k", "128k", "192k", "320k"], {"default": "128k"}),
                            },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Scromfy/Ace-Step/save"

    def save(self, audio, filename_prefix="ComfyUI", quality="128k", prompt=None, extra_pnginfo=None):
        res = scromfy_save_audio(self, audio, filename_prefix, "opus", prompt, extra_pnginfo, quality)
        return {"ui": res["ui"], "result": res["result"]}

class ScromfyPreviewAudio(ScromfySaveAudio):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"audio": ("AUDIO", ), },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    CATEGORY = "Scromfy/Ace-Step/audio"

NODE_CLASS_MAPPINGS = {
    "ScromfySaveAudio": ScromfySaveAudio,
    "ScromfySaveAudioMP3": ScromfySaveAudioMP3,
    "ScromfySaveAudioOpus": ScromfySaveAudioOpus,
    "ScromfyPreviewAudio": ScromfyPreviewAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScromfySaveAudio": "Scromfy Save Audio (FLAC/WAV)",
    "ScromfySaveAudioMP3": "Scromfy Save Audio (MP3)",
    "ScromfySaveAudioOpus": "Scromfy Save Audio (Opus)",
    "ScromfyPreviewAudio": "Scromfy Preview Audio",
}
