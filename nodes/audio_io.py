"""
Audio I/O nodes for ACE-Step
Ported from acestep_tweaked/nodes_audio.py
"""

from __future__ import annotations

import torchaudio
import torch
import comfy.model_management
import folder_paths
import os
import io
import json
import struct
import random
import hashlib
import node_helpers
from comfy.cli_args import args
from comfy.comfy_types import FileLocator


class EmptyLatentAudio:
    """Create empty audio latent space for generation"""
    
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seconds": ("FLOAT", {"default": 47.6, "min": 1.0, "max": 1000.0, "step": 0.1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "Scromfy/Ace-Step/audio"

    def generate(self, seconds, batch_size):
        # ACE-Step audio latents: 44100 Hz / 2048 hop / 2 downscale
        length = round((seconds * 44100 / 2048) / 2) * 2
        latent = torch.zeros([batch_size, 64, length], device=self.device)
        return ({"samples": latent, "type": "audio"}, )


class VAEEncodeAudio:
    """Encode audio waveform to latent space"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO", ),
                "vae": ("VAE", )
            }
        }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "Scromfy/Ace-Step/audio"

    def encode(self, vae, audio):
        sample_rate = audio["sample_rate"]
        # ACE-Step requires 44.1kHz
        if 44100 != sample_rate:
            waveform = torchaudio.functional.resample(audio["waveform"], sample_rate, 44100)
        else:
            waveform = audio["waveform"]

        t = vae.encode(waveform.movedim(1, -1))
        return ({"samples": t}, )


class VAEDecodeAudio:
    """Decode latent space to audio waveform"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT", ),
                "vae": ("VAE", )
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "decode"
    CATEGORY = "Scromfy/Ace-Step/audio"

    def decode(self, vae, samples):
        audio = vae.decode(samples["samples"]).movedim(-1, 1)
        # Normalize audio to prevent clipping
        std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
        std[std < 1.0] = 1.0
        audio /= std
        return ({"waveform": audio, "sample_rate": 44100}, )


def create_vorbis_comment_block(comment_dict, last_block):
    """Create FLAC vorbis comment metadata block"""
    vendor_string = b'ComfyUI-ACE-Step'
    vendor_length = len(vendor_string)

    comments = []
    for key, value in comment_dict.items():
        comment = f"{key}={value}".encode('utf-8')
        comments.append(struct.pack('<I', len(comment)) + comment)

    num_comments = len(comments)
    comments_data = b''.join(comments)

    block_data = (
        struct.pack('<I', vendor_length) + vendor_string +
        struct.pack('<I', num_comments) + comments_data
    )

    block_header = struct.pack('>I', len(block_data))[1:]
    if last_block:
        block_header = bytes([block_header[0] | 0x80]) + block_header[1:]
    block_header = bytes([4]) + block_header

    return block_header + block_data


class LoadAudio:
    """Load audio files (mp3, flac, wav, ogg)"""
    
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = folder_paths.filter_files_content_types(
            folder_paths.listdir(input_dir), 
            ["audio", "video"]
        )
        return {
            "required": {
                "audio": (sorted(files), {"audio_upload": True}),
            }
        }

    CATEGORY = "Scromfy/Ace-Step/audio"
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "load"

    def load(self, **kwargs):
        if 'audio' not in kwargs or kwargs['audio'] is None:
            raise ValueError("No audio file provided")
            
        audio_file = kwargs['audio']
        
        if isinstance(audio_file, FileLocator):
            audio_path = audio_file.to_local_path()
        elif isinstance(audio_file, str):
            audio_path = folder_paths.get_annotated_filepath(audio_file)
        else:
            raise ValueError(f"Unexpected audio file type: {type(audio_file)}")

        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert mono to stereo if needed
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        # Take first 2 channels if more than stereo
        elif waveform.shape[0] > 2:
            waveform = waveform[:2, :]
            
        # Normalize to stereo, 44.1kHz
        if sample_rate != 44100:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 44100)
            sample_rate = 44100

        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        return (audio,)

    @classmethod
    def IS_CHANGED(s, **kwargs):
        audio_file = kwargs.get('audio')
        if audio_file is None:
            return ""
            
        if isinstance(audio_file, FileLocator):
            m = hashlib.sha256()
            with open(audio_file.to_local_path(), 'rb') as f:
                m.update(f.read())
            return m.digest().hex()
        elif isinstance(audio_file, str):
            image_path = folder_paths.get_annotated_filepath(audio_file)
            m = hashlib.sha256()
            with open(image_path, 'rb') as f:
                m.update(f.read())
            return m.digest().hex()
        return ""

    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        if 'audio' not in kwargs or kwargs['audio'] is None:
            return "No audio file provided"
        return True


class SaveAudio:
    """Save audio with metadata support"""
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "ACE-Step"}),
            },
            "optional": {
                "format": (["flac", "mp3", "wav"], {"default": "flac"}),
                "metadata": ("STRING", {"default": "", "multiline": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_audio"
    OUTPUT_NODE = True
    CATEGORY = "Scromfy/Ace-Step/audio"

    def save_audio(self, audio, filename_prefix="ACE-Step", format="flac", metadata="", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir
        )
        results = list()

        metadata_dict = {}
        if metadata:
            try:
                metadata_dict = json.loads(metadata)
            except:
                # If not JSON, treat as simple key=value pairs
                for line in metadata.split('\n'):
                    if '=' in line:
                        key, val = line.split('=', 1)
                        metadata_dict[key.strip()] = val.strip()

        for batch_number, waveform in enumerate(audio["waveform"]):
            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.{format}"
            
            audio_path = os.path.join(full_output_folder, file)
            
            if format == "flac":
                # Add metadata to FLAC
                audio_io = io.BytesIO()
                torchaudio.save(audio_io, waveform, audio["sample_rate"], format="flac")
                audio_io.seek(0)
                
                flac_data = audio_io.read()
                
                # FLAC header is 4 bytes "fLaC", then metadata blocks
                if flac_data[:4] == b'fLaC':
                    # Find end of existing metadata blocks
                    pos = 4
                    while pos < len(flac_data):
                        is_last = flac_data[pos] & 0x80
                        pos += 4
                        block_size = int.from_bytes(flac_data[pos-3:pos], 'big')
                        if is_last:
                            break
                        pos += block_size
                    
                    # Insert vorbis comment
                    vorbis_block = create_vorbis_comment_block(metadata_dict, last_block=True)
                    flac_with_metadata = flac_data[:pos] + vorbis_block + flac_data[pos:]
                    
                    with open(audio_path, 'wb') as f:
                        f.write(flac_with_metadata)
                else:
                    # Fallback: save without metadata
                    torchaudio.save(audio_path, waveform, audio["sample_rate"], format="flac")
            else:
                # MP3, WAV don't support complex metadata easily
                torchaudio.save(audio_path, waveform, audio["sample_rate"], format=format)

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return {"ui": {"audio": results}}


class PreviewAudio:
    """Preview audio in ComfyUI interface"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"audio": ("AUDIO",), }}

    CATEGORY = "Scromfy/Ace-Step/audio"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview_audio"

    def preview_audio(self, audio):
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            "audio_preview", folder_paths.get_temp_directory()
        )
        
        filename_with_batch_num = filename.replace("%batch_num%", "0")
        file = f"{filename_with_batch_num}_{counter:05}_.flac"
        file_path = os.path.join(full_output_folder, file)
        
        torchaudio.save(file_path, audio["waveform"][0], audio["sample_rate"], format="flac")
        
        return {"ui": {"audio": [{"filename": file, "subfolder": subfolder, "type": "temp"}]}}


# Node registration
NODE_CLASS_MAPPINGS = {
    "EmptyLatentAudio": EmptyLatentAudio,
    "VAEEncodeAudio": VAEEncodeAudio,
    "VAEDecodeAudio": VAEDecodeAudio,
    "LoadAudio": LoadAudio,
    "SaveAudio": SaveAudio,
    "PreviewAudio": PreviewAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EmptyLatentAudio": "Empty Latent Audio",
    "VAEEncodeAudio": "VAE Encode (Audio)",
    "VAEDecodeAudio": "VAE Decode (Audio)",
    "LoadAudio": "Load Audio",
    "SaveAudio": "Save Audio",
    "PreviewAudio": "Preview Audio",
}
