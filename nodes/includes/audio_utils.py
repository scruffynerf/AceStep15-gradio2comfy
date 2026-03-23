"""Audio utilities for ACE-Step ComfyUI nodes.

Merged from audio_utils.py (FLAC metadata block generation)
and audio_save_utils.py (multi-format audio saving).
"""

from __future__ import annotations

import struct
import io
import json
import os

import av
import torch
import torchaudio
import folder_paths
from comfy.cli_args import args


# ─────────────────────────────────────────────────────────────────────────────
#  FLAC metadata helpers
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-format audio saving
# ─────────────────────────────────────────────────────────────────────────────

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

    primary_path = saved_filepaths[0] if saved_filepaths else ""
    return { "ui": { "audio": results }, "result": (primary_path,) }
