"""SaveAudio node for ACE-Step"""
import torchaudio
import folder_paths
import os
import io
import json
from .includes.audio_utils import create_vorbis_comment_block

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
                audio_io_buf = io.BytesIO()
                torchaudio.save(audio_io_buf, waveform, audio["sample_rate"], format="flac")
                audio_io_buf.seek(0)
                
                flac_data = audio_io_buf.read()
                
                if flac_data[:4] == b'fLaC':
                    pos = 4
                    while pos < len(flac_data):
                        is_last = flac_data[pos] & 0x80
                        pos += 4
                        block_size = int.from_bytes(flac_data[pos-3:pos], 'big')
                        if is_last:
                            break
                        pos += block_size
                    
                    vorbis_block = create_vorbis_comment_block(metadata_dict, last_block=True)
                    flac_with_metadata = flac_data[:pos] + vorbis_block + flac_data[pos:]
                    
                    with open(audio_path, 'wb') as f:
                        f.write(flac_with_metadata)
                else:
                    torchaudio.save(audio_path, waveform, audio["sample_rate"], format="flac")
            else:
                torchaudio.save(audio_path, waveform, audio["sample_rate"], format=format)

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return {"ui": {"audio": results}}


NODE_CLASS_MAPPINGS = {
    "SaveAudio": SaveAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveAudio": "Save Audio",
}
