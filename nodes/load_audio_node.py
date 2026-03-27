"""LoadAudio node for ACE-Step"""
import torchaudio
import torch
import os
import folder_paths
import hashlib
from comfy.comfy_types import FileLocator

class ScromfyLoadAudio:
    """Load audio files (mp3, flac, wav, ogg) from the ComfyUI input directory.
    
    Automatically resamples to the expected 48kHz stereo geometry required by ACE-Step.
    
    Inputs:
        audio (STRING): Dropdown of audio files located in the ComfyUI input directory.
        
    Outputs:
        audio (AUDIO): Standard audio dict {waveform, sample_rate}.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        
        # Ensure the directory exists to avoid FileNotFoundError
        if not os.path.exists(input_dir):
            os.makedirs(input_dir, exist_ok=True)
            
        files = folder_paths.filter_files_content_types(
            os.listdir(input_dir), 
            ["audio", "video"]
        )
        return {
            "required": {
                "audio": (sorted(files), {"audio_upload": True}),
            }
        }

    CATEGORY = "Scromfy/Ace-Step/Audio"
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
            
        # Normalize to stereo, 48kHz
        if sample_rate != 48000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 48000)
            sample_rate = 48000

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


NODE_CLASS_MAPPINGS = {
    "ScromfyLoadAudio": ScromfyLoadAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScromfyLoadAudio": "Scromfy Load Audio",
}
