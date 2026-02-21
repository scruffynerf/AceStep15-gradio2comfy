"""LoadAudio node for ACE-Step"""
import torchaudio
import torch
import folder_paths
import hashlib
from comfy.comfy_types import FileLocator

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


NODE_CLASS_MAPPINGS = {
    "LoadAudio": LoadAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAudio": "Load Audio",
}
