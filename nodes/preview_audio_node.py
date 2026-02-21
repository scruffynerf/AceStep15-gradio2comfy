"""PreviewAudio node for ACE-Step"""
import torchaudio
import folder_paths
import os

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


NODE_CLASS_MAPPINGS = {
    "PreviewAudio": PreviewAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PreviewAudio": "Preview Audio",
}
