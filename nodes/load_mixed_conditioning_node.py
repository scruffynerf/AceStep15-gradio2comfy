"""AceStepConditioningMixerLoader node for ACE-Step"""
import os
import json
import torch
import random
from safetensors.torch import load_file

def get_conditioning_files(suffix):
    base_path = "output/conditioning"
    if not os.path.exists(base_path):
        return ["none", "random"]
    
    files = [f for f in os.listdir(base_path) if f.endswith(suffix)]
    return sorted(files) + ["none", "random"]

class AceStepConditioningMixerLoader:
    """Load and mix specific conditioning components from saved files (safetensors/json)"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "timbre_tensor_file": (get_conditioning_files("_timbre.safetensors"),),
                "lyrics_file": (get_conditioning_files("_lyrics.safetensors"),),
                "audio_codes_file": (get_conditioning_files("_codes.json"),),
                "empty_mode": (["silence", "zeros", "ones", "random"], {"default": "silence"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("conditioning", "conditioning_info")
    FUNCTION = "load_and_mix"
    CATEGORY = "Scromfy/Ace-Step/loaders"

    @classmethod
    def IS_CHANGED(s, timbre_tensor_file, lyrics_file, audio_codes_file, empty_mode, seed):
        return f"{timbre_tensor_file}_{lyrics_file}_{audio_codes_file}_{empty_mode}_{seed}"

    def load_and_mix(self, timbre_tensor_file, lyrics_file, audio_codes_file, empty_mode, seed):
        base_path = "output/conditioning"
        rng = random.Random(seed)
        
        def pick_file(selected, suffix):
            if selected == "random":
                options = [f for f in os.listdir(base_path) if f.endswith(suffix)] if os.path.exists(base_path) else []
                if not options:
                    return "none"
                return rng.choice(options)
            return selected

        # Resolve randoms
        timbre_tensor_file = pick_file(timbre_tensor_file, "_timbre.safetensors")
        lyrics_file = pick_file(lyrics_file, "_lyrics.safetensors")
        audio_codes_file = pick_file(audio_codes_file, "_codes.json")

        # Determine Timbre Tensor (Required base or generated default)
        if timbre_tensor_file != "none":
            timbre_path = os.path.join(base_path, timbre_tensor_file)
            timbre_tensor = load_file(timbre_path).get("timbre")
            # If it's 2D [L, D], unsqueeze to [1, L, D]
            if timbre_tensor.dim() == 2:
                timbre_tensor = timbre_tensor.unsqueeze(0)
        else:
            timbre_tensor = None

        metadata = {}
        
        # Load other components
        lyrics = None
        if lyrics_file != "none":
            lyrics = load_file(os.path.join(base_path, lyrics_file)).get("lyrics")
            if lyrics.dim() == 2:
                lyrics = lyrics.unsqueeze(0)
            metadata["conditioning_lyrics"] = lyrics
            
        codes = None
        if audio_codes_file != "none":
            with open(os.path.join(base_path, audio_codes_file), "r") as f:
                codes = json.load(f)
                metadata["audio_codes"] = codes
        
        # Synchronize sequence lengths for missing components
        batch_size = 1
        seq_len = 1
        device = "cpu"
        
        if timbre_tensor is not None:
            batch_size = timbre_tensor.shape[0]
            seq_len = timbre_tensor.shape[1]
            device = timbre_tensor.device
        elif lyrics is not None:
            batch_size = lyrics.shape[0]
            seq_len = lyrics.shape[1]
            device = lyrics.device

        # Helper to create empty tensors
        def create_empty(b, l, d, dev, mode, s, is_lyrics=False):
            # Check for explicit empty lyrics file if requested or as primary fallback for lyrics
            if is_lyrics or mode == "silence":
                try:
                    # Resolve path relative to this file
                    base_dir = os.path.dirname(__file__)
                    if mode == "silence":
                        filename = "silence.safetensors"
                    else:
                        filename = "empty_lyrics.safetensors"
                    empty_path = os.path.join(base_dir, "includes", "emptytensors", filename)
                    if os.path.exists(empty_path):
                        from safetensors.torch import load_file
                        if filename == "silence.safetensors":
                            loaded = load_file(empty_path).get("silence")
                        else:
                            loaded = load_file(empty_path).get("lyrics")
                        if loaded is not None:
                            # Ensure it has batch dim
                            if loaded.dim() == 2:
                                loaded = loaded.unsqueeze(0)
                            
                            # Interpolate to match target sequence length if different
                            if loaded.shape[0] != b or loaded.shape[1] != l:
                                # [B, L, D] -> [B, D, L] for interpolate
                                loaded = loaded.permute(0, 2, 1)
                                loaded = torch.nn.functional.interpolate(loaded, size=l, mode='linear', align_corners=False)
                                # [B, D, L] -> [B, L, D]
                                loaded = loaded.permute(0, 2, 1)
                                
                            return loaded.to(dev)
                except Exception as e:
                    print(f"AceStepMixerLoader: Failed to load safetensors from includes/emptytensors: {e}")

            if mode == "zeros":
                return torch.zeros((b, l, d), device=dev)
            elif mode == "ones":
                return torch.ones((b, l, d), device=dev)
            elif mode == "random":
                generator = torch.Generator(device=dev)
                generator.manual_seed(s)
                return torch.randn((b, l, d), device=dev, generator=generator)
            return torch.zeros((b, l, d), device=dev)

        # If no timbre_tensor, generate a default
        if timbre_tensor is None:
            timbre_tensor = create_empty(batch_size, seq_len, 1024, device, empty_mode, seed)
            base_timbre = empty_mode
        else:
            base_timbre = timbre_tensor_file.replace("_timbre.safetensors", "")

        # Ensure lyrics is a tensor if missing
        if lyrics is None:
            lyrics = create_empty(batch_size, seq_len, 1024, device, empty_mode, seed, is_lyrics=True)
            metadata["conditioning_lyrics"] = lyrics
            
        # Construct filename-safe info string: timbre_lyrics_codes
        def get_base(filename, suffix):
            if filename == "none": return None
            return filename.replace(suffix, "")

        # base_timbre already defined above in the new logic
        base_lyrics = get_base(lyrics_file, "_lyrics.safetensors")
        base_codes = get_base(audio_codes_file, "_codes.json")
        
        parts = [base_timbre]
        parts.append(base_lyrics if base_lyrics else f"{empty_mode}_lyrics")
        parts.append(base_codes if base_codes else f"nocodes")
        
        filename_info = "_".join(parts)
            
        return ([[timbre_tensor, metadata]], filename_info)

NODE_CLASS_MAPPINGS = {
    "AceStepConditioningMixerLoader": AceStepConditioningMixerLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepConditioningMixerLoader": "Conditioning Mixer Loader",
}
