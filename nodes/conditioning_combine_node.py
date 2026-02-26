"""AceStepConditioningCombine node for ACE-Step"""
import torch
import os

class AceStepConditioningCombine:
    """Assemble separate components into a full ACE-Step conditioning"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "empty_mode": (["zeros", "ones", "random"], {"default": "zeros"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "timbre_tensor": ("TENSOR",),
                "pooled_output": ("TENSOR",),
                "lyrics_tensor": ("TENSOR",),
                "audio_codes": ("LIST",),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "combine"
    CATEGORY = "Scromfy/Ace-Step/advanced"

    def combine(self, empty_mode, seed, timbre_tensor=None, pooled_output=None, lyrics_tensor=None, audio_codes=None):
        # If both are missing, we use length 1
        # If one is present, we match its length
        
        batch_size = 1
        seq_len = 1
        device = "cpu"
        
        # 1. Inspect available tensors for dimensions
        if timbre_tensor is not None:
            if timbre_tensor.dim() == 2:
                timbre_tensor = timbre_tensor.unsqueeze(0)
            batch_size = timbre_tensor.shape[0]
            seq_len = timbre_tensor.shape[1]
            device = timbre_tensor.device
        elif lyrics_tensor is not None:
            if lyrics_tensor.dim() == 2:
                lyrics_tensor = lyrics_tensor.unsqueeze(0)
            batch_size = lyrics_tensor.shape[0]
            seq_len = lyrics_tensor.shape[1]
            device = lyrics_tensor.device
        elif pooled_output is not None:
            batch_size = pooled_output.shape[0]
            device = pooled_output.device

        # 2. Backfill missing tensors with matching dimensions and mode
        def create_empty(b, l, d, dev, mode, s, is_lyrics=False):
            # Check for explicit empty lyrics file if requested or as primary fallback for lyrics
            if is_lyrics:
                try:
                    # Resolve path relative to this file
                    base_dir = os.path.dirname(__file__)
                    empty_path = os.path.join(base_dir, "includes", "emptytensors", "empty_lyrics.safetensors")
                    if os.path.exists(empty_path):
                        from safetensors.torch import load_file
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
                    print(f"AceStepCombine: Failed to load empty_lyrics.safetensors: {e}")

            if mode == "zeros":
                return torch.zeros((b, l, d), device=dev)
            elif mode == "ones":
                return torch.ones((b, l, d), device=dev)
            elif mode == "random":
                generator = torch.Generator(device=dev)
                generator.manual_seed(s)
                return torch.randn((b, l, d), device=dev, generator=generator)
            return torch.zeros((b, l, d), device=dev)

        if timbre_tensor is None:
            timbre_tensor = create_empty(batch_size, seq_len, 1024, device, empty_mode, seed)
            
        if lyrics_tensor is None:
            lyrics_tensor = create_empty(batch_size, seq_len, 1024, device, empty_mode, seed + 1, is_lyrics=True)

        metadata = {
            "pooled_output": pooled_output,
            "conditioning_lyrics": lyrics_tensor,
            "audio_codes": audio_codes
        }
        
        return ([[timbre_tensor, metadata]],)

NODE_CLASS_MAPPINGS = {
    "AceStepConditioningCombine": AceStepConditioningCombine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepConditioningCombine": "Conditioning Component Combiner",
}
