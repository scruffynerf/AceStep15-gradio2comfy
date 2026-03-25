import os
import folder_paths
import comfy.sd
import comfy.utils
import torch

# Credit goes to https://github.com/jeankassio/ComfyUI-AceStep_SFT
# for his all-in-one SFT node implementation, I've split it into pieces.
# This is the model loader node.

class ScromfyAceStepModelLoader:
    """Specialized loader for AceStep 1.5 SFT model triplets.
    Handles loading the Diffusion model (DiT), dual CLIP encoders (Qwen), and VAE.
    
    Inputs:
        diffusion_model (STRING): Path to DiT model.
        text_encoder_1 (STRING): Path to text/lyrics Qwen.
        text_encoder_2 (STRING): Path to audio tags Qwen.
        vae_name (STRING): Path to audio VAE.
        
    Optional Inputs:
        lora_stack (ACESTEP_LORA): Stack of LoRAs to merge sequentially upon loading the base models.
        
    Outputs:
        model (MODEL): Base ACE-Step model.
        clip (CLIP): Dual text encoders.
        vae (VAE): Audio VAE.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "diffusion_model": (folder_paths.get_filename_list("diffusion_models"), {
                    "tooltip": "AceStep 1.5 diffusion model (DiT). e.g. Audio/acestep_v1.5_sft.safetensors",
                }),
                "text_encoder_1": (folder_paths.get_filename_list("text_encoders"), {
                    "tooltip": "Qwen3-0.6B encoder for captions/lyrics. e.g. Audio/qwen_0.6b_ace15.safetensors",
                }),
                "text_encoder_2": (folder_paths.get_filename_list("text_encoders"), {
                    "tooltip": "Qwen3 LLM for audio codes (1.7B or 4B). e.g. Audio/qwen_1.7b_ace15.safetensors",
                }),
                "vae_name": (folder_paths.get_filename_list("vae"), {
                    "tooltip": "AceStep 1.5 audio VAE. e.g. Audio/ace_1.5_vae.safetensors",
                }),
            },
            "optional": {
                "lora_stack": ("ACESTEP_LORA",),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load_models"
    CATEGORY = "Scromfy/Ace-Step/Loader"

    def load_models(self, diffusion_model, text_encoder_1, text_encoder_2, vae_name, lora_stack=None):
        # 1. Load Diffusion Model
        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", diffusion_model)
        model = comfy.sd.load_diffusion_model(unet_path)
        if hasattr(model, "model"):
            model.model.eval()

        # 2. Load Dual CLIP Encoders
        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", text_encoder_1)
        clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", text_encoder_2)
        
        # If they are the same, try loading only once as a single-element list.
        # This is an experiment to see if it avoids the CUDA crash for duplicates.
        ckpt_paths = [clip_path1] if text_encoder_1 == text_encoder_2 else [clip_path1, clip_path2]

        # ACE-Step uses a specialized CLIP type that takes checkpoints
        clip = comfy.sd.load_clip(
            ckpt_paths=ckpt_paths,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=comfy.sd.CLIPType.ACE,
        )
        if hasattr(clip, "cond_stage_model"):
            clip.cond_stage_model.eval()

        # 3. Apply LoRA stack if provided
        if lora_stack is not None:
            for lora_spec in lora_stack:
                lora_path = folder_paths.get_full_path_or_raise("loras", lora_spec["lora_name"])
                lora_data = comfy.utils.load_torch_file(lora_path, safe_load=True)
                model, clip = comfy.sd.load_lora_for_models(
                    model, clip, lora_data,
                    lora_spec["strength_model"], None
                )

        # 4. Load VAE
        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        vae_sd = comfy.utils.load_torch_file(vae_path)
        vae = comfy.sd.VAE(sd=vae_sd)
        if hasattr(vae, "first_stage_model"):
            vae.first_stage_model.eval()

        return (model, clip, vae)

NODE_CLASS_MAPPINGS = {
    "ScromfyAceStepModelLoader": ScromfyAceStepModelLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScromfyAceStepModelLoader": "Scromfy AceStep All-in-One Loader"
}
