"""AceStepInpaintSampler node for ACE-Step"""
import torch
import comfy.samplers
import comfy.sample
import comfy.model_management
from .includes.sampling_utils import apply_shift

class AceStepInpaintSampler:
    """Specialized KSampler for audio inpainting"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "source_latent": ("LATENT",),
                "mask": ("MASK",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "Scromfy/Ace-Step/sampling"

    def sample(self, model, source_latent, mask, positive, negative, seed, steps, cfg, sampler_name, scheduler, denoise, shift=1.0):
        samples = source_latent["samples"].clone()
        
        # Get sigmas
        device = comfy.model_management.get_torch_device()
        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps).to(device)
        
        if denoise < 1.0:
            sigmas = sigmas[int(len(sigmas) * (1.0 - denoise)):]
            
        sigmas = apply_shift(sigmas, shift)
        
        # Custom sampling
        out_samples = comfy.sample.sample_custom(
            model,
            seed,
            cfg,
            sampler_name,
            sigmas,
            positive,
            negative,
            samples,
            disable_noise=False
        )
        
        # Re-apply mask to preserve original regions
        mask = mask.to(samples.device)
        if len(mask.shape) == 2:
            mask_expanded = mask.unsqueeze(1).expand_as(samples)
        else:
            mask_expanded = mask.expand_as(samples)
            
        final_samples = samples * (1.0 - mask_expanded) + out_samples * mask_expanded
        
        return ({"samples": final_samples, "type": "audio"},)


NODE_CLASS_MAPPINGS = {
    "AceStepInpaintSampler": AceStepInpaintSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepInpaintSampler": "Inpaint Sampler (Audio)",
}
