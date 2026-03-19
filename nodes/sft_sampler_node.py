import torch
import torch.nn.functional as F
import torchaudio
import math
import json
import comfy.sample
import comfy.samplers
import comfy.model_management
import comfy.utils
import latent_preview
import node_helpers
from .includes.sft_sampling_utils import (
    MomentumBuffer, apg_guidance, adg_guidance, 
    apply_omega_scale, apply_erg_to_conditioning,
    clone_conditioning, zero_conditioning_value,
    build_text_only_conditioning, build_processed_text_only_conditioning
)

# Credit goes to https://github.com/jeankassio/ComfyUI-AceStep_SFT
# for his all-in-one SFT node implementation, I've split it into pieces.
# This is the sampler node.

class ScromfyAceStepSampler:
    """Specialized KSampler for AceStep 1.5 SFT.
    Supports APG (Adaptive Projected Guidance), ADG (Angle-based Dynamic Guidance),
    Omega/ERG scaling, guidance intervals, split guidance, and reference audio.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {
                    "default": 60, "min": 1, "max": 200,
                    "tooltip": "Diffusion inference steps. The official AceStep 1.5 quality baseline uses 60 steps for Base/SFT, 8 for Turbo",
                }),
                "cfg": ("FLOAT", {
                    "default": 15.0, "min": 0.0, "max": 100.0, "step": 0.1,
                    "tooltip": "Classifier-free guidance scale. The official AceStep 1.5 quality baseline uses 15.0 for Base/SFT, CFG should be 1.0 for Turbo",
                }),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {
                    "default": "euler",
                    "tooltip": "Official AceStep 1.5 quality baseline uses Euler sampling",
                }),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {
                    "default": "normal",
                    "tooltip": "Normal is recommended scheduler with Euler sampler",
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Denoise strength. 1.0 = full generation from noise. < 1.0 requires source_audio. Auto-set to 1.0 when reference_audio is provided",
                }),
            },
            "optional": {
                "vae": ("VAE",),
                "source_audio": ("AUDIO",),
                "reference_audio": ("AUDIO",),
                "reference_as_cover": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If False (default): learn style from reference, generate completely new music. If True: use reference as base for remix/cover",
                }),
                "audio_cover_strength": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Only used when reference_as_cover=True. How much reference content is preserved (0=remix, 1=exact cover)",
                }),
                "shift": ("FLOAT", {
                    "default": 3.0, "min": 1.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Timestep schedule shift. Stock default = 3.0",
                }),
                "custom_timesteps": ("STRING", {
                    "default": "",
                    "tooltip": "Custom comma-separated timesteps (overrides steps, shift and scheduler)",
                }),
                "sampler_settings": ("SCROMFY_SAMPLER_SETTINGS",),
                "vae_decode_settings": ("SCROMFY_VAE_SETTINGS",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("LATENT", "AUDIO")
    RETURN_NAMES = ("latent", "audio")
    FUNCTION = "sample"
    CATEGORY = "Scromfy/Ace-Step/Sampler"


    def sample(self, model, positive, negative, latent_image, seed, steps, cfg, 
               sampler_name, scheduler, denoise, shift=3.0, custom_timesteps="",
               vae=None, source_audio=None, reference_audio=None, 
               reference_as_cover=False, audio_cover_strength=0.0,
               sampler_settings=None, vae_decode_settings=None, mask=None):
        
        # --- 0. Advanced Settings Extraction ---
        ss = sampler_settings or {}
        guidance_mode = ss.get("guidance_mode", "apg")
        guidance_interval = ss.get("guidance_interval", 0.5)
        apg_momentum = ss.get("apg_momentum", -0.75)
        apg_norm_threshold = ss.get("apg_norm_threshold", 2.5)
        guidance_interval_decay = ss.get("guidance_interval_decay", 0.0)
        min_guidance_scale = ss.get("min_guidance_scale", 3.0)
        guidance_scale_text = ss.get("guidance_scale_text", -1.0)
        guidance_scale_lyric = ss.get("guidance_scale_lyric", -1.0)
        omega_scale = ss.get("omega_scale", 0.0)
        erg_scale = ss.get("erg_scale", 0.0)
        cfg_interval_start = ss.get("cfg_interval_start", 0.0)
        cfg_interval_end = ss.get("cfg_interval_end", 1.0)

        vs = vae_decode_settings or {}
        latent_shift = vs.get("latent_shift", 0.0)
        latent_rescale = vs.get("latent_rescale", 1.0)
        normalize_peak = vs.get("normalize_peak", False)
        voice_boost = vs.get("voice_boost", 0.0)

        batch_size = latent_image["samples"].shape[0]
        latent_length = latent_image["samples"].shape[-1]
        vae_sr = 44100 # Default for AceStep VAE

        # --- 1. Audio Conditioning (Source & Reference) ---
        if reference_audio is not None and vae is not None:
            ref_waveform = reference_audio["waveform"]
            ref_sr = reference_audio["sample_rate"]
            
            if ref_sr != vae_sr:
                ref_waveform = torchaudio.functional.resample(ref_waveform, ref_sr, vae_sr)
            
            if ref_waveform.shape[1] == 1:
                ref_waveform = ref_waveform.repeat(1, 2, 1)
            elif ref_waveform.shape[1] > 2:
                ref_waveform = ref_waveform[:, :2, :]
            
            target_samples = latent_length * 1920
            if ref_waveform.shape[-1] < target_samples:
                ref_waveform = F.pad(ref_waveform, (0, target_samples - ref_waveform.shape[-1]))
            elif ref_waveform.shape[-1] > target_samples:
                ref_waveform = ref_waveform[:, :, :target_samples]
            
            ref_latent = vae.encode(ref_waveform.movedim(1, -1))
            if ref_latent.shape[0] < batch_size:
                ref_latent = ref_latent.repeat(math.ceil(batch_size / ref_latent.shape[0]), 1, 1)[:batch_size]
            
            is_cover = reference_as_cover
            denoise = 1.0 # Parity: auto-set to 1.0 with reference audio
            positive = node_helpers.conditioning_set_values(positive, {
                "refer_audio_acoustic_hidden_states_packed": ref_latent,
                "refer_audio_order_mask": torch.arange(batch_size, device=ref_latent.device, dtype=torch.long),
                "is_covers": torch.full((batch_size,), is_cover, dtype=torch.bool, device=ref_latent.device),
                "audio_cover_strength": (audio_cover_strength if is_cover else 0.0),
            }, append=True)

        if source_audio is not None and vae is not None:
            # For source_audio denoising (img2img style)
            # This is typically handled by KSampler taking a starting latent.
            # If user provides source_audio, we should encode it and use it as latent_image.
            src_waveform = source_audio["waveform"]
            src_sr = source_audio["sample_rate"]
            if src_sr != vae_sr:
                src_waveform = torchaudio.functional.resample(src_waveform, src_sr, vae_sr)
            src_latent = vae.encode(src_waveform.movedim(1, -1))
            latent_image = {"samples": src_latent}

        # --- 2. Split Guidance Setup ---
        resolved_text_guidance = cfg if guidance_scale_text < 0.0 else guidance_scale_text
        resolved_lyric_guidance = cfg if guidance_scale_lyric < 0.0 else guidance_scale_lyric
        text_only_positive = build_text_only_conditioning(positive)
        split_guidance_active = (abs(resolved_text_guidance - cfg) > 1e-6 or abs(resolved_lyric_guidance - cfg) > 1e-6)

        # --- 3. Apply ERG Scale ---
        if abs(erg_scale) > 1e-8:
            positive = apply_erg_to_conditioning(positive, erg_scale)

        # --- 4. Sigmas and Intervals ---
        t = torch.linspace(1.0, 0.0, steps + 1)
        if custom_timesteps and custom_timesteps.strip():
            ts = [float(x.strip()) for x in custom_timesteps.split(",") if x.strip()]
            if not ts or ts[-1] != 0.0: ts.append(0.0)
            custom_sigmas = torch.FloatTensor(ts)
            steps = len(custom_sigmas) - 1
        else:
            custom_sigmas = shift * t / (1 + (shift - 1) * t) if shift != 1.0 else t

        use_official_interval = guidance_interval >= 0.0
        official_interval = max(0.0, min(1.0, guidance_interval))
        interval_step_start = int(steps * ((1.0 - official_interval) / 2.0))
        interval_step_end = int(steps * (official_interval / 2.0 + 0.5))
        cfg_interval_start, cfg_interval_end = sorted((cfg_interval_start, cfg_interval_end))

        # --- 5. Guidance Functions ---
        momentum_buf = MomentumBuffer(momentum=apg_momentum)
        schedule_state = {"index": 0, "last_sigma": None, "denom": max(steps - 1, 1)}
        branch_state = {"text_denoised": None}

        def get_step_context(sigma, cond_scale):
            sigma_value = float(sigma.flatten()[0])
            if schedule_state["last_sigma"] != sigma_value:
                if schedule_state["last_sigma"] is not None:
                    schedule_state["index"] = min(schedule_state["index"] + 1, schedule_state["denom"])
                schedule_state["last_sigma"] = sigma_value
            step_idx = schedule_state["index"]
            progress = step_idx / schedule_state["denom"]
            if use_official_interval:
                in_interval = interval_step_start <= step_idx < interval_step_end
            else:
                in_interval = cfg_interval_start <= progress <= cfg_interval_end
            
            curr_scale = cond_scale
            if guidance_interval_decay > 0.0:
                interval_span = max(interval_step_end - interval_step_start - 1, 1)
                interval_progress = min(max((step_idx - interval_step_start) / interval_span, 0.0), 1.0)
                curr_scale = cond_scale - ((cond_scale - min_guidance_scale) * interval_progress * guidance_interval_decay)
            return sigma_value, step_idx, progress, in_interval, curr_scale

        def calc_cond_batch_function(args):
            if not split_guidance_active:
                return comfy.samplers.calc_cond_batch(args["model"], args["conds"], args["input"], args["sigma"], args["model_options"])
            _, _, _, in_interval, _ = get_step_context(args["sigma"], cfg)
            if not in_interval:
                return comfy.samplers.calc_cond_batch(args["model"], args["conds"], args["input"], args["sigma"], args["model_options"])
            cond_out, uncond_out = comfy.samplers.calc_cond_batch(args["model"], args["conds"], args["input"], args["sigma"], args["model_options"])
            text_only_cond = build_processed_text_only_conditioning(args["conds"][0])
            text_out, _ = comfy.samplers.calc_cond_batch(args["model"], [text_only_cond, None], args["input"], args["sigma"], args["model_options"])
            branch_state["text_denoised"] = text_out
            return [cond_out, uncond_out]

        def guided_cfg_function(args):
            sigma_value, _, _, in_interval, current_guidance_scale = get_step_context(args["sigma"], args["cond_scale"])
            cond_denoised = args["cond_denoised"]
            uncond_denoised = args["uncond_denoised"]
            x = args["input"]
            
            if split_guidance_active and branch_state["text_denoised"] is not None:
                base = max(args["cond_scale"], 1e-8)
                text_unit = resolved_text_guidance / base
                lyric_unit = resolved_lyric_guidance / base
                text_denoised = branch_state["text_denoised"]
                blended = (x - uncond_denoised) + (text_denoised - cond_denoised) * text_unit + (cond_denoised - text_denoised) * lyric_unit
                cond_denoised = x - blended

            if not in_interval:
                return apply_omega_scale(x - cond_denoised, omega_scale)

            if guidance_mode == "standard_cfg" or current_guidance_scale <= 1.0:
                guided = uncond_denoised + (cond_denoised - uncond_denoised) * current_guidance_scale
                return apply_omega_scale(x - guided, omega_scale)

            sigma_r = args["sigma"].reshape(-1, *([1] * (x.ndim - 1))).clamp(min=1e-8)
            v_cond = (x - cond_denoised) / sigma_r
            v_uncond = (x - uncond_denoised) / sigma_r
            if guidance_mode == "adg":
                v_guided = adg_guidance(x.movedim(1, -1), v_cond.movedim(1, -1), v_uncond.movedim(1, -1), sigma_value, current_guidance_scale).movedim(-1, 1)
            else:
                v_guided = apg_guidance(v_cond, v_uncond, current_guidance_scale, momentum_buffer=momentum_buf, norm_threshold=apg_norm_threshold)
            return apply_omega_scale(v_guided * sigma_r, omega_scale)

        # --- 6. Patch and Sample ---
        patched_model = model.clone()
        if split_guidance_active:
            patched_model.set_model_sampler_calc_cond_batch_function(calc_cond_batch_function)
        patched_model.set_model_sampler_cfg_function(guided_cfg_function, disable_cfg1_optimization=True)

        pbar = comfy.utils.ProgressBar(steps)
        def callback(step, x0, x, total_steps):
            pbar.update_absolute(step + 1, total_steps)

        noise = comfy.sample.prepare_noise(latent_image["samples"], seed)
        samples = comfy.sample.sample(patched_model, noise, steps, cfg, sampler_name, scheduler,
                                      positive, negative, latent_image["samples"], denoise=denoise,
                                      sigmas=custom_sigmas, seed=seed, callback=callback)

        if latent_shift != 0.0 or latent_rescale != 1.0:
            samples = samples * latent_rescale + latent_shift

        # --- 7. Forced Inpainting Blending ---
        # If a mask is provided, we ensure unmasked regions are bit-identical to the source
        if mask is not None:
            # mask is likely [batch, height, width] - for audio it's [batch, 1, samples]
            # samples is [batch, channels, length]
            m = mask.to(samples.device)
            # Expand mask to match samples shape if needed
            if m.dim() == 2:
                # [B, T] -> [B, 1, T]
                m = m.unsqueeze(1)
            
            # Ensure mask matches sample channels and length
            if m.shape[1] != samples.shape[1]:
                m = m.expand(-1, samples.shape[1], -1)
            if m.shape[2] != samples.shape[2]:
                m = torch.nn.functional.interpolate(m, size=(samples.shape[2],), mode='nearest')
            
            # Apply blending formula: final = original * (1 - mask) + generated * mask
            original_latent = latent_image["samples"].to(samples.device)
            samples = original_latent * (1.0 - m) + samples * m

        out_latent = {"samples": samples, "type": "audio"}
        audio_output = None
        if vae is not None:
            audio = vae.decode(samples).movedim(-1, 1)
            if audio.dtype != torch.float32: audio = audio.float()
            if normalize_peak:
                peak = audio.abs().amax(dim=[1, 2], keepdim=True).clamp(min=1e-8)
                audio = audio / peak
            if voice_boost != 0.0:
                boost = 10.0 ** (voice_boost / 20.0)
                audio = torch.tanh(audio * boost * 0.99) / 0.99
            audio_output = {"waveform": audio, "sample_rate": vae_sr}

        return (out_latent, audio_output)

NODE_CLASS_MAPPINGS = {"ScromfyAceStepSampler": ScromfyAceStepSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"ScromfyAceStepSampler": "Scromfy AceStep Sampler"}
