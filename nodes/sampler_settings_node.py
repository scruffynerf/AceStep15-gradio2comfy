# Credit goes to https://github.com/jeankassio/ComfyUI-AceStep_SFT
# for his all-in-one SFT node implementation, I've split it into pieces.
# This node holds advanced sampling settings to keep the main Sampler node clean.

class ScromfyAceStepSamplerSettings:
    """Advanced sampling settings for ScromfyAceStepSampler.
    Encapsulates guidance modes, intervals, momentum, and scaling parameters.
    
    Inputs:
        guidance_mode (STRING): Type of guidance ("apg", "adg", "standard_cfg").
        guidance_interval (FLOAT): Centered interval applying guidance.
        apg_momentum (FLOAT): Momentum buffer coefficient for APG.
        apg_norm_threshold (FLOAT): Norm threshold for gradient clipping.
        guidance_interval_decay (FLOAT): Decay strength inside active interval.
        min_guidance_scale (FLOAT): Lowest bound for decayed guidance.
        guidance_scale_text (FLOAT): Independent CFG for text branch (-1 inherits).
        guidance_scale_lyric (FLOAT): Independent CFG for lyric branch (-1 inherits).
        omega_scale (FLOAT): Emulates AceStep omega scaling.
        erg_scale (FLOAT): Source energy reweighting.
        cfg_interval_start (FLOAT): Legacy explicit schedule start.
        cfg_interval_end (FLOAT): Legacy explicit schedule end.
        
    Outputs:
        sampler_settings (SCROMFY_SAMPLER_SETTINGS): Encapsulated settings payload.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guidance_mode": (["apg", "adg", "standard_cfg"], {
                    "default": "apg",
                    "tooltip": "APG = Adaptive Projected Guidance (AceStep default). ADG = Angle-based Dynamic Guidance. standard_cfg = regular CFG",
                }),
                "guidance_interval": ("FLOAT", {
                    "default": 0.5, "min": -1.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Official AceStep guidance interval width. 0.5 applies guidance in the centered middle band. Set to -1 to use legacy cfg_interval_start/end instead",
                }),
                "apg_momentum": ("FLOAT", {
                    "default": -0.75, "min": -1.0, "max": 1.0, "step": 0.05,
                    "tooltip": "APG momentum buffer coefficient",
                }),
                "apg_norm_threshold": ("FLOAT", {
                    "default": 2.5, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "APG norm threshold for gradient clipping",
                }),
                "guidance_interval_decay": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Linearly decays guidance inside the active interval toward min_guidance_scale, matching AceStep's official control",
                }),
                "min_guidance_scale": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 30.0, "step": 0.1,
                    "tooltip": "Lowest guidance scale reached when guidance_interval_decay is enabled",
                }),
                "guidance_scale_text": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 30.0, "step": 0.1,
                    "tooltip": "Independent text guidance scale. -1 inherits cfg. Works by adding a text-only conditioning branch inside the node",
                }),
                "guidance_scale_lyric": ("FLOAT", {
                    "default": -1.0, "min": -1.0, "max": 30.0, "step": 0.1,
                    "tooltip": "Independent lyric guidance scale. -1 inherits cfg. The full branch remains text+lyrics; this value controls the lyric-only delta against the text-only branch",
                }),
                "omega_scale": ("FLOAT", {
                    "default": 0.0, "min": -8.0, "max": 8.0, "step": 0.05,
                    "tooltip": "Mean-preserving output reweighting applied inside the node to emulate AceStep's omega_scale scheduler behavior",
                }),
                "erg_scale": ("FLOAT", {
                    "default": 0.0, "min": -0.9, "max": 2.0, "step": 0.05,
                    "tooltip": "Node-local ERG approximation. Reweights prompt and lyric conditioning energy before sampling to strengthen prompt adherence without changing ComfyUI core",
                }),
                "cfg_interval_start": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Start applying CFG/APG guidance at this fraction of the schedule",
                }),
                "cfg_interval_end": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Stop applying CFG/APG guidance at this fraction of the schedule",
                }),
            }
        }

    RETURN_TYPES = ("SCROMFY_SAMPLER_SETTINGS",)
    RETURN_NAMES = ("sampler_settings",)
    FUNCTION = "get_settings"
    CATEGORY = "Scromfy/Ace-Step/Sampler"

    def get_settings(self, **kwargs):
        return (kwargs,)

NODE_CLASS_MAPPINGS = {"ScromfyAceStepSamplerSettings": ScromfyAceStepSamplerSettings}
NODE_DISPLAY_NAME_MAPPINGS = {"ScromfyAceStepSamplerSettings": "Scromfy AceStep Sampler Settings"}
