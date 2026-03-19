"""AceStepZeroOutConditioning node for ACE-Step"""
import torch

class AceStepZeroOutConditioning:
    """Zero out conditioning tensors and remove audio_codes.
    
    Like ComfyUI's ConditioningZeroOut but also strips audio_codes
    from the conditioning dict, which the standard node does not do.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conditioning": ("CONDITIONING",)}}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "zero_out"
    CATEGORY = "Scromfy/Ace-Step/Conditioning"

    def zero_out(self, conditioning):
        c = []
        for t in conditioning:
            d = t[1].copy()
            pooled_output = d.get("pooled_output", None)
            if pooled_output is not None:
                d["pooled_output"] = torch.zeros_like(pooled_output)
            # Remove audio_codes so they don't leak into the negative/empty conditioning
            d.pop("audio_codes", None)
            n = [torch.zeros_like(t[0]), d]
            c.append(n)
        return (c,)


NODE_CLASS_MAPPINGS = {
    "AceStepZeroOutConditioning": AceStepZeroOutConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepZeroOutConditioning": "True Zero Out Conditioning",
}
