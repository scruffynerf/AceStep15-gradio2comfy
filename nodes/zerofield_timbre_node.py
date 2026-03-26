"""AceStepZeroFieldTimbre node for ACE-Step"""
import torch
import torch.nn.functional as F
import json
import logging

from .includes.zerobytes_utils import coherent_value, coherent_field

logger = logging.getLogger(__name__)


class AceStepZeroFieldTimbre:
    """Generate timbre_tensor as a continuous Zero-Field in 1024D latent space.

    The timbre is not derived from text or loaded from disk -- it is a
    continuous field that exists at every point in the embedding space.
    Instruments are entities that sample from this field via their salt.

    Each of the 1024 embedding dimensions evolves as an independent scalar
    field over the sequence length L, with coherent noise providing smooth
    temporal variation at configurable macro and micro scales.
    """

    INSTRUMENT_SALTS = {
        "piano":       10000,
        "guitar":      11000,
        "strings":     12000,
        "brass":       13000,
        "woodwinds":   14000,
        "synth_pad":   15000,
        "synth_lead":  16000,
        "bass":        17000,
        "drums":       18000,
        "vocals":      19000,
        "choir":       20000,
        "organ":       21000,
        "ambient":     22000,
        "electronic":  23000,
        "world":       24000,
        "custom":      25000,
    }

    @classmethod
    def INPUT_TYPES(s):
        instruments = list(AceStepZeroFieldTimbre.INSTRUMENT_SALTS.keys())
        return {
            "required": {
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFF}),
                "seq_length": ("INT", {"default": 77, "min": 1, "max": 512}),
                "instrument_mode": ([
                    "single", "blend", "layered", "territorial", "evolving"
                ], {"default": "single"}),
                "primary_instrument": (instruments, {"default": "piano"}),
            },
            "optional": {
                "secondary_instrument": (instruments, {"default": "strings"}),
                "reference_timbre": ("TENSOR",),
                "instrument_blend": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "field_scale_macro": ("FLOAT", {"default": 0.02, "min": 0.001, "max": 0.5, "step": 0.001}),
                "field_scale_micro": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 1.0, "step": 0.01}),
                "field_intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
                "warmth": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("TENSOR", "STRING")
    RETURN_NAMES = ("timbre_tensor", "field_stats")
    FUNCTION = "generate"
    CATEGORY = "Scromfy/Ace-Step/Conditioning"

    def generate(self, seed, seq_length, instrument_mode, primary_instrument,
                 secondary_instrument="strings", reference_timbre=None,
                 instrument_blend=0.5, field_scale_macro=0.02,
                 field_scale_micro=0.15, field_intensity=0.5,
                 warmth=0.0, brightness=0.0):

        salt_a = self.INSTRUMENT_SALTS[primary_instrument]
        field_a = self._generate_field(seed, salt_a, seq_length,
                                        field_scale_macro, field_scale_micro,
                                        field_intensity, warmth, brightness)

        if instrument_mode == "single":
            result = field_a

        elif instrument_mode == "blend":
            salt_b = self.INSTRUMENT_SALTS.get(secondary_instrument, 12000)
            field_b = self._generate_field(seed, salt_b, seq_length,
                                            field_scale_macro, field_scale_micro,
                                            field_intensity, warmth, brightness)
            result = field_a * (1 - instrument_blend) + field_b * instrument_blend

        elif instrument_mode == "layered":
            salt_b = self.INSTRUMENT_SALTS.get(secondary_instrument, 12000)
            field_b = self._generate_field(seed, salt_b, seq_length,
                                            field_scale_macro * 3, field_scale_micro * 2,
                                            field_intensity * 0.6, warmth, brightness)
            result = field_a * 0.7 + field_b * 0.3

        elif instrument_mode == "territorial":
            salt_b = self.INSTRUMENT_SALTS.get(secondary_instrument, 12000)
            field_b = self._generate_field(seed, salt_b, seq_length,
                                            field_scale_macro, field_scale_micro,
                                            field_intensity, warmth, brightness)
            mag_a = field_a.abs().mean(dim=-1, keepdim=True)
            mag_b = field_b.abs().mean(dim=-1, keepdim=True)
            ownership_a = (mag_a >= mag_b).float()
            result = field_a * ownership_a + field_b * (1.0 - ownership_a)

        elif instrument_mode == "evolving":
            salt_b = self.INSTRUMENT_SALTS.get(secondary_instrument, 12000)
            field_b = self._generate_field(seed, salt_b, seq_length,
                                            field_scale_macro, field_scale_micro,
                                            field_intensity, warmth, brightness)
            ramp = torch.linspace(0, 1, seq_length).unsqueeze(0).unsqueeze(-1)
            ramp = ramp * ramp * (3 - 2 * ramp)  # smoothstep
            result = field_a * (1 - ramp) + field_b * ramp
        else:
            result = field_a

        # Reference orbit
        if reference_timbre is not None:
            ref = reference_timbre
            if ref.dim() == 2:
                ref = ref.unsqueeze(0)
            if ref.shape[1] != seq_length:
                ref = ref.permute(0, 2, 1)
                ref = F.interpolate(ref, size=seq_length, mode='linear',
                                    align_corners=False)
                ref = ref.permute(0, 2, 1)
            result = ref + result

        stats = {
            "mode": instrument_mode,
            "primary": primary_instrument,
            "shape": list(result.shape),
            "mean": round(result.mean().item(), 4),
            "std": round(result.std().item(), 4),
            "min": round(result.min().item(), 4),
            "max": round(result.max().item(), 4),
        }
        return (result, json.dumps(stats))

    def _generate_field(self, seed, salt, seq_length, macro_freq, micro_freq,
                         intensity, warmth, brightness):
        """Generate [1, L, 1024] timbre field using vectorised coherent noise.

        Each embedding dimension is an independent scalar field over the
        sequence axis, with macro and micro frequency layers. The generation
        is batched per-dimension: all L positions are computed in one
        coherent_field call, eliminating the inner Python loop.
        """
        field = torch.zeros(1, seq_length, 1024)

        # Precompute spectral bias curve
        dim_positions = torch.linspace(0, 1, 1024)
        spectral_bias = (warmth * (1.0 - dim_positions) +
                         brightness * dim_positions) * 0.3

        # Precompute position arrays (shared across all dims)
        positions = list(range(seq_length))

        for dim in range(1024):
            dim_salt = salt + dim
            bias = spectral_bias[dim].item()

            # Batch all L positions for this dimension's macro layer
            macro_xs = [pos * macro_freq for pos in positions]
            macro_ys = [dim * 0.001] * seq_length
            macro_seeds = [seed + dim_salt] * seq_length
            macro_vals = coherent_field(macro_xs, macro_ys, macro_seeds, octaves=4)

            # Batch all L positions for this dimension's micro layer
            micro_xs = [pos * micro_freq for pos in positions]
            micro_ys = [dim * 0.01] * seq_length
            micro_seeds = [seed + dim_salt + 500] * seq_length
            micro_vals = coherent_field(micro_xs, micro_ys, micro_seeds, octaves=2)

            # Combine and write entire dimension column at once
            for pos in range(seq_length):
                field[0, pos, dim] = (macro_vals[pos] * 0.65 +
                                      micro_vals[pos] * 0.35 + bias) * intensity

        return field


NODE_CLASS_MAPPINGS = {
    "AceStepZeroFieldTimbre": AceStepZeroFieldTimbre,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepZeroFieldTimbre": "Zero Field Timbre Generator",
}
