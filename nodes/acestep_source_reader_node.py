import torch
import inspect
import numpy as np
from .includes.chord_utils import vae_encode_audio

class AceStepSourceReader:
    """
    ACE-Step ▸ Source Reader

    Reads the prepare_condition method from ace_step15.py and also tries
    injecting VAE latents directly as precomputed_lm_hints_25Hz.
    """
    CATEGORY = "Scromfy/AceStep/Conditioning"
    FUNCTION = "read"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "vae":          ("VAE",),
            "model":        ("MODEL",),
        }}

    def read(self, conditioning, vae, model):
        sep = "=" * 70

        # ── 1. Find and print prepare_condition source ────────────────────
        try:
            inner = getattr(model, "model", model)
            dm    = getattr(inner, "diffusion_model", None)
            if dm is not None:
                src = inspect.getsource(dm.__class__)
                # Find prepare_condition section
                idx = src.find("def prepare_condition")
                if idx >= 0:
                    chunk = src[idx:idx+3000]
                    print(f"""
{sep}
ace_step15 prepare_condition SOURCE:
{sep}""")
                    print(chunk[:3000])
                    print(sep)
                else:
                    print("prepare_condition not found in diffusion_model source")
                    
                # Also print detokenizer info
                detok = getattr(dm, "detokenizer", None)
                if detok is not None:
                    print(f"""
detokenizer: {type(detok).__name__}""")
                    embed = getattr(detok, "embed_tokens", None)
                    if embed is not None:
                        print(f"  embed_tokens: num_embeddings={embed.num_embeddings}  "
                              f"embedding_dim={embed.embedding_dim}")
                    print("detokenizer source:")
                    print(inspect.getsource(detok.__class__)[:2000])
            else:
                print("diffusion_model not found")
        except Exception as exc:
            print(f"source read error: {exc}")

        # ── 2. Try injecting VAE latents as precomputed_lm_hints_25Hz ─────
        print(f"""
{sep}
Trying precomputed_lm_hints_25Hz injection
{sep}""")
        try:
            # Synthesise 5s of C major as a quick test
            sr = 48000
            t  = np.linspace(0, 5.0, 5*sr, dtype=np.float32)
            test_audio = (0.3*np.sin(2*np.pi*261.63*t) +   # C
                          0.2*np.sin(2*np.pi*329.63*t) +   # E
                          0.2*np.sin(2*np.pi*392.00*t))    # G
            test_audio = test_audio.astype(np.float32)

            latents = vae_encode_audio(vae, test_audio)
            if latents is not None:
                print(f"  latents: {list(latents.shape)}")
                
                # Try as precomputed_lm_hints_25Hz
                out = []
                for tensor, d in conditioning:
                    nd = dict(d)
                    # Try the 25Hz hints key
                    nd["precomputed_lm_hints_25Hz"] = latents.float()
                    # Also remove audio_codes to avoid conflict
                    nd.pop("audio_codes", None)
                    out.append([tensor, nd])
                    print(f"  injected precomputed_lm_hints_25Hz {list(latents.shape)}")
                    print(f"  removed audio_codes")
                print(sep)
                return (out,)
            else:
                print("  latents generation failed")

        except Exception as exc:
            print(f"  precomputed_lm_hints_25Hz test failed: {exc}")
            print(sep)
            return (conditioning,)

        return (conditioning,)

NODE_CLASS_MAPPINGS = {
    "AceStepSourceReader": AceStepSourceReader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepSourceReader": "ACE-Step ▸ Source Reader"
}
