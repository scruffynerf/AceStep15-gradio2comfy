"""Sampling utilities for ACE-Step"""

def apply_shift(sigmas, shift):
    """
    Apply timestep shift formula: t' = shift * t / (1 + (shift - 1) * t)
    Used to adjust noise schedule for better quality in DiT/Flow models.
    """
    if shift == 1.0:
        return sigmas
    
    # Sigmas in Flow Matching usually correspond to t [0, 1]
    # We apply the shift to all sigmas except the last one (0.0)
    shifted_sigmas = sigmas.clone()
    t = sigmas[:-1]
    shifted_sigmas[:-1] = shift * t / (1 + (shift - 1) * t)
    return shifted_sigmas
