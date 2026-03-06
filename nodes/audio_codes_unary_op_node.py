"""AceStepAudioCodesUnaryOp node for ACE-Step"""
import torch
import torch.nn.functional as F
import logging
import hashlib
import math

import comfy.model_management
from .includes.fsq_utils import (
    parse_audio_codes, fsq_decode_indices, fsq_encode_to_indices, get_fsq_levels
)

logger = logging.getLogger(__name__)

class AceStepAudioCodesUnaryOp:
    """Operations that transform a single set of audio codes (A) in 6D FSQ space with optional masking"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_codes": ("LIST",),
                #"model": ("MODEL",),
                "mode": ([
                    "noop_visualize",
                    "gate",
                    "scale_masked",
                    "noise_masked",
                    "fade_out"
                ], {"default": "noop_visualize"}),
                "visualization_type": ([
                    "ssm",
                    "linear",
                    "pca_ribbon",
                    "music_radar",
                    "song_texture",
                    "spiral"
                ], {"default": "ssm"}),
                "ssm_blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "length_pct": ("FLOAT", {"default": 100.0, "min": 0.1, "max": 1000.0, "step": 0.1}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "sigma": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("LIST", "IMAGE")
    RETURN_NAMES = ("audio_codes", "visualization")
    FUNCTION = "process"
    CATEGORY = "Scromfy/Ace-Step/mixers"

    @classmethod
    def IS_CHANGED(s, audio_codes, mode, visualization_type, length_pct, strength, sigma, seed, ssm_blur, mask=None):
        try:
            # Hash some samples and parameters
            sample = str(audio_codes[0][:5]) if audio_codes else "e"
            info = f"{sample}_{mode}_{visualization_type}_{length_pct}_{strength}_{sigma}_{seed}_{ssm_blur}"
            if mask is not None:
                info += f"_{mask.abs().mean().item():.4f}"
            return hashlib.md5(info.encode()).hexdigest()
        except:
            return "random"

    def _linear_token_image(self, A, upscale=32):
        """
        Linear time visualization of tokens.
        A shape: [1, T, 6]
        """

        X = A[0]  # [T,6]

        # normalize [-1,1] -> [0,1]
        X = (X.clamp(-1,1) + 1.0) / 2.0

        # transpose so dims become rows
        img = X.T  # [6,T]

        # upscale vertically
        img = img.repeat_interleave(upscale, dim=0)

        # expand RGB
        img = img.unsqueeze(0).unsqueeze(-1).repeat(1,1,1,3)

        return img

    def _self_similarity_image(self, A, blur_sigma=0.0):
        """
        Create a Self-Similarity Matrix image from decoded tensor A.
        A shape: [1, T, 6]
        """
        X = A[0]  # [T,6]

        # normalize vectors
        X = torch.nn.functional.normalize(X, dim=1)

        # cosine similarity
        S = X @ X.T   # [T,T]

        # convert [-1,1] → [0,1]
        S = (S + 1.0) / 2.0

        # optional gaussian blur (the "cool trick")
        if blur_sigma > 0:
            k = int(blur_sigma * 4 + 1)
            if k % 2 == 0:
                k += 1

            coords = torch.arange(k, device=S.device) - k // 2
            g = torch.exp(-(coords**2) / (2 * blur_sigma**2))
            g = g / g.sum()

            kernel = torch.outer(g, g)
            kernel = kernel / kernel.sum()

            kernel = kernel.view(1,1,k,k)

            S = S.unsqueeze(0).unsqueeze(0)
            S = F.conv2d(S, kernel, padding=k//2)
            S = S[0,0]

        # convert to RGB image
        img = S.unsqueeze(0).unsqueeze(-1).repeat(1,1,1,3)
        return img

    def _pca_ribbon_image(self, A, height=192):
        """
        PCA ribbon visualization.
        A shape: [1,T,6]
        """

        X = A[0]  # [T,6]

        # center
        X = X - X.mean(dim=0, keepdim=True)

        # PCA via SVD
        U, S, V = torch.pca_lowrank(X, q=3)

        comps = X @ V[:, :3]  # [T,3]

        # normalize to [0,1]
        mins = comps.min(dim=0)[0]
        maxs = comps.max(dim=0)[0]
        comps = (comps - mins) / (maxs - mins + 1e-6)

        # [T,3] → [1,1,T,3]
        img = comps.unsqueeze(0).unsqueeze(1)

        # stretch vertically
        img = img.repeat(1, height, 1, 1)

        return img

    def _music_radar_image(self, A, size=128):
        """
        Radar-style visualization of token embeddings.
        A shape: [1,T,6]
        """

        X = A[0]  # [T,6]
        T = X.shape[0]

        # normalize
        X = X.clamp(-1,1)

        # angles for 6 dimensions
        angles = torch.linspace(0, 2*torch.pi, 7)[:-1]

        coords = torch.stack([
            torch.cos(angles),
            torch.sin(angles)
        ], dim=1).to(X.device)

        # convert embeddings to 2D radial points
        pts = X @ coords  # [T,2]

        # normalize to image space
        pts = (pts - pts.min(0)[0]) / (pts.max(0)[0] - pts.min(0)[0] + 1e-6)

        canvas = torch.zeros((size, T, 3), device=X.device)

        xs = (pts[:,0] * (size-1)).long()
        ys = (pts[:,1] * (size-1)).long()

        for t in range(T):
            canvas[xs[t], t] = torch.tensor([1.0,1.0,1.0], device=X.device)

        canvas = canvas.unsqueeze(0)

        return canvas

    def _song_texture_image(self, A, size=256):
        """
        Create a square 'song fingerprint' image.
        A shape: [1,T,6]
        """

        X = A[0]  # [T,6]

        # center
        X = X - X.mean(dim=0, keepdim=True)

        # PCA to RGB
        U, S, V = torch.pca_lowrank(X, q=3)
        comps = X @ V[:, :3]  # [T,3]

        # normalize
        mins = comps.min(dim=0)[0]
        maxs = comps.max(dim=0)[0]
        comps = (comps - mins) / (maxs - mins + 1e-6)

        # number of pixels
        N = size * size

        # pad or truncate sequence
        if comps.shape[0] < N:
            pad = torch.zeros(N - comps.shape[0], 3, device=comps.device)
            comps = torch.cat([comps, pad], dim=0)
        else:
            comps = comps[:N]

        img = comps.reshape(size, size, 3)

        return img.unsqueeze(0)

    def _spiral_music_image(self, A, size=256, tokens_per_turn=20, starburst_strength=2.5):
        """
        Spiral visualization of token embeddings with beat/starburst enhancement.

        A shape: [1, T, 6]

        size: output image size
        tokens_per_turn: tokens per spiral revolution
        starburst_strength: how strongly transients brighten the spiral
        """
        
        X = A[0]  # [T,6]
        T = X.shape[0]

        # --- PCA → RGB ---
        Xc = X - X.mean(dim=0, keepdim=True)

        U, S, V = torch.pca_lowrank(Xc, q=3)
        comps = Xc @ V[:, :3]

        mins = comps.min(dim=0)[0]
        maxs = comps.max(dim=0)[0]

        colors = (comps - mins) / (maxs - mins + 1e-6)

        # --- transient / beat energy ---
        diffs = torch.zeros(T, device=X.device)
        diffs[1:] = torch.norm(X[1:] - X[:-1], dim=1)

        diffs = diffs / (diffs.max() + 1e-6)
        energy = 1.0 + starburst_strength * diffs

        canvas = torch.zeros((size, size, 3), device=X.device)

        cx = size // 2
        cy = size // 2
        max_radius = size * 0.45

        for t in range(T):

            turn = t / tokens_per_turn

            angle = 2 * math.pi * turn
            radius = max_radius * (t / T)

            x = int(cx + radius * math.cos(angle))
            y = int(cy + radius * math.sin(angle))

            if 1 <= x < size-1 and 1 <= y < size-1:

                c = colors[t] * energy[t]

                # draw thicker point for visibility
                canvas[y, x] += c
                canvas[y+1, x] += c * 0.7
                canvas[y-1, x] += c * 0.7
                canvas[y, x+1] += c * 0.7
                canvas[y, x-1] += c * 0.7

        canvas = torch.clamp(canvas, 0, 1)

        return canvas.unsqueeze(0)

    def process(self, audio_codes, mode, length_pct, strength, sigma, seed, ssm_blur, visualization_type, mask=None):
        #inner_model = model.model
        #if hasattr(inner_model, "diffusion_model"):
        #    inner_model = inner_model.diffusion_model

        #comfy.model_management.load_model_gpu(model)
        device = comfy.model_management.get_torch_device()
        levels = get_fsq_levels(None)

        parsed = parse_audio_codes(audio_codes)

        if not parsed or not parsed[0]:
            logger.error("Empty audio_codes input")
            return (audio_codes,)

        ids = parsed[0]

        # Decode to 6D float space [1, T, 6]
        A = fsq_decode_indices(torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0), levels)

        # --- Visualization (time vs FSQ dimensions) ---
        if visualization_type == "ssm":
            vis_image = self._self_similarity_image(A, blur_sigma=ssm_blur)
        elif visualization_type == "pca_ribbon":
            vis_image = self._pca_ribbon_image(A, height=192)
        elif visualization_type == "music_radar":
            vis_image = self._music_radar_image(A, size=128)
        elif visualization_type == "song_texture":
            vis_image = self._song_texture_image(A, size=256)
        elif visualization_type == "spiral":
            vis_image = self._spiral_music_image(A, size=256, tokens_per_turn=20, starburst_strength=2.5)
        else:
            vis_image = self._linear_token_image(A, upscale=32)

        # Handle Length Scaling
        if length_pct != 100.0:
            curr_len = A.shape[1]
            new_len = max(1, int(curr_len * (length_pct / 100.0)))
            
            # [1, L, 6] -> [1, 6, L]
            A = A.transpose(1, 2)
            A = torch.nn.functional.interpolate(A, size=new_len, mode='linear', align_corners=False)
            A = A.transpose(1, 2)

        # Prepare mask
        target_len = A.shape[1]
        if mask is None:
            mask = torch.ones((1, target_len, 1), device=device)
        else:
            # Resize mask if needed [B, T] or [B, H, W] to [1, T, 1]
            mask = mask.to(device)
            if mask.dim() == 2: # [B, T]
                mask = mask.unsqueeze(-1)
            elif mask.dim() == 3: # [B, H, W]
                mask = mask.mean(dim=1).unsqueeze(-1) # Flatten spatial
            
            if mask.shape[1] != target_len:
                # Interpolate mask
                mask = mask.transpose(1, 2)
                mask = F.interpolate(mask, size=target_len, mode='linear', align_corners=False)
                mask = mask.transpose(1, 2)

        # Perform Operations in 6D space

        if mode == "noop_visualize":
            out = A

        elif mode == "gate":
            # Gating: A * mask (values move towards 0.0 midpoint)
            out = A * mask
            
        elif mode == "scale_masked":
            # Scale A only where mask > 0
            # formula: A * (1.0 + mask * (strength - 1.0))
            out = A * (1.0 + mask * (strength - 1.0))
            
        elif mode == "noise_masked":
            # Add noise only where mask allows
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            noise = torch.randn_like(A, generator=generator) * sigma
            out = A + mask * noise
            
        elif mode == "fade_out":
            # Special case: linear ramp multiply
            N = target_len
            fade = torch.linspace(1, 0, N, device=device)[None, :, None]
            # (1.0 - mask) * A + mask * (A * fade)
            out = A * (1.0 - mask) + (A * fade) * mask
            
        else:
            out = A

        # Clamp to FSQ range [-1, 1] before encoding
        out = out.clamp(-1.0, 1.0)
        
        # Encode back to indices
        result_ids = fsq_encode_to_indices(out, levels)[0].tolist()

        return ([result_ids], vis_image)

NODE_CLASS_MAPPINGS = {
    "AceStepAudioCodesUnaryOp": AceStepAudioCodesUnaryOp,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepAudioCodesUnaryOp": "Audio Codes Unary Operations",
}
