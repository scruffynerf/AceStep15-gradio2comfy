"""AceStepZeroConditioningInspector node for ACE-Step"""
import torch
import json
import logging

from .includes.fsq_utils import parse_audio_codes, fsq_decode_indices, get_fsq_levels
from .includes.zerobytes_utils import parse_section_map

logger = logging.getLogger(__name__)


class AceStepZeroConditioningInspector:
    """Visualize ZeroConditioning output for debugging and tuning.

    Renders diagnostic images of audio code sequences: per-dimension heatmaps,
    coherence plots, distribution histograms, and section overlays.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_codes": ("LIST",),
                "visualization": (["heatmap_6d", "coherence_plot", "histogram",
                                   "section_overlay"],
                                  {"default": "heatmap_6d"}),
            },
            "optional": {
                "section_map": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("visualization", "stats_json")
    FUNCTION = "inspect"
    CATEGORY = "Scromfy/Ace-Step/Conditioning"

    def inspect(self, audio_codes, visualization, section_map=""):
        levels = get_fsq_levels(None)
        parsed = parse_audio_codes(audio_codes)

        if not parsed or not parsed[0]:
            blank = torch.zeros(1, 64, 256, 3)
            return (blank, json.dumps({"error": "empty audio codes"}))

        ids = parsed[0]
        T = len(ids)
        indices = torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # [1, T]
        codes_6d = fsq_decode_indices(indices, levels)  # [1, T, 6]

        # Compute stats
        deltas = []
        for i in range(T - 1):
            deltas.append((codes_6d[0, i + 1, :] - codes_6d[0, i, :]).abs().mean().item())
        mean_delta = sum(deltas) / len(deltas) if deltas else 0.0

        per_dim_means = codes_6d[0].mean(dim=0).tolist()
        per_dim_stds = codes_6d[0].std(dim=0).tolist()

        stats = {
            "total_codes": T,
            "unique_codes": len(set(ids)),
            "mean_delta": round(mean_delta, 4),
            "per_dim_means": [round(v, 4) for v in per_dim_means],
            "per_dim_stds": [round(v, 4) for v in per_dim_stds],
        }

        # Render visualization
        if visualization == "heatmap_6d" or visualization == "section_overlay":
            img = self._render_heatmap(codes_6d, levels)
            if visualization == "section_overlay" and section_map:
                img = self._overlay_sections(img, parse_section_map(section_map), T)
        elif visualization == "coherence_plot":
            img = self._render_coherence(deltas)
        elif visualization == "histogram":
            img = self._render_histogram(ids)
        else:
            img = torch.zeros(1, 64, 256, 3)

        return (img, json.dumps(stats))

    def _render_heatmap(self, codes_6d, levels):
        """Render 6-row heatmap. Each row = one FSQ dimension over time."""
        T = codes_6d.shape[1]
        row_height = 60
        H = row_height * 6
        W = min(max(T * 4, 256), 1920)

        img = torch.zeros(H, W, 3)

        for d in range(6):
            # Normalize dim values from [-1, 1] to [0, 1]
            vals = (codes_6d[0, :, d] + 1.0) / 2.0  # [T]

            # Color: use distinct hue per dimension
            hues = [0.0, 0.15, 0.3, 0.55, 0.7, 0.85]
            hue = hues[d]

            for t_idx in range(T):
                x_start = int(t_idx / T * W)
                x_end = int((t_idx + 1) / T * W)
                if x_end <= x_start:
                    x_end = x_start + 1
                y_start = d * row_height
                y_end = y_start + row_height

                val = vals[t_idx].item()
                r, g, b = self._hsv_to_rgb(hue, 0.7, val)
                img[y_start:y_end, x_start:min(x_end, W), 0] = r
                img[y_start:y_end, x_start:min(x_end, W), 1] = g
                img[y_start:y_end, x_start:min(x_end, W), 2] = b

        return img.unsqueeze(0)  # [1, H, W, 3]

    def _overlay_sections(self, img, sections, T):
        """Draw red vertical lines at section boundaries."""
        if not sections:
            return img
        H = img.shape[1]
        W = img.shape[2]
        for sec in sections:
            boundary_t = int(sec["start"] * 5)
            if boundary_t > 0 and boundary_t < T:
                x = int(boundary_t / T * W)
                x = min(x, W - 1)
                img[0, :, x, 0] = 1.0  # Red
                img[0, :, x, 1] = 0.0
                img[0, :, x, 2] = 0.0
        return img

    def _render_coherence(self, deltas):
        """Render line plot of per-timestep deltas."""
        T = len(deltas)
        if T == 0:
            return torch.zeros(1, 200, 400, 3)
        H, W = 200, min(max(T * 4, 400), 1920)
        img = torch.ones(H, W, 3) * 0.1  # dark background

        max_delta = max(deltas) if deltas else 1.0
        if max_delta < 1e-6:
            max_delta = 1.0

        for i, delta in enumerate(deltas):
            x = int(i / T * W)
            y = int((1.0 - delta / max_delta) * (H - 10)) + 5
            x = min(x, W - 1)
            y = max(0, min(y, H - 1))
            # Draw point with small cross
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    px, py = x + dx, y + dy
                    if 0 <= px < W and 0 <= py < H:
                        img[py, px, 1] = 0.8  # Green dots

        return img.unsqueeze(0)

    def _render_histogram(self, ids):
        """Render histogram of code distribution."""
        H, W = 200, 400
        img = torch.ones(H, W, 3) * 0.1

        n_bins = min(100, max(1, len(set(ids))))
        hist = torch.histc(torch.tensor(ids, dtype=torch.float), bins=n_bins,
                           min=0, max=63999)
        max_count = hist.max().item()
        if max_count < 1:
            max_count = 1

        bin_width = max(1, W // n_bins)
        for i in range(n_bins):
            bar_height = int((hist[i].item() / max_count) * (H - 20))
            x_start = i * bin_width
            x_end = min(x_start + bin_width - 1, W - 1)
            y_start = H - 10 - bar_height
            y_end = H - 10
            if bar_height > 0 and x_end > x_start:
                img[y_start:y_end, x_start:x_end, 0] = 0.2
                img[y_start:y_end, x_start:x_end, 1] = 0.6
                img[y_start:y_end, x_start:x_end, 2] = 0.9

        return img.unsqueeze(0)

    @staticmethod
    def _hsv_to_rgb(h, s, v):
        """Simple HSV to RGB conversion."""
        if s == 0:
            return v, v, v
        i = int(h * 6)
        f = (h * 6) - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))
        i %= 6
        if i == 0: return v, t, p
        if i == 1: return q, v, p
        if i == 2: return p, v, t
        if i == 3: return p, q, v
        if i == 4: return t, p, v
        return v, p, q


NODE_CLASS_MAPPINGS = {
    "AceStepZeroConditioningInspector": AceStepZeroConditioningInspector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepZeroConditioningInspector": "Zero Conditioning Inspector",
}
