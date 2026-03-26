"""AceStepZeroConditioningGenerator node for ACE-Step"""
import json
import logging

from .includes.zerobytes_utils import (
    position_hash, hash_to_float, coherent_value,
    FSQ_LEVELS, DIM_SALTS, SECTION_TYPE_IDS, dims_to_composite,
    parse_section_map, build_default_section_map, lookup_section,
    section_start_token, all_previous_same_type,
    section_repetition_factor, motif_echo, call_response_influence,
)

logger = logging.getLogger(__name__)


class AceStepZeroConditioningGenerator:
    """Generate audio codes via Zerobyte Family position-is-seed determinism.

    No LLM required. The coordinate IS the seed. O(1) access to any timestep.
    Produces the same codes regardless of execution order.

    Uses three Zerobyte Family methodologies:
    - Zerobytes: hierarchical point hashing (song -> section -> measure -> beat)
    - Zero-Quadratic: section repetition, motif echo, call-and-response
    - Coherent noise: smooth transitions between adjacent timesteps
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFF}),
                "duration": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 600.0, "step": 0.1}),
                "bpm": ("FLOAT", {"default": 120.0, "min": 20.0, "max": 300.0, "step": 0.1}),
                "time_signature": (["4/4", "3/4", "6/8", "2/4", "5/4", "7/8"], {"default": "4/4"}),
                "coherence_fine": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "coherence_coarse": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "section_mode": (["auto", "manual", "none"], {"default": "auto"}),
            },
            "optional": {
                "section_map": ("STRING", {"default": "", "multiline": True}),
                "energy": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "density": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "crossfade_beats": ("INT", {"default": 2, "min": 0, "max": 8}),
            }
        }

    RETURN_TYPES = ("LIST", "STRING")
    RETURN_NAMES = ("audio_codes", "section_map_info")
    FUNCTION = "generate"
    CATEGORY = "Scromfy/Ace-Step/Conditioning"

    def generate(self, seed, duration, bpm, time_signature, coherence_fine,
                 coherence_coarse, section_mode, section_map="",
                 energy=0.5, density=0.5, crossfade_beats=2):
        # Parse time signature
        parts = time_signature.split("/")
        time_sig_num = int(parts[0])

        # Compute timing
        beats_per_second = bpm / 60.0
        tokens_per_beat = 5.0 / beats_per_second
        tokens_per_measure = tokens_per_beat * time_sig_num
        T = int(duration * 5)

        if T < 1:
            return ([[0]], json.dumps([]))

        # Build section map
        if section_mode == "manual" and section_map.strip():
            sections = parse_section_map(section_map)
            if not sections:
                sections = [{"type": "verse", "start": 0.0, "end": duration}]
        elif section_mode == "auto":
            sections = build_default_section_map(duration)
        else:
            sections = [{"type": "verse", "start": 0.0, "end": duration}]

        # Coherence frequencies based on user controls
        fine_freq = 0.15 + (1.0 - coherence_fine) * 0.15    # 0.15-0.30
        coarse_freq = 0.03 + (1.0 - coherence_coarse) * 0.07  # 0.03-0.10
        fine_octaves = 3 if coherence_fine > 0.3 else 2
        coarse_octaves = 2 if coherence_coarse > 0.3 else 1

        # Crossfade window in tokens
        crossfade_tokens = int(crossfade_beats * tokens_per_beat) if crossfade_beats > 0 else 0

        codes = []
        for t in range(T):
            dims = self._generate_hierarchical_dims(
                t, seed, sections, tokens_per_measure, tokens_per_beat,
                fine_freq, coarse_freq, fine_octaves, coarse_octaves,
                energy, density, duration)

            # Zero-Quadratic: section repetition
            dims = self._apply_section_repetition(
                t, dims, seed, sections, tokens_per_measure, tokens_per_beat,
                fine_freq, coarse_freq, fine_octaves, coarse_octaves,
                energy, density, duration)

            # Zero-Quadratic: motif echo
            dims = self._apply_motif_echo(
                t, dims, seed, sections, tokens_per_measure, tokens_per_beat,
                fine_freq, coarse_freq, fine_octaves, coarse_octaves,
                energy, density, duration)

            # Zero-Quadratic: call-and-response
            dims = self._apply_call_response(
                t, dims, seed, sections, tokens_per_measure, tokens_per_beat,
                fine_freq, coarse_freq, fine_octaves, coarse_octaves,
                energy, density, duration)

            # Crossfade at section boundaries
            if crossfade_tokens > 0:
                dims = self._apply_crossfade(
                    t, dims, seed, sections, crossfade_tokens)

            codes.append(dims_to_composite(dims))

        section_info = json.dumps(sections)
        logger.info(f"ZeroCond: generated {len(codes)} codes, seed={seed}, "
                     f"sections={len(sections)}")
        return ([codes], section_info)

    def _generate_hierarchical_dims(self, t, seed, sections, tpm, tpb,
                                     fine_freq, coarse_freq, fine_oct, coarse_oct,
                                     energy, density, duration):
        """Generate base 6D dims for timestep t using hierarchy + coherent noise."""
        sec_type_id, sec_idx = lookup_section(t, sections)
        sec_seed = position_hash(sec_type_id, sec_idx, 0x5EC, seed)

        measure_global = int(t / tpm) if tpm > 0 else 0
        # Find section type name for start_token lookup
        sec_type_name = None
        for s in sections:
            if SECTION_TYPE_IDS.get(s["type"], 1) == sec_type_id:
                sec_type_name = s["type"]
                break

        if sec_type_name:
            sec_start_t = section_start_token(sections, sec_type_name, sec_idx)
            measure_in_sec = int((t - sec_start_t) / tpm) if tpm > 0 else 0
        else:
            measure_in_sec = measure_global

        mea_seed = position_hash(measure_in_sec, 0, 0x4EA, sec_seed)
        beat_in_measure = int((t % tpm) / tpb) if tpb > 0 else 0

        dims = []
        for d in range(6):
            levels = FSQ_LEVELS[d]
            salt = DIM_SALTS[d]

            if d >= 3:
                # Coarse dims: 70% section seed, 30% coherent noise
                base = hash_to_float(position_hash(d, 0, salt, sec_seed))
                variation = coherent_value(t * coarse_freq, d * 0.1,
                                           mea_seed + salt, octaves=coarse_oct)
                combined = base * 0.7 + (variation * 0.5 + 0.5) * 0.3

                # Apply energy/density bias
                if d == 3:
                    combined = combined * 0.6 + energy * 0.4
                elif d == 4:
                    combined = combined * 0.7 + density * 0.3

                # Intro/outro envelope on d3 (energy)
                if d == 3:
                    time_s = t / 5.0
                    for sec in sections:
                        if sec["start"] <= time_s < sec["end"]:
                            sec_dur = max(0.1, sec["end"] - sec["start"])
                            if sec["type"] == "intro":
                                progress = (time_s - sec["start"]) / sec_dur
                                combined *= progress * progress * (3 - 2 * progress)
                            elif sec["type"] == "outro":
                                progress = (time_s - sec["start"]) / sec_dur
                                combined *= 1.0 - progress * progress * (3 - 2 * progress)
                            break

                dim_val = combined * (levels - 1)
            else:
                # Fine dims: 40% measure base, 60% coherent noise
                base = hash_to_float(position_hash(d, beat_in_measure, salt, mea_seed))
                variation = coherent_value(t * fine_freq, d * 0.1,
                                           seed + salt, octaves=fine_oct)
                combined = base * 0.4 + (variation * 0.5 + 0.5) * 0.6
                dim_val = combined * (levels - 1)

            dims.append(max(0.0, min(float(levels - 1), dim_val)))

        return dims

    def _apply_section_repetition(self, t, dims, seed, sections, tpm, tpb,
                                   fine_freq, coarse_freq, fine_oct, coarse_oct,
                                   energy, density, duration):
        """Blend dims with corresponding timestep from earlier same-type sections."""
        sec_type_id, sec_idx = lookup_section(t, sections)
        if sec_idx == 0:
            return dims  # first instance, nothing to repeat

        sec_type_name = None
        for s in sections:
            if SECTION_TYPE_IDS.get(s["type"], 1) == sec_type_id:
                sec_type_name = s["type"]
                break
        if not sec_type_name:
            return dims

        current_start = section_start_token(sections, sec_type_name, sec_idx)
        t_in_section = t - current_start

        for prev_type_id, prev_idx in all_previous_same_type(sections, sec_type_name, sec_idx):
            rep_factor = section_repetition_factor(
                prev_type_id, prev_idx, sec_type_id, sec_idx, seed)
            if rep_factor > 0.5:
                prev_start = section_start_token(sections, sec_type_name, prev_idx)
                prev_t = prev_start + t_in_section
                prev_dims = self._generate_hierarchical_dims(
                    prev_t, seed, sections, tpm, tpb,
                    fine_freq, coarse_freq, fine_oct, coarse_oct,
                    energy, density, duration)
                for d in range(6):
                    dims[d] = dims[d] * (1 - rep_factor) + prev_dims[d] * rep_factor
        return dims

    def _apply_motif_echo(self, t, dims, seed, sections, tpm, tpb,
                           fine_freq, coarse_freq, fine_oct, coarse_oct,
                           energy, density, duration):
        """Apply motif echo from previous measures."""
        current_measure = int(t / tpm) if tpm > 0 else 0
        lookback = 16

        for prev_measure in range(max(0, current_measure - lookback), current_measure):
            echo = motif_echo(prev_measure, current_measure, seed)
            if echo is not None:
                beat_offset = t - int(current_measure * tpm) if tpm > 0 else 0
                prev_t = int(prev_measure * tpm) + beat_offset
                if prev_t < 0:
                    continue
                prev_dims = self._generate_hierarchical_dims(
                    prev_t, seed, sections, tpm, tpb,
                    fine_freq, coarse_freq, fine_oct, coarse_oct,
                    energy, density, duration)
                blend = echo["strength"] * (1 - echo["variation"])
                for d in echo["echo_dims"]:
                    dims[d] = dims[d] * (1 - blend) + prev_dims[d] * blend
        return dims

    def _apply_call_response(self, t, dims, seed, sections, tpm, tpb,
                              fine_freq, coarse_freq, fine_oct, coarse_oct,
                              energy, density, duration):
        """Apply directional call-and-response influence from recent tokens."""
        lookback = 20
        for prev_t in range(max(0, t - lookback), t):
            influence = call_response_influence(prev_t, t, seed)
            if influence > 0.3:
                prev_dims = self._generate_hierarchical_dims(
                    prev_t, seed, sections, tpm, tpb,
                    fine_freq, coarse_freq, fine_oct, coarse_oct,
                    energy, density, duration)
                for d in range(3):  # Only fine dims respond
                    dims[d] = dims[d] * (1 - influence * 0.4) + prev_dims[d] * influence * 0.4
        return dims

    def _apply_crossfade(self, t, dims, seed, sections, crossfade_tokens):
        """Smoothstep crossfade at section boundaries."""
        time_s = t / 5.0
        for i, sec in enumerate(sections):
            if i + 1 < len(sections):
                boundary = sec["end"]
                boundary_t = int(boundary * 5)
                dist_to_boundary = boundary_t - t
                if 0 <= dist_to_boundary < crossfade_tokens:
                    # Approaching boundary: blend with next section's character
                    progress = 1.0 - (dist_to_boundary / crossfade_tokens)
                    progress = progress * progress * (3 - 2 * progress)  # smoothstep

                    next_sec = sections[i + 1]
                    next_type_id = SECTION_TYPE_IDS.get(next_sec["type"], 1)
                    next_sec_seed = position_hash(next_type_id, 0, 0x5EC, seed)
                    # Shift coarse dims toward next section character
                    for d in range(3, 6):
                        next_base = hash_to_float(
                            position_hash(d, 0, DIM_SALTS[d], next_sec_seed))
                        dims[d] = dims[d] * (1 - progress * 0.5) + \
                                  next_base * (FSQ_LEVELS[d] - 1) * progress * 0.5
                    break
        return dims


NODE_CLASS_MAPPINGS = {
    "AceStepZeroConditioningGenerator": AceStepZeroConditioningGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepZeroConditioningGenerator": "Zero Conditioning Generator",
}
