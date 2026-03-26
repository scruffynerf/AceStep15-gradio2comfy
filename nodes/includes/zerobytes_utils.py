"""Zerobyte Family utilities for AceStep ZeroConditioning nodes.

Position-is-seed procedural determinism: the coordinate IS the seed.
No global state, no iteration, no random module. Pure coordinate hashing.

Three hash families:
  - position_hash:  O(1) point properties (Zerobytes)
  - pair_hash:      O(N²) symmetric relationships (Zero-Quadratic)
  - asymmetric_pair_hash: O(N²) directional relationships (Zero-Quadratic)
"""
import struct
import math
import json

# ─── Hash foundation ──────────────────────────────────────────────────────────

try:
    import xxhash
    _HAS_XXHASH = True
except ImportError:
    import hashlib
    _HAS_XXHASH = False


def position_hash(a: int, b: int, salt: int, seed: int) -> int:
    """Pure coordinate hash. No state. No iteration. O(1)."""
    if _HAS_XXHASH:
        h = xxhash.xxh64(seed=seed ^ salt)
        h.update(struct.pack('<qq', a, b))
        return h.intdigest()
    else:
        data = struct.pack('<qqqi', a, b, salt, seed)
        return int(hashlib.blake2b(data, digest_size=8).hexdigest(), 16)


def pair_hash(ax: int, ay: int, bx: int, by: int, salt: int) -> int:
    """Symmetric pair hash: pair_hash(A, B) == pair_hash(B, A).
    Sort coordinates before packing to guarantee symmetry."""
    p1, p2 = sorted([(ax, ay), (bx, by)])
    if _HAS_XXHASH:
        h = xxhash.xxh64(seed=salt)
        h.update(struct.pack('<qqqq', p1[0], p1[1], p2[0], p2[1]))
        return h.intdigest()
    else:
        data = struct.pack('<qqqqi', p1[0], p1[1], p2[0], p2[1], salt)
        return int(hashlib.blake2b(data, digest_size=8).hexdigest(), 16)


def asymmetric_pair_hash(ax: int, ay: int, bx: int, by: int, salt: int) -> int:
    """Directional pair hash: rel(A->B) != rel(B->A). No sorting."""
    if _HAS_XXHASH:
        h = xxhash.xxh64(seed=salt)
        h.update(struct.pack('<qqqq', ax, ay, bx, by))
        return h.intdigest()
    else:
        data = struct.pack('<qqqqi', ax, ay, bx, by, salt)
        return int(hashlib.blake2b(data, digest_size=8).hexdigest(), 16)


def hash_to_float(h: int) -> float:
    """Map hash to [0.0, 1.0)."""
    return (h & 0xFFFFFFFF) / 0x100000000


def hash_to_level(h: int, num_levels: int) -> int:
    """Map hash to valid FSQ level index [0, num_levels - 1]."""
    return (h >> 16) % num_levels


# ─── Coherent noise ───────────────────────────────────────────────────────────

def coherent_value(x: float, y: float, seed: int, octaves: int = 4) -> float:
    """1D/2D coherent noise via hashed value interpolation. Returns [-1.0, 1.0].

    Uses smoothstep interpolation between hashed grid points. Each octave
    doubles frequency and halves amplitude (standard fBm).
    """
    value, amp, freq, max_amp = 0.0, 1.0, 1.0, 0.0
    for i in range(octaves):
        x0 = int(math.floor(x * freq))
        y0 = int(math.floor(y * freq))
        sx = (x * freq) - math.floor(x * freq)
        sy = (y * freq) - math.floor(y * freq)
        sx = sx * sx * (3 - 2 * sx)  # smoothstep
        sy = sy * sy * (3 - 2 * sy)

        n00 = hash_to_float(position_hash(x0,     y0,     0, seed + i)) * 2 - 1
        n10 = hash_to_float(position_hash(x0 + 1, y0,     0, seed + i)) * 2 - 1
        n01 = hash_to_float(position_hash(x0,     y0 + 1, 0, seed + i)) * 2 - 1
        n11 = hash_to_float(position_hash(x0 + 1, y0 + 1, 0, seed + i)) * 2 - 1

        nx0 = n00 * (1 - sx) + n10 * sx
        nx1 = n01 * (1 - sx) + n11 * sx
        value += amp * (nx0 * (1 - sy) + nx1 * sy)
        max_amp += amp
        amp *= 0.5
        freq *= 2.0
    return value / max_amp if max_amp > 0 else 0.0


def coherent_field(xs, ys, seeds, octaves=4):
    """Vectorised coherent noise for arrays of (x, y, seed) triples.

    Args:
        xs:    list/array of float x-coordinates  (length N)
        ys:    list/array of float y-coordinates  (length N)
        seeds: list/array of int seeds            (length N)
        octaves: int

    Returns:
        list of float values in [-1.0, 1.0], length N.

    ~10-50x faster than calling coherent_value() in a Python loop because
    the per-octave hash calls are batched and the interpolation is done
    in bulk via list comprehensions.
    """
    N = len(xs)
    values = [0.0] * N
    max_amp = 0.0
    amp = 1.0
    freq = 1.0

    for i in range(octaves):
        # Grid coordinates and fractional positions for all N points
        x0s = [int(math.floor(xs[j] * freq)) for j in range(N)]
        y0s = [int(math.floor(ys[j] * freq)) for j in range(N)]
        sxs = [(xs[j] * freq) - math.floor(xs[j] * freq) for j in range(N)]
        sys_ = [(ys[j] * freq) - math.floor(ys[j] * freq) for j in range(N)]
        # Smoothstep
        sxs = [s * s * (3 - 2 * s) for s in sxs]
        sys_ = [s * s * (3 - 2 * s) for s in sys_]

        # Hash all 4 corners for all N points
        n00 = [hash_to_float(position_hash(x0s[j],     y0s[j],     0, seeds[j] + i)) * 2 - 1 for j in range(N)]
        n10 = [hash_to_float(position_hash(x0s[j] + 1, y0s[j],     0, seeds[j] + i)) * 2 - 1 for j in range(N)]
        n01 = [hash_to_float(position_hash(x0s[j],     y0s[j] + 1, 0, seeds[j] + i)) * 2 - 1 for j in range(N)]
        n11 = [hash_to_float(position_hash(x0s[j] + 1, y0s[j] + 1, 0, seeds[j] + i)) * 2 - 1 for j in range(N)]

        # Interpolate all N points
        for j in range(N):
            nx0 = n00[j] * (1 - sxs[j]) + n10[j] * sxs[j]
            nx1 = n01[j] * (1 - sxs[j]) + n11[j] * sxs[j]
            values[j] += amp * (nx0 * (1 - sys_[j]) + nx1 * sys_[j])

        max_amp += amp
        amp *= 0.5
        freq *= 2.0

    if max_amp > 0:
        values = [v / max_amp for v in values]
    return values


# ─── FSQ constants ────────────────────────────────────────────────────────────

FSQ_LEVELS = [8, 8, 8, 5, 5, 5]
FSQ_VOCAB_SIZE = 64000  # product of levels

DIM_SALTS = [0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000]

SECTION_TYPE_IDS = {
    "intro": 0, "verse": 1, "chorus": 2, "bridge": 3,
    "outro": 4, "breakdown": 5, "drop": 6, "prechorus": 7,
}


def dims_to_composite(dims: list) -> int:
    """Convert 6 dimension values to a single composite index [0, 63999]."""
    composite = 0
    stride = 1
    for d_val, levels in zip(dims, FSQ_LEVELS):
        composite += max(0, min(levels - 1, int(d_val))) * stride
        stride *= levels
    return composite


# ─── Section map ──────────────────────────────────────────────────────────────

def parse_section_map(section_map_str: str) -> list:
    """Parse section map JSON string to list of dicts.
    Returns [{"type": str, "start": float, "end": float}, ...]."""
    if not section_map_str or not section_map_str.strip():
        return []
    try:
        sections = json.loads(section_map_str)
        if isinstance(sections, list):
            return sections
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def build_default_section_map(duration: float, form: str = "ABABCB",
                               intro_s: float = 8.0, outro_s: float = 8.0,
                               verse_weight: float = 1.0,
                               chorus_weight: float = 1.0) -> list:
    """Build a section map from a form template string."""
    form_map = {"A": "verse", "B": "chorus", "C": "bridge",
                "D": "breakdown", "E": "drop", "F": "prechorus"}
    body_sections = [form_map.get(c, "verse") for c in form.upper() if c in form_map]

    if not body_sections:
        body_sections = ["verse"]

    intro_s = max(0.0, min(intro_s, duration * 0.25))
    outro_s = max(0.0, min(outro_s, duration * 0.25))
    body_duration = duration - intro_s - outro_s

    if body_duration <= 0:
        return [{"type": "verse", "start": 0.0, "end": duration}]

    # Compute weighted section durations
    weights = []
    for sec in body_sections:
        if sec == "verse":
            weights.append(verse_weight)
        elif sec == "chorus":
            weights.append(chorus_weight)
        else:
            weights.append(1.0)
    total_weight = sum(weights)
    section_durations = [(w / total_weight) * body_duration for w in weights]

    result = []
    cursor = 0.0
    if intro_s > 0:
        result.append({"type": "intro", "start": 0.0, "end": round(intro_s, 2)})
        cursor = intro_s
    for sec_type, sec_dur in zip(body_sections, section_durations):
        result.append({"type": sec_type, "start": round(cursor, 2),
                       "end": round(cursor + sec_dur, 2)})
        cursor += sec_dur
    if outro_s > 0:
        result.append({"type": "outro", "start": round(cursor, 2),
                       "end": round(duration, 2)})
    return result


def lookup_section(t: int, sections: list, rate: float = 5.0):
    """Return (section_type_id, section_index) for timestep t.
    section_index counts per-type occurrences (verse 0, verse 1, etc.)."""
    time_s = t / rate
    type_counts = {}
    for sec in sections:
        sec_type = sec["type"]
        type_id = SECTION_TYPE_IDS.get(sec_type, 1)
        if sec_type not in type_counts:
            type_counts[sec_type] = 0
        if sec["start"] <= time_s < sec["end"]:
            return type_id, type_counts[sec_type]
        type_counts[sec_type] += 1
    # Fallback: last section
    if sections:
        last = sections[-1]
        type_id = SECTION_TYPE_IDS.get(last["type"], 1)
        count = sum(1 for s in sections if s["type"] == last["type"]) - 1
        return type_id, max(0, count)
    return 1, 0  # default: verse 0


def section_start_token(sections: list, sec_type: str, sec_idx: int,
                         rate: float = 5.0) -> int:
    """Return the starting token index for a specific section instance."""
    count = 0
    for sec in sections:
        if sec["type"] == sec_type:
            if count == sec_idx:
                return int(sec["start"] * rate)
            count += 1
    return 0


def all_previous_same_type(sections: list, current_type: str,
                            current_idx: int) -> list:
    """Return list of (type_id, index) for all earlier sections of the same type."""
    type_id = SECTION_TYPE_IDS.get(current_type, 1)
    result = []
    count = 0
    for sec in sections:
        if sec["type"] == current_type:
            if count < current_idx:
                result.append((type_id, count))
            count += 1
    return result


# ─── Zero-Quadratic relational hashes ────────────────────────────────────────

MOTIF_SALT = 0x40710
CALL_RESPONSE_SALT = 0xCA110
SEC_PAIR_SALT = 0x5ECA1


def section_repetition_factor(sec_type_a: int, sec_idx_a: int,
                               sec_type_b: int, sec_idx_b: int,
                               song_seed: int) -> float:
    """Returns 0.0-1.0. Same-type sections get 0.7-1.0 affinity."""
    ph = pair_hash(sec_type_a, sec_idx_a, sec_type_b, sec_idx_b,
                   song_seed ^ SEC_PAIR_SALT)
    if sec_type_a == sec_type_b:
        return 0.7 + 0.3 * hash_to_float(ph)
    else:
        return 0.1 * hash_to_float(ph)


def motif_echo(measure_a: int, measure_b: int,
               song_seed: int, threshold: float = 0.75) -> dict:
    """Returns echo dict or None if below threshold."""
    if measure_a == measure_b:
        return None
    strength = hash_to_float(pair_hash(measure_a, 0, measure_b, 0,
                                       song_seed ^ MOTIF_SALT))
    if strength < threshold:
        return None
    echo_dims = []
    for d in range(6):
        if hash_to_float(pair_hash(measure_a, d + 10, measure_b, d + 10,
                                    song_seed ^ MOTIF_SALT)) > 0.5:
            echo_dims.append(d)
    if not echo_dims:
        echo_dims = [0]  # at least one dim echoes
    variation = hash_to_float(pair_hash(measure_a, 99, measure_b, 99,
                                        song_seed ^ MOTIF_SALT)) * 0.3
    return {"strength": strength, "echo_dims": echo_dims, "variation": variation}


def call_response_influence(t_call: int, t_response: int,
                             song_seed: int, max_distance: int = 40) -> float:
    """Directional influence with quadratic temporal decay. Returns 0.0-1.0."""
    distance = abs(t_response - t_call)
    if distance > max_distance or t_response <= t_call:
        return 0.0
    base = hash_to_float(asymmetric_pair_hash(t_call, 0, t_response, 0,
                                               song_seed ^ CALL_RESPONSE_SALT))
    falloff = 1.0 - (distance / max_distance) ** 2
    return base * falloff
