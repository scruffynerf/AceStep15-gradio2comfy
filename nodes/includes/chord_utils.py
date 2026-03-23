"""
Chord and Music Theory utilities for ACE-Step Chord Injector.
Extracted from referencecode/chords/Ace-Step_chord_injector/nodes.py
"""

from __future__ import annotations
import re
from typing import List, Optional, Tuple
import numpy as np
from .fsq_utils import (
    get_fsq_levels, get_fsq_vocab_size,
    vae_encode_audio, get_tokenizer, unwrap_codes,
    extract_fsq_codes, patch_conditioning,
)

_SR        = 48_000
_FSQ_RATIO = 5        # 25 Hz → 5 Hz

# ─────────────────────────────────────────────────────────────────────────────
#  Music theory
# ─────────────────────────────────────────────────────────────────────────────

_NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
_NOTE_ALIASES = {
    "Db":"C#","Eb":"D#","Fb":"E","Gb":"F#","Ab":"G#","Bb":"A#","Cb":"B",
    "D♭":"C#","E♭":"D#","G♭":"F#","A♭":"G#","B♭":"A#",
}
_CHORD_INTERVALS = {
    "mM7":[0,3,7,11],"maj7":[0,4,7,11],"maj9":[0,4,7,11,14],
    "add9":[0,4,7,14],"dim7":[0,3,6,9],"m7b5":[0,3,6,10],
    "sus2":[0,2,7],"sus4":[0,5,7],"aug":[0,4,8],"dim":[0,3,6],
    "min":[0,3,7],"maj":[0,4,7],"m9":[0,3,7,10,14],"m7":[0,3,7,10],
    "m6":[0,3,7,9],"m":[0,3,7],"M7":[0,4,7,11],"9":[0,4,7,10,14],
    "13":[0,4,7,10,14,17,21],"11":[0,4,7,10,14,17],"7":[0,4,7,10],
    "6":[0,4,7,9],"5":[0,7],"":[0,4,7],
}


def parse_chord(s: str) -> Optional[Tuple[int, List[int]]]:
    s = s.strip()
    if not s or s in {"-","N","N.C.","rest","x"}:
        return None
    m = re.match(r'^([A-Ga-g][b#♭♯]?)', s)
    if not m:
        return None
    root_raw = m.group(1)
    qual = s[len(root_raw):].split("/")[0].strip()
    rn = root_raw[0].upper() + root_raw[1:].replace("♭","b").replace("♯","#")
    rn = _NOTE_ALIASES.get(rn, rn)
    if rn not in _NOTE_NAMES:
        return None
    pc  = _NOTE_NAMES.index(rn)
    ivs = next((v for q,v in _CHORD_INTERVALS.items()
                if qual == q or (q and qual.startswith(q))),
               _CHORD_INTERVALS[""])
    return 48 + pc, ivs


def midi_to_freq(m: int) -> float:
    return 440.0 * (2.0 ** ((m - 69) / 12.0))


def synth_note(freq: float, dur: float, kind: str, vel: float) -> np.ndarray:
    n   = max(1, int(_SR * dur))
    t   = np.linspace(0.0, dur, n, endpoint=False, dtype=np.float64)
    sig = np.zeros(n, dtype=np.float64)
    nyq = _SR / 2.0
    if kind == "piano":
        for p,a,d in zip([1,2,3,4,5,6,7,8],
                         [1.0,.50,.25,.15,.08,.05,.03,.02],
                         [2.5,3.5,5.0,6.5,8.0,9.5,11.,13.]):
            if freq*p >= nyq: break
            sig += a * np.exp(-t*d) * np.sin(2*np.pi*freq*p*t)
        att = min(int(_SR*.004), n)
        sig[:att] *= np.linspace(0, 1, att)
    elif kind == "organ":
        for p,a in [(1,1.0),(2,.8),(3,.6),(4,.4),(6,.2),(8,.1)]:
            if freq*p >= nyq: break
            sig += a * np.sin(2*np.pi*freq*p*t)
        att,rel = min(int(_SR*.012),n), min(int(_SR*.04),n)
        env = np.ones(n)
        env[:att]   = np.linspace(0,1,att)
        env[n-rel:] *= np.linspace(1,0,rel)
        sig *= env
    else:  # pad
        for p in [1,2,3]:
            if freq*p >= nyq: break
            sig += (1/p)*np.sin(2*np.pi*freq*p*t)
            sig += (.4/p)*np.sin(2*np.pi*freq*p*1.003*t)
        att,rel = min(int(_SR*.3),n//2), min(int(_SR*.3),n//2)
        env = np.ones(n)
        env[:att]   = np.linspace(0,1,att)
        env[n-rel:] *= np.linspace(1,0,rel)
        sig *= env
    return (sig * vel).astype(np.float32)


def parse_chord_tokens(text: str, default_beats: float):
    """Parse 'Am:2 F C G' → [(chord, beats), ...]"""
    result = []
    for tok in text.split():
        if not tok: continue
        beats = None
        if ":" in tok:
            c, b = tok.split(":",1)
            try: beats = float(b)
            except ValueError: c = tok
        elif "(" in tok:
            m = re.match(r'^([^(]+)\(([0-9.]+)\)$', tok)
            if m:
                c = m.group(1)
                try: beats = float(m.group(2))
                except ValueError: c = tok
            else: c = tok
        else: c = tok
        result.append((c, beats if beats is not None else default_beats))
    return result


def synthesise_region(chords, bpm, duration, kind, velocity):
    """Synthesise one region of audio for a given chord list and duration."""
    spb = 60.0 / bpm
    out = np.zeros(int(_SR * duration), dtype=np.float32)
    events, cur = [], 0.0
    for cs, beats in chords:
        events.append((cur, cs, beats * spb)); cur += beats * spb
    loop = cur
    if loop <= 0: return out
    offset = 0.0
    while offset < duration:
        for ev_start, cs, dur in events:
            abs_s = offset + ev_start
            if abs_s >= duration: break
            parsed = parse_chord(cs)
            if parsed is None: continue
            root, ivs = parsed
            rend = min(dur + 0.25, duration - abs_s)
            ss   = int(abs_s * _SR)
            for iv in ivs:
                nm = root + iv
                while nm > 72: nm -= 12
                while nm < 36: nm += 12
                note = synth_note(midi_to_freq(nm), rend, kind, velocity)
                es   = min(ss + len(note), len(out))
                out[ss:es] += note[:es-ss]
        offset += loop
    return out


def parse_lyrics_sections(lyrics: str) -> List[Tuple[str, int]]:
    """
    Parse lyrics text → ordered list of (section_name, content_line_count).
    Sections marked by [verse], [chorus], [bridge], [intro], [outro] etc.
    Blank lines don't count toward line count.
    Returns e.g. [('intro',2), ('verse',6), ('chorus',4), ('verse',6), ...]
    """
    sections = []
    current_section = "default"
    current_lines   = 0
    for raw_line in lyrics.splitlines():
        line = raw_line.strip()
        m = re.match(r'^\[([^\]]+)\]', line)
        if m:
            if current_lines > 0 or sections:
                sections.append((current_section, max(current_lines, 1)))
            current_section = m.group(1).lower()
            current_lines   = 0
        elif line:
            current_lines += 1
    if current_lines > 0 or not sections:
        sections.append((current_section, max(current_lines, 1)))
    return sections


def parse_chord_map(chord_map: str, default_beats: float) -> dict:
    """
    Parse chord map text → {section_name: [(chord, beats), ...]}

    Format:
        [verse]  Am F C G       ← bracketed section name
        [chorus] F C G Am
        default  Am F C G       ← bare word 'default' → stored as 'default'
        Am F C G                ← bare chords with no prefix → 'default'
    """
    result  = {}
    current = "default"
    for raw in chord_map.splitlines():
        line = raw.strip()
        if not line: continue
        m = re.match(r'^\[([^\]]+)\]\s*(.*)', line)
        if m:
            # [section] tag
            current = m.group(1).lower()
            rest    = m.group(2).strip()
            if rest:
                result[current] = parse_chord_tokens(rest, default_beats)
        else:
            # Bare lines: check if first word is a known section keyword
            _SECTION_KEYWORDS = {
                "default","verse","chorus","bridge","intro","outro",
                "pre","post","hook","refrain","solo","interlude","tag",
            }
            parts = line.split(None, 1)
            first = parts[0].lower()
            rest  = parts[1].strip() if len(parts) > 1 else ""
            if first in _SECTION_KEYWORDS and rest:
                current = first
                result[current] = parse_chord_tokens(rest, default_beats)
            else:
                # Pure chord line → append to current section
                tokens = parse_chord_tokens(line, default_beats)
                if tokens:
                    result[current] = tokens
    return result


def build_chord_audio(
    lyrics:        str,
    chord_map_txt: str,
    bpm:           float,
    beats_per_chord: float,
    total_dur:     float,
    synth_type:    str,
    velocity:      float,
) -> np.ndarray:
    """
    Build full-length chord audio, mapping chords to lyrics sections.
    """
    chord_map  = parse_chord_map(chord_map_txt, beats_per_chord)
    sections   = parse_lyrics_sections(lyrics) if lyrics.strip() else []

    print(f"  [chord] sections from lyrics: {sections}")
    print(f"  [chord] chord map keys: {list(chord_map.keys())}")

    # ── No lyrics / no sections → use default for full duration ──────────
    if not sections:
        default_chords = chord_map.get("default", list(chord_map.values())[0]
                                       if chord_map else [("C",4),("G",4),("Am",4),("F",4)])
        audio = synthesise_region(default_chords, bpm, total_dur, synth_type, velocity)
        pk = np.max(np.abs(audio))
        if pk > 1e-6: audio *= 0.85 / pk
        return audio

    # ── Distribute duration proportionally by line count ─────────────────
    total_lines = max(sum(lc for _, lc in sections), 1)
    out = np.zeros(int(_SR * total_dur), dtype=np.float32)
    cursor = 0.0

    for i, (sec_name, line_count) in enumerate(sections):
        # Last section gets whatever time remains
        if i == len(sections) - 1:
            sec_dur = total_dur - cursor
        else:
            sec_dur = (line_count / total_lines) * total_dur

        if sec_dur <= 0:
            continue

        # Look up chords: exact match → base name match → default → fallback
        chords = None
        for key in [sec_name,
                    re.sub(r'\s*\d+$', '', sec_name),  # "verse 2" → "verse"
                    "default"]:
            if key in chord_map:
                chords = chord_map[key]
                break
        if chords is None:
            chords = list(chord_map.values())[0] if chord_map else [("C",4),("G",4),("Am",4),("F",4)]

        print(f"  [chord]   [{sec_name}] {sec_dur:.1f}s → {[c[0] for c in chords]}")
        seg = synthesise_region(chords, bpm, sec_dur, synth_type, velocity)
        start = int(cursor * _SR)
        end   = min(start + len(seg), len(out))
        out[start:end] += seg[:end-start]
        cursor += sec_dur

    pk = np.max(np.abs(out))
    if pk > 1e-6: out *= 0.85 / pk
    return out
