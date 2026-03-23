<!-- markdownlint-disable MD024 -->
# Audio & Post-Processing Nodes

The Audio category handles the decoding of mathematical latents into physical waveforms, the ingestion of external audio, DSP/AI analysis, and high-fidelity file exports.

---

## 1. Scromfy Audio VAE Decode PLUSPLUS

*File: `nodes/audio_vae_decode_plusplus_node.py`*

An enhanced replacement for standard VAE Decode nodes. It perfectly replicates the specific VAE decoding logic required for ACE-Step architecture, while exposing local parameter overrides for latent shifting, rescaling, peak normalization, and vocal boosting.

### Inputs

- **`samples`** *(Required, LATENT)*
- **`vae`** *(Required, VAE)*

### Options

- **`latent_shift`**: Additive shift on DiT latents before VAE decode to combat clipping.
- **`latent_rescale`**: Multiplicative scale applied to the latents.
- **`normalize_peak`**: Normalizes the final waveform to maximum amplitude.
- **`voice_boost`**: Boosts the vocal presence in dB (post-processing).
- **`vae_decode_settings`** *(Optional)*: Allows inheriting these settings from a global master node.

### Outputs

- **`audio`** (`AUDIO`): The standard ComfyUI audio dictionary (waveform and sample rate).

---

## 2. Scromfy AceStep Music Analyzer (LLM)

*File: `nodes/sft_music_analyzer_node.py`*

AI-powered audio analysis. Uses massive language models (like Qwen2.5 Omni, MERT, AST, or Whisper) to 'listen' to an audio input and generate descriptive musical tags. It also automatically estimates BPM and keyscale.

*Note: Models are automatically downloaded to `models/LLM/` on first use.*

### Inputs

- **`audio`** *(Required, AUDIO)*

### Options

- **`model`**: Dropdown of 9 different analysis models.
- **`get_tags`**, **`get_bpm`**, **`get_keyscale`**: Toggles for what data to extract.
- **Generative Params**: `max_new_tokens`, `audio_duration`, `use_flash_attn`, `temperature`, `top_p/k`.
- **`unload_model`**: Flushes the LLM from VRAM after analysis to free space for generation.

### Outputs

- **`tags`** (`STRING`)
- **`bpm`** (`INT`)
- **`keyscale`** (`STRING`)
- **`duration`** (`FLOAT`)
- **`music_infos`** (`STRING`): A JSON string dump of all extracted data.

---

## 3. Audio Analyzer (No LLM)

*File: `nodes/audio_analyzer_node.py`*

A lightweight, purely DSP-based metadata extraction node for BPM, keyscale, and duration. Excellent for fast analysis on low-VRAM machines where loading a large LLM is impossible.

### Outputs

- **`bpm`** (`INT`)
- **`keyscale`** (`STRING`)
- **`duration`** (`FLOAT`)

---

## 4. Scromfy Audio Post Process

*File: `nodes/audio_post_process_node.py`*

A mastering macro node specifically tuned for generated audio. It applies Short-Time Fourier Transform (STFT) manipulation to smooth artifacts.

### Inputs

- **`audio`** *(Required, AUDIO)*

### Options

- **`de_esser_strength`**: Reduces high-frequency (6kHz+) harshness common in generated vocals.
- **`spectral_smoothing`**: Convolutional smoothing across frequency bands to reduce "robotic" noise.

### Outputs

- **`audio`** (`AUDIO`)

---

## 5. Scromfy Save Audio

*Files: `nodes/save_audio_node.py`, `nodes/save_audio_mp3_node.py`, `nodes/save_audio_opus_node.py`*

High-fidelity multi-format export nodes. Crucially, these nodes natively embed your ComfyUI generation `prompt` and `metadata` directly into the file's ID3/metadata tags.

### Nodes Available

- **`Scromfy Save Audio (FLAC/WAV)`**
- **`Scromfy Save Audio (MP3)`**
- **`Scromfy Save Audio (Opus)`**

### Options

- **`filename_prefix`**: Where to save the file.
- **`quality`**: Bitrate specific format options (e.g. `128k`, `V0`).

### Outputs

- **`filepath`** (`STRING`): The absolute path to the saved file.

---

## 6. AceStepLoadAudio

*File: `nodes/load_audio_node.py`*

Ingests local audio files and safely resamples them to the expected sample rate geometry for ComfyUI.

---

## 7. Matchering (Audio Matching & Mastering)

*Files: `nodes/matchering_node.py`, `nodes/matchering_advanced_node.py`, `nodes/matchering_limiter_config_node.py`*

A powerful audio matching and mastering tool. It 'masters' a target audio track by matching its RMS, frequency response, and peak levels to a reference track.

**Credits:**
- **Matchering Library:** Created by [Sergree (Sergey Grishakov)](https://github.com/sergree/matchering) (GPLv3).
- **ComfyUI Adaptation:** Originally developed by [MuziekMagie](https://github.com/MuziekMagie/ComfyUI-Matchering).

### Nodes Available

- **`Matchering`**: Simple two-input node (Target + Reference).
- **`Matchering (Advanced)`**: Full control over FFT size, RMS correction steps, LOWESS smoothing, etc.
- **`Matchering Limiter Config`**: Detailed configuration for the brickwall limiter (attack, hold, release, filters).

### Outputs

- **`Result`**: Mastered audio with limiter and normalization.
- **`Result (no limiter)`**: Raw matched audio before the final limiter stage.
- **`Result (no limiter, normalized)`**: Matched audio with peak normalization but no compression/limiting.
