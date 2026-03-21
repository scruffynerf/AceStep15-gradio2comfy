# ACE-Step Nodes - Repository Index & Build Strategy

Welcome to the central index for the **ScromfyUI-AceStep** project. This document defines the project's focus, status, and provides a structural map of every code file and node category in the repository. For deep-dives into specific node functionality, follow the links to the detailed category documents.

## Project Focus

**Primary Goal:** Replicate 100% of the ACE-Step Gradio app functionality in ComfyUI nodes, while vastly surpassing its capabilities via modularity and explicit conditioning manipulation.

**Success Criteria:**

- Load ACE-Step models via standard ComfyUI loaders ✅
- Generate music in all 4 modes (Simple/Custom/Cover/Repaint) ✅
- Access all advanced parameters (LM CFG, temperature, BPM, etc.) ✅
- Use LoRA adapters ✅
- Generate lyrics via multiple LLM APIs ✅
- Apply post-processing (de-esser, spectral smoothing) ✅
- Save with proper metadata ✅

## Project Status

- **Node Implementation:** 100% Complete.
- **Native Support:** Confirmed ComfyUI native ACE-Step support. Checkpoint loaders work out-of-the-box.
- **Frontend Extensions:** Available for WebAmp, Radio, and custom lyric syncing.

---

## 📂 Repository File Index

The root directory consists of `__init__.py` which handles dynamic node scanning. All nodes reside in `nodes/`.

### Prompting Nodes ([Detailed Specs ➡](nodes/Prompt.md))

Centralize text formatting, LM prompting strategies (like Enriched CoT), and metadata building.

- `text_encoder_plusplus_node.py` — **ScromfyAceStepTextEncoderPlusPlus**: The definitive ACE-Step 1.5 text encoder.
- `metadata_builder_node.py` — **AceStepMetadataBuilder**: Formats the music metadata dictionary.
- `prompt_gen_node.py` — **AceStepPromptGen**: Dynamic multi-category prompt generator using weighted tags.
- `random_prompt_node.py` — **AceStepRandomPrompt**: Randomized music prompt generator.
- `prompt_freeform_node.py` — **Prompt Freeform**: Allows freeform text with dynamic wildcard resolution.

### Conditioning Manipulation ([Detailed Specs ➡](nodes/Conditioning.md))

Raw tensor manipulation, 5Hz audio code editing, splitting, combining, and custom masking logic.

- `audio_codes_mixer_node.py` — **AceStepAudioCodesMixer**: Binary toolbox for mixing two sets of audio codes in 6D FSQ space.
- `audio_codes_unary_op_node.py` — **AceStepAudioCodesUnaryOp**: Unary operations on audio codes with length scaling/masking.
- `conditioning_combine_node.py` — **AceStepConditioningCombine**: Assemble individual tensors and codes into a full conditioning object.
- `conditioning_dual_mixer_node.py` — **AceStepConditioningMixer**: Selectively mix components from two conditioning sources.
- `conditioning_split_node.py` — **AceStepConditioningSplitter**: Decompose conditioning into raw components.
- `audio_codes_to_semantic_hints_node.py` — **AceStepAudioCodesToSemanticHints**: Convert 5Hz audio codes to 25Hz semantic hints.
- `semantic_hints_to_audio_codes_node.py` — **AceStepSemanticHintsToAudioCodes**: Convert 25Hz semantic hints back to 5Hz audio codes.
- `audio_codes_decode_node.py` — **AceStepAudioCodesUnderstand**: Reconstruct metadata and lyrics from 5Hz token IDs.
- `conditioning_zero_out_node.py` — **AceStepConditioningZeroOut**: Zero out conditioning for negative/unconditional input.
- `conditioning_view_node.py` — **AceStepConditioningExplore**: Deep introspection and debugging of conditioning data.
- `load_audio_codes_node.py` — **AceStepAudioCodesLoader**: Load 5Hz audio code tensors from disk.
- `load_conditioning_node.py` — **AceStepConditioningLoad**: Load saved conditioning components.
- `load_lyrics_tensor_node.py` — **AceStepLyricsTensorLoader**: Load lyrics conditioning tensors.
- `load_mixed_conditioning_node.py` — **AceStepConditioningMixerLoader**: Mix saved components during load.
- `load_timbre_tensor_node.py` — **AceStepTimbreTensorLoader**: Load timbre conditioning tensors.
- `audio_mask_node.py` — **AceStepAudioMask**: Time-to-step mask generator.
- `tensor_mask_node.py` — **AceStepTensorMaskGenerator**: Primitive mask generator (fraction, range, window).
- `tensor_mixer_node.py` — **AceStepTensorMixer**: Mix two tensors with masking.
- `tensor_unary_op_node.py` — **AceStepTensorUnaryOp**: Transform single tensors.
- `save_conditioning_node.py` — **AceStepConditioningSave**: Component saver.
- `save_tensor_node.py` — **AceStepTensorSave**: Raw tensor saver.

### Audio & Post-Processing ([Detailed Specs ➡](nodes/Audio.md))

Decoding latents with extended VAE features, analyzing external audio, and post-processing tools.

- `audio_analyzer_node.py` — **Audio Analyzer (No LLM)**: DSP-based BPM, key, and duration extraction.
- `sft_music_analyzer_node.py` — **ScromfyAceStepMusicAnalyzer**: AI-powered analyzer (Whisper/Qwen) for tags and theory.
- `audio_post_process_node.py` — **AceStepPostProcess**: Audio enhancement (de-esser, spectral smoothing).
- `audio_vae_decode_plusplus_node.py` — **Scromfy Audio VAE Decode PLUSPLUS**: Advanced VAE decoder with local logic overrides.
- `save_audio_node.py` — **Scromfy Save Audio**: High-fidelity multi-format audio saver (handles metadata).
- `load_audio_node.py` — **AceStepLoadAudio**: Audio loader with auto-resampling.

### Samplers ([Detailed Specs ➡](nodes/Sampler.md))

Overriding core implementations for specific features like masking.

- `sft_sampler_node.py` — **ScromfyAceStepSampler**: The primary sampler with APG/ADG guidance and native mask-based inpainting.

### Lyrics Generation & Formatting ([Detailed Specs ➡](nodes/Lyrics.md))

AI interactions, BPM calculations, and formatters to properly align text.

- `lyrics_formatter_node.py` — **AceStepLyricsFormatter**: Structure lyrics with required formatting tags.
- `lyrics_genius_search_node.py` — **AceStepGeniusLyricsSearch**: Fetch particular lyrics from Genius.
- `lyrics_genius_random_node.py` — **AceStepRandomLyrics**: Fetch random Genius lyrics.
- `lyrics_duration_node.py` — **AceStepLyricsBPMCalculator**: BPM/Duration estimation for lyrics structure.
- `lyrics_claude_node.py` / `lyrics_gemini_node.py` / `lyrics_groq_node.py` / `lyrics_openai_node.py` / `lyrics_perplexity_node.py` / `lyrics_generic_ai_node.py` — API integrations for procedural generation.

### Radio & Playback ([Detailed Specs ➡](nodes/Radio.md))

In-UI playback experiences for Comfy.

- `webamp_node.py` — **AceStepWebAmpRadio**: Full Winamp integration.
- `radio_node.py` — **RadioPlayer**: Lightweight in-UI player.

### LoRA Loading ([Detailed Specs ➡](nodes/Lora.md))

- `load_lora_node.py` — **AceStepLoRALoader**: Standard ACE-Step 1.5 LoRA loader.
- `sft_lora_loader_node.py` — **Scromfy AceStep Lora Stack**: Advanced multi-LoRA stacking.

### Whisper Transcription ([Detailed Specs ➡](nodes/Whisper.md))

- `faster_whisper_node.py` — Multi-purpose file hosting: **Faster Whisper Loader**, **Transcribe** (VAD-enabled), and **Save Subtitle/Lyrics** (SRT/VTT/LRC generation).

### Misc & Utilities ([Detailed Specs ➡](nodes/Misc.md))

- `wikipedia_node.py` — **WikipediaRandomNode**: Pull random page content.
- `emoji_spinner_node.py` — **ScromfyEmojiSpinner**: Iconify/SVG rendering to masks.
- `mask_picker_node.py` — **ScromfyMaskPicker**: Recursive mask directory browser.

### Visualizers ([Detailed Specs ➡](nodes/Visualizers.md))

- `flex_audio_visualizer_circular_node.py`, `flex_audio_visualizer_contour_node.py`, `flex_audio_visualizer_line_node.py`, `flex_lyrics_node.py`, `emoji_spinner_visualizer_node.py`, `lyric_settings_node.py`, `visualizer_settings_node.py`

### Shared Utility Modules (`nodes/includes/`)

- `analysis_utils.py`: FSQ quantization logic and dependency checks.
- `audio_save_utils.py`: Centralized audio saving, resampling, and format transcoding math.
- `audio_utils.py`: FLAC metadata block generation.
- `emoji_utils.py`: Iconify fetching, SVG-to-Mask conversion (svglib), and caching.
- `flex_utils.py`: Dynamic layout parsing and styling logic for visualizers.
- `fsq_utils.py`: Low-level FSQ encoding/decoding math.
- `icon_collections.py`: Static categorization lists for icons mapping to genres/moods.
- `lyrics_utils.py`: Prompt builders and markdown cleaning.
- `mapping_utils.py`: Shared dictionaries (languages, time signatures) and dropdown wrappers.
- `prompt_utils.py`: Dynamic wildcard expansion and UI-weight sorting.
- `sampling_utils.py`: Noise schedule shift formulas.
- `visualizer_utils.py`: Core rendering mechanics, font-loading, and mathematical plotting for visualizers.
- `whisper_utils.py`: Model discovery, language mappings, and subtitle/LRC formatting logic.
