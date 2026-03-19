# ACE-Step Nodes - Technical Specifications

All nodes live in `nodes/` and are auto-registered by `__init__.py`.
Shared logic is in `nodes/includes/`. Frontend extensions are in `web/`.

**82 node files total** — 64 active, 18 obsolete.

---

## Prompt (Scromfy/Ace-Step/Prompt)

1. **ScromfyAceStepTextEncoderPlusPlus** (`text_encoder_plusplus_node.py`): The definitive ACE-Step 1.5 text encoder. Merges high-level SFT "Enriched CoT" formatting with granular base controls.
2. **AceStepMetadataBuilder** (`metadata_builder_node.py`): Format music metadata dictionary (BPM, duration, key, etc.).
3. **AceStepPromptGen** (`prompt_gen_node.py`): Dynamic multi-category prompt generator using weighted tags.
4. **AceStepRandomPrompt** (`random_prompt_node.py`): Randomized music prompt generator.
5. **Prompt Freeform** (`prompt_freeform_node.py`): Allows freeform text with dynamic wildcard resolution.

## Conditioning (Scromfy/Ace-Step/Conditioning)

1. **AceStepAudioCodesMixer** (`audio_codes_mixer_node.py`): Binary toolbox for mixing two sets of audio codes in 6D FSQ space.
2. **AceStepAudioCodesUnaryOp** (`audio_codes_unary_op_node.py`): Unary operations on audio codes with length scaling and optional masking.
3. **AceStepConditioningCombine** (`conditioning_combine_node.py`): Assemble individual tensors and codes into a full conditioning object.
4. **AceStepConditioningMixer** (`conditioning_dual_mixer_node.py`): Selectively mix components from two conditioning sources.
5. **AceStepConditioningSplitter** (`conditioning_split_node.py`): Decompose conditioning into components.
6. **AceStepAudioCodesToSemanticHints** (`audio_codes_to_semantic_hints_node.py`): Convert 5Hz audio codes to 25Hz semantic hints.
7. **AceStepSemanticHintsToAudioCodes** (`semantic_hints_to_audio_codes_node.py`): Convert 25Hz semantic hints back to 5Hz audio codes.
8. **AceStepConditioningZeroOut** (`conditioning_zero_out_node.py`): Zero out conditioning for negative/unconditional input.
9. **AceStepAudioCodesUnderstand** (`audio_codes_decode_node.py`): Reconstruct metadata and lyrics from 5Hz token IDs.
10. **AceStepConditioningExplore** (`conditioning_view_node.py`): Deep introspection and debugging of conditioning data.
11. **AceStepAudioCodesLoader** (`load_audio_codes_node.py`): Load 5Hz audio code tensors from disk.
12. **AceStepConditioningLoad** (`load_conditioning_node.py`): Load saved conditioning components.
13. **AceStepLyricsTensorLoader** (`load_lyrics_tensor_node.py`): Load lyrics conditioning tensors.
14. **AceStepConditioningMixerLoader** (`load_mixed_conditioning_node.py`): Mix saved components during load.
15. **AceStepTimbreTensorLoader** (`load_timbre_tensor_node.py`): Load timbre conditioning tensors.
16. **AceStepAudioMask** (`audio_mask_node.py`): Time-to-step mask generator.
17. **AceStepTensorMaskGenerator** (`tensor_mask_node.py`): Primitive mask generator (fraction, range, window).
18. **AceStepTensorMixer** (`tensor_mixer_node.py`): Mix two tensors with masking.
19. **AceStepTensorUnaryOp** (`tensor_unary_op_node.py`): Transform single tensors.

## Sampler (Scromfy/Ace-Step/Sampler)

1. **ScromfyAceStepSampler** (`sft_sampler_node.py`): The primary SFT sampler with APG/ADG guidance and native mask-based inpainting.

## Audio (Scromfy/Ace-Step/Audio)

1. **Audio Analyzer (No LLM)** (`audio_analyzer_node.py`): DSP-based BPM, key, and duration extraction.
2. **ScromfyAceStepMusicAnalyzer** (`sft_music_analyzer_node.py`): AI-powered analyzer (Whisper/Qwen) for tags and theory.
3. **AceStepPostProcess** (`audio_post_process_node.py`): Audio enhancement (de-esser, smoothing).
4. **Scromfy Audio VAE Decode PLUSPLUS** (`audio_vae_decode_plusplus_node.py`): Advanced VAE decoder with local logic overrides.
5. **Scromfy Save Audio** (`save_audio_node.py`): High-fidelity multi-format audio saver.
6. **AceStepLoadAudio** (`load_audio_node.py`): Audio loader with auto-resampling.

## Lyrics (Scromfy/Ace-Step/Lyrics)

1. **AceStepLyricsFormatter** (`lyrics_formatter_node.py`): Structure lyrics with required tags.
2. **AceStepGeniusLyricsSearch** (`lyrics_genius_search_node.py`): Fetch lyrics from Genius.
3. **AceStepRandomLyrics** (`lyrics_genius_random_node.py`): Fetch random Genius lyrics.
4. **AceStepLyricsBPMCalculator** (`lyrics_duration_node.py`): BPM/Duration estimation for lyrics.
5. **AceStepClaudeLyrics** (`lyrics_claude_node.py`): Anthropic Claude integration.
6. **AceStepGeminiLyrics** (`lyrics_gemini_node.py`): Google Gemini integration.
7. **AceStepGroqLyrics** (`lyrics_groq_node.py`): Groq API integration.
8. **AceStepOpenAILyrics** (`lyrics_openai_node.py`): OpenAI API integration.
9. **AceStepPerplexityLyrics** (`lyrics_perplexity_node.py`): Perplexity API integration.
10. **AceStepGenericAILyrics** (`lyrics_generic_ai_node.py`): OpenAI-compatible local/remote LLMs.

## Visualizers (Scromfy/Ace-Step/Visualizers)

1. **ScromfyFlexAudioVisualizerCircular** (`flex_audio_visualizer_circular_node.py`): Circular wave/spectrum.
2. **ScromfyFlexAudioVisualizerContour** (`flex_audio_visualizer_contour_node.py`): Mask-filling contours.
3. **ScromfyFlexAudioVisualizerLine** (`flex_audio_visualizer_line_node.py`): Linear wave/spectrum.
4. **ScromfyFlexLyrics** (`flex_lyrics_node.py`): Timed lyrics overlay.
5. **ScromfyEmojiSpinnerVisualizer** (`emoji_spinner_visualizer_node.py`): Visualizer for icon strips.
6. **Lyric Settings** (`lyric_settings_node.py`): Visualizer specific lyric formatting.

## Radio (Scromfy/Ace-Step/Radio)

1. **AceStepWebAmpRadio** (`webamp_node.py`): Full Winamp integration.
2. **RadioPlayer** (`radio_node.py`): Lightweight in-UI player.

## Lora (Scromfy/Ace-Step/Lora)

1. **AceStepLoRALoader** (`load_lora_node.py`): Standard ACE-Step 1.5 LoRA loader.
2. **Scromfy AceStep Lora Stack** (`sft_lora_loader_node.py`): Advanced multi-LoRA stacking.

## Whisper (Scromfy/Ace-Step/Whisper)

1. **Faster Whisper Loader** (`faster_whisper_node.py`): Model loader.
2. **Faster Whisper Transcribe** (`faster_whisper_node.py`): VAD-enabled transcription.
3. **Save Subtitle/Lyrics** (`faster_whisper_node.py`): SRT/VTT/LRC generation.

## Misc (Scromfy/Ace-Step/Misc)

1. **AceStep5HzLMConfig** (`lm_config_node.py`): 5Hz LM parameters.
2. **WikipediaRandomNode** (`wikipedia_node.py`): Random page content.
3. **ScromfyEmojiSpinner** (`emoji_spinner_node.py`): Iconify/SVG rendering.
4. **ScromfyMaskPicker** (`mask_picker_node.py`): Recursive mask browser.

## Persistence (Scromfy/Ace-Step/Save)

1. **AceStepConditioningSave** (`save_conditioning_node.py`): Component saver.
2. **AceStepTensorSave** (`save_tensor_node.py`): Raw tensor saver.

---

## Obsolete Nodes (Scromfy/Ace-Step/obsolete)

These are deprecated and will be removed in a future version.

| Class | File |
| --- | --- |
| ObsoleteAceStepInpaintSampler | `obsolete_inpaint_sampler_node.py` |
| ObsoleteAceStepMetadataBuilder | `obsolete_metadata_builder_node.py` |
| ObsoleteAceStepModeSelector | `obsolete_mode_selector_node.py` |
| ObsoleteAceStepAudioCodesToSemanticHints | `obsolete_audio_codes_to_latent_node.py` |
| ObsoleteAceStepAudioToCodec | `obsolete_audio_to_codec_node.py` |
| ObsoleteAceStepCLIPTextEncode | `obsolete_clip_text_encode_node.py` |
| ObsoleteAceStepCodecToLatent | `obsolete_codec_to_latent_node.py` |
| ObsoleteAceStepConditioning | `obsolete_conditioning_node.py` |
| ObsoleteAceStepCustomTimesteps | `obsolete_custom_timesteps_node.py` |
| ObsoleteAceStepKSamplerAdvanced | `obsolete_ksampler_advanced_node.py` |
| ObsoleteAceStepKSampler | `obsolete_ksampler_node.py` |
| ObsoleteAceStepLatentToAudioCodes | `obsolete_latent_to_audio_codes_node.py` |
| ObsoleteAceStepLoRAStatus | `obsolete_lora_status_node.py` |
| ObsoleteEmptyLatentAudio | `obsolete_empty_latent_audio_node.py` |
| ObsoleteSaveText | `obsolete_save_text_node.py` |
| ObsoleteVAEDecodeAudio | `obsolete_vae_decode_audio_node.py` |
| ObsoleteVAEEncodeAudio | `obsolete_vae_encode_audio_node.py` |
| ScromfyAceStepTextEncode (SFT) | `obsolete_sft_text_encode_node.py` |
| ScromfyACEStep15TaskTextEncode (Base) | `obsolete_text_encode_ace15_node.py` |

---

## Shared Utility Modules (nodes/includes/)

* **analysis_utils.py**: FSQ quantization logic and dependency checks.
* **audio_utils.py**: FLAC metadata block generation.
* **emoji_utils.py**: Iconify fetching, SVG-to-Mask conversion (svglib), and caching.
* **fsq_utils.py**: Low-level FSQ encoding/decoding math.
* **lyrics_utils.py**: Prompt builders and markdown cleaning.
* **prompt_utils.py**: Dynamic wildcard expansion and UI-weight sorting.
* **sampling_utils.py**: Noise schedule shift formulas.
* **whisper_utils.py**: Model discovery, language mappings, and subtitle/LRC formatting logic.

## AI Instructions (AIinstructions/)

* **systemprompt.default.txt**: The master system prompt used for all AI lyric generation.
* **systemprompt.txt**: User-created override.

## Prompt Components (prompt_components/)

* **WEIGHTS.default.json**: Master default weights for UI sorting.
* **TOTALIGNORE.default.list**: Master default ignore list.
* **LOADBUTNOTSHOW.default.list**: Master default "hide from UI" list.
* **REPLACE.default.list**: Master default replacement map.
* **Customization**: Users can create versions without `.default` to apply overrides.

## Frontend Extensions (web/)

* **radio_player.js**: Lightweight widget for RadioPlayer.
* **webamp_player.js**: Premium Winamp widget for WebAmpRadio.
