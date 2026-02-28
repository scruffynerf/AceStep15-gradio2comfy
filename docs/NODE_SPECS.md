# ACE-Step Nodes - Technical Specifications

All nodes live in `nodes/` and are auto-registered by `__init__.py`.
Shared logic is in `nodes/includes/`. Frontend extensions are in `web/`.

**60 node files total** — 45 active, 15 obsolete (46 active node types).

---

## Prompt & Encoding (Scromfy/Ace-Step/prompt)

1. **ScromfyACEStep15TaskTextEncode** (`text_encode_ace15_node.py`): Full ACE-Step 1.5 text encoder with human-readable dropdowns for language, key signature, and time signature. Supports LLM audio code generation toggle.
2. **AceStepPromptGen** (`prompt_gen_node.py`): Multi-category prompt generator with 8 dropdowns (style, mood, adjective, culture, genre, vocal, performer, instrument) — each offering none/random/random2 plus all items. Outputs individual category strings and a combined prompt.
3. **AceStepRandomPrompt** (`random_prompt_node.py`): Randomized music prompt generator.

## Metadata & Analysis (Scromfy/Ace-Step/metadata)

4. **AceStepAudioAnalyzer** (`audio_analyzer_node.py`): Extract BPM, key, and duration from audio.
5. **AceStepAudioCodesUnderstand** (`audio_codes_decode_node.py`): Generatively reconstruct metadata and lyrics from 5Hz token IDs.
6. **AceStepConditioningExplore** (`conditioning_view_node.py`): Deep-introspection debug node — recursively explores conditioning data with circular-reference protection, MRO display, and lovely-tensors summaries.
7. **AceStepMetadataBuilder** (`metadata_builder_node.py`): Format music metadata dictionary.

## Mixers & Transformers (Scromfy/Ace-Step/mixers)

8. **AceStepAudioCodesMixer** (`audio_codes_mixer_node.py`): Binary toolbox for mixing two sets of audio codes in 6D FSQ space.
9. **AceStepAudioCodesUnaryOp** (`audio_codes_unary_op_node.py`): Unary operations on audio codes with length scaling and optional masking.
10. **AceStepConditioningCombine** (`conditioning_combine_node.py`): Assemble individual tensors and codes into a full conditioning object.
11. **AceStepConditioningMixer** (`conditioning_dual_mixer_node.py`): Selectively mix components from two conditioning sources.
12. **AceStepConditioningSplitter** (`conditioning_split_node.py`): Inverse of the Combiner — decompose conditioning into components.

## Mixing & Masking (Scromfy/Ace-Step/mixing)

13. **AceStepAudioMask** (`audio_mask_node.py`): Time-based mask generator (maps seconds to latent steps).
14. **AceStepTensorMaskGenerator** (`tensor_mask_node.py`): Generates `[1, N, 1]` masks using various modes (all, none, fraction, range, ramp, window).
15. **AceStepTensorMixer** (`tensor_mixer_node.py`): Binary toolbox for mixing two tensors with optional masking and scaling.
16. **AceStepTensorUnaryOp** (`tensor_unary_op_node.py`): Transforms a single tensor with optional mask and length scaling.

## Advanced & Semantic (Scromfy/Ace-Step/advanced)

17. **AceStepAudioCodesToSemanticHints** (`audio_codes_to_semantic_hints_node.py`): Convert 5Hz audio codes to 25Hz semantic hints for DiT conditioning.
18. **AceStepSemanticHintsToAudioCodes** (`semantic_hints_to_audio_codes_node.py`): Convert 25Hz semantic hints back to 5Hz audio codes (lossy).

## Audio & Effects (Scromfy/Ace-Step/audio)

19. **AceStepPostProcess** (`audio_post_process_node.py`): De-esser and spectral smoothing for generated audio.

## Essential (Scromfy/Ace-Step/essential)

20. **AceStepConditioningZeroOut** (`conditioning_zero_out_node.py`): Zero out conditioning for negative/unconditional input.
21. **EmptyLatentAudio** (`empty_latent_audio_node.py`): Create empty 64-channel latents for generation.

## Loaders (Scromfy/Ace-Step/loaders)

22. **AceStepAudioCodesLoader** (`load_audio_codes_node.py`): Load audio codes from disk.
23. **AceStepConditioningLoad** (`load_conditioning_node.py`): Load and reconstruct conditioning from saved component files.
24. **AceStepLLMLoader** (`load_llm_node.py`): Specialized loader for the 5Hz LLM.
25. **AceStepLoRALoader** (`load_lora_node.py`): Specialized LoRA loader for ACE-Step 1.5.
26. **AceStepLyricsTensorLoader** (`load_lyrics_tensor_node.py`): Load a lyrics conditioning tensor from disk.
27. **AceStepConditioningMixerLoader** (`load_mixed_conditioning_node.py`): Granularly mix saved components from different files.
28. **AceStepTimbreTensorLoader** (`load_timbre_tensor_node.py`): Load a timbre conditioning tensor from disk.

## Lyrics (Scromfy/Ace-Step/lyrics)

29. **AceStepLyricsFormatter** (`lyrics_formatter_node.py`): Auto-format lyrics with required section tags.
30. **AceStepGeniusLyricsSearch** (`lyrics_genius_search_node.py`): Fetch lyrics by artist + title from Genius.com.
31. **AceStepRandomLyrics** (`lyrics_genius_random_node.py`): Pick a random Genius song and fetch its lyrics.
32. **AceStepLyricsBPMCalculator** (`lyrics_duration_node.py`): Estimate duration and suggested BPM ranges (Low/Mid/High) based on line count and word count. Filter [tags] automatically.

### AI-Powered Lyrics (Scromfy/Ace-Step/lyrics/AI)

API keys stored in `keys/*.txt` — see [keys/README.md](../keys/README.md).

33. **AceStepClaudeLyrics** (`lyrics_claude_node.py`): Anthropic Claude API.
34. **AceStepGeminiLyrics** (`lyrics_gemini_node.py`): Google Gemini API.
35. **AceStepGroqLyrics** (`lyrics_groq_node.py`): Groq API (uses official `groq` library).
36. **AceStepOpenAILyrics** (`lyrics_openai_node.py`): OpenAI API.
37. **AceStepPerplexityLyrics** (`lyrics_perplexity_node.py`): Perplexity API.
38. **AceStepGenericAILyrics** (`lyrics_generic_ai_node.py`): OpenAI-compatible API (Ollama, LM Studio, etc.) with custom `api_url`.
39. **AceStepGenericModelList** (`lyrics_generic_ai_node.py`): Fetch available model IDs from an OpenAI-compatible `/v1/models` endpoint.

## Persistence / Save (Scromfy/Ace-Step/save)

40. **AceStepConditioningSave** (`save_conditioning_node.py`): Save conditioning components to separate files.
41. **AceStepTensorSave** (`save_tensor_node.py`): Save a raw tensor to disk.

## Miscellaneous (Scromfy/Ace-Step/misc)

42. **AceStep5HzLMConfig** (`llm_config_node.py`): LLM parameter configuration.

## Radio (Scromfy/Ace-Step/Radio)

43. **RadioPlayer** (`radio_node.py`): In-UI audio player that scans an output folder and plays tracks with polling. Frontend in `web/radio_player.js`.

## TBD / Uncategorized (Scromfy/Ace-Step/TBD)

44. **AceStepInpaintSampler** (`inpaint_sampler_node.py`): Specialized sampler for audio inpainting.
45. **AceStepLoadAudio** (`load_audio_node.py`): Load audio files with auto-resampling.
46. **AceStepModeSelector** (`mode_selector_node.py`): 4-in-1 mode routing.
47. **AceStepRandomPrompt** (`random_prompt_node.py`): Randomized music prompts.

---

## Obsolete Nodes (Scromfy/Ace-Step/obsolete)

These are deprecated and will be removed in a future version.

| # | Class | File |
|---|-------|------|
| 48 | ObsoleteAceStepAudioCodesToSemanticHints | `obsolete_audio_codes_to_latent_node.py` |
| 49 | ObsoleteAceStepAudioToCodec | `obsolete_audio_to_codec_node.py` |
| 50 | ObsoleteAceStepCLIPTextEncode | `obsolete_clip_text_encode_node.py` |
| 51 | ObsoleteAceStepCodecToLatent | `obsolete_codec_to_latent_node.py` |
| 52 | ObsoleteAceStepConditioning | `obsolete_conditioning_node.py` |
| 53 | ObsoleteAceStepCustomTimesteps | `obsolete_custom_timesteps_node.py` |
| 54 | ObsoleteAceStepKSamplerAdvanced | `obsolete_ksampler_advanced_node.py` |
| 55 | ObsoleteAceStepKSampler | `obsolete_ksampler_node.py` |
| 56 | ObsoleteAceStepLatentToAudioCodes | `obsolete_latent_to_audio_codes_node.py` |
| 57 | ObsoleteAceStepLoRAStatus | `obsolete_lora_status_node.py` |
| 58 | ObsoleteFlacPreviewAudio | `obsolete_preview_audio_node.py` |
| 59 | ObsoleteSaveAudio | `obsolete_save_audio_node.py` |
| 60 | ObsoleteSaveText | `obsolete_save_text_node.py` |
| 61 | ObsoleteVAEDecodeAudio | `obsolete_vae_decode_audio_node.py` |
| 62 | ObsoleteVAEEncodeAudio | `obsolete_vae_encode_audio_node.py` |

---

## Shared Utility Modules (nodes/includes/)

- **analysis_utils.py**: FSQ quantization logic and dependency checks.
- **audio_utils.py**: FLAC metadata block generation.
- **fsq_utils.py**: Low-level FSQ encoding/decoding math.
- **lyrics_utils.py**: Prompt builders and markdown cleaning.
- **prompt_utils.py**: Style presets, genres, moods, adjectives, cultures, instruments, performers, vocal qualities.
- **sampling_utils.py**: Noise schedule shift formulas.

## Frontend Extensions (web/)

- **radio_player.js**: ComfyUI widget for the RadioPlayer node — in-browser audio player with queue, polling, and transport controls.
