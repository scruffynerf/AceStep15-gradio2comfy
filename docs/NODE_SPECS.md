# ACE-Step Nodes - Technical Specifications

All nodes have been implemented and refactored into individual files within the `nodes/` directory.

## Implementation Overview

- **Nodes**: 45 custom nodes implemented.
- **Shared Logic**: Isolated in `nodes/includes/`.
- **Dynamic Loading**: `__init__.py` automatically registers all `*_node.py` files.

---

## Audio I/O Nodes (nodes/*_node.py)

1. **LoadAudio** (`load_audio_node.py`): Load audio files with auto-resampling to 44.1kHz.
2. **SaveAudio** (`save_audio_node.py`): Save audio with FLAC metadata support.
3. **PreviewAudio** (`preview_audio_node.py`): Temporary audio preview in UI.
4. **EmptyLatentAudio** (`empty_latent_audio_node.py`): Create empty 64-channel latents.
5. **VAEEncodeAudio** (`vae_encode_audio_node.py`): Waveform → latents.
6. **VAEDecodeAudio** (`vae_decode_audio_node.py`): Latents → waveform.

## Lyrics Generation Nodes (nodes/*_node.py)

7. **AceStepGeminiLyrics** (`gemini_lyrics_node.py`): Google Gemini API.
8. **AceStepGroqLyrics** (`groq_lyrics_node.py`): Groq API.
9. **AceStepOpenAILyrics** (`openai_lyrics_node.py`): OpenAI API.
10. **AceStepClaudeLyrics** (`claude_lyrics_node.py`): Anthropic Claude API.
11. **AceStepPerplexityLyrics** (`perplexity_lyrics_node.py`): Perplexity API.
12. **AceStepLyricsFormatter** (`lyrics_formatter_node.py`): Auto-formats lyrics with required tags.
13. **SaveText** (`save_text_node.py`): Save text output to file.

## Prompts & Post-Processing Nodes (nodes/*_node.py)

14. **AceStepPromptGen** (`prompt_gen_node.py`): 200+ music style presets.
15. **AceStepRandomPrompt** (`random_prompt_node.py`): Randomized genre/mood/instrument prompts.
16. **AceStepPostProcess** (`post_process_node.py`): De-esser and spectral smoothing.

## Sampling Nodes (nodes/*_node.py)

17. **AceStepKSampler** (`ksampler_node.py`): Audio-optimized with `shift` support.
18. **AceStepKSamplerAdvanced** (`ksampler_advanced_node.py`): Advanced controls with `shift`.
19. **AceStepInpaintSampler** (`inpaint_sampler_node.py`): Specialized sampler for audio inpainting/masking.

## Audio Analysis & Codec Nodes (nodes/*_node.py)

20. **AceStepAudioAnalyzer** (`audio_analyzer_node.py`): Extract BPM, key, and duration.
21. **AceStepAudioToCodec** (`audio_to_codec_node.py`): Waveform → FSQ codes for conditioning.
22. **AceStepCodecToLatent** (`codec_to_latent_node.py`): FSQ codes → latents.

## Text & Conditioning Nodes (nodes/*_node.py)

23. **AceStepMetadataBuilder** (`metadata_builder_node.py`): Format music metadata dictionary.
24. **AceStepCLIPTextEncode** (`clip_text_encode_node.py`): Encode text with metadata integration.
25. **AceStepConditioning** (`conditioning_node.py`): Combine multiple conditioning types.

## Advanced & UI Nodes (nodes/*_node.py)

26. **AceStepModeSelector** (`mode_selector_node.py`): 4-in-1 mode routing.
27. **AceStepAudioMask** (`audio_mask_node.py`): Time-based latent masking.
28. **AceStep5HzLMConfig** (`lm_config_node.py`): LM parameter configuration.
29. **AceStepCustomTimesteps** (`custom_timesteps_node.py`): Parse custom sigma schedules.
30. **AceStepLoRAStatus** (`lora_status_node.py`): Display LoRA loading info.
31. **AceStepLoRALoader** (`lora_loader_node.py`): Specialized LoRA loader for the ACE-Step 1.5 LoRA using additional json files.
32. **AceStepConditioningExplore** (`conditioning_explore_node.py`): Debug node to explore conditioning data structure with pretty JSON and tensor summaries.
33. **AceStepConditioningMixer** (`conditioning_mixer_node.py`): Selectively mix components (main tensor, pooled output, lyrics, audio codes) from two conditioning sources.
34. **AceStepConditioningSave** (`conditioning_save_node.py`): Save individual conditioning components to separate files on disk.
35. **AceStepConditioningLoad** (`conditioning_load_node.py`): Load and reconstruct conditioning from saved component files.
36. **AceStepConditioningMixerLoader** (`conditioning_mixer_loader_node.py`): Granularly mix saved components from different files.
    - **Optional Base**: `tune_tensor_file` can be set to "none", generating a zero/ones/random fallback.
    - **Fallback Modes**: `empty_mode` (zeros, ones, random) for missing tensors.
    - **Explicit Lyrics Fallback**: Prioritizes `nodes/includes/emptytensors/empty_lyrics.safetensors` for missing lyrics, with sequence length synchronization.
    - **Automation**: Supports "random" selection, seeds, and outputs a string summary.
37. **AceStepTuneTensorLoader** (`tune_tensor_loader_node.py`): Load a tune conditioning tensor (formerly "main") from disk with random/seed support.
38. **AceStepLyricsTensorLoader** (`lyrics_loader_node.py`): Load a lyrics conditioning tensor from disk with random/seed support.
39. **AceStepAudioCodesLoader** (`audio_codes_loader_node.py`): Load audio codes from disk with random/seed support.
40. **AceStepTensorMixer** (`tensor_mixer_node.py`): Advanced tensor processing node. Supports linear blending (with ratio), concatenation, and arithmetic operations (add, multiply, etc.). Includes sophisticated scaling/interpolation to handle sequence length mismatches.
41. **AceStepConditioningCombine** (`conditioning_combine_node.py`): Assemble individual tensors and codes into a full conditioning object.
    - **Optional Inputs**: All inputs are optional.
    - **Dimensions**: Intelligently infers batch size and device from any provided component.
    - **Length Sync**: Fallback tensors (zeros/ones/random) automatically synchronize their sequence length to match other provided components.
    - **Explicit Fallback**: Inherits the `empty_lyrics.safetensors` prioritization logic.
42. **AceStepConditioningSplitter** (`conditioning_split_node.py`): Inverse of the Combiner. Splits a `CONDITIONING` object into Tune Tensor, Pooled Output, Lyrics Tensor, and Audio Codes list for individual processing.
43. **AceStepTensorMaskGenerator** (`tensor_mask_nodes.py`): Generates a `[1, N, 1]` mask using various modes (`all`, `none`, `fraction`, `range`, `ramp`, `window`). Supports soft edges and reversals.
44. **AceStepTensorUnaryOp** (`tensor_unary_op_node.py`): Transforms a single input tensor `A` with an optional mask. Modes: `gate`, `scale_masked`, `noise_masked`, `fade_out`.
45. **AceStepGeniusLyricsSearch** (`genius_lyrics_node.py`): Fetches song lyrics from Genius.com using their official API. Requires a personal access token in `keys/genius_api_key.txt`.
46. **AceStepTensorMixer** (`tensor_mixer_node.py`): Consolidated Binary Toolbox for mixing two tensors `A` and `B`. Features 12+ modes (blend, inject, lerp, average, concat, etc.), universal masking, and silent sequence length synchronization.

---

## Shared Utility Modules (nodes/includes/)

- **analysis_utils.py**: FSQ quantization logic and dependency checks.
- **audio_utils.py**: FLAC metadata block generation.
- **lyrics_utils.py**: Prompt builders and markdown cleaning.
- **prompt_utils.py**: Style presets and musical descriptors.
- **sampling_utils.py**: Noise schedule shift formulas.
