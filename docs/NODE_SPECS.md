# ACE-Step Nodes - Technical Specifications

All nodes have been implemented and refactored into individual files within the `nodes/` directory.

## Implementation Overview

- **Nodes**: 30 custom nodes implemented.
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

---

## Shared Utility Modules (nodes/includes/)

- **analysis_utils.py**: FSQ quantization logic and dependency checks.
- **audio_utils.py**: FLAC metadata block generation.
- **lyrics_utils.py**: Prompt builders and markdown cleaning.
- **prompt_utils.py**: Style presets and musical descriptors.
- **sampling_utils.py**: Noise schedule shift formulas.
