# ACE-Step Nodes - Complete Implementation Guide

## Copy-Paste Nodes (15 total)

### Audio I/O (6 nodes) - ✅ COMPLETE
**File**: `nodes/audio_io.py`  
**Source**: `acestep_tweaked/nodes_audio.py`

1. ✅ **LoadAudio** - Load mp3/flac/wav/ogg files
2. ✅ **SaveAudio** - Save audio with FLAC metadata support
3. ✅ **PreviewAudio** - Preview in ComfyUI UI
4. ✅ **EmptyLatentAudio** - Create empty 64-channel 44.1kHz latents
5. ✅ **VAEEncodeAudio** - Waveform → latents (auto-resamples to 44.1kHz)
6. ✅ **VAEDecodeAudio** - Latents → waveform (normalized)

---

###Lyrics Generation (6 nodes) - 🔄 IN PROGRESS
**File**: `nodes/lyrics_gen.py`  
**Sources**: 
- `JK-AceStep-Nodes/gemini_nodes.py`
- `JK-AceStep-Nodes/groq_nodes.py`
- `JK-AceStep-Nodes/lyrics_nodes.py`

7. ⏳ **AceStepGeminiLyrics** - Google Gemini API
8. ⏳ **AceStepGroqLyrics** - Groq API (fast)
9. ⏳ **AceStepOpenAILyrics** - OpenAI API
10. ⏳ **AceStepClaudeLyrics** - Anthropic Claude API
11. ⏳ **AceStepPerplexityLyrics** - Perplexity API
12. ⏳ **SaveText** - Save lyrics/prompts to .txt

---

### Prompts & Post-Processing (3 nodes) - ⏳ PENDING
**File**: `nodes/prompts.py` + `nodes/post_process.py`  
**Sources**:
- `JK-AceStep-Nodes/ace_step_prompt_gen.py`
- `JK-AceStep-Nodes/ace_step_post_process.py`
- `JK-AceStep-Nodes/ace_step_vocoder_adapter.py`

13. ⏳ **AceStepPromptGen** - 200+ music style presets
14. ⏳ **AceStepPostProcess** - De-esser + spectral smoothing
15. ⏳ **AceStepVocoderAdapter** - Vocoder interface (optional)

---

## Adapt Nodes (5 total)

### Sampling Nodes (2 nodes) - ⏳ PENDING
**File**: `nodes/sampling.py`  
**Source**: `JK-AceStep-Nodes/ace_step_ksampler.py`

16. ⏳ **AceStepKSampler** 
   - **Existing code**: Lines 267-418
   - **Changes needed**: 
     - Remove auto-steps calculation (user controls steps)
     - Keep audio-specific memory optimization
     - Keep progress bar
     - Add category "ACE-Step/sampling"

17. ⏳ **AceStepKSamplerAdvanced**
   - **Existing code**: Lines 421-565
   - **Changes needed**:
     - Add shift parameter input
     - Remove auto-steps
     - Keep start_at_step, end_at_step, return_with_leftover_noise
     - Add category "ACE-Step/sampling"

---

### Text & Prompts (1 node) - ⏳ PENDING
**File**: `nodes/prompts.py`  
**Source**: Gradio `handler.py:1372-1407`

18. ⏳ **AceStepRandomPrompt**
   - **Existing code**: Uses Gradio's instruction templates
   - **Changes needed**:
     - Extract genre/mood/instrument lists
     - Port random combination logic
     - Add seed control
     - Remove Gradio-specific DIT instruction handling
     - Output plain STRING

---

### Audio Analysis (2 nodes) - ⏳ PENDING
**File**: `nodes/audio_analysis.py`  
**Source**: Gradio `handler.py`

19. ⏳ **AceStepAudioAnalyzer**
   - **Existing code**: Lines 751-775, 1409-1462
   - **Changes needed**:
     - Port librosa BPM detection
     - Port key/scale detection
     - Remove Gradio UI update code
     - Return: bpm (INT), keyscale (STRING), duration (FLOAT)

20. ⏳ **AceStepAudioToCodec**
   - **Existing code**: Lines 1484-1540
   - **Changes needed**:
     - Port VAE encoding
     - Port FSQ quantization
     - Remove Gradio progress callbacks
     - Format codes as comma-separated STRING
     - Return audio codes range 0-46000

---

## New Nodes (11 total)

### Text Encoding (2 nodes) - ⏳ PENDING
**File**: `nodes/text_encode.py`

21. ⏳ **AceStepMetadataBuilder**
   - **Why new**: No existing node formats ace15 kwargs
   - **Inputs**: 
     - bpm (INT, default=0, 0=auto)
     - duration (FLOAT, default=-1, -1=auto)
     - keyscale (STRING, default="", ""=auto)
     - timesignature ([2,3,4], default=4)
     - language (LANGUAGES list, default="en")
     - instrumental (BOOLEAN, default=False)
   - **Output**: DICT (metadata kwargs)
   - **Logic**: Format as kwargs dict for CLIPTextEncode

22. ⏳ **AceStepCLIPTextEncode**
   - **Why new**: Standard CLIPTextEncode doesn't pass kwargs
   - **Inputs**:
     - clip (CLIP)
     - text (STRING) - main description
     - lyrics (STRING, optional) - lyric content
     - metadata (DICT, from MetadataBuilder)
   - **Output**: CONDITIONING
   - **Logic**: Call `clip.tokenize_with_weights(text, **metadata)` then `clip.encode_from_tokens()`

---

### Audio Processing (4 nodes) - ⏳ PENDING
**File**: `nodes/audio_analysis.py`

23. ⏳ **AceStepLyricsFormatter**
   - **Why new**: No existing formatter with ACE-Step tag rules
   - **Input**: lyrics (STRING, multiline)
   - **Output**: formatted lyrics (STRING)
   - **Logic**:
     - Detect missing [Intro]/[Outro]
     - Validate [Verse]/[Chorus]/[Bridge] placement
     - Ensure line length < 80 chars
     - Balance vocal/instrumental sections

24. ⏳ **AceStepCodecToLatent**
   - **Why new**: Opposite of AudioToCodec, no existing implementation
   - **Inputs**:
     - audio_codes (STRING, comma-separated)
     - vae (VAE)
   - **Output**: LATENT
   - **Logic**:
     - Parse codes → tokens
     - FSQ decode
     - Return latent for conditioning

25. ⏳ **AceStepAudioMask**
   - **Why new**: Time-based audio masking doesn't exist
   - **Inputs**:
     - audio (AUDIO)
     - start_seconds (FLOAT, default=0.0)
     - end_seconds (FLOAT, default=-1, -1=end)
   - **Output**: MASK
   - **Logic**:
     - Create binary mask matching latent dimensions
     - 1 = regenerate, 0 = keep original
     - Convert time → latent frame indices

26. ⏳ **AceStepInpaintSampler**
   - **Why new**: Standard KSampler doesn't support audio inpainting
   - **Inputs**:
     - model (MODEL)
     - source_latent (LATENT)
     - mask (MASK)
     - positive (CONDITIONING)
     - negative (CONDITIONING)
     - seed, steps, cfg, sampler_name, scheduler, denoise
   - **Output**: LATENT
   - **Logic**:
     - Standard sampling loop
     - Apply mask during denoising
     - Preserve unmasked regions from source

---

### Advanced Features (5 nodes) - ⏳ PENDING
**File**: `nodes/advanced.py`

27. ⏳ **AceStepModeSelector**
   - **Why new**: 4-in-1 UI convenience, no equivalent
   - **Inputs**:
     - mode (["Simple", "Custom", "Cover", "Repaint"])
     - model, vae, clip
     - description (STRING, for Simple)
     - prompt (STRING, for Custom)
     - lyrics (STRING, for Custom/Simple)
     - reference_audio (AUDIO, for Custom/Cover)
     - source_audio (AUDIO, for Cover/Repaint)
     - repaint_start (FLOAT, for Repaint)
     - repaint_end (FLOAT, for Repaint)
   - **Outputs**: CONDITIONING, LATENT, MASK (optional)
   - **Logic**: Route based on mode, call appropriate sub-nodes

28. ⏳ **AceStep5HzLMConfig**
   - **Why new**: ace15.py hardcodes these, no UI exposure
   - **Inputs**:
     - temperature (FLOAT, 0.0-2.0, default=0.85)
     - cfg_scale (FLOAT, 1.0-5.0, default=2.0)
     - top_k (INT, 0-100, default=0, 0=disabled)
     - top_p (FLOAT, 0.0-1.0, default=0.9)
     - negative_prompt (STRING, default="NO USER INPUT")
   - **Output**: LM_CONFIG (DICT)
   - **Logic**: Format as dict to merge into metadata
   - **Note**: Requires patching ace15.py to read these from kwargs

29. ⏳ **AceStepCustomTimesteps**
   - **Why new**: Custom sigma schedules not exposed
   - **Input**: timesteps (STRING, comma-separated, default="0.97,0.76,0.615,0.5,0.395,0.28,0.18,0.085,0")
   - **Output**: SIGMAS
   - **Logic**: Parse string → tensor for KSampler

30. ⏳ **AceStepLoRAStatus**
   - **Why new**: Display convenience, no existing status node
   - **Input**: lora_path (STRING, optional)
   - **Output**: STRING (status message)
   - **Logic**: Check if LoRA loaded, return info

31. ⏳ **AceStepConditioning**
   - **Why new**: Combine text+lyrics+timbre conditioning
   - **Inputs**:
     - text_cond (CONDITIONING)
     - lyrics (STRING, optional)
     - timbre_audio (AUDIO, optional)
   - **Output**: CONDITIONING (combined)
   - **Logic**: Merge conditioning dicts with extras

---

## Core ComfyUI Nodes (Native - Already Exist)

These don't need implementation, list for workflow reference:

- **CheckpointLoaderSimple** - Loads ACE-Step .safetensors
- **VAELoader** - Separate VAE loading
- **CLIPLoader** - Separate text encoder loading
- **KSampler** - Standard diffusion sampling
- **KSamplerAdvanced** - Advanced sampling params
- **ConditioningCombine** - Combine multiple conditionings
- **ConditioningConcat** - Concatenate conditioning
- **EmptyLatentImage** - (not used for audio, but available)

---

## Implementation Priority

**Phase 1 - Core Functionality** (Nodes 1-15 + 21-22):
- ✅ Audio I/O (6 nodes complete)
- 🔄 Lyrics generation (6 nodes in progress)
- ⏳ Prompt gen + post-process (3 nodes pending)
- ⏳ Metadata + text encoding (2 new nodes pending)

**Phase 2 - Generation Modes** (Nodes 16-20, 23-26):
- ⏳ Sampling nodes (2 adapt)
- ⏳ Audio analysis + codec (2 adapt)
- ⏳ Lyrics formatter (1 new)
- ⏳ Inpainting (2 new)

**Phase 3 - Advanced** (Nodes 27-31):
- ⏳ Mode selector (1 new)  
- ⏳ LM config (1 new)
- ⏳ Custom timesteps (1 new)
- ⏳ LoRA status (1 new)
- ⏳ Advanced conditioning (1 new)

---

## Current Status
- **Complete**: 6/31 nodes (19%)
- **In Progress**: 6/31 nodes (19%)
- **Pending**: 19/31 nodes (62%)
