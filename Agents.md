# Agents.md - ACE-Step ComfyUI Node Development

## Core Principles & Rules

### Rule #1: Never Reinvent the Wheel
**NEVER** create new code when existing ComfyUI infrastructure can be reused. Always:
1. **Check existing nodes first** (JK-AceStep-Nodes, acestep_tweaked, patched-text_encoder)
2. **Check ComfyUI core** (native ACE-Step support exists!)
3. **Ask if unsure** - Don't assume you need to build from scratch
4. **Extend, don't replace** - Wrap or extend existing functionality

### Rule #2: Follow ComfyUI's Integration Patterns
Study how ComfyUI natively integrates ACE-Step:
- **Text Encoders**: See `comfy/text_encoders/ace15.py` for the pattern
- **Model Architecture**: See `comfy/ldm/ace/` for DiT/VAE implementations  
- **Node Structure**: Follow existing node conventions from `acestep_tweaked`

### Rule #3: Keep It Modular
Build **single-purpose nodes** that can be composed:
- ✅ Good: `AceStepMetadataBuilder` outputs STRING
- ❌ Bad: `AceStepAllInOneGenerator` does everything

### Rule #4: Preserve Workflow Flexibility
Users should be able to:
- Mix ACE-Step nodes with standard ComfyUI nodes
- Use existing loaders (CheckpointLoader, VAELoader, CLIPLoader)
- Build custom pipelines beyond the 4 Gradio modes

---

## Project Focus

### Primary Goal
**Replicate 100% of Gradio app functionality in ComfyUI nodes**

NOT by porting the entire app, but by:
1. Identifying what features **already exist** in ComfyUI
2. Creating **minimal new nodes** to fill gaps
3. Providing **example workflows** showing how to combine nodes

### Success Criteria
A user can:
- ✅ Load ACE-Step models via standard ComfyUI loaders
- ✅ Generate music in all 4 modes (Simple/Custom/Cover/Repaint)
- ✅ Access all advanced parameters (LM CFG, temperature, BPM, etc.)
- ✅ Use LoRA adapters
- ✅ Generate lyrics via LLM APIs
- ✅ Apply post-processing (de-esser, spectral smoothing)
- ✅ Save with proper metadata

---

## Build Strategy

### Phase 1: Leverage Native Support (COMPLETE)
- ✅ Confirmed ComfyUI has built-in ACE-Step 1.5 support
- ✅ Confirmed checkpoints are already converted (Comfy-Org)
- ✅ Identified existing reusable nodes (~15 nodes)

### Phase 2: Gap Analysis (COMPLETE)
- ✅ Mapped every Gradio feature to ComfyUI equivalents
- ✅ Identified 8-15 missing nodes needed
- ✅ Prioritized into Core/UX/Optional tiers

### Phase 3: Node Development (COMPLETE)
82 nodes implemented across these categories:

**Conditioning** (19 nodes): `AceStepAudioCodesMixer`, `AceStepAudioCodesUnaryOp`, `AceStepConditioningCombine`, `AceStepConditioningMixer`, `AceStepConditioningSplitter`, `AceStepAudioCodesToSemanticHints`, `AceStepSemanticHintsToAudioCodes`, `AceStepConditioningZeroOut`, `AceStepAudioCodesUnderstand`, `AceStepConditioningExplore`, `AceStepAudioCodesLoader`, `AceStepConditioningLoad`, `AceStepLyricsTensorLoader`, `AceStepConditioningMixerLoader`, `AceStepTimbreTensorLoader`, `AceStepAudioMask`, `AceStepTensorMaskGenerator`, `AceStepTensorMixer`, `AceStepTensorUnaryOp`

**Sampler** (1 node): `ScromfyAceStepSampler`

**Audio** (6 nodes): `Audio Analyzer (No LLM)`, `ScromfyAceStepMusicAnalyzer`, `AceStepPostProcess`, `Scromfy Audio VAE Decode PLUSPLUS`, `Scromfy Save Audio`, `AceStepLoadAudio`

**Lyrics** (10 nodes): `AceStepLyricsFormatter`, `AceStepGeniusLyricsSearch`, `AceStepRandomLyrics`, `AceStepLyricsBPMCalculator`, `AceStepClaudeLyrics`, `AceStepGeminiLyrics`, `AceStepGroqLyrics`, `AceStepOpenAILyrics`, `AceStepPerplexityLyrics`, `AceStepGenericAILyrics`

**Visualizers** (6 nodes): `ScromfyFlexAudioVisualizerCircular`, `ScromfyFlexAudioVisualizerContour`, `ScromfyFlexAudioVisualizerLine`, `ScromfyFlexLyrics`, `ScromfyEmojiSpinnerVisualizer`, `Lyric Settings`

**Radio** (2 nodes): `AceStepWebAmpRadio`, `RadioPlayer`

**Lora** (2 nodes): `AceStepLoRALoader`, `Scromfy AceStep Lora Stack`

**Whisper** (3 nodes): `Faster Whisper Loader`, `Faster Whisper Transcribe`, `Save Subtitle/Lyrics`

**Prompt** (5 nodes): `ScromfyAceStepTextEncoderPlusPlus`, `AceStepMetadataBuilder`, `AceStepPromptGen`, `AceStepRandomPrompt`, `Prompt Freeform`

**Misc** (4 nodes): `AceStep5HzLMConfig`, `WikipediaRandomNode`, `ScromfyEmojiSpinner`, `ScromfyMaskPicker`

**Persistence** (2 nodes): `AceStepConditioningSave`, `AceStepTensorSave`

**Obsolete** (18 nodes): All prefixed with `obsolete_` in filename.

### Phase 4: Documentation & Examples
- Create 4 example workflows (one per mode)
- Write usage guides for each workflow
- Document parameter mappings (Gradio → ComfyUI)

---

## Technical Architecture

### How ACE-Step Integrates with ComfyUI

#### Text Encoding Flow
```
User Input (prompt, lyrics, metadata)
    ↓
ACE15Tokenizer.tokenize_with_weights()
    ├─ Creates lm_prompt (for 5Hz LM)
    ├─ Creates qwen3_06b tokens (for genre/style)
    └─ Creates lyrics tokens (for vocal guidance)
    ↓
ACE15TEModel.encode_token_weights()
    ├─ Encodes genre/style with Qwen3-0.6B
    ├─ Encodes lyrics with Qwen3-0.6B layer 0
    └─ Generates audio codes with 5Hz LM (Qwen3-2B or 4B)
    ↓
Returns: (base_embeddings, pooled, extras)
    extras = {
        "conditioning_lyrics": lyrics_embeds,
        "audio_codes": [generated_codes]
    }
```

**Key Insight**: The `ace15.py` tokenizer handles:
- BPM, duration, keyscale, timesignature in metadata
- Chain-of-Thought prompting for LM
- Constrained decoding (audio tokens only)
- CFG-guided generation

**Important kwargs for `tokenize_with_weights()`**:
- `lyrics`: String with lyric content
- `bpm`: Integer (default 120)
- `duration`: Float in seconds (default 120)
- `keyscale`: String like "C major" (default "C major")
- `timesignature`: Integer 2/3/4 (default 2)
- `language`: String like "en" (default "en")
- `seed`: Integer for LM generation (default 0)

#### Model Loading Flow
```
CheckpointLoader (standard ComfyUI)
    ↓
Loads: MODEL, CLIP, VAE
    ├─ MODEL: ACEStepTransformer2DModel (from comfy/ldm/ace/)
    ├─ CLIP: ACE15TEModel (from comfy/text_encoders/ace15.py)
    └─ VAE: AutoencoderOobleck (from comfy/ldm/ace/vae/)
```

**No custom loader needed!** Standard ComfyUI loaders work because:
- ComfyUI has native ACE-Step support
- Checkpoints are already converted to ComfyUI format
- Architecture definitions exist in core

---

## Node Development Guidelines

### Input/Output Types
Match ComfyUI conventions:
- `MODEL`, `VAE`, `CLIP` - Standard model types
- `CONDITIONING` - For conditioned generation
- `LATENT` - For latent audio
- `AUDIO` - For waveform audio
- `STRING` - For text (prompts, lyrics, metadata)
- `INT`, `FLOAT`, `BOOLEAN` - For parameters

### Metadata Formatting
Use YAML format (as ace15.py expects):
```python
meta = 'bpm: {}\\nduration: {}\\nkeyscale: {}\\ntimesignature: {}'.format(
    bpm, duration, keyscale, timesignature
)
```

### Audio Code Format
Audio codes are **integers offset from audio_start_id (151669)**:
```python
# Generated codes are relative (0-based from audio range)
# ace15.py handles the offset internally
audio_codes = [1234, 5678, 910]  # These map to 152903, 157347, 152579
```

### LM Generation Parameters
Exposed in `sample_manual_loop_no_classes()`:
- `cfg_scale`: Float (default 2.0)
- `temperature`: Float (default 0.85)
- `top_p`: Float (default 0.9)
- `top_k`: Int (default None)
- `seed`: Int (default 1)
- `min_tokens`: Int (duration * 5 for 5Hz)
- `max_new_tokens`: Int (same as min_tokens for exact length)

---

## What We're NOT Building

### ❌ Custom Model Loaders
**Why**: ComfyUI's standard loaders work with converted checkpoints

### ❌ Custom DiT Implementation
**Why**: `comfy/ldm/ace/ace_step15.py` has full implementation

### ❌ Custom VAE
**Why**: `comfy/ldm/ace/vae/` has AutoencoderOobleck

### ❌ Custom Text Encoder
**Why**: `comfy/text_encoders/ace15.py` is complete

### ❌ Custom Samplers (mostly)
**Why**: Existing KSampler works, only need inpainting variant

---

## Questions to Ask Before Building

1. **Does a similar node exist in JK-AceStep-Nodes?**
   - If yes → Reuse or extend it

2. **Does ComfyUI core provide this?**
   - If yes → Use the core functionality

3. **Can this be done with existing nodes connected together?**
   - If yes → Create a workflow example instead

4. **Is this feature used by >50% of users?**
   - If no → Consider making it optional/extensible

5. **Can I pass this as kwargs to ace15.tokenize_with_weights()?**
   - If yes → Create a simple wrapper node

---

## Current Status

### What We Have
- ✅ 54 custom nodes implemented (100% complete)
- ✅ Native ComfyUI ACE-Step support confirmed
- ✅ Converted checkpoints available
- ✅ Dynamic node loading via `__init__.py` scanner
- ✅ 6 shared utility modules in `nodes/includes/`
- ✅ AI lyrics nodes load API keys from `keys/*.txt` files
- ✅ All obsolete nodes prefixed with `obsolete_` in filename

### Next Steps
1. [ ] Create example workflows for all 4 major generation modes
2. [ ] Performance optimization for long audio generations
3. [ ] Final verification with a large-scale music production workflow

---

## Code Style & Standards

### Follow Existing Patterns
Study these reference implementations:
- `acestep_tweaked/nodes_audio.py` - Audio I/O nodes
- `JK-AceStep-Nodes/ace_step_ksampler.py` - Sampling nodes
- `comfy/text_encoders/ace15.py` - Text encoding

### Node Template
```python
class AceStepNodeName:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "param": ("TYPE", {"default": value}),
            },
            "optional": {
                "optional_param": ("TYPE",),
            }
        }
    
    RETURN_TYPES = ("TYPE1", "TYPE2")
    RETURN_NAMES = ("output1", "output2")
    FUNCTION = "execute"
    CATEGORY = "ACE-Step"
    
    def execute(self, param, optional_param=None):
        # Implementation
        return (output1, output2)
```

### Documentation
Every node should have:
- Clear docstring explaining purpose
- Input parameter descriptions
- Output descriptions
- Example usage in comments

---

## Project File Structure

```text
scromfyUI-AceStep/
├── __init__.py           # Dynamic node scanner
├── nodes/
│   ├── includes/         # Shared utility modules
│   │   ├── analysis_utils.py
│   │   ├── audio_utils.py
│   │   ├── fsq_utils.py
│   │   ├── lyrics_utils.py
│   │   ├── prompt_utils.py
│   │   └── sampling_utils.py
│   ├── *_node.py         # 82 individual node files
├── keys/                 # API key files (git-ignored)
├── workflows/            # Example workflows (.json)
├── docs/                 # Documentation
│   ├── NODE_SPECS.md     # Technical reference (authoritative node list)
│   └── PROGRESS.md       # Implementation tracker
└── Agents.md             # This file
```

---

## Version History

- **2026-02-04**: Initial strategy document created
  - Confirmed native ComfyUI support
  - Identified 8-15 nodes needed
  - Established "never reinvent" principle
