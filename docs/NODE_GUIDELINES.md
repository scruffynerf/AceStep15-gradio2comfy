# Node Development Guidelines

## Input/Output Types

Match ComfyUI conventions:

- `MODEL`, `VAE`, `CLIP` - Standard model types
- `CONDITIONING` - For conditioned generation
- `LATENT` - For latent audio
- `AUDIO` - For waveform audio
- `STRING` - For text (prompts, lyrics, metadata)
- `INT`, `FLOAT`, `BOOLEAN` - For parameters

## Metadata Formatting

Use YAML format (as ace15.py expects):

```python
meta = 'bpm: {}\\nduration: {}\\nkeyscale: {}\\ntimesignature: {}'.format(
    bpm, duration, keyscale, timesignature
)
```

## Audio Code Format

Audio codes are **integers offset from audio_start_id (151669)**:

```python
# Generated codes are relative (0-based from audio range)
# ace15.py handles the offset internally
audio_codes = [1234, 5678, 910]  # These map to 152903, 157347, 152579
```

## LM Generation Parameters

Exposed in `sample_manual_loop_no_classes()`:

- `cfg_scale`: Float (default 2.0)
- `temperature`: Float (default 0.85)
- `top_p`: Float (default 0.9)
- `top_k`: Int (default None)
- `seed`: Int (default 1)
- `min_tokens`: Int (duration * 5 for 5Hz)
- `max_new_tokens`: Int (same as min_tokens for exact length)

---

## Questions to Ask Before Building

1. **Does a similar node exists**
   - If yes → Reuse or extend it

2. **Does ComfyUI core provide this?**
   - If yes → Use the core functionality

3. **Can this be done with existing nodes connected together?**
   - If yes → Create a workflow example instead

4. **Is this feature used by >50% of users?**
   - If no → Consider making it optional/extensible

5. **Can I pass this as kwargs?**
   - If yes → Create a simple wrapper node

---

## Code Style & Standards

### Follow Existing Patterns

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
    CATEGORY = "Scromfy/Ace-Step/"
    
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
