"""
ACE-Step Custom Nodes for ComfyUI
Complete self-contained node package for ACE-Step 1.5 music generation
"""

from .nodes import audio_io, lyrics_gen, prompts, util, sampling, audio_analysis, text_encode, advanced

# Collect all node mappings
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Register audio I/O nodes
NODE_CLASS_MAPPINGS.update(audio_io.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(audio_io.NODE_DISPLAY_NAME_MAPPINGS)

# Register lyrics generation nodes
NODE_CLASS_MAPPINGS.update(lyrics_gen.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(lyrics_gen.NODE_DISPLAY_NAME_MAPPINGS)

# Register prompt/post-processing nodes
NODE_CLASS_MAPPINGS.update(prompts.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(prompts.NODE_DISPLAY_NAME_MAPPINGS)

# Register utility nodes
NODE_CLASS_MAPPINGS.update(util.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(util.NODE_DISPLAY_NAME_MAPPINGS)

# Register sampling nodes (ADAPT)
NODE_CLASS_MAPPINGS.update(sampling.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(sampling.NODE_DISPLAY_NAME_MAPPINGS)

# Register audio analysis nodes (ADAPT)
NODE_CLASS_MAPPINGS.update(audio_analysis.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(audio_analysis.NODE_DISPLAY_NAME_MAPPINGS)

# Register text encoding nodes (NEW)
NODE_CLASS_MAPPINGS.update(text_encode.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(text_encode.NODE_DISPLAY_NAME_MAPPINGS)

# Register advanced nodes (NEW)
NODE_CLASS_MAPPINGS.update(advanced.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(advanced.NODE_DISPLAY_NAME_MAPPINGS)

# Register prompt/post-processing nodes
NODE_CLASS_MAPPINGS.update(prompts.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(prompts.NODE_DISPLAY_NAME_MAPPINGS)

# Register utility nodes
NODE_CLASS_MAPPINGS.update(util.NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(util.NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print(f"🎵 ACE-Step Nodes loaded: {len(NODE_CLASS_MAPPINGS)} nodes registered")
