# scromfyUI-AceStep

Advanced ACE-Step 1.5 music generation nodes for ComfyUI. 

This repository allows you to use the powerful ACE-Step 1.5 music generation models directly within ComfyUI, providing full control over lyrics, styles, reference audio, and advanced sampling parameters.

## Features
- **82 Specialized Nodes**: Comprehensive support for the ACE-Step 1.5 workflow (64 active + 18 obsolete/deprecated).
- **Full Text Encoder**: Human-readable dropdowns for language, key signature, and time signature with LLM audio code generation toggle.
- **Multi-Category Prompt Generator**: 8 independent category dropdowns (style, mood, adjective, culture, genre, vocal, performer, instrument) with random/random2 options.
- **Multi-API Lyrics Generation**: Integrated nodes for Genius search, random Genius lyrics, plus AI generation via Gemini, Groq, OpenAI, Claude, and Perplexity.
- **Advanced Conditioning**: Mixers, splitters, and combiners for fine-grained control over timbre, lyrics, and audio code conditioning.
- **Native Masking & Inpainting**: Specialized nodes for selective audio regeneration with time-based and tensor-based masks.
- **Radio Player**: In-UI audio player widget that scans output folders and plays tracks with auto-polling.
- **Deep Debug Tools**: Conditioning explorer with recursive introspection, circular-reference protection, and lovely-tensors summaries.

## Installation
1. Clone this repository into your `ComfyUI/custom_nodes` folder:
   ```bash
   git clone https://github.com/scruffynerf/scromfyUI-AceStep
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## API Keys

Several lyrics nodes require API keys. See [keys/README.md](keys/README.md) for setup instructions and links to obtain keys for Genius, OpenAI, Claude, Gemini, Groq, and Perplexity.

## Usage
Look for nodes under the `Scromfy/Ace-Step` category in the ComfyUI node menu. The repository uses a dynamic scanning system that automatically loads all nodes from the `nodes/` directory.

## Progress & Specs
- [PROGRESS.md](docs/PROGRESS.md) — Current implementation status.
- [NODE_SPECS.md](docs/NODE_SPECS.md) — Detailed node documentation.

## Credits
- [ACE-Step-v1.5 lora loader](https://github.com/Neyroslav/ComfyUI-ACE-Step-1.5_LoRA_Loader) — adapting his code for this repo, with his blessing
- [JK-AceStep-Nodes](https://github.com/jeankassio/JK-AceStep-Nodes) — adapted some code from this repo