# scromfyUI-AceStep

Advanced ACE-Step 1.5 music generation nodes for ComfyUI.

This repository allows you to use the powerful ACE-Step 1.5 music generation models directly within ComfyUI, providing full control over lyrics, styles, reference audio, and advanced sampling parameters.

## Features

- **Specialized Nodes**: Comprehensive support for the ACE-Step 1.5 workflow.
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

## Documentation & Specs

- [Detailed Node Guides](docs/nodes/) — In-depth reference for each node category:
  - [Audio Nodes](docs/nodes/Audio.md)
  - [Conditioning Nodes](docs/nodes/Conditioning.md)
  - [Lyrics Nodes](docs/nodes/Lyrics.md)
  - [Prompt Nodes](docs/nodes/Prompt.md)
  - [Sampler Nodes](docs/nodes/Sampler.md)
  - [Radio System](docs/nodes/Radio.md)
  - [LoRA Nodes](docs/nodes/Lora.md)
  - [Whisper Nodes](docs/nodes/Whisper.md)
  - [Misc Nodes](docs/nodes/Misc.md)
- [Repository Index & Strategy](docs/REPO_INDEX.md) — Canonical file index and project focus.
- [Node Development Guidelines](docs/NODE_GUIDELINES.md) — Standards and templates for new nodes.
- [Technical Architecture](docs/ARCHITECTURE.md) — Under-the-hood details on conditioning and VAE flows.

## Credits

- [ACE-Step-v1.5 lora loader](https://github.com/Neyroslav/ComfyUI-ACE-Step-1.5_LoRA_Loader) — adapted his code with his blessing
- [Jean Kassio](https://github.com/jeankassio) — adapted some code from his repos
- <https://github.com/jeankassio/JK-AceStep-Nodes>
- <https://github.com/jeankassio/ComfyUI-AceStep_SFT>
- [Matchering Library](https://github.com/sergree/matchering) — Created by Sergree (Sergey Grishakov), GPLv3.
- [ComfyUI-Matchering](https://github.com/MuziekMagie/ComfyUI-Matchering) — Original ComfyUI node adaptation by MuziekMagie.
