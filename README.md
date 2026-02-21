# AceStep15-gradio2comfy

Advanced ACE-Step 1.5 music generation nodes for ComfyUI. 

This repository allows you to use the powerful ACE-Step 1.5 music generation models directly within ComfyUI, providing full control over lyrics, styles, reference audio, and advanced sampling parameters.

## Features
- **30+ Specialized Nodes**: Comprehensive support for the ACE-Step 1.5 workflow.
- **Audio-Optimized Sampling**: KSamplers with memory optimization and precision "shift" parameter control.
- **Multi-API Lyrics Generation**: Integrated nodes for Gemini, Groq, OpenAI, Claude, and Perplexity.
- **Advanced Conditioning**: Control over BPM, Key, Lyrics Strength, and Timbre references.
- **Native Masking & Inpainting**: specialized nodes for selective audio regeneration.

## Installation
1. Clone this repository into your `ComfyUI/custom_nodes` folder:
   ```bash
   git clone https://github.com/scruffynerf/AceStep15-gradio2comfy
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Look for nodes under the `Scromfy/Ace-Step` category in the ComfyUI node menu. The repository uses a dynamic scanning system that automatically loads all nodes from the `nodes/` directory.

## Progress & Specs
- [PROGRESS.md](docs/PROGRESS.md) - Current implementation status.
- [NODE_SPECS.md](docs/NODE_SPECS.md) - Detailed node documentation.