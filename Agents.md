# Agents.md - ACE-Step ComfyUI Node Development

## Core Principles & Rules

### Rule #1: Never Reinvent the Wheel

**NEVER** create new code when existing ComfyUI infrastructure can be reused. Always:

1. **Check existing nodes first**
2. **Check ComfyUI core** files in referencecode
3. **Ask if unsure** - Don't assume you need to build from scratch
4. **Extend, don't replace** - Wrap or extend existing functionality

### Rule #2: Follow ComfyUI's Integration Patterns

Study how ComfyUI natively integrates ACE-Step:

- **Text Encoders**: See `comfy/text_encoders/ace15.py` for the pattern
- **Model Architecture**: See `comfy/ldm/ace/` for DiT/VAE implementations  
- **Node Structure**: Follow existing node conventions

### Rule #3: Keep It Modular

Build **single-purpose nodes** that can be composed:

- ✅ Good: `AceStepMetadataBuilder` outputs STRING
- ❌ Bad: `AceStepAllInOneGenerator` does everything

### Rule #4: Preserve Workflow Flexibility

Users should be able to:

- Mix ACE-Step nodes with standard ComfyUI nodes
- Use existing loaders (CheckpointLoader, VAELoader, CLIPLoader)

---

## Agent Environment & Operations Rules

### 🚫 Do NOT Run ComfyUI Directly

- **NEVER** attempt to start or run ComfyUI directly from the terminal. This applies to both inference and server startup.

### 🐍 ALWAYS Use the Venv Environment for Python

- When running Python scripts for testing or utility purposes, **YOU MUST** use the virtual environment, if one exists (e.g. `./venv/bin/python`). Do not use the system Python unless absolutely required.

### 🌐 Use Online Resources Instead of Local Files

- Do **NOT** search the host laptop's local system for resources like fonts, images, or third-party assets (e.g., trying to read system font directories like `C:\Windows\Fonts` or `/System/Library/Fonts`).
- Instead, go online (using web search tools) to find, reference, and download the appropriate files/resources when needed.

---

## Project Structure & Documentation

Detailed background information has been separated into subject-specific documentation. For deep-dives into these areas, consult the specific documents in the `/docs` folder:

- [Technical Architecture](docs/ARCHITECTURE.md): Details on text encoding flow, model loading, and integration patterns.
- [Node Development Guidelines](docs/NODE_GUIDELINES.md): Input/output types, metadata formatting, code style, and node templates.
- [Repository Index & Strategy](docs/REPO_INDEX.md): Canonical file index, project focus, and node category map.
- [Node Documentation](docs/nodes/): Deep-dive reference guides for the specific inputs, outputs, and behaviors of every custom node in each category.

**Directory Structure Overview**:

```text
scromfyUI-AceStep/
├── AI_instructions/       # Additional AI prompts and context
├── Agents.md             # This file (Agent capabilities and rules)
├── __init__.py           # Dynamic node scanner
├── color_schemas/        # Color definitions for visualizers
├── docs/                 # Documentation
│   ├── ARCHITECTURE.md   # Technical architecture
│   ├── BUILD_STRATEGY.md # Project strategy and status
│   ├── NODE_GUIDELINES.md# Development guidelines and style
│   ├── NODE_SPECS.md     # Technical reference (authoritative node list)
│   ├── PROGRESS.md       # Implementation tracker
│   └── nodes/            # Detailed I/O documentation for each node category
├── fonts/                # Local font files for visualizers
├── keys/                 # API key files (git-ignored)
├── lyrics/               # Output or parsed lyrics
├── masks/                # Image masks
├── nodes/                # ComfyUI node definitions
│   ├── includes/         # Shared utility modules
│   └── *_node.py         # 82+ individual node files
├── prompt_components/    # Text/JSON assets for prompting (genres, lyrics templates, etc.)
├── referencecode/        # Original reference scripts from ACE-Step, ComfyUI, etc.
├── scripts/              # UI or backend utility scripts
├── web/                  # ComfyUI frontend web extensions
├── webamp_skins/         # Skins for WebAmp player
├── webamp_visualizers/   # Additional WebAmp visualizer modules
└── workflows/            # Example ComfyUI workflows (.json)
```
