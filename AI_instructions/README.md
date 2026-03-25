# AI Instructions

This directory contains the system prompts used by the AI lyric generation nodes.

## System Prompt Hierarchy

The AI nodes use a dual-file system to ensure safety during updates while allowing full user customization.

- **systemprompt.default.txt**: The master default prompt provided with the extension. **Do not modify this file directly**, as it may be overwritten during updates.
- **systemprompt.txt**: Create this file to override the default prompt with your own custom instructions. If this file exists, it will be used instead of the default.

## Features

- **Role Splitting**: All AI nodes automatically split the prompt into "system" and "user" roles for cleaner results and better adherence to instructions.
- **Reasoning Support**: Automated `<think>` block removal is applied to models like DeepSeek R1 to ensure only the final lyrics are returned to ComfyUI.
- **Dynamic Context**: The nodes automatically inject the Title, Artist, and Genre into the prompt context for the AI.

## Customization Tips

When creating your own `systemprompt.txt`, consider including:
1.  **Section Tags**: Define which square-bracket tags the AI should use (e.g., `[Intro]`, `[Chorus]`).
2.  **Formatting Rules**: Instruct the AI to avoid markdown, code fences, or bolding.
3.  **Creative Constraints**: Tell the AI to focus on specific themes or rhyming schemes.
