# Prompt Components System

This directory contains the data used by the **Prompt Generator** node. You can add, remove, or modify files here to customize your music prompt generation.

## 📂 File Types

- **`.txt`**: A flat list of items, one per line. Every line becomes a choice in the dropdown.
- **`.json`**: A dictionary of key-value pairs. The **key** appears in the dropdown, and the **value** is what gets inserted into the final prompt.
  - *Example (`STYLE_PRESETS.json`)*: `"Synthwave": "Vintage analog synths, neon pads..."`

## 🪄 Wildcards

You can use `__FILENAME__` (double underscores) inside any text file or JSON value to pull a random item from another file.

- **Recursive**: If a picked item contains its own wildcard, it will expand those too (up to 5 levels deep).
- **Plural Fallback**: If you use `__ADJECTIVE__` but the file is named `ADJECTIVES.txt`, it will automatically find the correct list.
- **Example**: A mood in `MOODS.txt` could be `A __CULTURE__ inspired __GENRE__ track`.

---

## 🛠️ Control Files

Three special files allow you to manage how components are loaded and displayed:

### 1. `TOTALIGNORE.list` (TXT)
Files listed here are completely ignored by the system. Use this to disable repo-default lists without deleting them.
- **Example**: `ADJECTIVES.txt`

### 2. `LOADBUTNOTSHOW.list` (TXT)
Files listed here are loaded (available for wildcards) but **hidden** from the Prompt Generator's UI dropdowns.
- **Example**: If you want `MOODS` to use `__ADJECTIVE__` wildcards, but you don't want a separate "Adjectives" dropdown in your ComfyUI node, add `ADJECTIVES.txt` here.

### 3. `REPLACE.list` (JSON)
Allows you to substitute an existing component name with a custom file.
- **Format**: `{"ORIGINAL_NAME": "YOUR_NEW_FILENAME"}`
- **Example**: `{"ADJECTIVES": "MY_PROMPT_LIST"}`
  - This ignores the default `ADJECTIVES.txt`.
  - It loads `MY_PROMPT_LIST.txt` instead.
  - It assigns it the name `ADJECTIVES` (so wildcards and UI labels remain unchanged).

### 4. `WEIGHTS.json` (JSON)
Controls the **order** in which components appear in the UI and their position in the final combined prompt.
- **Format**: `{"COMPONENT_NAME": numerical_weight}`
- **Rules**:
  - Higher weights appear first.
  - Items not listed default to weight `0`.
  - Tied weights are sorted alphabetically.
- **Example**: `{"STYLE_PRESETS": 100, "INSTRUMENTS": 10}` ensures Style is always at the top.

---

## 🔄 Refreshing
To see changes (new files or list updates), simply **Refresh** your ComfyUI browser page. The `Prompt Generator` node will dynamically update its inputs and outputs based on the current files in this directory.
