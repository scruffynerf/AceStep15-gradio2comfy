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
- **Case Insensitive**: The lookup will match `__GENRE__` to `genres.txt`.
- **Friendly Display**: The node dropdowns will show `(wildcard)` in lowercase for items like `random` or `none` when appropriate.

---

## 🛠️ Control Files (Safe Overrides)

The following files allow you to manage how components are loaded and displayed. Each has a **`.default`** version provided by the extension. **Do not modify the `.default` files**, as they may be overwritten during updates. Instead, create a version without `.default` to apply your own settings.

### 1. `TOTALIGNORE.list` (TXT)
Files listed here are completely ignored by the system. Use this to disable repo-default lists without deleting them.

### 2. Visibility Control (Directory-Based)
The visibility of components in the **Prompt Generator** dropdown is determined by their file location:

- **Visible Dropdowns**: Files located directly in the root of `prompt_components/` (e.g. `GENRE.txt`). These are also available as wildcards (`__GENRE__`).
- **Wildcard Only**: Files located in **subdirectories** (e.g. `wildcards/PLACES.txt`). These will **not** appear in the UI dropdowns but are fully functional as wildcards (`__PLACES__`).

### 3. `REPLACE.list` (JSON)
Allows you to substitute an existing component name with a custom file.
- **Example**: `{"ADJECTIVES": "MY_PROMPT_LIST"}`

### 4. `WEIGHTS.json` (JSON)
Controls the **order** in which components appear in the UI and their position in the final combined prompt.
- **Higher weights appear first**.
- Items not listed default to weight `0`.

---

## 🔄 Refreshing
To see changes (new files or list updates), simply **Refresh** your ComfyUI browser page. The `Prompt Generator` node will dynamically update its inputs and outputs based on the current files in this directory.
