# Flex Audio Visualizers Reference

The Scromfy Flex Visualizer suite follows a strict **Global vs. Local** parameter split to eliminate UI redundancy and provide clear control over the "Soul" vs. "Shape" of the visualization.

## Architecture Overview

- **Global Settings (Master Node)**: Defines audio analysis, color logic, and global motion.
- **Local Settings (Individual Nodes)**: Defines the physical shape (radius, height, curvature) and placement (position, rotation).
- **Reference Mask**: If a `source_mask` is provided to any Flex node, it defines the geometric area. By enabling **Use Mask as Visibility Filter** in the Settings node, it also acts as a strict clipping filter—visualizer elements will only appear where the mask is non-black. This allows you to "confine" a visualizer to a specific region of the screen while keeping the background clean.

---

## 1. Global Settings

*Located in the `Flex Visualizer Settings (Scromfy)` node.*

### System

| Parameter | Description |
| :--- | :--- |
| `randomize` | Automatically vary parameters for each batch. |
| `seed` | Deterministic seed for randomization. |
| `loop_background` | Toggle between looping the background (True) or clamping to the last frame (False). |
| `use_mask_as_visibility_filter` | **Blocking**: If True, visualizer elements only appear where the `source_mask` is white. (Default: False). |

### Audio Processing

| Parameter | Description |
| :--- | :--- |
| `visualization_feature` | Switch between `frequency` (FFT) and `waveform` analysis. |
| `num_points` | The total number of points/bars to generate (affects sampling resolution). |
| `smoothing` | Temporal smoothing of the audio data. |
| `fft_size` | Window size for FFT resolution. |
| `min_frequency` | Lower bound of frequency analysis (Bass). |
| `max_frequency` | Upper bound of frequency analysis (Treble). |

### Color & Style

| Parameter | Description |
| :--- | :--- |
| `color_mode` | Selection logic for colors (see **Color Modes** section below). |
| `color_schema` | Preset palette from `color_schemas/`. **Used only in 'Schema' mode.** |
| `custom_color` | Static hex color. **Used only in 'Custom' mode.** |
| `color_shift` | Hue offset (0.0 to 1.0) applied as a cycle. |
| `saturation` | Color intensity (0.0 to 1.0). |
| `brightness` | Overall brilliance (0.0 to 1.0). |
| `line_width` | Drawing thickness in **pixels**. |

### Motion & Direction

| Parameter | Description |
| :--- | :--- |
| `visualization_method` | `bar` (discrete) vs `line` (smooth). |
| `direction` | Vector orientation (Outward, Inward, Both, Centroid, Starburst). |
| `sequence_direction` | Data flow mapping (Left, Right, Centered, Both Ends). |
| `direction_skew` | Global tilt of all drawing vectors in **degrees**. |
| `centroid_offset_x/y` | Shift the focal point in **% of screen size** (-1.0 to 1.0). |

---

## Color Modes

The `color_mode` setting defines how colors are assigned to individual points or bars.

| Mode | Logic |
| :--- | :--- |
| **White** | All elements are pure white. |
| **Spectrum** | Colors are derived from the audio frequency (Low=Red, High=Violet). |
| **Custom** | Every element uses the `custom_color` hex value. |
| **Schema** | Uses the selected `.json` palette from `color_schemas/`. **User-Addable.** |
| **Amplitude** | Color is based on the volume of the specific point (mapping 0-1 to the Hue circle). |
| **Radial** | Color changes based on distance from the center. |
| **Angular** | Color maps to the angle (0-360) around the center. |
| **Path** | Color transitions linearly from the first point to the last point. |
| **Screen** | Color is based on the X/Y coordinate on the screen. |

> [!TIP]
> **Custom Schemas**: You can add your own `.json` palettes to the `/color_schemas` directory. They will appear automatically in the `color_schema` dropdown. See the [Schema Guide](file:///Users/scohn/code/AceStep15-gradio2comfy/color_schemas/README.md) for details.

---

## 2. Node-Specific Settings

*Exclusive to individual visualizer nodes.*

### Circular Node (Shape/Placement)

| Parameter | Description |
| :--- | :--- |
| `radius` | Maximum expansion radius in **pixels**. |
| `base_radius` | The resting radius in **pixels**. |
| `amplitude_scale` | Sensitivity multiplier for audio movement. |
| `bar_length_mode` | `absolute` (pixels) vs `relative` (% of base_radius). |
| `position_x / y` | Center coordinate (0.5 = center of screen). |
| `rotation` | Global rotation of the circle in **degrees**. |

### Line Node (Shape/Placement)

| Parameter | Description |
| :--- | :--- |
| `max_height / min` | Height bounds for bars in **pixels** (or **%** if relative). |
| `bar_length_mode` | `absolute` vs `relative` (% of screen height). |
| `length` | Physical length of the base line in **pixels** (0 = full width). |
| `separation` | Gap between bars in **pixels**. |
| `curvature` | Roundness of bar ends (visual only). |
| `curve_smoothing` | Smoothness factor for `line` method (0.0 to 1.0). |
| `position_x / y` | Midpoint coordinate of the line (0.0 to 1.0). |
| `rotation` | Angle of the line in **degrees**. |

### Contour Node (Shape/Source)

| Parameter | Description |
| :--- | :--- |
| `installed_mask` | Select a built-in mask or "random". |
| `mask_scale` | Size of the mask content (0.01 to 1.0 multiplier). |
| `mask_top_margin` | Vertical offset of mask content (0.0 to 0.5 percentage). |
| `bar_length` | Visualizer height in **pixels** (or **%** of mask scale). |
| `bar_length_mode` | `absolute` vs `relative` (% of mask scale). |
| `distribute_by` | Strategy for spreading points across the contour hierarchy. |
| `contour_layers` | Comma-separated list of target layers (e.g. `0,1`). |
| `contour_color_shift` | Hue variance between different layers (0.0 to 1.0). |
| `adaptive_point_density`| If True, `num_points` is scaled by the mask perimeter length. |
| `min_contour_area` | Ignores shapes smaller than this area in **square pixels**. |
| `max_contours` | Maximum number of individual shapes to process. |
| `ghost_mask_strength` | Opacity of the original mask shape preview (0.0 to 1.0). |
| `ghost_mode` | `None`, `White`, `Color`, `Outline White`, `Outline Color`, `Outline White (Thin)`, or `Outline Color (Thin)`. |
| `contour_smoothing` | Level of geometric simplification (0 to 50). |

#### Contour Outputs

| Name | Type | Description |
| :--- | :--- | :--- |
| `IMAGE` | Image | The final audio visualization. |
| `MASK` | Mask | The alpha mask of the visualization. |
| `SETTINGS` | String | Debug string of active settings. |
| `SOURCE_MASK` | Mask | The original input/randomly selected mask. |
| `LAYER_MAP` | Image | A color-coded visualization of the mask hierarchy (labels L0, L1, etc). |

---

## 3. Lyrics Integration

*The Lyrics system consists of an overlay node and a dedicated settings node.*

### Lyric Settings Node

*A standalone master node for text styling.*

| Parameter | Category | Description |
| :--- | :--- | :--- |
| `lrc_text` | Source | The raw LRC or SRT formatted lyric text. |
| `font_name` | Style | Selection from the `fonts/` directory. |
| `font_size` | Style | Base font size. |
| `highlight_color` | Style | Color of the current active line. |
| `normal_color` | Style | Color of pending/past lines. |
| `background_alpha` | Style | Transparency of the shadow/blur region behind text. |
| `blur_radius` | Style | Softness of the text shadow. |
| `active_blur` | Style | Extra background blur behind the currently active lyric. |
| `y_position` | Placement | Vertical alignment (0.75 = lower third). |
| `max_lines` | Layout | Maximum number of lines to display at once. |
| `line_spacing` | Layout | Gap between lines of text. |

### Flex Lyrics Node (The Overlay)

*This node combines a background (image/video) with the styling from a Lyric Settings node.*

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `audio` | Input | Used for timing synchronization with LRC/SRT. |
| `opt_video` | Input | Background source (image or video). |
| `lyric_settings` | Input | Link to a **Lyric Settings** node for global styling. |
| `strength` | Logic | Opacity/intensity of the lyric overlay. |
| `feature_param` | Logic | Which visualizer parameter controls visibility (optional). |
