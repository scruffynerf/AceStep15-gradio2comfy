# Flex Audio Visualizers Reference

The Scromfy Flex Visualizer suite follows a strict **Global vs. Local** parameter split to eliminate UI redundancy and provide clear control over the "Soul" vs. "Shape" of the visualization.

## Architecture Overview

- **Global Settings (Master Node)**: Defines audio analysis, color logic, and global motion.
- **Local Settings (Individual Nodes)**: Defines the physical shape (radius, height, curvature) and placement (position, rotation).

---

## 1. Global Settings
*Located in the `Flex Visualizer Settings (Scromfy)` node.*

### System
| Parameter | Description |
| :--- | :--- |
| `randomize` | Automatically vary parameters for each batch. |
| `seed` | Deterministic seed for randomization. |
| `loop_background` | Toggle between looping the background (True) or clamping to the last frame (False). |

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
| `color_mode` | Mode: Spectrum, Custom, Amplitude, Radial, Angular, Path, Screen. |
| `color_schema` | Preset palette selection (Ice, Fire, Sunset, etc.). |
| `custom_color` | Static color for `custom` mode. |
| `color_shift` | Offsets the hue mapping along the sequence. |
| `saturation` | Intensity of colors. |
| `brightness` | Overall brilliance of colors. |
| `line_width` | Global thickness for all drawn lines and bars. |

### Motion & Direction
| Parameter | Description |
| :--- | :--- |
| `visualization_method` | `bar` (discrete segments) vs `line` (smooth connected path). |
| `direction` | `outward`, `inward`, `both`, `centroid`, `starburst`. |
| `sequence_direction` | `left`, `right`, `centered`, `both ends`. |
| `direction_skew` | Rotation of individual drawing vectors (in degrees). |
| `centroid_offset_x/y` | Manual offset for the expansion/contraction center. |

---

## 2. Node-Specific Settings
*Exclusive to individual visualizer nodes.*

### Circular Node (Shape/Placement)
| Parameter | Description |
| :--- | :--- |
| `radius` | Maximum expansion radius. |
| `base_radius` | The resting radius of the circle. |
| `amplitude_scale` | Sensitivity to audio volume. |
| `bar_length_mode` | `absolute` (pixels) vs `relative` (% of base_radius). |
| `position_x / y` | Screen placement (0.5 = center). |
| `rotation` | Overall rotation of the circular shape. |

### Line Node (Shape/Placement)
| Parameter | Description |
| :--- | :--- |
| `num_bars` | (Local Override) Number of elements to draw. |
| `max_height` | Ceiling for bar/amplitude length. |
| `min_height` | Floor for bar/amplitude length. |
| `bar_length_mode` | `absolute` vs `relative` (% of screen). |
| `length` | Physical length of the base line (0 = full screen). |
| `separation` | Gap between bars. |
| `curvature` | Visual rounding of bar caps. |
| `curve_smoothing` | Path smoothing for `line` method. |
| `position_x / y` | Screen placement. |
| `rotation` | Horizontal, Vertical, or Diagonal line angle. |

### Contour Node (Shape/Source)
| Parameter | Description |
| :--- | :--- |
| `installed_mask` | Preset mask selection. |
| `mask_scale` | Size of the mask relative to the screen. |
| `mask_top_margin` | Vertical shift of the mask. |
| `bar_length` | Height of the visualizer elements. |
| `bar_length_mode` | `absolute` vs `relative` (% of mask scale). |
| `distribute_by` | Strategy: `area`, `perimeter`, `equal`, `angular`. |
| `contour_layers` | Hierarchy selection (0 = outer only, or specific indices). |
| `contour_color_shift` | Color variance between distinct shapes. |
| `adaptive_point_density`| Scales point count based on perimeter length. |
| `min_contour_area` | Noise reduction threshold. |
| `max_contours` | Maximum number of shapes to process. |
| `ghost_mask_strength` | Alpha for the original shape preview. |
| `ghost_use_custom_color`| Toggle custom ghost color vs white. |
| `contour_smoothing` | Simplifies complex mask paths. |

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
