import numpy as np
import cv2
import torch
from abc import abstractmethod
from .flex_utils import FlexBase

class BaseAudioProcessor:
    def __init__(self, audio, num_frames, height, width, frame_rate):
        """
        Base class to process audio data.
        """
        # Convert waveform tensor to mono numpy array
        self.audio = audio['waveform'].squeeze(0).mean(axis=0).cpu().numpy()
        self.sample_rate = audio['sample_rate']
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.frame_rate = frame_rate

        self.audio_duration = len(self.audio) / self.sample_rate
        self.frame_duration = 1 / self.frame_rate if self.frame_rate > 0 else self.audio_duration / self.num_frames

        self.spectrum = None  # Initialize spectrum
        self.current_frame = 0

    def _normalize(self, data):
        return (data - data.min()) / (data.max() - data.min())

    def _enhance_contrast(self, data, power=0.3):
        return np.power(data, power)

    def _resize(self, data, new_width, new_height):
        return cv2.resize(data, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    def _get_audio_frame(self, frame_index):
        start_time = frame_index * self.frame_duration
        end_time = (frame_index + 1) * self.frame_duration
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        return self.audio[start_sample:end_sample]

    def compute_spectrum(self, frame_index, fft_size, min_frequency, max_frequency):
        audio_frame = self._get_audio_frame(frame_index)
        if len(audio_frame) < fft_size:
            audio_frame = np.pad(audio_frame, (0, fft_size - len(audio_frame)), mode='constant')

        # Apply window function
        window = np.hanning(len(audio_frame))
        audio_frame = audio_frame * window

        # Compute FFT
        spectrum = np.abs(np.fft.rfft(audio_frame, n=fft_size))

        # Extract desired frequency range
        freqs = np.fft.rfftfreq(fft_size, d=1.0 / self.sample_rate)
        freq_indices = np.where((freqs >= min_frequency) & (freqs <= max_frequency))[0]
        
        # Store active frequencies for color mapping
        self.active_frequencies = freqs[freq_indices]
        spectrum = spectrum[freq_indices]

        # Check if spectrum is not empty
        if spectrum.size > 0:
            # Apply logarithmic scaling
            spectrum = np.log1p(spectrum)

            # Normalize
            max_spectrum = np.max(spectrum)
            if max_spectrum != 0:
                spectrum = spectrum / max_spectrum
            else:
                spectrum = np.zeros_like(spectrum)
        else:
            # Return zeros if spectrum is empty
            spectrum = np.zeros(1)
            self.active_frequencies = np.array([min_frequency])

        return spectrum

    def update_spectrum(self, new_spectrum, smoothing):
        if self.spectrum is None or len(self.spectrum) != len(new_spectrum):
            self.spectrum = np.zeros(len(new_spectrum))

        # Apply smoothing
        self.spectrum = smoothing * self.spectrum + (1 - smoothing) * new_spectrum

import colorsys

def get_color_for_frequency(freq, shift=0.0, saturation=1.0, brightness=1.0):
    """
    Maps a frequency to a color in the HSL spectrum.
    Uses log scale so octaves match.
    """
    if freq <= 0:
        return (0.0, 0.0, 0.0)
    
    # log2(freq) gives us a linear scale where +1 is one octave
    # Multiply by 1.0 (or whatever) to map an octave to a range
    # 1.0 means one octave wraps around the full 360 degrees (0.0 to 1.0)
    # Most musical-visual mappings use this "circular" property.
    hue = (np.log2(freq) + shift) % 1.0
    
    # Convert HLS (Hue, Lightness, Saturation) to RGB
    # We use lightness = brightness * 0.5 (since 0.5 is full color, 1.0 is white)
    r, g, b = colorsys.hls_to_rgb(hue, brightness * 0.5, saturation)
    return (r, g, b)

class FlexAudioVisualizerBase(FlexBase):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        base_required = base_inputs.get("required", {})
        base_optional = base_inputs.get("optional", {})

        new_inputs = {
            "required": {
                "audio": ("AUDIO",),
                "frame_rate": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 240.0, "step": 1.0}),
                "screen_width": ("INT", {"default": 768, "min": 100, "max": 1920, "step": 1}),
                "screen_height": ("INT", {"default": 464, "min": 100, "max": 1080, "step": 1}),
                "position_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "position_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "color_mode": (["white", "spectrum"], {"default": "spectrum"}),
                "color_shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Shifts the starting point of the color spectrum."}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

        required = {**new_inputs["required"], **base_required}
        optional = {**base_optional}

        return {
            "required": required,
            "optional": optional
        }

    CATEGORY = "Scromfy/Ace-Step/Visualizers"
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "apply_effect"

    @classmethod
    @abstractmethod
    def get_modifiable_params(cls):
        """Return a list of parameter names that can be modulated."""
        pass

    def validate_param(self, param_name, param_value):
        """
        Ensure that modulated parameter values stay within valid ranges.
        """
        valid_params = {
            'fft_size': lambda x: max(256, int(2 ** np.round(np.log2(x)))) if x > 0 else 256,
            'min_frequency': lambda x: max(20.0, min(x, 20000.0)),
            'max_frequency': lambda x: max(20.0, min(x, 20000.0)),
            'num_bars': lambda x: max(1, int(x)),
            'num_points': lambda x: max(3, int(x)),
            'smoothing': lambda x: np.clip(x, 0.0, 1.0),
            'rotation': lambda x: x % 360.0,
            'curvature': lambda x: max(0.0, x),
            'separation': lambda x: max(0.0, x),
            'max_height': lambda x: max(10.0, x),
            'min_height': lambda x: max(0.0, x),
            'position_x': lambda x: np.clip(x, 0.0, 1.0),
            'position_y': lambda x: np.clip(x, 0.0, 1.0),
            'reflect': lambda x: bool(x),
            'line_width': lambda x: max(1, int(x)),
            'radius': lambda x: max(1.0, x),
            'base_radius': lambda x: max(1.0, x),
            'amplitude_scale': lambda x: max(0.0, x),
            'color_shift': lambda x: x % 1.0,
            'saturation': lambda x: np.clip(x, 0.0, 1.0),
            'brightness': lambda x: np.clip(x, 0.0, 1.0),
        }

        if param_name in valid_params:
            return valid_params[param_name](param_value)
        else:
            return param_value

    def rotate_image(self, image, angle):
        """Rotate the image by the given angle."""
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, M, (w, h))

        return rotated_image

    @abstractmethod
    def get_audio_data(self, processor: BaseAudioProcessor, frame_index, **kwargs):
        """
        Abstract method to get audio data for visualization at a specific frame index.
        """
        pass

    @abstractmethod
    def apply_effect_internal(self, processor: BaseAudioProcessor, **kwargs) -> np.ndarray:
        """
        Abstract method to generate the image for the current frame.
        """
        pass

    def process_audio_data(self, processor: BaseAudioProcessor, frame_index, visualization_feature, num_points, smoothing, fft_size, min_frequency, max_frequency):
        if visualization_feature == 'frequency':
            spectrum = processor.compute_spectrum(frame_index, fft_size, min_frequency, max_frequency)

            # Resample the spectrum to match the number of points
            data = np.interp(
                np.linspace(0, len(spectrum), num_points, endpoint=False),
                np.arange(len(spectrum)),
                spectrum,
            )

        elif visualization_feature == 'waveform':
            audio_frame = processor._get_audio_frame(frame_index)
            if len(audio_frame) < 1:
                data = np.zeros(num_points)
            else:
                # Use the waveform data directly
                data = np.interp(
                    np.linspace(0, len(audio_frame), num_points, endpoint=False),
                    np.arange(len(audio_frame)),
                    audio_frame,
                )
                # Normalize the waveform to [-1, 1]
                max_abs_value = np.max(np.abs(data))
                if max_abs_value != 0:
                    data = data / max_abs_value
                else:
                    data = np.zeros_like(data)
        else:
            data = np.zeros(num_points)

        # Update processor's spectrum with smoothing
        if processor.spectrum is None or len(processor.spectrum) != len(data):
            processor.spectrum = np.zeros(len(data))
        processor.update_spectrum(data, smoothing)

        # Return updated data and frequency mapping
        feature_value = np.mean(np.abs(processor.spectrum))
        
        # We need to resample frequencies as well to match num_points
        if visualization_feature == 'frequency' and hasattr(processor, 'active_frequencies'):
            item_freqs = np.interp(
                np.linspace(0, len(processor.active_frequencies), num_points, endpoint=False),
                np.arange(len(processor.active_frequencies)),
                processor.active_frequencies,
            )
        else:
            item_freqs = np.zeros(num_points)
            
        return processor.spectrum.copy(), feature_value, item_freqs

    def apply_effect(self, audio, frame_rate, screen_width, screen_height, 
                     strength, feature_param, feature_mode, feature_threshold,
                    opt_feature=None, **kwargs):
        # Calculate num_frames based on audio duration and frame_rate
        audio_duration = len(audio['waveform'].squeeze(0).mean(axis=0)) / audio['sample_rate']
        num_frames = int(audio_duration * frame_rate)

        # Initialize the audio processor
        processor = BaseAudioProcessor(audio, num_frames, screen_height, screen_width, frame_rate)

        # Initialize results list
        result = []

        self.start_progress(num_frames, desc=f"Applying {self.__class__.__name__}")

        for i in range(num_frames):
            processor.current_frame = i
            
            # First process parameters to get the correct values for this frame
            processed_kwargs = self.process_parameters(
                frame_index=i,
                feature_value=self.get_feature_value(i, opt_feature) if opt_feature is not None else None,
                feature_param=feature_param,
                feature_mode=feature_mode,
                strength=strength,
                feature_threshold=feature_threshold,
                **kwargs
            )
            processed_kwargs["frame_index"] = i
            processed_kwargs["screen_width"] = screen_width
            processed_kwargs["screen_height"] = screen_height
            
            # Get audio data using the processed parameters
            num_points = processed_kwargs.get('num_points', processed_kwargs.get('num_bars', 64))
            spectrum, _, item_freqs = self.process_audio_data(
                processor, 
                i,
                processed_kwargs.get('visualization_feature', 'frequency'),
                num_points,
                processed_kwargs.get('smoothing', 0.5),
                processed_kwargs.get('fft_size', 2048),
                processed_kwargs.get('min_frequency', 20.0),
                processed_kwargs.get('max_frequency', 8000.0)
            )
            processed_kwargs["item_freqs"] = item_freqs

            # Generate the image for the current frame
            image = self.apply_effect_internal(processor, **processed_kwargs)
            result.append(image)

            self.update_progress()

        self.end_progress()

        # Convert result to tensor
        result_np = np.stack(result)
        result_tensor = torch.from_numpy(result_np).float()
        mask = result_tensor[:, :, :, 0]
        
        return (result_tensor, mask,)
