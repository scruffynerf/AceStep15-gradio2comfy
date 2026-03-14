import numpy as np
import cv2
import torch
import torchaudio.transforms as T
from .flex_audio_visualizer_contour_node import ScromfyFlexAudioVisualizerContourNode

class ScromfyEmojiSpinnerVisualizerNode(ScromfyFlexAudioVisualizerContourNode):
    @classmethod
    def INPUT_TYPES(cls):
        base_inputs = super().INPUT_TYPES()
        required = base_inputs.get("required", {}).copy()
        optional = base_inputs.get("optional", {}).copy()
        
        # Remove inputs that we will override or provide automatically
        if "installed_mask" in required:
            del required["installed_mask"]
            
        if "mask" in optional:
            del optional["mask"]
            
        # Add Spinner specific inputs
        required["spinner_frames"] = ("IMAGE",)
        required["spinner_audio"] = ("AUDIO",)
        required["spinner_mask"] = ("MASK",)
        
        return {
            "required": required,
            "optional": optional
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "MASK", "IMAGE", "AUDIO")
    RETURN_NAMES = ("IMAGE", "MASK", "SETTINGS", "SOURCE_MASK", "LAYER_MAP", "AUDIO")
    FUNCTION = "apply_spinner_effect"
    CATEGORY = "Scromfy/Ace-Step/Visualizers"

    def apply_spinner_effect(self, audio, frame_rate, screen_width, screen_height, strength, feature_param,
                     feature_mode, feature_threshold, spinner_frames, spinner_audio, spinner_mask, opt_video=None, opt_feature=None, **kwargs):
        
        spin_num_frames = spinner_frames.shape[0]
        s_height, s_width = spinner_frames.shape[1], spinner_frames.shape[2]
        
        out_w = screen_width
        out_h = screen_height
        if opt_video is not None:
            out_h, out_w = opt_video.shape[1], opt_video.shape[2]
            
        mask_scale = kwargs.get("mask_scale", 0.60)
        mask_top_margin = kwargs.get("mask_top_margin", 0.05)
        
        new_w = int(s_width * mask_scale)
        new_h = int(s_height * mask_scale)
        
        # Phase 1: Overlay spinner_frames onto background
        phase1_frames = []
        for i in range(spin_num_frames):
            frame_np = spinner_frames[i].cpu().numpy()
            
            # Step 1: Scale within spinner frame original size (s_width, s_height) 
            if new_w > 0 and new_h > 0:
                m_resized = cv2.resize(frame_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
                canvas = np.zeros((s_height, s_width, 3), dtype=np.float32)
                x_offset = (s_width - new_w) // 2
                y_offset = int(s_height * mask_top_margin)
                y_end = min(y_offset + new_h, s_height)
                x_end = min(x_offset + new_w, s_width)
                h_to_copy = y_end - y_offset
                w_to_copy = x_end - x_offset
                canvas[y_offset:y_end, x_offset:x_end] = m_resized[:h_to_copy, :w_to_copy]
            else:
                canvas = frame_np.copy()
                
            # Step 2: Stretch to final output dimensions (like the visualizer base does)
            if s_height != out_h or s_width != out_w:
                canvas = cv2.resize(canvas, (out_w, out_h))
                
            # Delta mask from black background. Black is literally 0,0,0
            # Be slightly lenient: sum across RGB channels > 0.05
            alpha = (np.sum(canvas, axis=-1) > 0.05).astype(np.float32)[..., np.newaxis]
            
            bg = np.zeros((out_h, out_w, 3), dtype=np.float32)
            if opt_video is not None:
                loop_bg = kwargs.get("loop_background", True)
                v_idx = (i % opt_video.shape[0]) if loop_bg else min(i, opt_video.shape[0]-1)
                bg = opt_video[v_idx].cpu().numpy()
                if bg.shape[0] != out_h or bg.shape[1] != out_w:
                    bg = cv2.resize(bg, (out_w, out_h))
            
            blended = canvas * alpha + bg * (1.0 - alpha)
            phase1_frames.append(torch.from_numpy(blended))
            
        phase1_tensor = torch.stack(phase1_frames)
        
        # Phase 2: Visualize remaining audio
        # Note: opt_video needs to start AFTER the spin phase for Phase 2 to maintain seamless looping
        phase2_opt_video = None
        if opt_video is not None:
            loop_bg = kwargs.get("loop_background", True)
            if loop_bg:
                num_v_frames = opt_video.shape[0]
                start_idx = spin_num_frames % num_v_frames
                phase2_opt_video = torch.cat([opt_video[start_idx:], opt_video[:start_idx]], dim=0)
            else:
                if opt_video.shape[0] > spin_num_frames:
                    phase2_opt_video = opt_video[spin_num_frames:]
                else:
                    # Last frame repeated
                    phase2_opt_video = opt_video[-1:]
                    
        # We pass the original unscaled spinner_mask. The Contour node will apply 
        # mask_scale and mask_top_margin automatically.
        images, masks, settings, source_mask_out, layer_map = super().apply_effect(
            audio, frame_rate, screen_width, screen_height, strength, feature_param,
            feature_mode, feature_threshold, mask=spinner_mask, opt_video=phase2_opt_video,
            opt_feature=opt_feature, **kwargs
        )
        
        # Combine frames
        full_images = torch.cat([phase1_tensor, images], dim=0)
        
        # Combine masks (Phase 1 generates no Contour masks, so we pad with zeros)
        b2, h2, w2 = masks.shape if len(masks.shape) == 3 else (1, *masks.shape)
        phase1_masks = torch.zeros((spin_num_frames, out_h, out_w), dtype=torch.float32)
        full_masks = torch.cat([phase1_masks, masks.view(b2, out_h, out_w)], dim=0)
        
        # Combine layer maps
        b2_lm, h2_lm, w2_lm, c2_lm = layer_map.shape 
        phase1_layer_maps = torch.zeros((spin_num_frames, out_h, out_w, 3), dtype=torch.float32)
        full_layer_maps = torch.cat([phase1_layer_maps, layer_map.view(b2_lm, out_h, out_w, 3)], dim=0)
        
        # Concatenate Audio
        s_wav = spinner_audio['waveform']
        s_sr = spinner_audio['sample_rate']
        m_wav = audio['waveform']
        m_sr = audio['sample_rate']
        
        if s_sr != m_sr:
            resampler = T.Resample(orig_freq=s_sr, new_freq=m_sr, dtype=s_wav.dtype)
            s_wav = resampler(s_wav)
            
        if s_wav.shape[1] != m_wav.shape[1]:
            # force to 2 channels
            if s_wav.shape[1] == 1: s_wav = s_wav.repeat(1, 2, 1)
            if m_wav.shape[1] == 1: m_wav = m_wav.repeat(1, 2, 1)
            
        combined_wav = torch.cat([s_wav, m_wav], dim=-1)
        combined_audio = {"waveform": combined_wav, "sample_rate": m_sr}
        
        return (full_images, full_masks, settings, source_mask_out, full_layer_maps, combined_audio)

NODE_CLASS_MAPPINGS = {
    "ScromfyEmojiSpinnerVisualizer": ScromfyEmojiSpinnerVisualizerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ScromfyEmojiSpinnerVisualizer": "Emoji Spinner Visualizer (Scromfy)",
}
