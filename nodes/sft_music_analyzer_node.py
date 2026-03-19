import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import os
import json
import re
import gc
import warnings
from huggingface_hub import snapshot_download

# Credit goes to https://github.com/jeankassio/ComfyUI-AceStep_SFT
# for his all-in-one SFT node implementation, I've split it into pieces.
# This is the music analyzer node.

# Global singleton cache for analysis models
_audio_model = None
_audio_processor = None
_audio_model_name = None

_ANALYSIS_MODELS = {
    "Qwen2.5-Omni-3B": "Qwen/Qwen2.5-Omni-3B",
    "Qwen2-Audio-7B-Instruct": "Qwen/Qwen2-Audio-7B-Instruct",
    "Ke-Omni-R-3B": "KE-Team/Ke-Omni-R-3B",
    "Qwen2.5-Omni-7B": "Qwen/Qwen2.5-Omni-7B",
    "AST-AudioSet": "MIT/ast-finetuned-audioset-10-10-0.4593",
    "MERT-v1-330M": "m-a-p/MERT-v1-330M",
    "Whisper-large-v2-audio-captioning": "MU-NLPC/whisper-large-v2-audio-captioning",
    "Whisper-small-audio-captioning": "MU-NLPC/whisper-small-audio-captioning",
    "Whisper-tiny-audio-captioning": "MU-NLPC/whisper-tiny-audio-captioning",
}

class ScromfySFTMusicAnalyzer:
    """Analyzes audio to extract descriptive tags, BPM and key/scale.
    Full port of AceStepSFTMusicAnalyzer with all 9 supported models.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"tooltip": "Audio to analyze for tags, BPM and key/scale."}),
                "model": (list(_ANALYSIS_MODELS.keys()), {
                    "default": "Qwen2.5-Omni-3B",
                    "tooltip": "Select the AI model for audio analysis (e.g. Qwen2.5 Omni for detailed tagging).",
                }),
                "get_tags": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Extract descriptive tags (genre, mood, etc.).",
                }),
                "get_bpm": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Estimate BPM (Beats Per Minute).",
                }),
                "get_keyscale": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Detect the musical key and scale.",
                }),
            },
            "optional": {
                "max_new_tokens": ("INT", {
                    "default": 100, "min": 50, "max": 1000, "step": 10,
                    "tooltip": "Maximum tokens for the AI model to generate.",
                }),
                "audio_duration": ("INT", {
                    "default": 30, "min": 10, "max": 120, "step": 5,
                    "tooltip": "Duration of audio (in seconds) to pass to the model for analysis.",
                }),
                "unload_model": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "VRAM management: Unload the analysis model from VRAM after completion.",
                }),
                "use_flash_attn": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable Flash Attention 2 (requires compatible GPU/drivers).",
                }),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 200, "step": 1}),
                "repetition_penalty": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 3.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("tags", "bpm", "keyscale", "music_infos")
    FUNCTION = "analyze"
    CATEGORY = "Scromfy/Ace-Step/SFT"

    def analyze(self, audio, model, get_tags, get_bpm, get_keyscale,
                max_new_tokens=100, audio_duration=30, unload_model=True, use_flash_attn=False,
                temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.5, seed=0):
        
        tags = ""
        detected_bpm = 0
        keyscale = ""
        
        gen_kwargs = self._build_gen_kwargs(temperature, top_p, top_k, repetition_penalty, seed)

        if get_tags:
            try:
                tags = self._extract_tags(audio, model, max_new_tokens, audio_duration, use_flash_attn, gen_kwargs)
                print(f"[ScromfySFT] Extracted tags: {tags}")
            except Exception as e:
                print(f"[ScromfySFT] Tag extraction failed: {e}")

        if get_bpm or get_keyscale:
            try:
                dsp = self._detect_bpm_keyscale(audio)
                if get_bpm: detected_bpm = dsp["bpm"]
                if get_keyscale: keyscale = dsp["keyscale"]
                print(f"[ScromfySFT] Detected BPM: {dsp['bpm']} | Key: {dsp['keyscale']}")
            except Exception as e:
                print(f"[ScromfySFT] DSP detection failed: {e}")

        if unload_model and get_tags:
            self._unload_audio_model()

        music_infos = json.dumps({
            "tags": tags,
            "bpm": f"{detected_bpm}bpm",
            "keyscale": keyscale,
        }, ensure_ascii=False, indent=4)

        return (tags, detected_bpm, keyscale, music_infos)

    def _build_gen_kwargs(self, temperature, top_p, top_k, repetition_penalty, seed):
        kwargs = {}
        if temperature > 0:
            kwargs["do_sample"] = True
            kwargs["temperature"] = temperature
        else:
            kwargs["do_sample"] = False
        if top_p < 1.0: kwargs["top_p"] = top_p
        if top_k > 0: kwargs["top_k"] = top_k
        if repetition_penalty != 1.0: kwargs["repetition_penalty"] = repetition_penalty
        torch.manual_seed(seed)
        return kwargs

    def _extract_tags(self, audio_dict, model_key, max_new_tokens, audio_duration, use_flash_attn, gen_kwargs):
        model, processor = self._load_audio_model(model_key, use_flash_attn)
        
        if model_key.startswith("Qwen2.5-Omni") or model_key == "Ke-Omni-R-3B":
            return self._tag_qwen_omni(audio_dict, model, processor, max_new_tokens, audio_duration, gen_kwargs)
        elif model_key == "Qwen2-Audio-7B-Instruct":
            return self._tag_qwen2_audio(audio_dict, model, processor, max_new_tokens, audio_duration, gen_kwargs)
        elif model_key == "MERT-v1-330M":
            return self._tag_mert(audio_dict, model, processor)
        elif model_key == "AST-AudioSet":
            return self._tag_ast(audio_dict, model, processor)
        elif "audio-captioning" in model_key:
            return self._tag_whisper_captioning(audio_dict, model, processor, max_new_tokens, audio_duration, gen_kwargs)
        return ""

    def _load_audio_model(self, model_key, use_flash_attn):
        global _audio_model, _audio_processor, _audio_model_name
        if _audio_model is not None and _audio_model_name == model_key:
            return _audio_model, _audio_processor
        if _audio_model is not None:
            self._unload_audio_model()

        repo_id = _ANALYSIS_MODELS[model_key]
        model_dir = snapshot_download(repo_id)
        load_kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
        if use_flash_attn: load_kwargs["attn_implementation"] = "flash_attention_2"

        if model_key.startswith("Qwen2.5-Omni") or model_key == "Ke-Omni-R-3B":
            from transformers import AutoProcessor
            if model_key == "Ke-Omni-R-3B":
                from transformers import Qwen2_5_OmniThinkerForConditionalGeneration as ModelClass
            else:
                from transformers import Qwen2_5OmniForConditionalGeneration as ModelClass
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*Flash Attention 2.*")
                _audio_model = ModelClass.from_pretrained(model_dir, **load_kwargs)
            if hasattr(_audio_model, "disable_talker"): _audio_model.disable_talker()
            _audio_processor = AutoProcessor.from_pretrained(model_dir, use_fast=False)
        elif model_key == "Qwen2-Audio-7B-Instruct":
            from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
            _audio_model = Qwen2AudioForConditionalGeneration.from_pretrained(model_dir, **load_kwargs)
            _audio_processor = AutoProcessor.from_pretrained(model_dir)
        elif "audio-captioning" in model_key:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            _audio_model = WhisperForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.float32, device_map="auto")
            _audio_processor = WhisperProcessor.from_pretrained(model_dir)
        elif model_key == "MERT-v1-330M":
            from transformers import AutoModel, Wav2Vec2FeatureExtractor
            _audio_model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.float32, device_map="auto", trust_remote_code=True)
            _audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir, trust_remote_code=True)
        elif model_key == "AST-AudioSet":
            from transformers import ASTForAudioClassification, AutoFeatureExtractor
            _audio_model = ASTForAudioClassification.from_pretrained(model_dir, torch_dtype=torch.float32, device_map="auto")
            _audio_processor = AutoFeatureExtractor.from_pretrained(model_dir)

        _audio_model.eval()
        _audio_model_name = model_key
        return _audio_model, _audio_processor

    def _unload_audio_model(self):
        global _audio_model, _audio_processor, _audio_model_name
        if _audio_model is not None:
            try: _audio_model.to("cpu")
            except: pass
            del _audio_model
            _audio_model = None
        if _audio_processor is not None:
            del _audio_processor
            _audio_processor = None
        _audio_model_name = None
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def _tag_qwen_omni(self, audio_dict, model, processor, max_new_tokens, audio_duration, gen_kwargs):
        y = self._prepare_audio_mono(audio_dict, 16000, audio_duration)
        instruction = "Describe this music in short, descriptive tags separated by commas. Focus on genre, mood, instruments, and style."
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "audio", "audio": y, "sampling_rate": 16000}, {"type": "text", "text": instruction}]}
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=prompt, audio=[y], sampling_rate=16000, return_tensors="pt", padding=True).to(model.device).to(model.dtype)
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, **gen_kwargs)
        result = processor.batch_decode(output_ids[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)[0]
        return self._clean_tags(result)

    def _tag_mert(self, audio_dict, model, processor):
        labels = ["drums", "bass", "guitar", "piano", "synth", "strings", "vocals", "electronic", "acoustic", "happy", "sad", "pop", "rock", "jazz", "hip hop"]
        y = self._prepare_audio_mono(audio_dict, 24000, 30)
        inputs = processor(y, sampling_rate=24000, return_tensors="pt").to(model.device)
        with torch.no_grad():
            features = model(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        top_indices = features.argsort()[::-1][:10]
        return ", ".join([labels[i % len(labels)] for i in top_indices])

    def _tag_ast(self, audio_dict, model, processor):
        y = self._prepare_audio_mono(audio_dict, 16000, 30)
        inputs = processor(y, sampling_rate=16000, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        probs = torch.sigmoid(logits)
        top_indices = probs.argsort(descending=True)[:10].cpu().numpy()
        labels = model.config.id2label
        return ", ".join([labels[int(i)] for i in top_indices if probs[i] > 0.1])

    def _tag_whisper_captioning(self, audio_dict, model, processor, max_new_tokens, audio_duration, gen_kwargs):
        y = self._prepare_audio_mono(audio_dict, 16000, audio_duration)
        inputs = processor(y, sampling_rate=16000, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, **gen_kwargs)
        return processor.batch_decode(gen, skip_special_tokens=True)[0]

    def _prepare_audio_mono(self, audio_dict, target_sr, max_seconds):
        waveform = audio_dict["waveform"]
        sr = audio_dict["sample_rate"]
        if waveform.dim() == 3: y = waveform[0].mean(dim=0)
        elif waveform.dim() == 2: y = waveform.mean(dim=0)
        else: y = waveform
        if sr != target_sr: y = torchaudio.functional.resample(y, sr, target_sr)
        y = y.cpu().numpy()
        max_samples = target_sr * max_seconds
        if len(y) > max_samples:
            start = (len(y) - max_samples) // 2
            y = y[start:start + max_samples]
        return y

    def _clean_tags(self, text):
        tags = [t.strip().lower() for t in text.split(",") if t.strip()]
        unique = []
        for t in tags:
            if t not in unique: unique.append(t)
        return ", ".join(unique[:20])

    def _detect_bpm_keyscale(self, audio_dict):
        import librosa
        waveform = audio_dict["waveform"]
        sr = audio_dict["sample_rate"]
        if waveform.dim() == 3: y = waveform[0].mean(dim=0).cpu().numpy()
        else: y = waveform.mean(dim=0).cpu().numpy()
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = int(round(float(tempo if not isinstance(tempo, np.ndarray) else tempo[0])))
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
        pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        best_idx = np.argmax(chroma)
        is_minor = chroma[(best_idx + 3) % 12] > chroma[(best_idx + 4) % 12]
        return {"bpm": bpm, "keyscale": f"{pitch_names[best_idx]} {'minor' if is_minor else 'major'}"}

NODE_CLASS_MAPPINGS = {"ScromfySFTMusicAnalyzer": ScromfySFTMusicAnalyzer}
NODE_DISPLAY_NAME_MAPPINGS = {"ScromfySFTMusicAnalyzer": "ScromfySFT Music Analyzer"}
