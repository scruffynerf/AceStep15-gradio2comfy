"""AceStepAudioCodesUnderstand node for ACE-Step"""
import torch
import re

DEFAULT_LM_UNDERSTAND_INSTRUCTION = "Understand the given musical conditions and describe the audio semantics accordingly:"

class AceStepAudioCodesUnderstand:
    """Generatively reconstruct metadata and lyrics from Audio Codes using the 5Hz LLM"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "audio_codes": ("LIST",),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 100}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "DICT")
    RETURN_NAMES = ("full_output", "lyrics", "metadata")
    FUNCTION = "understand"
    CATEGORY = "Scromfy/Ace-Step/text"

    def understand(self, clip, audio_codes, temperature, top_k, top_p, max_new_tokens):
        if not audio_codes:
            return ("No audio codes provided.", "", {})

        # 1. Format codes into the string format the LLM expects
        # audio_codes might be a list of ints or tensors
        flat_codes = []
        for item in audio_codes:
            if isinstance(item, (int, float)):
                flat_codes.append(int(item))
            elif torch.is_tensor(item):
                if item.numel() == 1:
                    flat_codes.append(int(item.item()))
                else:
                    flat_codes.extend(item.flatten().tolist())
            elif isinstance(item, list):
                flat_codes.extend(item)
        
        code_str = "".join([f"<|audio_code_{c}|>" for c in flat_codes])
        
        # 2. Build the chat prompt
        # Qwen-style chat template: <|im_start|>system\n...\n<|im_end|>\n<|im_start|>user\n...\n<|im_end|>\n<|im_start|>assistant\n
        prompt = f"<|im_start|>system\n# Instruction\n{DEFAULT_LM_UNDERSTAND_INSTRUCTION}\n\n<|im_end|>\n"
        prompt += f"<|im_start|>user\n{code_str}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n"
        
        # 3. Access the underlying model and tokenizer
        # In ComfyUI, clip is a wrapper. We need the transformer model.
        # This part is environment-dependent, but usually clip.tokenizer and clip.patcher.model exist.
        tokenizer = clip.tokenizer
        model_patcher = clip.patcher
        model = model_patcher.model
        device = model_patcher.load_device
        
        # 4. Tokenize the prompt
        # We need to use the tokenizer's internal methods because it's a wrapper
        # Accessing the underlying transformers tokenizer if possible
        raw_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        inputs = raw_tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        
        # 5. Generative Loop
        # We'll do a simple sampling loop to avoid complex transformers dependencies
        generated_ids = input_ids.clone()
        
        # Set model to eval mode
        model.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                # Note: ComfyUI's model wrapper might expect specific arguments
                # For Qwen/CLIP models, we usually want the logits
                outputs = model(generated_ids)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                next_token_logits = logits[:, -1, :]
                
                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply Top-K
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply Top-P
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

                # Sample
                if temperature > 0:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                # Check for EOS
                if next_token.item() in [raw_tokenizer.eos_token_id, 151645]: # <|im_end|> is often 151645
                    break
                    
        # 6. Decode output
        output_ids = generated_ids[0, input_ids.shape[1]:]
        output_text = raw_tokenizer.decode(output_ids, skip_special_tokens=True)
        
        # 7. Parse Metadata and Lyrics
        # Simple extraction based on </think> tag and format
        metadata = {}
        lyrics = ""
        
        # Extract everything after </think> as lyrics
        think_match = re.search(r'</think>', output_text)
        if think_match:
            lyrics = output_text[think_match.end():].strip()
            # Clean up "# Lyric" header
            lyrics = re.sub(r'^#\s*Lyri[c|cs]?\s*\n', '', lyrics, flags=re.IGNORECASE).strip()
            
            # Metadata is usually inside the <think> or before it
            pre_lyrics = output_text[:think_match.start()]
            # Extract fields like "bpm: 120", "caption: ..."
            for field in ["bpm", "caption", "duration", "keyscale", "language", "timesignature"]:
                m = re.search(rf'{field}:\s*(.*?)(?:\n|$)', pre_lyrics, re.IGNORECASE)
                if m:
                    metadata[field] = m.group(1).strip()
        else:
            # Fallback if no think tag
            lyrics = output_text
            
        return (output_text, lyrics, metadata)

NODE_CLASS_MAPPINGS = {
    "AceStepAudioCodesUnderstand": AceStepAudioCodesUnderstand,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepAudioCodesUnderstand": "Audio Codes Understanding",
}
