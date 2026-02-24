"""AceStepAudioCodesUnderstand node for ACE-Step"""
import torch
import re
import logging

logger = logging.getLogger(__name__)

DEFAULT_LM_UNDERSTAND_INSTRUCTION = "Understand the given musical conditions and describe the audio semantics accordingly:"

class AceStepAudioCodesUnderstand:
    """Generatively reconstruct metadata and lyrics from Audio Codes using a standalone LLM"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "llm": ("ACE_LLM",),
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

    def understand(self, audio_codes, temperature, top_k, top_p, max_new_tokens, llm=None, clip=None):
        if llm is None and clip is not None:
            llm = clip
            
        if llm is None:
            return ("Error: No LLM/CLIP model provided. Please ensure the node is connected.", "", {})
            
        if not audio_codes:
            return ("No audio codes provided.", "", {})

        model = llm["model"]
        tokenizer = llm["tokenizer"]
        device = llm["device"]

        # 1. Format codes into the string format the LLM expects
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
        prompt = f"<|im_start|>system\n# Instruction\n{DEFAULT_LM_UNDERSTAND_INSTRUCTION}\n\n<|im_end|>\n"
        prompt += f"<|im_start|>user\n{code_str}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n"
        
        # 3. Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        
        # 4. Generative Loop
        generated_ids = input_ids.clone()
        model.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
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
                if next_token.item() in [tokenizer.eos_token_id, 151645]: # <|im_end|> is often 151645
                    break
                    
        # 5. Decode output
        output_ids = generated_ids[0, input_ids.shape[1]:]
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        # 6. Parse Metadata and Lyrics
        metadata = {}
        lyrics = ""
        
        think_match = re.search(r'</think>', output_text)
        if think_match:
            lyrics = output_text[think_match.end():].strip()
            lyrics = re.sub(r'^#\s*Lyri[c|cs]?\s*\n', '', lyrics, flags=re.IGNORECASE).strip()
            
            pre_lyrics = output_text[:think_match.start()]
            for field in ["bpm", "caption", "duration", "keyscale", "language", "timesignature"]:
                m = re.search(rf'{field}:\s*(.*?)(?:\n|$)', pre_lyrics, re.IGNORECASE)
                if m:
                    metadata[field] = m.group(1).strip()
        else:
            lyrics = output_text
            
        return (output_text, lyrics, metadata)

NODE_CLASS_MAPPINGS = {
    "AceStepAudioCodesUnderstand": AceStepAudioCodesUnderstand,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStepAudioCodesUnderstand": "Audio Codes Understanding",
}
