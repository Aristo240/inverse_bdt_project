import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

class OpenLLM:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2", device="cuda:0"):
        print(f"Loading {model_name} in float16 on {device}...")
        
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model in standard FP16 (No Quantization)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        ).to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_choice(self, user_prompt: str) -> int:
        """
        Returns: 1 for A, 0 for B, -1 if parsing failed.
        """
        # Manual Formatting for Mistral
        formatted_prompt = f"[INST] {user_prompt} [/INST]\nMy choice is Option"

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        # Force inputs to the correct single device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=5,     
                min_new_tokens=1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        clean_text = re.sub(r"[^\w\s]", "", response_text).lower()
        
        if "a" in clean_text: return 1
        if "b" in clean_text: return 0
        
        print(f"\n[DEBUG] Raw Output: '{response_text}'")
        return -1