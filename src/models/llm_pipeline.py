import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class OpenLLM:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2"):
        print(f"Loading {model_name} with 4-bit quantization...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto" # Handles multi-GPU automatically
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_choice(self, user_prompt: str) -> int:
        """
        Returns: 1 for A, 0 for B, -1 if parsing failed.
        """
        # 1. Robust Prompt Formatting using Chat Template
        # This handles system prompts and special tokens automatically
        messages = [{"role": "user", "content": user_prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # 2. Tokenize and move to correct device
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        # .to(model.device) is safer than .to("cuda") with device_map="auto"
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=10, 
                do_sample=False, # Deterministic (Greedy) for Pilot
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        # 3. Robust Slicing (Decode ONLY the new tokens)
        # We slice the output tensor, not the string
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # 4. robust Parsing
        # Normalize text to handle "Option A." vs "option a"
        clean_text = re.sub(r"[^\w\s]", "", response_text).lower()
        
        # Check start of string first (most likely)
        if clean_text.startswith("option a") or clean_text == "a":
            return 1
        if clean_text.startswith("option b") or clean_text == "b":
            return 0
            
        # Fallback search
        if "option a" in clean_text: return 1
        if "option b" in clean_text: return 0
            
        return -1