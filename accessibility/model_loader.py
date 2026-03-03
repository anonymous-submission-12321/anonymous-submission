# accessibility/model_loader.py
"""Load fine-tuned QLoRA models for inference."""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Optional, Tuple
import os

class QLoRALoader:
    def __init__(self, checkpoint_path: str):
        self.path = checkpoint_path
        self.model = None
        self.tokenizer = None
        
    def load(self) -> Tuple[Optional[AutoModelForCausalLM], Optional[AutoTokenizer]]:
        """Load fine-tuned QLoRA checkpoint"""
        if not os.path.exists(self.path):
            print(f"Error: Checkpoint path {self.path} does not exist")
            return None, None
            
        print("Loading model...")  # Orca-readable
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.path,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            self.model.eval()
            
            # Ensure padding token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("Model loaded.")  # Orca-readable
            return self.model, self.tokenizer
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None
        
    def unload(self):
        """Free memory"""
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model = None
        self.tokenizer = None
