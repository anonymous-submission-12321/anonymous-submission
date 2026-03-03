# accessibility/command_recognizer.py
"""Convert natural language to shell commands using fine-tuned model."""
import torch
from typing import Optional

class CommandRecognizer:
    def __init__(self, model, tokenizer, max_length: int = 128):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def recognize(self, user_input: str) -> str:
        """Convert natural language to shell command"""
        if not self.model or not self.tokenizer:
            return "# Error: Model not loaded"
            
        prompt = f"### Instruction: Convert to shell command\n### Input: {user_input}\n### Command:"
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            # Move to same device as model
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    temperature=0.1,  # Low temp for deterministic output
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            command = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the command part after the prompt
            if "### Command:" in command:
                command = command.split("### Command:")[-1].strip()
            else:
                # Fallback: take everything after the last newline
                command = command.split('\n')[-1].strip()
            
            return command
            
        except Exception as e:
            return f"# Error generating command: {e}"
