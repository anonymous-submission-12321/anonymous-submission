# evaluation/benchmark.py
"""
Inference speed benchmarking utilities.
"""
import torch
import time
from typing import List, Dict, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)

class SpeedBenchmark:
    """Benchmark inference speed of models."""
    
    def __init__(self, 
                 model: PreTrainedModel, 
                 tokenizer: PreTrainedTokenizer,
                 device: Optional[torch.device] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device
        
        # Set padding side for generation
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def benchmark(self,
                 prompts: List[str],
                 num_samples: int = 100,
                 batch_size: int = 1,
                 max_new_tokens: int = 100,
                 warmup: int = 3) -> Dict[str, float]:
        """
        Run speed benchmark.
        
        Returns:
            Dictionary with tokens/sec, samples/sec metrics
        """
        # Repeat prompts to reach num_samples
        all_prompts = (prompts * (num_samples // len(prompts) + 1))[:num_samples]
        
        # Warmup
        logger.info(f"Warming up ({warmup} runs)...")
        for i in range(warmup):
            _ = self.model.generate(
                **self.tokenizer(all_prompts[i:i+1], return_tensors="pt").to(self.device),
                max_new_tokens=10
            )
        
        # Benchmark
        logger.info(f"Benchmarking with {num_samples} samples...")
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        total_tokens = 0
        
        for i in range(0, len(all_prompts), batch_size):
            batch = all_prompts[i:i+batch_size]
            inputs = self.tokenizer(
                batch, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Count generated tokens (excluding input)
            input_len = inputs['input_ids'].shape[1]
            for output in outputs:
                total_tokens += len(output) - input_len
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start_time
        
        return {
            "total_samples": num_samples,
            "total_tokens": total_tokens,
            "elapsed_seconds": elapsed,
            "tokens_per_second": total_tokens / elapsed,
            "samples_per_second": num_samples / elapsed,
            "avg_tokens_per_sample": total_tokens / num_samples
        }
    
    @staticmethod
    def get_default_prompts() -> List[str]:
        """Return default set of test prompts."""
        return [
            "Convert to Linux command: Check CPU scaling",
            "Convert to Linux command: Find all Python files larger than 1MB",
            "Convert to Linux command: Show disk usage in human readable format",
            "Convert to Linux command: Kill process running on port 8080",
            "Convert to Linux command: Monitor system logs in real time",
            "Convert to Linux command: Compress directory into tar.gz",
            "Convert to Linux command: Check memory usage every 2 seconds",
            "Convert to Linux command: Find files modified in last 24 hours",
            "Convert to Linux command: Show open network connections",
            "Convert to Linux command: Backup home directory to external drive",
        ]
