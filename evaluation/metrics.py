# evaluation/metrics.py
"""
Metrics computation for command generation evaluation.
"""
import evaluate
import numpy as np
from typing import List, Dict, Tuple
import re
import logging

logger = logging.getLogger(__name__)

# Load metrics once
_bleu = evaluate.load("bleu")
_rouge = evaluate.load("rouge")
_exact_match = evaluate.load("exact_match")

class CommandMetrics:
    """Compute evaluation metrics for command generation."""
    
    @staticmethod
    def normalize_command(cmd: str) -> str:
        """Normalize command for comparison."""
        if not cmd:
            return ""
        return ' '.join(cmd.strip().lower().split())
    
    @staticmethod
    def extract_command(response: str, input_text: str = None) -> str:
        """
        Extract clean command from model response.
        
        Args:
            response: Raw model output
            input_text: Original input prompt (for cleanup)
        
        Returns:
            Cleaned command string
        """
        prompt_prefix = "Convert to Linux command: "
        
        # Strategy 1: Remove prompt prefix if at beginning
        if response.startswith(prompt_prefix):
            command = response[len(prompt_prefix):].strip()
        else:
            command = response.strip()
        
        # Strategy 2: Remove input text if at beginning
        if input_text and command.startswith(input_text):
            command = command[len(input_text):].lstrip()
        
        # Strategy 3: Handle prompt in middle
        if prompt_prefix in command:
            parts = command.split(prompt_prefix)
            command = parts[-1].strip()
        
        # Take first line only
        command = command.split('\n')[0].strip()
        
        # Remove quotes and artifacts
        command = command.strip('"').strip("'").strip('`').strip()
        command = re.sub(r'\s+', ' ', command)
        
        # Handle repetitive garbage
        words = command.split()
        if len(words) > 20 and len(set(words)) / len(words) < 0.3:
            command = ' '.join(words[:5])
        
        return command
    
    def compute_metrics(self, 
                       predictions: List[str], 
                       references: List[str]) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Args:
            predictions: List of generated commands
            references: List of ground truth commands
        
        Returns:
            Dictionary with metric names and values
        """
        logger.info("Computing evaluation metrics...")
        
        # Exact Match
        norm_preds = [self.normalize_command(p) for p in predictions]
        norm_refs = [self.normalize_command(r) for r in references]
        exact_match = _exact_match.compute(
            predictions=norm_preds,
            references=norm_refs
        )["exact_match"]
        
        # BLEU
        bleu = _bleu.compute(
            predictions=predictions,
            references=[[r] for r in references]
        )["bleu"]
        
        # ROUGE
        rouge = _rouge.compute(
            predictions=predictions,
            references=references
        )
        
        return {
            "exact_match": exact_match,
            "bleu": bleu,
            "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"],
            "rougeL": rouge["rougeL"]
        }
    
    def compute_char_bleu(self, 
                         predictions: List[str], 
                         references: List[str]) -> float:
        """Compute character-level BLEU score."""
        char_preds = [" ".join(list(pred)) for pred in predictions]
        char_refs = [[" ".join(list(ref))] for ref in references]
        char_bleu = _bleu.compute(predictions=char_preds, references=char_refs)
        return char_bleu["bleu"]
