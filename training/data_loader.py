# training/data_loader.py
"""
Data loading and preprocessing utilities.
"""
import json
import os
import logging
from datasets import Dataset, DatasetDict, load_from_disk
from typing import Tuple, List, Dict, Optional
import torch

logger = logging.getLogger(__name__)

class DataLoader:
    """Handles dataset loading, tokenization, and caching."""
    
    def __init__(self, cache_file: str, tokenized_cache_dir: str = "./tokenized_cache"):
        self.cache_file = cache_file
        self.tokenized_cache_dir = tokenized_cache_dir
        os.makedirs(tokenized_cache_dir, exist_ok=True)
    
    def load_raw_data(self) -> DatasetDict:
        """Load raw dataset from cache file."""
        logger.info(f"Loading raw data from {self.cache_file}")
        with open(self.cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        return DatasetDict({
            split: Dataset.from_dict(data)
            for split, data in cache_data.items()
        })
    
    def tokenize_dataset(self, 
                         tokenizer, 
                         max_seq_length: int = 512,
                         force_reprocess: bool = False) -> DatasetDict:
        """
        Tokenize dataset with caching.
        
        Args:
            tokenizer: HuggingFace tokenizer
            max_seq_length: Maximum sequence length
            force_reprocess: If True, ignore cached version
        
        Returns:
            Tokenized dataset
        """
        cache_path = os.path.join(
            self.tokenized_cache_dir, 
            f"seq_{max_seq_length}"
        )
        
        if os.path.exists(cache_path) and not force_reprocess:
            logger.info(f"Loading cached dataset from {cache_path}")
            return load_from_disk(cache_path)
        
        logger.info("Tokenizing dataset...")
        dataset = self.load_raw_data()
        
        def tokenize_function(examples):
            texts = [f"Convert to Linux command: {t} {c}" 
                     for t, c in zip(examples["text"], examples["command"])]
            tokenized = tokenizer(
                texts, 
                truncation=True, 
                padding="max_length", 
                max_length=max_seq_length, 
                return_tensors=None
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        tokenized = dataset.map(
            tokenize_function, 
            batched=True, 
            batch_size=1000,
            remove_columns=dataset["train"].column_names
        )
        
        # Save to cache
        tokenized.save_to_disk(cache_path)
        logger.info(f"Tokenized dataset saved to {cache_path}")
        
        return tokenized
    
    def load_test_data(self, split: str = "test") -> Tuple[List[str], List[str], List[str]]:
        """Load test split with prompts and commands."""
        logger.info(f"Loading {split} data from {self.cache_file}")
        with open(self.cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        split_data = cache_data[split]
        texts = split_data["text"]
        commands = split_data["command"]
        input_texts = split_data.get("input_texts", texts)
        
        logger.info(f"Loaded {len(texts)} {split} examples")
        return texts, commands, input_texts
    
    @staticmethod
    def format_prompt(text: str) -> str:
        """Format input text for inference."""
        return f"Convert to Linux command: {text}"
