# training/qlora_trainer.py
"""
QLoRA trainer module for Linux command generation.
Consolidates training logic from train.py
"""
import torch
import logging
import gc
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset, DatasetDict, load_from_disk
from typing import Dict, Optional, Tuple
import json
import os

logger = logging.getLogger(__name__)

class QLoRATrainer:
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def setup_tokenizer(self, model_path: str):
        """Initialize and configure tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer
    
    def setup_model(self, model_path: str, model_config: Dict):
        """Initialize model with QLoRA configuration."""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"
        )
        
        model = prepare_model_for_kbit_training(model)
        
        # Enable gradient checkpointing with explicit use_reentrant=False
        if not model_config.get("disable_gradient_checkpointing", False):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.config.use_cache = False
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        return get_peft_model(model, lora_config)
    
    def train(self, 
              model_path: str,
              model_name: str,
              train_dataset,
              eval_dataset,
              output_dir: str,
              batch_size: int = 8,
              num_epochs: int = 3,
              learning_rate: float = 2e-4,
              warmup_steps: int = 100,
              save_steps: int = 500,
              eval_steps: int = 500):
        """Run training loop."""
        
        # Calculate gradient accumulation steps based on batch size
        if batch_size == 8:
            grad_accum = 2
        elif batch_size == 4:
            grad_accum = 4
        else:
            grad_accum = 8
        
        training_args = TrainingArguments(
            output_dir=f"{output_dir}/{model_name}",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            warmup_steps=warmup_steps,
            logging_steps=10,
            save_steps=save_steps,
            eval_steps=eval_steps,
            learning_rate=learning_rate,
            fp16=True,
            eval_strategy="steps",
            save_total_limit=2,
            load_best_model_at_end=True,
            remove_unused_columns=False,
            dataloader_num_workers=4,
            optim="adamw_8bit",
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )
        
        logger.info("Starting training...")
        trainer.train()
        trainer.save_model(f"{output_dir}/{model_name}/final")
        
        # Clean up
        gc.collect()
        torch.cuda.empty_cache()
        
        return trainer
