"""
Optimized Whisper trainer for operator conversion
Includes specialized training strategies and evaluation metrics
"""

import os
import torch
import numpy as np
import evaluate
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from typing import Dict, List, Optional, Any, Tuple
import logging
import wandb
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OperatorTrainingConfig:
    """Configuration for operator-focused training"""
    model_name: str = "openai/whisper-small"
    output_dir: str = "./results"
    
    # Training hyperparameters optimized for operator conversion
    learning_rate: float = 1e-5  # Lower LR for fine-tuning
    warmup_steps: int = 500
    max_steps: int = 5000
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 100
    
    # Batch sizes
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    
    # Optimization
    optim: str = "adamw_torch"
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Regularization for fine-tuning
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Generation parameters
    generation_max_length: int = 225
    generation_num_beams: int = 1  # Faster inference
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Operator-specific settings
    operator_loss_weight: float = 2.0  # Higher weight for operator tokens
    symbol_accuracy_weight: float = 1.5  # Weight symbol accuracy in evaluation

class OperatorWhisperTrainer:
    """Specialized trainer for operator conversion with Whisper"""
    
    def __init__(self, config: OperatorTrainingConfig):
        self.config = config
        self.processor = WhisperProcessor.from_pretrained(config.model_name)
        self.model = None
        self.trainer = None
        
        # Load evaluation metrics
        self.wer_metric = evaluate.load("wer")
        self.bleu_metric = evaluate.load("bleu")
        
        # Operator symbols for specialized evaluation
        self.operator_symbols = set("+-×÷=<>%$@#&*()[]{},.!?;:")
        
        # Setup logging
        if wandb.run is None:
            wandb.init(project="speech2symbol", config=config.__dict__)
    
    def prepare_model(self, freeze_encoder: bool = True) -> WhisperForConditionalGeneration:
        """Prepare model with optimal settings for operator conversion"""
        logger.info(f"Loading model: {self.config.model_name}")
        
        model = WhisperForConditionalGeneration.from_pretrained(
            self.config.model_name,
            dropout=self.config.dropout,
            attention_dropout=self.config.attention_dropout
        )
        
        # Freeze encoder for more stable fine-tuning
        if freeze_encoder:
            logger.info("Freezing encoder parameters")
            for param in model.model.encoder.parameters():
                param.requires_grad = False
        
        # Enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()
        
        self.model = model
        return model
    
    def create_training_arguments(self) -> TrainingArguments:
        """Create optimized training arguments"""
        return TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            logging_steps=self.config.logging_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            optim=self.config.optim,
            weight_decay=self.config.weight_decay,
            adam_beta1=self.config.adam_beta1,
            adam_beta2=self.config.adam_beta2,
            adam_epsilon=self.config.adam_epsilon,
            max_grad_norm=self.config.max_grad_norm,
            fp16=torch.cuda.is_available(),  # Use mixed precision if available
            dataloader_pin_memory=True,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            label_names=["labels"],
            load_best_model_at_end=True,
            metric_for_best_model="eval_symbol_accuracy",
            greater_is_better=True,
            report_to="wandb",
        )
    
    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """Compute comprehensive metrics including symbol-level accuracy"""
        predictions, labels = eval_pred
        
        # Handle case where predictions might be logits
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        # Decode predictions and labels
        decoded_preds = self.processor.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in labels (ignore index)
        labels = np.where(labels != -100, labels, self.processor.tokenizer.pad_token_id)
        decoded_labels = self.processor.batch_decode(labels, skip_special_tokens=True)
        
        # Compute WER
        wer = self.wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
        
        # Compute BLEU
        bleu = self.bleu_metric.compute(
            predictions=decoded_preds,
            references=[[label] for label in decoded_labels]
        )
        
        # Compute symbol-level accuracy
        symbol_accuracy = self._compute_symbol_accuracy(decoded_preds, decoded_labels)
        
        # Compute operator-specific accuracy
        operator_accuracy = self._compute_operator_accuracy(decoded_preds, decoded_labels)
        
        # Compute character-level accuracy
        char_accuracy = self._compute_character_accuracy(decoded_preds, decoded_labels)
        
        return {
            "wer": wer,
            "bleu": bleu["bleu"],
            "symbol_accuracy": symbol_accuracy,
            "operator_accuracy": operator_accuracy,
            "char_accuracy": char_accuracy,
            "combined_score": (
                (1 - wer) * 0.3 + 
                bleu["bleu"] * 0.2 + 
                symbol_accuracy * 0.3 + 
                operator_accuracy * 0.2
            )
        }
    
    def _compute_symbol_accuracy(self, predictions: List[str], references: List[str]) -> float:
        """Compute accuracy specifically for symbol conversion"""
        correct_symbols = 0
        total_symbols = 0
        
        for pred, ref in zip(predictions, references):
            # Extract symbols from both
            pred_symbols = [char for char in pred if char in self.operator_symbols]
            ref_symbols = [char for char in ref if char in self.operator_symbols]
            
            # Count correct symbols (order matters)
            min_len = min(len(pred_symbols), len(ref_symbols))
            for i in range(min_len):
                if pred_symbols[i] == ref_symbols[i]:
                    correct_symbols += 1
            
            total_symbols += max(len(pred_symbols), len(ref_symbols))
        
        return correct_symbols / max(total_symbols, 1)
    
    def _compute_operator_accuracy(self, predictions: List[str], references: List[str]) -> float:
        """Compute accuracy for mathematical operators specifically"""
        math_operators = set("+-×÷=<>%")
        correct_ops = 0
        total_ops = 0
        
        for pred, ref in zip(predictions, references):
            pred_ops = [char for char in pred if char in math_operators]
            ref_ops = [char for char in ref if char in math_operators]
            
            min_len = min(len(pred_ops), len(ref_ops))
            for i in range(min_len):
                if pred_ops[i] == ref_ops[i]:
                    correct_ops += 1
            
            total_ops += max(len(pred_ops), len(ref_ops))
        
        return correct_ops / max(total_ops, 1)
    
    def _compute_character_accuracy(self, predictions: List[str], references: List[str]) -> float:
        """Compute character-level accuracy"""
        correct_chars = 0
        total_chars = 0
        
        for pred, ref in zip(predictions, references):
            min_len = min(len(pred), len(ref))
            correct_chars += sum(1 for i in range(min_len) if pred[i] == ref[i])
            total_chars += max(len(pred), len(ref))
        
        return correct_chars / max(total_chars, 1)
    
    def create_trainer(self, train_dataset, eval_dataset, data_collator) -> Trainer:
        """Create trainer with custom callbacks"""
        
        training_args = self.create_training_arguments()
        
        # Custom callbacks
        callbacks = [
            EarlyStoppingCallback(
                early_stopping_patience=self.config.early_stopping_patience,
                early_stopping_threshold=self.config.early_stopping_threshold
            ),
            OperatorFocusedCallback()
        ]
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.processor.feature_extractor,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks,
        )
        
        return self.trainer
    
    def train(self, train_dataset, eval_dataset, data_collator):
        """Train the model"""
        logger.info("Starting training...")
        
        if self.model is None:
            self.prepare_model()
        
        if self.trainer is None:
            self.create_trainer(train_dataset, eval_dataset, data_collator)
        
        # Train
        train_result = self.trainer.train()
        
        # Save the final model
        self.trainer.save_model()
        self.processor.save_pretrained(self.config.output_dir)
        
        # Log final metrics
        logger.info(f"Training completed. Final metrics: {train_result.metrics}")
        
        return train_result
    
    def eval_on_test_set(self, test_dataset) -> Dict[str, float]:
        """Evaluate on test set with detailed metrics"""
        logger.info("Evaluating on test set...")
        
        # Generate predictions
        predictions = []
        references = []
        
        for example in test_dataset:
            # Generate prediction
            inputs = {
                'input_features': example['input_features'].unsqueeze(0)
            }
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs['input_features'],
                    max_length=self.config.generation_max_length,
                    num_beams=self.config.generation_num_beams,
                    do_sample=False
                )
            
            prediction = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            
            predictions.append(prediction)
            references.append(example['text'])
        
        # Compute comprehensive metrics
        metrics = self.compute_metrics((
            self.processor.tokenizer(predictions, return_tensors="pt", padding=True).input_ids,
            self.processor.tokenizer(references, return_tensors="pt", padding=True).input_ids
        ))
        
        logger.info(f"Test set evaluation: {metrics}")
        return metrics

class OperatorFocusedCallback(TrainerCallback):
    """Custom callback for operator-focused training"""
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Log additional operator-specific metrics"""
        if logs and "eval_symbol_accuracy" in logs:
            wandb.log({
                "symbol_accuracy": logs["eval_symbol_accuracy"],
                "operator_accuracy": logs.get("eval_operator_accuracy", 0),
                "combined_score": logs.get("eval_combined_score", 0)
            })
    
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """Additional evaluation logic if needed"""
        pass 