"""
Dataset loader and preprocessor for LibriHeavy dataset
Handles loading, filtering, and preprocessing for operator conversion training
"""

import os
import re
import torch
import torchaudio
import librosa
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from transformers import WhisperProcessor, WhisperFeatureExtractor
from typing import Dict, List, Optional, Tuple, Any, cast
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OperatorDatasetLoader:
    """Loads and preprocesses LibriHeavy dataset for operator conversion"""
    
    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        sample_rate: int = 16000,
        max_duration: float = 30.0,
        min_duration: float = 1.0,
        operator_focus: bool = True
    ):
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.operator_focus = operator_focus
        
        # Initialize processor
        self.processor: WhisperProcessor = WhisperProcessor.from_pretrained(model_name)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        
        # Define operator mappings
        self.operator_mappings = {
            # Mathematical operators
            "plus": "+", "add": "+", "added to": "+",
            "minus": "-", "subtract": "-", "subtracted from": "-", "negative": "-",
            "times": "×", "multiply": "×", "multiplied by": "×",
            "divide": "÷", "divided by": "÷", "over": "/",
            "equals": "=", "equal to": "=", "is equal to": "=",
            "greater than": ">", "less than": "<",
            "greater than or equal": "≥", "less than or equal": "≤",
            "percent": "%", "percentage": "%",
            
            # Punctuation
            "comma": ",", "period": ".", "dot": ".",
            "question mark": "?", "exclamation": "!", "exclamation mark": "!",
            "semicolon": ";", "colon": ":",
            "apostrophe": "'", "quote": '"', "quotation mark": '"',
            "left parenthesis": "(", "right parenthesis": ")",
            "left bracket": "[", "right bracket": "]",
            "left brace": "{", "right brace": "}",
            
            # Currency and symbols
            "dollar": "$", "cent": "¢", "pound": "£", "euro": "€",
            "at sign": "@", "hashtag": "#", "ampersand": "&",
            "asterisk": "*", "underscore": "_", "hyphen": "-",
        }
        
        # Create reverse mapping for data augmentation
        self.reverse_mappings = {v: k for k, v in self.operator_mappings.items()}
        
        # Operator-heavy phrases for filtering
        self.operator_patterns = [
            r'\b(plus|minus|times|divide|equals|greater|less)\b',
            r'\b(comma|period|question mark|exclamation)\b',
            r'\b(dollar|percent|at sign|hashtag)\b',
            r'[+\-×÷=<>%$@#&*()[\]{}]'
        ]
    
    def load_dataset(
        self, 
        subset_percentage: float = 1.0,
        split: str = "train",
        cache_dir: Optional[str] = None
    ) -> Dataset:
        """Load LibriHeavy dataset with optional filtering"""
        logger.info(f"Loading LibriHeavy dataset, {split} split")
        
        # Load the dataset
        dataset = load_dataset(
            "pkufool/libriheavy_long", 
            split=split, 
            cache_dir=cache_dir,
            streaming=False
        )
        
        # Apply subset if specified
        if subset_percentage < 1.0:
            num_samples = int(len(dataset) * subset_percentage)
            dataset = dataset.select(range(num_samples))
            logger.info(f"Using {num_samples} samples ({subset_percentage*100}% of dataset)")
        
        # Filter for operator-heavy samples if specified
        if self.operator_focus:
            dataset = dataset.filter(
                self._contains_operators,
                desc="Filtering for operator-heavy samples"
            )
            logger.info(f"Filtered to {len(dataset)} operator-heavy samples")
        
        return dataset
    
    def _contains_operators(self, example: Dict) -> bool:
        """Check if text contains operators or punctuation terms"""
        text = example.get('text', '').lower()
        
        # Check for spoken operator terms
        for pattern in self.operator_patterns:
            if re.search(pattern, text):
                return True
        
        # Check for actual symbols (indicating good training targets)
        symbol_count = sum(1 for char in text if char in "+-×÷=<>%$@#&*()[]{},.!?;:")
        return symbol_count >= 2  # At least 2 symbols
    
    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """Preprocess the dataset for training"""
        logger.info("Preprocessing dataset...")
        
        # Apply preprocessing function
        dataset = dataset.map(
            self._preprocess_function,
            remove_columns=dataset.column_names,
            desc="Preprocessing"
        )
        
        # Filter out samples that are too long or short
        dataset = dataset.filter(
            lambda x: self.min_duration <= x['duration'] <= self.max_duration,
            desc="Filtering by duration"
        )
        
        logger.info(f"Preprocessed dataset size: {len(dataset)}")
        return dataset
    
    def _preprocess_function(self, example: Dict) -> Dict:
        """Preprocess individual sample"""
        # Load and process audio
        audio_array = example['audio']['array']
        sampling_rate = example['audio']['sampling_rate']
        
        # Resample if necessary
        if sampling_rate != self.sample_rate:
            audio_array = librosa.resample(
                audio_array, 
                orig_sr=sampling_rate, 
                target_sr=self.sample_rate
            )
        
        # Calculate duration
        duration = len(audio_array) / self.sample_rate
        
        # Extract features
        inputs = self.feature_extractor(
            audio_array, 
            sampling_rate=self.sample_rate, 
            return_tensors="pt"
        )
        
        # Process text
        text = example['text']
        original_text = text  # Keep original for comparison
        
        # Create augmented version with spoken operators
        augmented_text = self._create_augmented_text(text)
        
        # Tokenize both versions  
        original_labels = self.processor.tokenizer(  # type: ignore
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )["input_ids"].squeeze()
        
        augmented_labels = self.processor.tokenizer(  # type: ignore
            augmented_text,
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )["input_ids"].squeeze()
        
        return {
            'input_features': inputs.input_features.squeeze(),
            'labels': original_labels,
            'text': text,
            'original_text': original_text,
            'augmented_text': augmented_text,
            'augmented_labels': augmented_labels,
            'duration': duration
        }
    
    def _create_augmented_text(self, text: str) -> str:
        """Create augmented version by replacing some symbols with spoken forms"""
        augmented = text
        
        # Randomly replace some symbols with spoken equivalents
        # This creates training pairs where model learns both directions
        import random
        
        for symbol, spoken in self.reverse_mappings.items():
            if symbol in augmented and random.random() < 0.3:  # 30% chance
                # Replace some occurrences (not all) with spoken form
                occurrences = [m.start() for m in re.finditer(re.escape(symbol), augmented)]
                if occurrences:
                    # Replace random subset of occurrences
                    to_replace = random.sample(
                        occurrences, 
                        max(1, len(occurrences) // 2)
                    )
                    # Replace from right to left to maintain indices
                    for pos in sorted(to_replace, reverse=True):
                        augmented = (augmented[:pos] + 
                                   f" {spoken} " + 
                                   augmented[pos + len(symbol):])
        
        return augmented.strip()
    
    def create_operator_dataset(
        self, 
        additional_phrases: Optional[List[str]] = None
    ) -> Dataset:
        """Create custom dataset focusing on operator conversion"""
        
        # Base operator phrases
        operator_phrases = [
            "two plus three equals five",
            "ten minus four is six", 
            "five times seven equals thirty five",
            "twenty divided by four is five",
            "x is greater than zero",
            "y is less than or equal to ten",
            "the result is fifty percent",
            "it costs five dollars and twenty cents",
            "send email to john at company dot com",
            "press the hashtag key to continue",
            "use parentheses to group the terms",
            "add a comma after each item",
            "end the sentence with a period",
            "is this correct question mark",
            "that's amazing exclamation mark",
        ]
        
        if additional_phrases:
            operator_phrases.extend(additional_phrases)
        
        # Convert to dataset format (would need TTS for audio)
        # For now, return text-only for evaluation
        data = []
        for phrase in operator_phrases:
            # Create both versions
            spoken_version = phrase
            symbol_version = self._convert_spoken_to_symbols(phrase)
            
            data.append({
                'text': symbol_version,
                'spoken_text': spoken_version,
                'has_operators': True
            })
        
        return Dataset.from_list(data)
    
    def _convert_spoken_to_symbols(self, text: str) -> str:
        """Convert spoken operators to symbols"""
        result = text
        
        # Sort by length (longer first) to handle overlapping patterns
        sorted_mappings = sorted(
            self.operator_mappings.items(), 
            key=lambda x: len(x[0]), 
            reverse=True
        )
        
        for spoken, symbol in sorted_mappings:
            # Use word boundaries for better matching
            pattern = r'\b' + re.escape(spoken) + r'\b'
            result = re.sub(pattern, symbol, result, flags=re.IGNORECASE)
        
        # Clean up multiple spaces
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result
    
    def get_data_collator(self):
        """Return data collator for training"""
        from transformers import DataCollatorForSeq2Seq
        
        return DataCollatorForSeq2Seq(
            self.processor.tokenizer,  # type: ignore
            model=None,  # Will be set during training
            padding=True,
            return_tensors="pt"
        ) 