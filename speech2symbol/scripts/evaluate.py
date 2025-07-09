"""
Comprehensive evaluation script for speech-to-symbol conversion
Includes symbol-level accuracy, operator-specific metrics, and error analysis
"""

import os
import sys
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.whisper_trainer import OperatorWhisperTrainer, OperatorTrainingConfig
from postprocessing.symbol_converter import ComprehensiveSymbolConverter
from data.dataset_loader import OperatorDatasetLoader

import torch
import evaluate
from jiwer import wer, cer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveEvaluator:
    """Comprehensive evaluator for speech-to-symbol conversion"""
    
    def __init__(self, model_path: str, use_postprocessing: bool = True):
        self.model_path = model_path
        self.use_postprocessing = use_postprocessing
        
        # Load model and processor
        config = OperatorTrainingConfig(output_dir=model_path)
        self.trainer = OperatorWhisperTrainer(config)
        self.trainer.prepare_model()
        
        # Load trained weights
        model_file = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(model_file):
            self.trainer.model.load_state_dict(torch.load(model_file, map_location='cpu'))
            logger.info(f"Loaded model from {model_file}")
        
        # Initialize post-processor
        if use_postprocessing:
            self.converter = ComprehensiveSymbolConverter()
        
        # Initialize metrics
        self.wer_metric = evaluate.load("wer")
        self.bleu_metric = evaluate.load("bleu")
        
        # Operator categories for detailed analysis
        self.operator_categories = {
            'mathematical': ['+', '-', '×', '÷', '=', '<', '>', '≥', '≤', '%'],
            'punctuation': [',', '.', '!', '?', ';', ':', "'", '"'],
            'grouping': ['(', ')', '[', ']', '{', '}'],
            'currency': ['$', '¢', '£', '€'],
            'special': ['@', '#', '&', '*', '_']
        }
    
    def evaluate_dataset(self, test_dataset, sample_size: int = None) -> Dict:
        """Evaluate model on test dataset"""
        if sample_size and sample_size < len(test_dataset):
            test_dataset = test_dataset.select(range(sample_size))
        
        logger.info(f"Evaluating on {len(test_dataset)} samples...")
        
        predictions = []
        references = []
        raw_predictions = []
        postprocessed_predictions = []
        
        for i, example in enumerate(test_dataset):
            if i % 50 == 0:
                logger.info(f"Processing sample {i}/{len(test_dataset)}")
            
            # Generate prediction
            inputs = {'input_features': example['input_features'].unsqueeze(0)}
            
            with torch.no_grad():
                generated_ids = self.trainer.model.generate(
                    inputs['input_features'],
                    max_length=225,
                    num_beams=1,
                    do_sample=False
                )
            
            raw_pred = self.trainer.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            
            raw_predictions.append(raw_pred)
            references.append(example['text'])
            
            # Apply post-processing if enabled
            if self.use_postprocessing:
                processed_pred, _ = self.converter.convert_text(raw_pred)
                postprocessed_predictions.append(processed_pred)
                predictions.append(processed_pred)
            else:
                predictions.append(raw_pred)
        
        # Compute comprehensive metrics
        results = self._compute_comprehensive_metrics(
            predictions, references, raw_predictions, postprocessed_predictions
        )
        
        return results
    
    def _compute_comprehensive_metrics(self, predictions: List[str], references: List[str],
                                     raw_predictions: List[str], postprocessed_predictions: List[str]) -> Dict:
        """Compute comprehensive evaluation metrics"""
        results = {}
        
        # Basic ASR metrics
        results['wer'] = wer(references, predictions)
        results['cer'] = cer(references, predictions)
        
        # BLEU score
        bleu = self.bleu_metric.compute(
            predictions=predictions,
            references=[[ref] for ref in references]
        )
        results['bleu'] = bleu['bleu']
        
        # Symbol-level accuracy
        results['symbol_accuracy'] = self._compute_symbol_accuracy(predictions, references)
        
        # Category-specific accuracies
        for category, symbols in self.operator_categories.items():
            results[f'{category}_accuracy'] = self._compute_category_accuracy(
                predictions, references, symbols
            )
        
        # Position-based accuracy (symbols at different positions)
        results['position_accuracy'] = self._compute_position_accuracy(predictions, references)
        
        # Context-aware accuracy
        results['context_accuracy'] = self._compute_context_accuracy(predictions, references)
        
        # Post-processing improvement (if applicable)
        if self.use_postprocessing and postprocessed_predictions:
            results['postprocessing_improvement'] = self._compute_postprocessing_improvement(
                raw_predictions, postprocessed_predictions, references
            )
        
        # Error analysis
        results['error_analysis'] = self._analyze_errors(predictions, references)
        
        return results
    
    def _compute_symbol_accuracy(self, predictions: List[str], references: List[str]) -> float:
        """Compute symbol-level accuracy"""
        all_symbols = set(''.join(self.operator_categories.values()))
        correct = 0
        total = 0
        
        for pred, ref in zip(predictions, references):
            pred_symbols = [c for c in pred if c in all_symbols]
            ref_symbols = [c for c in ref if c in all_symbols]
            
            # Alignment-based comparison
            for i in range(min(len(pred_symbols), len(ref_symbols))):
                if pred_symbols[i] == ref_symbols[i]:
                    correct += 1
                total += 1
            
            # Account for length differences
            total += abs(len(pred_symbols) - len(ref_symbols))
        
        return correct / max(total, 1)
    
    def _compute_category_accuracy(self, predictions: List[str], references: List[str], 
                                 category_symbols: List[str]) -> Dict:
        """Compute accuracy for specific symbol category"""
        category_set = set(category_symbols)
        correct = 0
        total = 0
        precision_tp = 0
        precision_fp = 0
        recall_fn = 0
        
        for pred, ref in zip(predictions, references):
            pred_symbols = [c for c in pred if c in category_set]
            ref_symbols = [c for c in ref if c in category_set]
            
            # True positives (correct symbols)
            for i in range(min(len(pred_symbols), len(ref_symbols))):
                if pred_symbols[i] == ref_symbols[i]:
                    correct += 1
                    precision_tp += 1
                else:
                    precision_fp += 1
                total += 1
            
            # False positives (extra predicted symbols)
            if len(pred_symbols) > len(ref_symbols):
                precision_fp += len(pred_symbols) - len(ref_symbols)
                total += len(pred_symbols) - len(ref_symbols)
            
            # False negatives (missed symbols)
            if len(ref_symbols) > len(pred_symbols):
                recall_fn += len(ref_symbols) - len(pred_symbols)
                total += len(ref_symbols) - len(pred_symbols)
        
        accuracy = correct / max(total, 1)
        precision = precision_tp / max(precision_tp + precision_fp, 1)
        recall = precision_tp / max(precision_tp + recall_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'total_symbols': total
        }
    
    def _compute_position_accuracy(self, predictions: List[str], references: List[str]) -> Dict:
        """Compute accuracy based on symbol position in text"""
        position_stats = {'beginning': {'correct': 0, 'total': 0},
                         'middle': {'correct': 0, 'total': 0},
                         'end': {'correct': 0, 'total': 0}}
        
        all_symbols = set(''.join(self.operator_categories.values()))
        
        for pred, ref in zip(predictions, references):
            ref_len = len(ref)
            
            for i, char in enumerate(ref):
                if char in all_symbols:
                    # Determine position
                    if i < ref_len * 0.33:
                        position = 'beginning'
                    elif i > ref_len * 0.67:
                        position = 'end'
                    else:
                        position = 'middle'
                    
                    position_stats[position]['total'] += 1
                    
                    # Check if correctly predicted
                    if i < len(pred) and pred[i] == char:
                        position_stats[position]['correct'] += 1
        
        # Calculate accuracies
        for position in position_stats:
            total = position_stats[position]['total']
            if total > 0:
                position_stats[position]['accuracy'] = position_stats[position]['correct'] / total
            else:
                position_stats[position]['accuracy'] = 0.0
        
        return position_stats
    
    def _compute_context_accuracy(self, predictions: List[str], references: List[str]) -> Dict:
        """Compute accuracy based on surrounding context"""
        context_stats = {
            'with_numbers': {'correct': 0, 'total': 0},
            'without_numbers': {'correct': 0, 'total': 0},
            'sentence_start': {'correct': 0, 'total': 0},
            'sentence_end': {'correct': 0, 'total': 0}
        }
        
        # Implementation would analyze context around symbols
        # This is a simplified version
        import re
        
        for pred, ref in zip(predictions, references):
            has_numbers = bool(re.search(r'\d', ref))
            context_type = 'with_numbers' if has_numbers else 'without_numbers'
            
            # Simple symbol matching for context analysis
            all_symbols = set(''.join(self.operator_categories.values()))
            ref_symbols = [c for c in ref if c in all_symbols]
            pred_symbols = [c for c in pred if c in all_symbols]
            
            for i in range(min(len(ref_symbols), len(pred_symbols))):
                context_stats[context_type]['total'] += 1
                if ref_symbols[i] == pred_symbols[i]:
                    context_stats[context_type]['correct'] += 1
        
        # Calculate accuracies
        for context in context_stats:
            total = context_stats[context]['total']
            if total > 0:
                context_stats[context]['accuracy'] = context_stats[context]['correct'] / total
            else:
                context_stats[context]['accuracy'] = 0.0
        
        return context_stats
    
    def _compute_postprocessing_improvement(self, raw_predictions: List[str], 
                                          postprocessed_predictions: List[str],
                                          references: List[str]) -> Dict:
        """Compute improvement from post-processing"""
        raw_wer = wer(references, raw_predictions)
        processed_wer = wer(references, postprocessed_predictions)
        
        raw_symbol_acc = self._compute_symbol_accuracy(raw_predictions, references)
        processed_symbol_acc = self._compute_symbol_accuracy(postprocessed_predictions, references)
        
        return {
            'wer_improvement': raw_wer - processed_wer,
            'symbol_accuracy_improvement': processed_symbol_acc - raw_symbol_acc,
            'relative_improvement': (processed_symbol_acc - raw_symbol_acc) / max(raw_symbol_acc, 1e-8)
        }
    
    def _analyze_errors(self, predictions: List[str], references: List[str]) -> Dict:
        """Analyze common error patterns"""
        error_analysis = {
            'substitution_errors': {},
            'insertion_errors': {},
            'deletion_errors': {},
            'common_mistakes': []
        }
        
        all_symbols = set(''.join(self.operator_categories.values()))
        
        for pred, ref in zip(predictions, references):
            pred_symbols = [c for c in pred if c in all_symbols]
            ref_symbols = [c for c in ref if c in all_symbols]
            
            # Simple error analysis (could be made more sophisticated)
            min_len = min(len(pred_symbols), len(ref_symbols))
            
            for i in range(min_len):
                if pred_symbols[i] != ref_symbols[i]:
                    error_key = f"{ref_symbols[i]}->{pred_symbols[i]}"
                    error_analysis['substitution_errors'][error_key] = error_analysis['substitution_errors'].get(error_key, 0) + 1
        
        return error_analysis
    
    def create_evaluation_report(self, results: Dict, output_dir: str):
        """Create comprehensive evaluation report with visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results to JSON
        report_path = os.path.join(output_dir, 'evaluation_results.json')
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create visualizations
        self._create_visualizations(results, output_dir)
        
        # Create summary report
        self._create_summary_report(results, output_dir)
        
        logger.info(f"Evaluation report saved to {output_dir}")
    
    def _create_visualizations(self, results: Dict, output_dir: str):
        """Create evaluation visualizations"""
        plt.style.use('seaborn-v0_8')
        
        # Category accuracies
        categories = []
        accuracies = []
        
        for key, value in results.items():
            if key.endswith('_accuracy') and isinstance(value, dict) and 'accuracy' in value:
                categories.append(key.replace('_accuracy', ''))
                accuracies.append(value['accuracy'])
        
        if categories:
            plt.figure(figsize=(10, 6))
            plt.bar(categories, accuracies)
            plt.title('Symbol Category Accuracies')
            plt.ylabel('Accuracy')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'category_accuracies.png'))
            plt.close()
        
        # Position accuracy
        if 'position_accuracy' in results:
            pos_data = results['position_accuracy']
            positions = list(pos_data.keys())
            pos_accuracies = [pos_data[pos]['accuracy'] for pos in positions]
            
            plt.figure(figsize=(8, 6))
            plt.bar(positions, pos_accuracies)
            plt.title('Symbol Accuracy by Position')
            plt.ylabel('Accuracy')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'position_accuracies.png'))
            plt.close()
    
    def _create_summary_report(self, results: Dict, output_dir: str):
        """Create summary text report"""
        report_lines = [
            "=== Speech-to-Symbol Conversion Evaluation Report ===\n",
            f"Overall WER: {results.get('wer', 0):.3f}",
            f"Overall CER: {results.get('cer', 0):.3f}",
            f"BLEU Score: {results.get('bleu', 0):.3f}",
            f"Symbol Accuracy: {results.get('symbol_accuracy', 0):.3f}\n",
            "Category-specific Results:",
        ]
        
        for category in self.operator_categories.keys():
            key = f"{category}_accuracy"
            if key in results and isinstance(results[key], dict):
                acc = results[key]['accuracy']
                prec = results[key]['precision']
                recall = results[key]['recall']
                f1 = results[key]['f1']
                report_lines.append(f"  {category.capitalize()}: Acc={acc:.3f}, P={prec:.3f}, R={recall:.3f}, F1={f1:.3f}")
        
        if 'postprocessing_improvement' in results:
            pp_imp = results['postprocessing_improvement']
            report_lines.extend([
                "\nPost-processing Improvements:",
                f"  WER Improvement: {pp_imp['wer_improvement']:.3f}",
                f"  Symbol Accuracy Improvement: {pp_imp['symbol_accuracy_improvement']:.3f}",
                f"  Relative Improvement: {pp_imp['relative_improvement']:.1%}"
            ])
        
        summary_path = os.path.join(output_dir, 'evaluation_summary.txt')
        with open(summary_path, 'w') as f:
            f.write('\n'.join(report_lines))

def main():
    parser = argparse.ArgumentParser(description="Evaluate speech-to-symbol conversion model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--test_data_path", type=str, help="Path to test dataset")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Output directory")
    parser.add_argument("--sample_size", type=int, help="Number of samples to evaluate")
    parser.add_argument("--use_postprocessing", action="store_true", help="Use post-processing")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(args.model_path, args.use_postprocessing)
    
    # Load test dataset
    dataset_loader = OperatorDatasetLoader()
    test_dataset = dataset_loader.load_dataset(split="test")
    test_dataset = dataset_loader.preprocess_dataset(test_dataset)
    
    # Run evaluation
    results = evaluator.evaluate_dataset(test_dataset, args.sample_size)
    
    # Create report
    evaluator.create_evaluation_report(results, args.output_dir)
    
    print(f"Evaluation completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 