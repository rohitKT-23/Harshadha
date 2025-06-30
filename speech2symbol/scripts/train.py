"""
Main training script for speech-to-symbol conversion
Integrates dataset loading, model training, and evaluation
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset_loader import OperatorDatasetLoader
from models.whisper_trainer import OperatorWhisperTrainer, OperatorTrainingConfig
from postprocessing.symbol_converter import ContextAwareSymbolConverter

import torch
import wandb
from datasets import DatasetDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train speech-to-symbol conversion model")
    
    # Model and dataset arguments
    parser.add_argument("--model_name", type=str, default="openai/whisper-small",
                       help="Base model to fine-tune")
    parser.add_argument("--dataset_percentage", type=float, default=0.01,
                       help="Percentage of dataset to use (0.01 = 1%)")
    parser.add_argument("--operator_focus", action="store_true", default=True,
                       help="Filter for operator-heavy samples")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./results",
                       help="Output directory for model and results")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate for training")
    parser.add_argument("--max_steps", type=int, default=2000,
                       help="Maximum training steps")
    parser.add_argument("--eval_steps", type=int, default=200,
                       help="Evaluation frequency")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--freeze_encoder", action="store_true", default=True,
                       help="Freeze encoder parameters")
    
    # Evaluation arguments
    parser.add_argument("--test_postprocessing", action="store_true",
                       help="Test post-processing on evaluation set")
    parser.add_argument("--confidence_threshold", type=float, default=0.7,
                       help="Confidence threshold for post-processing")
    
    # Logging and monitoring
    parser.add_argument("--wandb_project", type=str, default="speech2symbol",
                       help="Weights & Biases project name")
    parser.add_argument("--run_name", type=str, default=None,
                       help="Run name for logging")
    
    return parser.parse_args()

def setup_logging_and_monitoring(args):
    """Setup logging and monitoring"""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        config=vars(args)
    )
    
    logger.info(f"Starting training with arguments: {vars(args)}")

def load_and_prepare_datasets(args):
    """Load and prepare training datasets"""
    logger.info("Loading and preparing datasets...")
    
    # Initialize dataset loader
    dataset_loader = OperatorDatasetLoader(
        model_name=args.model_name,
        operator_focus=args.operator_focus
    )
    
    # Load datasets
    logger.info("Loading training set...")
    train_dataset = dataset_loader.load_dataset(
        subset_percentage=args.dataset_percentage,
        split="train"
    )
    
    logger.info("Loading validation set...")
    eval_dataset = dataset_loader.load_dataset(
        subset_percentage=args.dataset_percentage * 0.5,  # Use half for eval
        split="validation"
    )
    
    # Preprocess datasets
    train_dataset = dataset_loader.preprocess_dataset(train_dataset)
    eval_dataset = dataset_loader.preprocess_dataset(eval_dataset)
    
    # Get data collator
    data_collator = dataset_loader.get_data_collator()
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Evaluation samples: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset, data_collator, dataset_loader

def create_and_train_model(args, train_dataset, eval_dataset, data_collator):
    """Create and train the model"""
    logger.info("Creating and configuring model...")
    
    # Create training configuration
    config = OperatorTrainingConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size
    )
    
    # Initialize trainer
    trainer = OperatorWhisperTrainer(config)
    
    # Prepare model
    model = trainer.prepare_model(freeze_encoder=args.freeze_encoder)
    
    # Create trainer
    trainer.create_trainer(train_dataset, eval_dataset, data_collator)
    
    # Start training
    logger.info("Starting training...")
    train_result = trainer.train(train_dataset, eval_dataset, data_collator)
    
    return trainer, train_result

def evaluate_with_postprocessing(args, trainer, eval_dataset, dataset_loader):
    """Evaluate model with post-processing"""
    if not args.test_postprocessing:
        return {}
    
    logger.info("Evaluating with post-processing...")
    
    # Initialize post-processor
    converter = ContextAwareSymbolConverter(use_spacy=True)
    
    # Generate predictions for a subset
    sample_size = min(100, len(eval_dataset))
    eval_samples = eval_dataset.select(range(sample_size))
    
    predictions = []
    references = []
    postprocessed_predictions = []
    
    for example in eval_samples:
        # Generate raw prediction
        inputs = {
            'input_features': example['input_features'].unsqueeze(0)
        }
        
        with torch.no_grad():
            generated_ids = trainer.model.generate(
                inputs['input_features'],
                max_length=225,
                num_beams=1,
                do_sample=False
            )
        
        raw_prediction = trainer.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        
        # Apply post-processing
        processed_prediction, metadata = converter.convert_text(
            raw_prediction, confidence_threshold=args.confidence_threshold
        )
        
        predictions.append(raw_prediction)
        postprocessed_predictions.append(processed_prediction)
        references.append(example['text'])
    
    # Compute metrics for both raw and post-processed predictions
    raw_metrics = trainer.compute_metrics((
        trainer.processor.tokenizer(predictions, return_tensors="pt", padding=True).input_ids,
        trainer.processor.tokenizer(references, return_tensors="pt", padding=True).input_ids
    ))
    
    processed_metrics = trainer.compute_metrics((
        trainer.processor.tokenizer(postprocessed_predictions, return_tensors="pt", padding=True).input_ids,
        trainer.processor.tokenizer(references, return_tensors="pt", padding=True).input_ids
    ))
    
    # Get post-processing statistics
    pp_stats = converter.get_conversion_statistics(predictions)
    
    postprocessing_results = {
        'raw_metrics': raw_metrics,
        'processed_metrics': processed_metrics,
        'postprocessing_stats': pp_stats,
        'improvement': {
            'symbol_accuracy': processed_metrics['symbol_accuracy'] - raw_metrics['symbol_accuracy'],
            'operator_accuracy': processed_metrics['operator_accuracy'] - raw_metrics['operator_accuracy'],
            'wer_improvement': raw_metrics['wer'] - processed_metrics['wer']
        }
    }
    
    logger.info("Post-processing evaluation results:")
    logger.info(f"Symbol accuracy improvement: {postprocessing_results['improvement']['symbol_accuracy']:.3f}")
    logger.info(f"Operator accuracy improvement: {postprocessing_results['improvement']['operator_accuracy']:.3f}")
    logger.info(f"WER improvement: {postprocessing_results['improvement']['wer_improvement']:.3f}")
    
    # Log to wandb
    wandb.log({
        "postprocessing/symbol_accuracy_improvement": postprocessing_results['improvement']['symbol_accuracy'],
        "postprocessing/operator_accuracy_improvement": postprocessing_results['improvement']['operator_accuracy'],
        "postprocessing/wer_improvement": postprocessing_results['improvement']['wer_improvement'],
        "postprocessing/conversion_rate": pp_stats['conversion_rate'],
        "postprocessing/average_confidence": pp_stats['average_confidence']
    })
    
    return postprocessing_results

def create_evaluation_report(args, train_result, postprocessing_results):
    """Create comprehensive evaluation report"""
    logger.info("Creating evaluation report...")
    
    report = {
        'training_config': vars(args),
        'training_results': train_result.metrics if train_result else {},
        'postprocessing_results': postprocessing_results,
        'model_path': args.output_dir
    }
    
    # Save report
    import json
    report_path = os.path.join(args.output_dir, 'evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Evaluation report saved to: {report_path}")
    
    return report

def main():
    """Main training function"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging and monitoring
    setup_logging_and_monitoring(args)
    
    try:
        # Load and prepare datasets
        train_dataset, eval_dataset, data_collator, dataset_loader = load_and_prepare_datasets(args)
        
        # Create and train model
        trainer, train_result = create_and_train_model(args, train_dataset, eval_dataset, data_collator)
        
        # Evaluate with post-processing
        postprocessing_results = evaluate_with_postprocessing(args, trainer, eval_dataset, dataset_loader)
        
        # Create evaluation report
        report = create_evaluation_report(args, train_result, postprocessing_results)
        
        logger.info("Training completed successfully!")
        
        # Final logging
        wandb.log({
            "training/final_eval_loss": train_result.metrics.get('eval_loss', 0) if train_result else 0,
            "training/final_symbol_accuracy": train_result.metrics.get('eval_symbol_accuracy', 0) if train_result else 0,
        })
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise
    
    finally:
        wandb.finish()

if __name__ == "__main__":
    main() 