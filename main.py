"""
Speech-to-Symbol Conversion System
Main entry point for training, evaluation, and inference

This system improves speech-to-text accuracy by correctly converting 
spoken operators and punctuation terms into their corresponding symbols.

Usage:
    python main.py train --dataset_percentage 0.01 --max_steps 2000
    python main.py evaluate --model_path ./results
    python main.py demo --text "two plus three equals five"
    python main.py audio --file path/to/audio.wav
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the speech2symbol package to the path
sys.path.append(str(Path(__file__).parent / "speech2symbol"))

from speech2symbol.scripts.train import main as train_main
from speech2symbol.scripts.evaluate import main as evaluate_main
from speech2symbol.postprocessing.symbol_converter import ComprehensiveSymbolConverter

try:
    from speech2symbol.pipeline.audio_processor import Speech2SymbolPipeline
except ImportError:
    Speech2SymbolPipeline = None  # Will be handled gracefully

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_conversion(text: str, model_path: str | None = None):
    """Demonstrate the conversion system with text input"""
    logger.info("=== Speech-to-Symbol Conversion Demo ===")
    
    # Initialize the post-processing converter
    converter = ComprehensiveSymbolConverter()
    
    logger.info(f"Input text: '{text}'")
    
    # Convert the text
    converted_text, metadata = converter.convert_text(text)
    
    logger.info(f"Converted text: '{converted_text}'")
    
    if metadata['conversions']:
        logger.info("Conversions made:")
        for conv in metadata['conversions']:
            logger.info(f"  '{conv['original']}' -> '{conv['converted']}' (priority: {conv['priority']})")
    else:
        logger.info("No conversions were made.")
    
    return converted_text, metadata

def process_audio_file(audio_path: str, model_path: str | None = None, confidence: float = 0.7):
    """Process audio file through complete pipeline"""
    logger.info("=== Complete Audio Processing Pipeline ===")
    
    try:
        # Create pipeline
        pipeline = Speech2SymbolPipeline(trained_model_path=model_path)
        
        # Process audio
        result = pipeline.process_audio_complete(audio_path, confidence)
        
        if result["status"] == "success":
            print(f"\n🎤 Audio Input: {audio_path}")
            print(f"📝 Raw ASR Output: '{result['raw_transcription']}'")
            print(f"✨ Final Output: '{result['final_output']}'")
            print(f"🔄 Conversions Made: {result['total_conversions']}")
            
            if result['conversions_made']:
                print("\n📋 Detailed Conversions:")
                for conv in result['conversions_made']:
                    print(f"  '{conv['original']}' → '{conv['converted']}' (priority: {conv['priority']})")
            
            if result['total_conversions'] > 0:
                print(f"📊 Total Conversions: {result['total_conversions']}")
                
        else:
            logger.error(f"Processing failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Audio processing failed: {str(e)}")

def show_examples():
    """Show example conversions"""
    examples = [
        "two plus three equals five",
        "ten minus four is six", 
        "five times seven equals thirty five",
        "twenty divided by four is five",
        "x is greater than zero",
        "the result is fifty percent",
        "it costs five dollars and twenty cents",
        "send email to john at company dot com",
        "add a comma after each item",
        "end the sentence with a period",
        "is this correct question mark",
        "that's amazing exclamation mark",
    ]
    
    print("\n=== Example Conversions ===")
    converter = ComprehensiveSymbolConverter()
    
    for example in examples:
        converted, _ = converter.convert_text(example)
        print(f"'{example}' -> '{converted}'")

def test_pipeline():
    """Test complete pipeline with sample data"""
    logger.info("=== Testing Complete Pipeline ===")
    
    # Test text samples
    test_samples = [
        {"text": "two plus three equals five", "expected": "two + three = five"},
        {"text": "ten percent of fifty", "expected": "ten % of fifty"},
        {"text": "send email to john at company dot com", "expected": "send email to john at company. com"},
        {"text": "is this correct question mark", "expected": "is this correct?"}
    ]
    
    pipeline = Speech2SymbolPipeline()
    
    print("\n🧪 Testing Text Processing:")
    for i, sample in enumerate(test_samples, 1):
        result = pipeline.process_text_only(sample["text"])
        
        correct = result["final_output"] == sample["expected"]
        status = "✅ PASS" if correct else "❌ FAIL"
        
        print(f"\nTest {i}: {status}")
        print(f"  Input: '{sample['text']}'")
        print(f"  Expected: '{sample['expected']}'")
        print(f"  Got: '{result['final_output']}'")
        print(f"  Conversions: {result['total_conversions']}")

def create_sample_audio():
    """Create instructions for creating sample audio files"""
    print("\n🎵 To test audio processing, create audio files with content like:")
    print("  - 'Two plus three equals five'")
    print("  - 'The result is fifty percent'") 
    print("  - 'Send email to john at company dot com'")
    print("\nSave as .wav files and use: python main.py audio --file your_audio.wav")
    print("\n💡 You can use text-to-speech tools or record yourself speaking these phrases.")

def main():
    parser = argparse.ArgumentParser(
        description="Speech-to-Symbol Conversion System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model on 1% of dataset
  python main.py train --dataset_percentage 0.01 --max_steps 2000
  
  # Evaluate trained model
  python main.py evaluate --model_path ./results --use_postprocessing
  
  # Demo text conversion
  python main.py demo --text "two plus three equals five"
  
  # Process audio file (complete pipeline)
  python main.py audio --file audio.wav --model ./results
  
  # Show example conversions
  python main.py examples
  
  # Test complete pipeline
  python main.py test
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument("--model_name", type=str, default="openai/whisper-small",
                             help="Base model to fine-tune")
    train_parser.add_argument("--dataset_percentage", type=float, default=0.01,
                             help="Percentage of dataset to use")
    train_parser.add_argument("--output_dir", type=str, default="./results",
                             help="Output directory")
    train_parser.add_argument("--learning_rate", type=float, default=1e-5,
                             help="Learning rate")
    train_parser.add_argument("--max_steps", type=int, default=2000,
                             help="Maximum training steps")
    train_parser.add_argument("--batch_size", type=int, default=4,
                             help="Training batch size")
    train_parser.add_argument("--test_postprocessing", action="store_true",
                             help="Test post-processing during evaluation")
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model')
    eval_parser.add_argument("--model_path", type=str, required=True,
                            help="Path to trained model")
    eval_parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                            help="Output directory for results")
    eval_parser.add_argument("--sample_size", type=int,
                            help="Number of samples to evaluate")
    eval_parser.add_argument("--use_postprocessing", action="store_true",
                            help="Use post-processing")
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Demo the conversion system')
    demo_parser.add_argument("--text", type=str, required=True,
                            help="Text to convert")
    demo_parser.add_argument("--model_path", type=str,
                            help="Path to trained model (optional)")
    
    # Audio processing command  
    audio_parser = subparsers.add_parser('audio', help='Process audio file')
    audio_parser.add_argument("--file", type=str, required=True,
                             help="Audio file path")
    audio_parser.add_argument("--model", type=str,
                             help="Path to trained model (optional)")
    audio_parser.add_argument("--confidence", type=float, default=0.7,
                             help="Confidence threshold for conversion")
    
    # Examples command
    subparsers.add_parser('examples', help='Show example conversions')
    
    # Test command
    subparsers.add_parser('test', help='Test complete pipeline')
    
    # Sample audio command
    subparsers.add_parser('sample-audio', help='Instructions for creating sample audio')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        logger.info("Starting training...")
        # Prepare arguments for training script
        train_args = [
            '--model_name', args.model_name,
            '--dataset_percentage', str(args.dataset_percentage),
            '--output_dir', args.output_dir,
            '--learning_rate', str(args.learning_rate),
            '--max_steps', str(args.max_steps),
            '--batch_size', str(args.batch_size),
            '--operator_focus'  # Always use operator focus
        ]
        
        if args.test_postprocessing:
            train_args.append('--test_postprocessing')
        
        # Override sys.argv for the training script
        original_argv = sys.argv
        sys.argv = ['train.py'] + train_args
        
        try:
            train_main()
        finally:
            sys.argv = original_argv
    
    elif args.command == 'evaluate':
        logger.info("Starting evaluation...")
        # Prepare arguments for evaluation script
        eval_args = [
            '--model_path', args.model_path,
            '--output_dir', args.output_dir
        ]
        
        if args.sample_size:
            eval_args.extend(['--sample_size', str(args.sample_size)])
        
        if args.use_postprocessing:
            eval_args.append('--use_postprocessing')
        
        # Override sys.argv for the evaluation script
        original_argv = sys.argv
        sys.argv = ['evaluate.py'] + eval_args
        
        try:
            evaluate_main()
        finally:
            sys.argv = original_argv
    
    elif args.command == 'demo':
        demo_conversion(args.text, args.model_path)
    
    elif args.command == 'audio':
        process_audio_file(args.file, args.model, args.confidence)
    
    elif args.command == 'examples':
        show_examples()
    
    elif args.command == 'test':
        test_pipeline()
    
    elif args.command == 'sample-audio':
        create_sample_audio()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
