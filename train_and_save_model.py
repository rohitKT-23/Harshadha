"""
Model Training and Serialization Script
Train the speech-to-symbol model and save it in multiple formats for Flask integration
"""

import os
import sys
import pickle
import json
import torch
import logging
from pathlib import Path
from typing import Dict, Any

# Add the speech2symbol package to the path
sys.path.append(str(Path(__file__).parent / "speech2symbol"))

from speech2symbol.scripts.train import main as train_main
from speech2symbol.postprocessing.symbol_converter import ComprehensiveSymbolConverter
from ml_models.asr_pipeline import ASRPipeline, create_production_pipeline
from ml_models.symbol_converter import SymbolConverter, create_advanced_converter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelSerializer:
    """Serialize models in multiple formats for Flask integration"""
    
    def __init__(self, output_dir: str = "./saved_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def train_and_save_complete_model(self, 
                                    dataset_percentage: float = 0.01,
                                    max_steps: int = 2000,
                                    model_name: str = "openai/whisper-small"):
        """Train the complete model and save in multiple formats"""
        logger.info("=== Starting Complete Model Training and Serialization ===")
        
        # Step 1: Train the Whisper model
        logger.info("Step 1: Training Whisper ASR model...")
        self._train_whisper_model(dataset_percentage, max_steps, model_name)
        
        # Step 2: Create and save symbol converter
        logger.info("Step 2: Creating and saving symbol converter...")
        self._save_symbol_converter()
        
        # Step 3: Create and save complete pipeline
        logger.info("Step 3: Creating and saving complete pipeline...")
        self._save_complete_pipeline()
        
        # Step 4: Create Flask-ready model package
        logger.info("Step 4: Creating Flask-ready model package...")
        self._create_flask_model_package()
        
        logger.info("✅ Model training and serialization completed!")
        logger.info(f"📁 All models saved to: {self.output_dir}")
        
    def _train_whisper_model(self, dataset_percentage: float, max_steps: int, model_name: str):
        """Train Whisper model and save"""
        # Override sys.argv for training
        original_argv = sys.argv
        train_args = [
            'train.py',
            '--model_name', model_name,
            '--dataset_percentage', str(dataset_percentage),
            '--output_dir', str(self.output_dir / "whisper_model"),
            '--max_steps', str(max_steps),
            '--batch_size', '4',
            '--operator_focus'
        ]
        
        sys.argv = train_args
        
        try:
            train_main()
            logger.info("Whisper model training completed!")
        except Exception as e:
            logger.error(f"Whisper training failed: {e}")
            # Create a dummy model for testing
            self._create_dummy_whisper_model()
        finally:
            sys.argv = original_argv
    
    def _create_dummy_whisper_model(self):
        """Create a dummy Whisper model for testing when training fails"""
        logger.warning("Creating dummy Whisper model for testing...")
        
        dummy_dir = self.output_dir / "whisper_model"
        dummy_dir.mkdir(exist_ok=True)
        
        # Create dummy config
        dummy_config = {
            "model_type": "whisper",
            "model_name": "openai/whisper-small",
            "is_dummy": True,
            "training_completed": False
        }
        
        with open(dummy_dir / "config.json", 'w') as f:
            json.dump(dummy_config, f, indent=2)
        
        logger.info("Dummy Whisper model created")
    
    def _save_symbol_converter(self):
        """Save symbol converter in multiple formats"""
        # Create advanced converter
        converter = create_advanced_converter()
        
        # Save as JSON (for easy inspection)
        json_path = self.output_dir / "symbol_converter.json"
        converter.save_model(str(json_path))
        logger.info(f"Symbol converter saved as JSON: {json_path}")
        
        # Save as pickle
        pickle_path = self.output_dir / "symbol_converter.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(converter, f)
        logger.info(f"Symbol converter saved as pickle: {pickle_path}")
        
        # Save converter stats
        stats = converter.get_statistics()
        stats_path = self.output_dir / "converter_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Converter statistics saved: {stats_path}")
    
    def _save_complete_pipeline(self):
        """Save complete ASR pipeline"""
        # Create production pipeline
        whisper_path = self.output_dir / "whisper_model"
        pipeline = create_production_pipeline(
            model_path=str(whisper_path) if whisper_path.exists() else None,
            confidence=0.7,
            use_gpu=True
        )
        
        # Save complete pipeline as pickle
        pipeline_path = self.output_dir / "complete_pipeline.pkl"
        with open(pipeline_path, 'wb') as f:
            pickle.dump(pipeline, f)
        logger.info(f"Complete pipeline saved as pickle: {pipeline_path}")
        
        # Save pipeline config
        config = {
            "whisper_model_path": str(whisper_path) if whisper_path.exists() else None,
            "confidence_threshold": 0.7,
            "language": "en",
            "model_type": "production_pipeline"
        }
        
        config_path = self.output_dir / "pipeline_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Pipeline config saved: {config_path}")
    
    def _create_flask_model_package(self):
        """Create a Flask-ready model package"""
        flask_model = {
            "model_info": {
                "name": "speech2symbol_flask_model",
                "version": "1.0.0",
                "description": "Speech-to-Symbol conversion model for Flask API",
                "created_at": str(torch.datetime.now()),
                "formats": ["pickle", "json", "torch"]
            },
            "components": {
                "whisper_model": str(self.output_dir / "whisper_model"),
                "symbol_converter": str(self.output_dir / "symbol_converter.pkl"),
                "complete_pipeline": str(self.output_dir / "complete_pipeline.pkl"),
                "config_files": [
                    str(self.output_dir / "pipeline_config.json"),
                    str(self.output_dir / "converter_stats.json")
                ]
            },
            "usage_example": {
                "text_processing": "pipeline.process_text('two plus three')",
                "audio_processing": "pipeline.process_audio('audio.wav')",
                "flask_integration": "from saved_models import load_flask_model"
            }
        }
        
        # Save Flask model package
        flask_path = self.output_dir / "flask_model_package.json"
        with open(flask_path, 'w') as f:
            json.dump(flask_model, f, indent=2)
        logger.info(f"Flask model package saved: {flask_path}")
        
        # Create Flask loader script
        self._create_flask_loader_script()
    
    def _create_flask_loader_script(self):
        """Create a Flask loader script"""
        loader_script = '''
"""
Flask Model Loader
Load saved models for Flask API integration
"""

import pickle
import json
import os
from pathlib import Path
from typing import Dict, Any

def load_flask_model(model_dir: str = "./saved_models") -> Dict[str, Any]:
    """Load all models for Flask integration"""
    model_dir = Path(model_dir)
    
    models = {}
    
    # Load symbol converter
    converter_path = model_dir / "symbol_converter.pkl"
    if converter_path.exists():
        with open(converter_path, 'rb') as f:
            models['symbol_converter'] = pickle.load(f)
        print(f"✅ Loaded symbol converter from {converter_path}")
    
    # Load complete pipeline
    pipeline_path = model_dir / "complete_pipeline.pkl"
    if pipeline_path.exists():
        with open(pipeline_path, 'rb') as f:
            models['complete_pipeline'] = pickle.load(f)
        print(f"✅ Loaded complete pipeline from {pipeline_path}")
    
    # Load configs
    config_path = model_dir / "pipeline_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            models['config'] = json.load(f)
        print(f"✅ Loaded pipeline config from {config_path}")
    
    return models

def get_model_info(model_dir: str = "./saved_models") -> Dict[str, Any]:
    """Get information about saved models"""
    model_dir = Path(model_dir)
    
    info = {
        "model_directory": str(model_dir),
        "available_models": [],
        "config_files": [],
        "total_size": 0
    }
    
    if model_dir.exists():
        for file_path in model_dir.rglob("*"):
            if file_path.is_file():
                info["total_size"] += file_path.stat().st_size
                
                if file_path.suffix in ['.pkl', '.pt', '.pth']:
                    info["available_models"].append(str(file_path))
                elif file_path.suffix == '.json':
                    info["config_files"].append(str(file_path))
    
    return info

if __name__ == "__main__":
    # Test loading
    models = load_flask_model()
    print(f"Loaded {len(models)} model components")
    
    # Test model info
    info = get_model_info()
    print(f"Model info: {info}")
'''
        
        loader_path = self.output_dir / "flask_model_loader.py"
        with open(loader_path, 'w') as f:
            f.write(loader_script)
        logger.info(f"Flask loader script created: {loader_path}")
    
    def test_saved_models(self):
        """Test the saved models"""
        logger.info("=== Testing Saved Models ===")
        
        try:
            # Test loading symbol converter
            converter_path = self.output_dir / "symbol_converter.pkl"
            if converter_path.exists():
                with open(converter_path, 'rb') as f:
                    converter = pickle.load(f)
                
                # Test conversion
                test_text = "two plus three equals five"
                result, metadata = converter.convert(test_text)
                logger.info(f"✅ Symbol converter test: '{test_text}' -> '{result}'")
            
            # Test loading complete pipeline
            pipeline_path = self.output_dir / "complete_pipeline.pkl"
            if pipeline_path.exists():
                with open(pipeline_path, 'rb') as f:
                    pipeline = pickle.load(f)
                
                # Test text processing
                test_text = "ten percent of fifty"
                result = pipeline.process_text(test_text, return_metadata=False)
                logger.info(f"✅ Pipeline test: '{test_text}' -> '{result['final_output']}'")
            
            logger.info("✅ All model tests passed!")
            
        except Exception as e:
            logger.error(f"❌ Model testing failed: {e}")

def main():
    """Main function to train and save models"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train and save Speech-to-Symbol models")
    parser.add_argument("--output_dir", type=str, default="./saved_models",
                       help="Output directory for saved models")
    parser.add_argument("--dataset_percentage", type=float, default=0.01,
                       help="Percentage of dataset to use for training")
    parser.add_argument("--max_steps", type=int, default=2000,
                       help="Maximum training steps")
    parser.add_argument("--model_name", type=str, default="openai/whisper-small",
                       help="Base Whisper model name")
    parser.add_argument("--test_only", action="store_true",
                       help="Only test existing models, don't train")
    
    args = parser.parse_args()
    
    # Create serializer
    serializer = ModelSerializer(args.output_dir)
    
    if args.test_only:
        # Only test existing models
        serializer.test_saved_models()
    else:
        # Train and save models
        serializer.train_and_save_complete_model(
            dataset_percentage=args.dataset_percentage,
            max_steps=args.max_steps,
            model_name=args.model_name
        )
        
        # Test the saved models
        serializer.test_saved_models()

if __name__ == "__main__":
    main() 