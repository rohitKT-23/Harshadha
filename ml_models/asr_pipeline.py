"""
Production-Ready ASR Pipeline
Complete Speech-to-Symbol conversion pipeline combining Whisper ASR and intelligent symbol conversion
"""

import time
import logging
from typing import Dict, List, Optional, Union, Tuple, TYPE_CHECKING
from pathlib import Path
import json
import pickle
import os

# Import our modular components
from .whisper_wrapper import WhisperWrapper, create_base_whisper, create_custom_whisper
from .symbol_converter import SymbolConverter, create_advanced_converter, ConversionResult

# Type checking imports
if TYPE_CHECKING:
    import numpy as np
    import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ASRPipeline:
    """
    Production-ready ASR Pipeline for Speech-to-Symbol conversion
    
    Features:
    - Modular Whisper ASR (base or fine-tuned models)
    - Context-aware symbol conversion
    - Batch processing capabilities
    - Performance monitoring
    - Model serialization
    - API-ready interface
    """
    
    def __init__(
        self,
        whisper_model: str = "openai/whisper-small",
        trained_model_path: Optional[str] = None,
        symbol_converter: Optional[SymbolConverter] = None,
        confidence_threshold: float = 0.7,
        use_gpu: bool = True,
        language: str = "en"
    ):
        """
        Initialize ASR Pipeline
        
        Args:
            whisper_model: Base Whisper model name
            trained_model_path: Path to fine-tuned model (optional)
            symbol_converter: Custom symbol converter (optional)
            confidence_threshold: Minimum confidence for symbol conversion
            use_gpu: Use GPU if available
            language: Target language
        """
        self.whisper_model = whisper_model
        self.trained_model_path = trained_model_path
        self.confidence_threshold = confidence_threshold
        self.language = language
        
        # Initialize components
        logger.info("Initializing ASR Pipeline...")
        
        # Setup Whisper ASR
        self.asr = self._setup_whisper(use_gpu)
        
        # Setup Symbol Converter
        self.converter = symbol_converter or create_advanced_converter()
        self.converter.confidence_threshold = confidence_threshold
        
        # Performance tracking
        self.stats = {
            'total_processed': 0,
            'total_conversions': 0,
            'average_processing_time': 0.0,
            'average_confidence': 0.0
        }
        
        logger.info("ASR Pipeline initialized successfully!")
    
    def _setup_whisper(self, use_gpu: bool) -> WhisperWrapper:
        """Setup Whisper ASR component"""
        if self.trained_model_path:
            logger.info(f"Loading custom Whisper model: {self.trained_model_path}")
            return create_custom_whisper(self.trained_model_path, use_gpu)
        else:
            logger.info(f"Loading base Whisper model: {self.whisper_model}")
            model_size = self.whisper_model.split('-')[-1] if 'whisper-' in self.whisper_model else 'small'
            return create_base_whisper(model_size, use_gpu)
    
    def process_audio(
        self, 
        audio_input: Union[str, 'np.ndarray', 'torch.Tensor'],
        return_metadata: bool = True,
        custom_confidence: Optional[float] = None
    ) -> Dict:
        """
        Process audio through complete pipeline
        
        Args:
            audio_input: Audio file path, array, or tensor
            return_metadata: Include detailed metadata
            custom_confidence: Override default confidence threshold
            
        Returns:
            Dict with transcription, conversion results, and metadata
        """
        start_time = time.time()
        confidence = custom_confidence or self.confidence_threshold
        
        try:
            # Step 1: Speech Recognition
            logger.info("🎤 Starting ASR transcription...")
            asr_result = self.asr.transcribe(audio_input)
            
            if asr_result["status"] != "success":
                return {
                    "status": "error",
                    "error": asr_result.get("error", "ASR failed"),
                    "stage": "transcription"
                }
            
            raw_transcription = asr_result["transcription"]
            logger.info(f"📝 Raw transcription: '{raw_transcription}'")
            
            # Step 2: Symbol Conversion
            logger.info("🔄 Starting symbol conversion...")
            converted_text, conversion_metadata = self.converter.convert(
                raw_transcription, confidence
            )
            logger.info(f"✨ Final output: '{converted_text}'")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update statistics
            self._update_stats(processing_time, conversion_metadata)
            
            # Prepare result
            result = {
                "status": "success",
                "input": str(audio_input) if isinstance(audio_input, str) else "audio_data",
                "raw_transcription": raw_transcription,
                "final_output": converted_text,
                "total_conversions": conversion_metadata["total_conversions"],
                "processing_time": processing_time
            }
            
            if return_metadata:
                result.update({
                    "conversions_made": conversion_metadata["conversions"],
                    "average_confidence": conversion_metadata["average_confidence"],
                    "context_analysis": conversion_metadata.get("context_analysis", {}),
                    "asr_metadata": {
                        "model_type": asr_result.get("model_type", "unknown"),
                        "device": asr_result.get("device", "unknown")
                    }
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "stage": "pipeline"
            }
    
    def process_text(
        self, 
        text: str, 
        custom_confidence: Optional[float] = None,
        return_metadata: bool = True
    ) -> Dict:
        """
        Process text-only through symbol conversion
        
        Args:
            text: Input text to convert
            custom_confidence: Override default confidence threshold
            return_metadata: Include detailed metadata
            
        Returns:
            Dict with conversion results and metadata
        """
        start_time = time.time()
        confidence = custom_confidence or self.confidence_threshold
        
        try:
            logger.info(f"📝 Processing text: '{text}'")
            
            # Symbol conversion
            converted_text, conversion_metadata = self.converter.convert(text, confidence)
            
            processing_time = time.time() - start_time
            
            # Update statistics (text-only)
            self._update_stats(processing_time, conversion_metadata)
            
            result = {
                "status": "success",
                "input_text": text,
                "final_output": converted_text,
                "total_conversions": conversion_metadata["total_conversions"],
                "processing_time": processing_time
            }
            
            if return_metadata:
                result.update({
                    "conversions_made": conversion_metadata["conversions"],
                    "average_confidence": conversion_metadata["average_confidence"],
                    "context_analysis": conversion_metadata.get("context_analysis", {})
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Text processing failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "stage": "text_conversion"
            }
    
    def batch_process_audio(
        self, 
        audio_files: List[str],
        return_metadata: bool = False
    ) -> List[Dict]:
        """
        Process multiple audio files
        
        Args:
            audio_files: List of audio file paths
            return_metadata: Include detailed metadata for each
            
        Returns:
            List of processing results
        """
        logger.info(f"🔄 Batch processing {len(audio_files)} audio files...")
        
        results = []
        for i, audio_file in enumerate(audio_files, 1):
            logger.info(f"Processing file {i}/{len(audio_files)}: {audio_file}")
            
            result = self.process_audio(audio_file, return_metadata)
            result["file_index"] = i
            result["file_path"] = audio_file
            results.append(result)
        
        # Batch statistics
        successful = [r for r in results if r["status"] == "success"]
        total_conversions = sum(r.get("total_conversions", 0) for r in successful)
        avg_time = sum(r.get("processing_time", 0) for r in successful) / max(len(successful), 1)
        
        logger.info(f"✅ Batch complete: {len(successful)}/{len(audio_files)} successful")
        logger.info(f"📊 Total conversions: {total_conversions}, Avg time: {avg_time:.2f}s")
        
        return results
    
    def batch_process_text(
        self, 
        texts: List[str],
        return_metadata: bool = False
    ) -> List[Dict]:
        """
        Process multiple texts
        
        Args:
            texts: List of input texts
            return_metadata: Include detailed metadata
            
        Returns:
            List of processing results
        """
        logger.info(f"📝 Batch processing {len(texts)} texts...")
        
        results = []
        for i, text in enumerate(texts, 1):
            result = self.process_text(text, return_metadata=return_metadata)
            result["text_index"] = i
            results.append(result)
        
        return results
    
    def evaluate_on_samples(
        self, 
        test_samples: List[Dict],
        metrics: List[str] = ["accuracy", "conversion_rate"]
    ) -> Dict:
        """
        Evaluate pipeline on test samples
        
        Args:
            test_samples: List of {"input": audio/text, "expected": str} dicts
            metrics: List of metrics to calculate
            
        Returns:
            Evaluation results
        """
        logger.info(f"🧪 Evaluating on {len(test_samples)} samples...")
        
        results = []
        correct = 0
        total_conversions = 0
        
        for sample in test_samples:
            if "audio" in sample:
                result = self.process_audio(sample["audio"], return_metadata=False)
            else:
                result = self.process_text(sample["input"], return_metadata=False)
            
            if result["status"] == "success":
                output = result["final_output"]
                expected = sample["expected"]
                is_correct = output.strip() == expected.strip()
                
                if is_correct:
                    correct += 1
                
                total_conversions += result.get("total_conversions", 0)
                
                results.append({
                    "input": sample.get("input", sample.get("audio", "")),
                    "expected": expected,
                    "actual": output,
                    "correct": is_correct,
                    "conversions": result.get("total_conversions", 0)
                })
        
        # Calculate metrics
        accuracy = correct / len(test_samples) if test_samples else 0
        conversion_rate = total_conversions / len(test_samples) if test_samples else 0
        
        evaluation = {
            "total_samples": len(test_samples),
            "correct_predictions": correct,
            "accuracy": accuracy,
            "conversion_rate": conversion_rate,
            "total_conversions": total_conversions,
            "detailed_results": results
        }
        
        logger.info(f"📊 Evaluation complete: {accuracy:.2%} accuracy, {conversion_rate:.1f} avg conversions")
        
        return evaluation
    
    def _update_stats(self, processing_time: float, conversion_metadata: Dict):
        """Update pipeline statistics"""
        self.stats["total_processed"] += 1
        self.stats["total_conversions"] += conversion_metadata["total_conversions"]
        
        # Update rolling averages
        n = self.stats["total_processed"]
        self.stats["average_processing_time"] = (
            (self.stats["average_processing_time"] * (n-1) + processing_time) / n
        )
        
        if conversion_metadata["total_conversions"] > 0:
            self.stats["average_confidence"] = (
                (self.stats["average_confidence"] * (n-1) + conversion_metadata["average_confidence"]) / n
            )
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        return {
            **self.stats,
            "asr_info": self.asr.get_model_info(),
            "converter_info": self.converter.get_statistics(),
            "pipeline_config": {
                "whisper_model": self.whisper_model,
                "trained_model_path": self.trained_model_path,
                "confidence_threshold": self.confidence_threshold,
                "language": self.language
            }
        }
    
    def save_pipeline(self, save_path: str):
        """Save entire pipeline configuration and models"""
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save pipeline config
        config = {
            "whisper_model": self.whisper_model,
            "trained_model_path": self.trained_model_path,
            "confidence_threshold": self.confidence_threshold,
            "language": self.language,
            "stats": self.stats
        }
        
        config_path = save_dir / "pipeline_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save Whisper model if custom
        if self.trained_model_path:
            whisper_dir = save_dir / "whisper_model"
            self.asr.save_model(str(whisper_dir))
        
        # Save symbol converter
        converter_path = save_dir / "symbol_converter.json"
        self.converter.save_model(str(converter_path))
        
        logger.info(f"Pipeline saved to: {save_path}")
    
    @classmethod
    def load_pipeline(cls, load_path: str) -> 'ASRPipeline':
        """Load pipeline from saved configuration"""
        load_dir = Path(load_path)
        
        # Load config
        config_path = load_dir / "pipeline_config.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check for custom Whisper model
        whisper_dir = load_dir / "whisper_model"
        trained_model_path = str(whisper_dir) if whisper_dir.exists() else None
        
        # Load symbol converter
        converter_path = load_dir / "symbol_converter.json"
        converter = SymbolConverter.load_model(str(converter_path))
        
        # Create pipeline
        pipeline = cls(
            whisper_model=config["whisper_model"],
            trained_model_path=trained_model_path,
            symbol_converter=converter,
            confidence_threshold=config["confidence_threshold"],
            language=config["language"]
        )
        
        # Restore stats
        pipeline.stats = config.get("stats", pipeline.stats)
        
        logger.info(f"Pipeline loaded from: {load_path}")
        return pipeline
    
    def update_confidence_threshold(self, new_threshold: float):
        """Update confidence threshold for symbol conversion"""
        self.confidence_threshold = new_threshold
        self.converter.confidence_threshold = new_threshold
        logger.info(f"Updated confidence threshold to: {new_threshold}")
    
    def add_custom_conversion_rule(self, pattern: str, replacement: str, priority: int = 5):
        """Add custom conversion rule"""
        from .symbol_converter import ConversionRule
        
        rule = ConversionRule(pattern=pattern, replacement=replacement, priority=priority)
        self.converter.add_custom_rule(rule)
        logger.info(f"Added custom rule: '{pattern}' -> '{replacement}'")
    
    def __call__(self, audio_input: Union[str, 'np.ndarray', 'torch.Tensor']) -> str:
        """Make pipeline callable for simple use"""
        result = self.process_audio(audio_input, return_metadata=False)
        return result.get("final_output", "")

# Factory functions for common configurations
def create_basic_pipeline(**kwargs) -> ASRPipeline:
    """Create basic ASR pipeline with default settings"""
    return ASRPipeline(**kwargs)

def create_production_pipeline(
    model_path: Optional[str] = None,
    confidence: float = 0.7,
    use_gpu: bool = True
) -> ASRPipeline:
    """Create production-ready ASR pipeline"""
    return ASRPipeline(
        trained_model_path=model_path,
        confidence_threshold=confidence,
        use_gpu=use_gpu
    )

def create_api_pipeline(
    whisper_model: str = "openai/whisper-small",
    confidence: float = 0.8
) -> ASRPipeline:
    """Create API-optimized pipeline"""
    return ASRPipeline(
        whisper_model=whisper_model,
        confidence_threshold=confidence,
        use_gpu=True
    )

# Quick test function
def test_pipeline():
    """Quick test of the pipeline"""
    pipeline = create_basic_pipeline()
    
    test_texts = [
        "two plus three equals five",
        "ten percent of fifty",
        "list items comma separated by comma",
        "is this correct question mark"
    ]
    
    print("🧪 Testing ASR Pipeline with sample texts:")
    for text in test_texts:
        result = pipeline.process_text(text, return_metadata=False)
        print(f"'{text}' -> '{result['final_output']}'")
    
    print(f"\n📊 Pipeline Stats: {pipeline.get_stats()}")

if __name__ == "__main__":
    test_pipeline()
