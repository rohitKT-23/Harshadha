"""
Complete Audio Processing Pipeline
Audio Input → ASR (Whisper) → Symbol Conversion → Final Output
"""

import torch
import torchaudio
import numpy as np
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
from typing import Dict, List, Optional, Tuple
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from postprocessing.symbol_converter import ComprehensiveSymbolConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Speech2SymbolPipeline:
    """Complete pipeline for speech to symbol conversion"""
    
    def __init__(
        self, 
        model_name: str = "openai/whisper-small",
        trained_model_path: Optional[str] = None,
        use_gpu: bool = True
    ):
        self.model_name = model_name
        self.trained_model_path = trained_model_path
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        
        logger.info(f"Initializing pipeline with device: {self.device}")
        
        # Initialize ASR pipeline
        self._setup_asr_pipeline()
        
        # Initialize symbol converter
        self.symbol_converter = ComprehensiveSymbolConverter()
        
        logger.info("Pipeline initialized successfully!")
    
    def _setup_asr_pipeline(self):
        """Setup ASR pipeline with trained or base model"""
        try:
            if self.trained_model_path and os.path.exists(self.trained_model_path):
                logger.info(f"Loading trained model from {self.trained_model_path}")
                
                # Load custom trained model
                self.processor = WhisperProcessor.from_pretrained(self.trained_model_path)
                self.model = WhisperForConditionalGeneration.from_pretrained(self.trained_model_path)
                self.model.to(self.device)
                
                # Create custom pipeline
                self.asr_pipeline = None  # We'll use model directly
                
            else:
                logger.info(f"Loading base model: {self.model_name}")
                
                # Use Hugging Face pipeline for base model
                self.asr_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=self.model_name,
                    device=0 if self.device == "cuda" else -1
                )
                self.processor = None
                self.model = None
                
        except Exception as e:
            logger.warning(f"Error loading model: {e}. Falling back to base model.")
            self.asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-small",
                device=0 if self.device == "cuda" else -1
            )
    
    def transcribe_audio(self, audio_path: str) -> Dict:
        """Transcribe audio file to text"""
        logger.info(f"Transcribing audio: {audio_path}")
        
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            if self.asr_pipeline:
                # Use pipeline for base model
                result = self.asr_pipeline(audio_path)
                transcription = result["text"]
                
            else:
                # Use custom trained model
                transcription = self._transcribe_with_custom_model(audio_path)
            
            logger.info(f"Transcription: {transcription}")
            return {
                "transcription": transcription,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {
                "transcription": "",
                "status": "error",
                "error": str(e)
            }
    
    def _transcribe_with_custom_model(self, audio_path: str) -> str:
        """Transcribe using custom trained model"""
        # Load and preprocess audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Prepare inputs
        inputs = self.processor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate transcription
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs["input_features"],
                max_length=225,
                num_beams=1,
                do_sample=False
            )
        
        # Decode
        transcription = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        
        return transcription
    
    def convert_symbols(self, text: str, confidence_threshold: float = 0.7) -> Dict:
        """Convert spoken operators to symbols"""
        logger.info(f"Converting symbols in: {text}")
        
        converted_text, metadata = self.symbol_converter.convert_text(text)
        
        logger.info(f"Converted: {converted_text}")
        
        return {
            "original_text": text,
            "converted_text": converted_text,
            "conversions": metadata.get("conversions", []),
            "total_conversions": len(metadata.get("conversions", [])),
            "priority_scores": [conv["priority"] for conv in metadata.get("conversions", [])]
        }
    
    def process_audio_complete(
        self, 
        audio_path: str, 
        confidence_threshold: float = 0.7
    ) -> Dict:
        """Complete pipeline: Audio → ASR → Symbol Conversion"""
        logger.info("=== Complete Audio Processing Pipeline ===")
        logger.info(f"Input: {audio_path}")
        
        # Step 1: Transcribe audio
        asr_result = self.transcribe_audio(audio_path)
        
        if asr_result["status"] != "success":
            return {
                "status": "error",
                "error": asr_result.get("error", "ASR failed"),
                "pipeline_stage": "transcription"
            }
        
        # Step 2: Convert symbols
        conversion_result = self.convert_symbols(
            asr_result["transcription"], 
            confidence_threshold
        )
        
        # Step 3: Combine results
        final_result = {
            "status": "success",
            "input_audio": audio_path,
            "raw_transcription": asr_result["transcription"],
            "final_output": conversion_result["converted_text"],
            "conversions_made": conversion_result["conversions"],
            "total_conversions": conversion_result["total_conversions"],
            "average_priority": np.mean(conversion_result["priority_scores"]) if conversion_result["priority_scores"] else 0.0
        }
        
        # Log results
        logger.info(f"Raw ASR: {final_result['raw_transcription']}")
        logger.info(f"Final Output: {final_result['final_output']}")
        logger.info(f"Conversions: {final_result['total_conversions']}")
        
        return final_result
    
    def process_text_only(self, text: str, confidence_threshold: float = 0.7) -> Dict:
        """Process text without ASR (for testing)"""
        logger.info("=== Text-Only Processing ===")
        
        conversion_result = self.convert_symbols(text, confidence_threshold)
        
        return {
            "status": "success",
            "input_text": text,
            "final_output": conversion_result["converted_text"],
            "conversions_made": conversion_result["conversions"],
            "total_conversions": conversion_result["total_conversions"]
        }
    
    def evaluate_on_samples(self, test_samples: List[Dict]) -> Dict:
        """Evaluate pipeline on test samples"""
        results = []
        
        for sample in test_samples:
            if "audio_path" in sample:
                result = self.process_audio_complete(sample["audio_path"])
            else:
                result = self.process_text_only(sample["text"])
            
            results.append({
                "input": sample,
                "output": result,
                "expected": sample.get("expected", ""),
                "correct": result.get("final_output", "") == sample.get("expected", "")
            })
        
        # Calculate accuracy
        correct_count = sum(1 for r in results if r["correct"])
        accuracy = correct_count / len(results) if results else 0.0
        
        return {
            "accuracy": accuracy,
            "total_samples": len(results),
            "correct_predictions": correct_count,
            "detailed_results": results
        }


# Helper function for easy usage
def create_pipeline(model_path: Optional[str] = None) -> Speech2SymbolPipeline:
    """Create and return a speech2symbol pipeline"""
    return Speech2SymbolPipeline(trained_model_path=model_path)


# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Speech-to-Symbol Pipeline")
    parser.add_argument("--audio", type=str, help="Audio file path")
    parser.add_argument("--text", type=str, help="Text input (no ASR)")
    parser.add_argument("--model", type=str, help="Trained model path")
    parser.add_argument("--confidence", type=float, default=0.7, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = Speech2SymbolPipeline(trained_model_path=args.model)
    
    if args.audio:
        result = pipeline.process_audio_complete(args.audio, args.confidence)
        print(f"Final Result: {result['final_output']}")
        
    elif args.text:
        result = pipeline.process_text_only(args.text, args.confidence)
        print(f"Final Result: {result['final_output']}")
        
    else:
        print("Please provide --audio or --text input") 