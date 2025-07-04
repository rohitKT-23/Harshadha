"""
Production-Ready Whisper ASR Wrapper
Modular class for speech-to-text conversion with support for both base and fine-tuned models
"""

import torch
import torchaudio
import numpy as np
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
from typing import Dict, List, Optional, Union
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperWrapper:
    """
    Production-ready Whisper ASR wrapper with support for:
    - Base Whisper models (openai/whisper-small, medium, large)
    - Fine-tuned custom models
    - GPU/CPU inference
    - Batch processing
    - Model serialization
    """
    
    def __init__(
        self, 
        model_name: str = "openai/whisper-small",
        trained_model_path: Optional[str] = None,
        use_gpu: bool = True,
        language: str = "en"
    ):
        """
        Initialize Whisper ASR model
        
        Args:
            model_name: Base model name (openai/whisper-small, medium, large)
            trained_model_path: Path to fine-tuned model (optional)
            use_gpu: Use GPU if available
            language: Target language for transcription
        """
        self.model_name = model_name
        self.trained_model_path = trained_model_path
        self.language = language
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        
        logger.info(f"Initializing Whisper on device: {self.device}")
        
        # Model components
        self.asr_pipeline = None
        self.processor = None
        self.model = None
        
        # Initialize model
        self._setup_model()
        
        logger.info("Whisper wrapper initialized successfully!")
    
    def _setup_model(self):
        """Setup ASR model (base or fine-tuned)"""
        try:
            if self.trained_model_path and os.path.exists(self.trained_model_path):
                self._load_custom_model()
            else:
                self._load_base_model()
                
        except Exception as e:
            logger.warning(f"Error loading model: {e}. Falling back to base model.")
            self._load_base_model()
    
    def _load_custom_model(self):
        """Load fine-tuned Whisper model"""
        logger.info(f"Loading custom model from: {self.trained_model_path}")
        
        self.processor = WhisperProcessor.from_pretrained(self.trained_model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(self.trained_model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Clear pipeline since we'll use model directly
        self.asr_pipeline = None
        
        logger.info("Custom model loaded successfully")
    
    def _load_base_model(self):
        """Load base Whisper model via pipeline"""
        logger.info(f"Loading base model: {self.model_name}")
        
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model_name,
            device=0 if self.device == "cuda" else -1
        )
        
        # Clear custom model components
        self.processor = None
        self.model = None
        
        logger.info("Base model loaded successfully")
    
    def transcribe(
        self, 
        audio_input: Union[str, np.ndarray, torch.Tensor],
        return_timestamps: bool = False,
        return_confidence: bool = False
    ) -> Dict:
        """
        Transcribe audio to text
        
        Args:
            audio_input: Audio file path, numpy array, or torch tensor
            return_timestamps: Include word-level timestamps
            return_confidence: Include confidence scores
            
        Returns:
            Dict with transcription and metadata
        """
        try:
            if isinstance(audio_input, str):
                # File path
                if not os.path.exists(audio_input):
                    raise FileNotFoundError(f"Audio file not found: {audio_input}")
                transcription = self._transcribe_file(audio_input)
            else:
                # Array/tensor input
                transcription = self._transcribe_array(audio_input)
            
            return {
                "transcription": transcription.strip(),
                "status": "success",
                "model_type": "custom" if self.model else "base",
                "device": self.device
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {
                "transcription": "",
                "status": "error",
                "error": str(e)
            }
    
    def _transcribe_file(self, audio_path: str) -> str:
        """Transcribe audio file"""
        if self.asr_pipeline:
            # Use pipeline for base model
            result = self.asr_pipeline(audio_path)
            return result["text"]
        else:
            # Use custom model
            return self._transcribe_with_custom_model(audio_path)
    
    def _transcribe_array(self, audio_array: Union[np.ndarray, torch.Tensor]) -> str:
        """Transcribe audio array/tensor"""
        if self.asr_pipeline:
            # Convert to numpy if needed
            if isinstance(audio_array, torch.Tensor):
                audio_array = audio_array.cpu().numpy()
            
            result = self.asr_pipeline(audio_array)
            return result["text"]
        else:
            # Use custom model
            return self._transcribe_array_custom(audio_array)
    
    def _transcribe_with_custom_model(self, audio_path: str) -> str:
        """Transcribe using custom trained model"""
        # Load and preprocess audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Preprocess audio
        waveform = self._preprocess_audio(waveform, sample_rate)
        
        # Generate transcription
        return self._generate_transcription(waveform)
    
    def _transcribe_array_custom(self, audio_array: Union[np.ndarray, torch.Tensor]) -> str:
        """Transcribe audio array with custom model"""
        # Convert to tensor if needed
        if isinstance(audio_array, np.ndarray):
            waveform = torch.from_numpy(audio_array)
        else:
            waveform = audio_array
        
        # Ensure correct shape
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Preprocess and transcribe
        waveform = self._preprocess_audio(waveform, 16000)
        return self._generate_transcription(waveform)
    
    def _preprocess_audio(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Preprocess audio for Whisper"""
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform
    
    def _generate_transcription(self, waveform: torch.Tensor) -> str:
        """Generate transcription using custom model"""
        # Prepare inputs
        inputs = self.processor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs["input_features"],
                max_length=225,
                num_beams=1,
                do_sample=False,
                language=self.language
            )
        
        # Decode
        transcription = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        
        return transcription
    
    def batch_transcribe(self, audio_files: List[str]) -> List[Dict]:
        """Transcribe multiple audio files"""
        results = []
        
        for audio_file in audio_files:
            logger.info(f"Transcribing: {audio_file}")
            result = self.transcribe(audio_file)
            results.append({
                "file": audio_file,
                **result
            })
        
        return results
    
    def save_model(self, save_path: str):
        """Save custom model to disk"""
        if self.model and self.processor:
            os.makedirs(save_path, exist_ok=True)
            
            self.model.save_pretrained(save_path)
            self.processor.save_pretrained(save_path)
            
            logger.info(f"Model saved to: {save_path}")
        else:
            logger.warning("No custom model to save")
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "trained_model_path": self.trained_model_path,
            "model_type": "custom" if self.model else "base",
            "device": self.device,
            "language": self.language,
            "parameters": self._count_parameters() if self.model else "Unknown"
        }
    
    def _count_parameters(self) -> int:
        """Count model parameters"""
        if self.model:
            return sum(p.numel() for p in self.model.parameters())
        return 0
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> 'WhisperWrapper':
        """Load from pretrained model path"""
        return cls(trained_model_path=model_path, **kwargs)
    
    def __call__(self, audio_input: Union[str, np.ndarray, torch.Tensor]) -> str:
        """Make the class callable"""
        result = self.transcribe(audio_input)
        return result.get("transcription", "")

# Factory functions for common configurations
def create_base_whisper(model_size: str = "small", use_gpu: bool = True) -> WhisperWrapper:
    """Create base Whisper model"""
    model_name = f"openai/whisper-{model_size}"
    return WhisperWrapper(model_name=model_name, use_gpu=use_gpu)

def create_custom_whisper(model_path: str, use_gpu: bool = True) -> WhisperWrapper:
    """Create custom fine-tuned Whisper model"""
    return WhisperWrapper.from_pretrained(model_path, use_gpu=use_gpu)
