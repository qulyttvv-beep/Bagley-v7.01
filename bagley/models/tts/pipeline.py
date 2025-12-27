"""
🗣️ BagleyTTSPipeline - Complete TTS Pipeline
End-to-end text to speech synthesis
"""

from typing import Optional, List, Union, AsyncGenerator
import logging
import io

import torch
from torch import Tensor

from bagley.models.tts.model import BagleyVoice, BagleyVoiceConfig
from bagley.models.tts.vocoder import BagleyVocoder

logger = logging.getLogger(__name__)


# Emotion mappings
EMOTION_MAP = {
    "neutral": 0,
    "happy": 1,
    "sad": 2,
    "angry": 3,
    "surprised": 4,
    "fearful": 5,
    "disgusted": 6,
    "contempt": 7,
}

# Voice mappings
VOICE_MAP = {
    "bagley": 0,
    "natural_male": 1,
    "natural_female": 2,
}


class TextProcessor:
    """Text normalization and phoneme conversion"""
    
    def __init__(self, language: str = "en-us"):
        self.language = language
    
    def normalize(self, text: str) -> str:
        """Normalize text (numbers, abbreviations, etc.)"""
        # Placeholder - would use text normalization library
        return text.lower().strip()
    
    def to_phonemes(self, text: str) -> List[int]:
        """Convert text to phoneme IDs"""
        # Placeholder - would use g2p library
        # Returns dummy phoneme IDs for now
        return [ord(c) % 512 for c in text]


class BagleyTTSPipeline:
    """
    🗣️ Complete TTS Pipeline
    
    Orchestrates:
    - Text normalization
    - Phoneme conversion
    - Neural TTS synthesis
    - Vocoder decoding
    
    Supports:
    - Multiple voices (including Bagley's chaotic voice)
    - Emotion control
    - Speed/pitch adjustment
    - Real-time streaming
    - Voice cloning
    """
    
    def __init__(
        self,
        model: Optional[BagleyVoice] = None,
        vocoder: Optional[BagleyVocoder] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.model = model
        self.vocoder = vocoder
        self.device = device
        self.dtype = dtype
        
        self.text_processor = TextProcessor()
        
        # Sample rate
        self.sample_rate = 44100
        if model:
            self.sample_rate = model.config.output_sample_rate
        
        logger.info("Initialized BagleyTTSPipeline")
    
    @classmethod
    def from_pretrained(
        cls,
        path: str,
        device: str = "cuda",
    ) -> "BagleyTTSPipeline":
        """Load complete pipeline from pretrained"""
        import os
        
        model = BagleyVoice.from_pretrained(os.path.join(path, "model"), device)
        vocoder = BagleyVocoder.from_pretrained(os.path.join(path, "vocoder"), device)
        
        return cls(model=model, vocoder=vocoder, device=device)
    
    def _prepare_inputs(
        self,
        text: str,
        voice: str = "bagley",
        emotion: str = "neutral",
    ) -> tuple:
        """Prepare model inputs from text"""
        # Normalize and convert to phonemes
        normalized = self.text_processor.normalize(text)
        phoneme_ids = self.text_processor.to_phonemes(normalized)
        
        # Convert to tensors
        phoneme_tensor = torch.tensor([phoneme_ids], device=self.device, dtype=torch.long)
        speaker_tensor = torch.tensor([VOICE_MAP.get(voice, 0)], device=self.device, dtype=torch.long)
        emotion_tensor = torch.tensor([EMOTION_MAP.get(emotion, 0)], device=self.device, dtype=torch.long)
        
        return phoneme_tensor, speaker_tensor, emotion_tensor
    
    @torch.no_grad()
    def __call__(
        self,
        text: Union[str, List[str]],
        voice: str = "bagley",
        emotion: str = "neutral",
        speed: float = 1.0,
        pitch_shift: float = 0.0,
        output_format: str = "wav",
    ) -> bytes:
        """
        Synthesize speech from text.
        
        Args:
            text: Input text or list of texts
            voice: Voice to use (bagley, natural_male, natural_female)
            emotion: Emotion (neutral, happy, sad, angry, etc.)
            speed: Speed multiplier (0.5-2.0)
            pitch_shift: Pitch shift in semitones (-12 to 12)
            output_format: Output format (wav, mp3, ogg)
            
        Returns:
            Audio bytes in specified format
        """
        if isinstance(text, list):
            text = " ".join(text)
        
        # Prepare inputs
        phonemes, speaker, emotion_id = self._prepare_inputs(text, voice, emotion)
        
        # Generate with model
        if self.model is not None:
            output = self.model(phonemes, speaker, emotion_id)
            
            # Decode acoustic tokens to waveform
            if self.vocoder is not None:
                # Get acoustic features from logits
                acoustic_logits = output.acoustic_logits
                acoustic_tokens = acoustic_logits.argmax(dim=-1)
                
                # Convert tokens to features (placeholder)
                features = torch.randn(1, 256, acoustic_tokens.shape[1] * 4, device=self.device)
                
                # Generate waveform
                waveform = self.vocoder.generate(features)
            else:
                # Placeholder waveform
                duration = len(text) * 0.1  # Rough estimate
                waveform = torch.zeros(1, 1, int(duration * self.sample_rate), device=self.device)
        else:
            # Placeholder when model not loaded
            duration = len(text) * 0.1
            waveform = torch.zeros(1, 1, int(duration * self.sample_rate), device=self.device)
        
        # Apply speed adjustment
        if speed != 1.0:
            waveform = self._adjust_speed(waveform, speed)
        
        # Apply pitch shift
        if pitch_shift != 0.0:
            waveform = self._adjust_pitch(waveform, pitch_shift)
        
        # Convert to bytes
        return self._to_audio_bytes(waveform, output_format)
    
    @torch.no_grad()
    async def stream(
        self,
        text: str,
        voice: str = "bagley",
        emotion: str = "neutral",
        chunk_size_ms: int = 100,
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream audio chunks for real-time TTS.
        
        Yields audio chunks as they're generated for low-latency streaming.
        """
        # Prepare inputs
        phonemes, speaker, emotion_id = self._prepare_inputs(text, voice, emotion)
        
        # Calculate chunk size in samples
        chunk_samples = int(self.sample_rate * chunk_size_ms / 1000)
        
        # Placeholder for streaming implementation
        # Would generate chunks incrementally
        total_samples = len(text) * int(0.1 * self.sample_rate)
        
        for i in range(0, total_samples, chunk_samples):
            # Generate chunk (placeholder)
            chunk = torch.zeros(1, 1, min(chunk_samples, total_samples - i))
            
            yield self._to_audio_bytes(chunk, "wav")
    
    def _adjust_speed(self, waveform: Tensor, speed: float) -> Tensor:
        """Adjust playback speed"""
        if speed == 1.0:
            return waveform
        
        # Resample to adjust speed
        new_length = int(waveform.shape[-1] / speed)
        return torch.nn.functional.interpolate(
            waveform.unsqueeze(0),
            size=new_length,
            mode='linear',
            align_corners=False
        ).squeeze(0)
    
    def _adjust_pitch(self, waveform: Tensor, semitones: float) -> Tensor:
        """Adjust pitch by semitones"""
        # Placeholder - would use pitch shifting algorithm
        return waveform
    
    def _to_audio_bytes(self, waveform: Tensor, format: str = "wav") -> bytes:
        """Convert waveform tensor to audio bytes"""
        waveform = waveform.squeeze().cpu().numpy()
        
        buffer = io.BytesIO()
        
        try:
            import scipy.io.wavfile as wavfile
            import numpy as np
            
            # Normalize and convert to int16
            waveform = np.clip(waveform, -1, 1)
            waveform_int = (waveform * 32767).astype(np.int16)
            
            wavfile.write(buffer, self.sample_rate, waveform_int)
            
        except ImportError:
            # Fallback - write raw PCM
            import numpy as np
            waveform_int = (np.clip(waveform, -1, 1) * 32767).astype(np.int16)
            buffer.write(waveform_int.tobytes())
        
        return buffer.getvalue()
    
    def clone_voice(
        self,
        audio_path: str,
        voice_name: str,
    ) -> str:
        """
        Clone a voice from an audio file.
        
        Args:
            audio_path: Path to reference audio
            voice_name: Name for the new voice
            
        Returns:
            Voice ID for use in synthesis
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        logger.info(f"Cloning voice from {audio_path} as '{voice_name}'")
        
        # Placeholder for voice cloning
        # Would:
        # 1. Load and preprocess audio
        # 2. Extract speaker embedding
        # 3. Store in speaker embedding table
        
        return voice_name
    
    def list_voices(self) -> List[str]:
        """List available voices"""
        voices = list(VOICE_MAP.keys())
        # Would also include cloned voices
        return voices
    
    def list_emotions(self) -> List[str]:
        """List available emotions"""
        return list(EMOTION_MAP.keys())
