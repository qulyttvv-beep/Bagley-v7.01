"""
⚙️ BagleyVoice Configuration
Hyperparameters for the custom TTS model
"""

from dataclasses import dataclass, field
from typing import Optional, List, Literal


@dataclass
class BagleyVoiceConfig:
    """
    Configuration for BagleyVoice - Custom TTS Model
    
    Architecture based on:
    - Fish Speech: DualAR for parallel token generation
    - Chatterbox: Streaming low-latency output
    - XTTS: Voice cloning capabilities
    """
    
    # ==================== Model Architecture ====================
    
    # Phoneme encoder
    phoneme_vocab_size: int = 512
    phoneme_embedding_dim: int = 512
    
    # Hidden dimensions
    hidden_size: int = 1024
    num_attention_heads: int = 16
    head_dim: int = 64
    
    # Encoder
    encoder_layers: int = 6
    
    # DualAR Decoder
    semantic_decoder_layers: int = 12  # Coarse audio tokens
    acoustic_decoder_layers: int = 6   # Fine audio tokens
    
    # Audio tokens
    num_semantic_tokens: int = 1024   # Coarse tokens
    num_acoustic_tokens: int = 2048   # Fine tokens per frame
    num_quantizers: int = 8           # RVQ codebook layers
    
    # ==================== Prosody Prediction ====================
    
    # Prosody predictor
    prosody_hidden_dim: int = 256
    num_prosody_features: int = 5  # pitch, energy, duration, speed, emphasis
    
    # Emotion embedding
    num_emotions: int = 8  # neutral, happy, sad, angry, surprised, fearful, disgusted, contempt
    emotion_embedding_dim: int = 128
    
    # ==================== Voice Configuration ====================
    
    # Speaker embedding for voice cloning
    speaker_embedding_dim: int = 512
    max_speaker_embeddings: int = 1000
    
    # Pre-defined voices
    predefined_voices: List[str] = field(default_factory=lambda: [
        "bagley",        # Chaotic, expressive Bagley voice
        "natural_male",  # Natural male for video narration
        "natural_female", # Natural female for video narration
    ])
    
    # Voice cloning
    enable_voice_cloning: bool = True
    min_clone_audio_seconds: float = 3.0
    max_clone_audio_seconds: float = 30.0
    
    # ==================== Audio Configuration ====================
    
    sample_rate: int = 44100
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 128
    
    # Output
    output_sample_rate: int = 44100
    
    # ==================== Text Processing ====================
    
    # Phoneme conversion
    phoneme_language: str = "en-us"
    enable_multilingual: bool = True
    supported_languages: List[str] = field(default_factory=lambda: [
        "en", "nl", "de", "fr", "es", "it", "pt", "zh", "ja", "ko"
    ])
    
    # Text normalization
    normalize_numbers: bool = True
    normalize_abbreviations: bool = True
    
    # ==================== Streaming Configuration ====================
    
    # Real-time streaming
    enable_streaming: bool = True
    chunk_size_tokens: int = 50
    lookahead_tokens: int = 10
    
    # Latency optimization
    initial_latency_ms: int = 100
    target_rtf: float = 0.3  # Real-time factor (lower = faster)
    
    # ==================== Normalization ====================
    
    norm_type: str = "layer_norm"
    norm_eps: float = 1e-6
    
    # ==================== Training ====================
    
    initializer_range: float = 0.02
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    
    # ==================== Performance ====================
    
    use_flash_attention: bool = True
    use_kv_cache: bool = True
    
    # ==================== Computed Properties ====================
    
    @property
    def total_parameters(self) -> int:
        """Estimate total parameters"""
        # Embeddings
        phoneme_embed = self.phoneme_vocab_size * self.phoneme_embedding_dim
        speaker_embed = self.max_speaker_embeddings * self.speaker_embedding_dim
        emotion_embed = self.num_emotions * self.emotion_embedding_dim
        
        # Encoder
        encoder_per_layer = self.hidden_size * self.hidden_size * 4 + self.hidden_size * self.hidden_size * 4
        encoder_params = encoder_per_layer * self.encoder_layers
        
        # Semantic decoder
        semantic_per_layer = self.hidden_size * self.hidden_size * 8
        semantic_params = semantic_per_layer * self.semantic_decoder_layers
        
        # Acoustic decoder
        acoustic_per_layer = self.hidden_size * self.hidden_size * 8
        acoustic_params = acoustic_per_layer * self.acoustic_decoder_layers
        
        # Output heads
        output_heads = self.hidden_size * self.num_semantic_tokens + self.hidden_size * self.num_acoustic_tokens * self.num_quantizers
        
        return phoneme_embed + speaker_embed + emotion_embed + encoder_params + semantic_params + acoustic_params + output_heads
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.hidden_size % self.num_attention_heads == 0
        self.head_dim = self.hidden_size // self.num_attention_heads
