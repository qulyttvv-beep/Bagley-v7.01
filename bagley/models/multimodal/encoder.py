"""
🔗 Multimodal Encoder - Unified Embedding Space
Encodes all modalities into a shared representation
"""

from typing import Optional, List, Dict, Any, Union
import logging

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class ImageEncoder(nn.Module):
    """Vision encoder (CLIP/SigLIP style)"""
    
    def __init__(self, hidden_size: int = 1024, patch_size: int = 14, num_layers: int = 24):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, hidden_size, patch_size, stride=patch_size)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, 257, hidden_size) * 0.02)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=16,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final projection
        self.proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, images: Tensor) -> Tensor:
        batch_size = images.shape[0]
        
        # Patch embedding
        x = self.patch_embed(images)  # [B, D, H, W]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        
        # Add CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)
        
        # Add position embedding (truncate or interpolate if needed)
        x = x + self.pos_embed[:, :x.shape[1]]
        
        # Encode
        x = self.encoder(x)
        
        # Project CLS token
        return self.proj(x[:, 0])


class AudioEncoder(nn.Module):
    """Audio encoder (Whisper-style)"""
    
    def __init__(self, hidden_size: int = 1024, num_layers: int = 12):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Mel spectrogram projection
        self.conv1 = nn.Conv1d(80, hidden_size, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, 3, stride=2, padding=1)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, 1500, hidden_size) * 0.02)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=16,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, mel: Tensor) -> Tensor:
        # mel: [B, 80, T]
        x = torch.relu(self.conv1(mel))
        x = torch.relu(self.conv2(x))
        x = x.transpose(1, 2)  # [B, T, D]
        
        # Position embedding
        x = x + self.pos_embed[:, :x.shape[1]]
        
        # Encode
        x = self.encoder(x)
        
        # Pool and project
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.pool(x).squeeze(-1)  # [B, D]
        
        return self.proj(x)


class TextEncoder(nn.Module):
    """Text encoder (simple transformer)"""
    
    def __init__(self, vocab_size: int = 50000, hidden_size: int = 1024, num_layers: int = 12):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Parameter(torch.randn(1, 512, hidden_size) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=16,
            dim_feedforward=hidden_size * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, input_ids: Tensor) -> Tensor:
        x = self.embed(input_ids)
        x = x + self.pos_embed[:, :x.shape[1]]
        x = self.encoder(x)
        
        # Mean pool
        x = x.mean(dim=1)
        return self.proj(x)


class MultimodalEncoder(nn.Module):
    """
    🔗 Unified Multimodal Encoder
    
    Projects all modalities into a shared embedding space:
    - Images → 1024-dim embedding
    - Audio → 1024-dim embedding
    - Text → 1024-dim embedding
    - Video → 1024-dim embedding (via frame aggregation)
    
    Enables Bagley to understand and reason about any input type.
    """
    
    def __init__(
        self,
        hidden_size: int = 1024,
        num_image_layers: int = 24,
        num_audio_layers: int = 12,
        num_text_layers: int = 12,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Modality-specific encoders
        self.image_encoder = ImageEncoder(hidden_size, num_layers=num_image_layers)
        self.audio_encoder = AudioEncoder(hidden_size, num_layers=num_audio_layers)
        self.text_encoder = TextEncoder(hidden_size=hidden_size, num_layers=num_text_layers)
        
        # Video is encoded via image encoder + temporal aggregation
        self.temporal_pool = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, 8, hidden_size * 4, batch_first=True),
            num_layers=2,
        )
        
        # Modality projection to shared space
        self.image_proj = nn.Linear(hidden_size, hidden_size)
        self.audio_proj = nn.Linear(hidden_size, hidden_size)
        self.text_proj = nn.Linear(hidden_size, hidden_size)
        self.video_proj = nn.Linear(hidden_size, hidden_size)
        
        logger.info("Initialized MultimodalEncoder")
    
    def encode_image(self, images: Tensor) -> Tensor:
        """Encode images to embedding space"""
        emb = self.image_encoder(images)
        return self.image_proj(emb)
    
    def encode_audio(self, mel: Tensor) -> Tensor:
        """Encode audio (mel spectrogram) to embedding space"""
        emb = self.audio_encoder(mel)
        return self.audio_proj(emb)
    
    def encode_text(self, input_ids: Tensor) -> Tensor:
        """Encode text to embedding space"""
        emb = self.text_encoder(input_ids)
        return self.text_proj(emb)
    
    def encode_video(self, video: Tensor) -> Tensor:
        """
        Encode video to embedding space.
        
        Args:
            video: [B, T, C, H, W] or [B, C, T, H, W]
        """
        if video.dim() == 5 and video.shape[2] == 3:
            # [B, T, C, H, W] format
            batch_size, num_frames = video.shape[:2]
            video_flat = video.view(-1, *video.shape[2:])
        else:
            # [B, C, T, H, W] format
            video = video.permute(0, 2, 1, 3, 4)  # to [B, T, C, H, W]
            batch_size, num_frames = video.shape[:2]
            video_flat = video.view(-1, *video.shape[2:])
        
        # Encode each frame
        frame_embs = self.image_encoder(video_flat)  # [B*T, D]
        frame_embs = frame_embs.view(batch_size, num_frames, -1)  # [B, T, D]
        
        # Temporal aggregation
        video_emb = self.temporal_pool(frame_embs)
        video_emb = video_emb.mean(dim=1)  # [B, D]
        
        return self.video_proj(video_emb)
    
    def forward(
        self,
        images: Optional[Tensor] = None,
        audio: Optional[Tensor] = None,
        text_ids: Optional[Tensor] = None,
        video: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Encode any combination of modalities.
        
        Returns dict with embeddings for each provided modality.
        """
        embeddings = {}
        
        if images is not None:
            embeddings['image'] = self.encode_image(images)
        
        if audio is not None:
            embeddings['audio'] = self.encode_audio(audio)
        
        if text_ids is not None:
            embeddings['text'] = self.encode_text(text_ids)
        
        if video is not None:
            embeddings['video'] = self.encode_video(video)
        
        return embeddings
    
    @classmethod
    def from_pretrained(cls, path: str, device: str = "cuda") -> "MultimodalEncoder":
        import os
        from safetensors.torch import load_file
        
        model = cls()
        
        weights_path = os.path.join(path, "multimodal_encoder.safetensors")
        if os.path.exists(weights_path):
            model.load_state_dict(load_file(weights_path))
        
        return model.to(device)
