"""
📊 Data Loading for Bagley Training
Dataset configurations and efficient data loading
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Iterator
from pathlib import Path
import logging

import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader, IterableDataset

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """
    📊 Dataset Configuration
    
    Supports:
    - Local files (JSON, Parquet, Arrow)
    - Streaming from disk
    - WebDataset format
    - Custom formats
    """
    
    # Data sources
    train_files: List[str] = field(default_factory=list)
    eval_files: List[str] = field(default_factory=list)
    
    # Format
    data_format: str = "jsonl"  # jsonl, parquet, arrow, webdataset
    
    # Columns/fields
    text_column: str = "text"
    image_column: Optional[str] = "image"
    audio_column: Optional[str] = "audio"
    
    # Processing
    max_seq_length: int = 8192
    truncation: bool = True
    padding: str = "max_length"  # max_length, longest, do_not_pad
    
    # Streaming
    streaming: bool = True
    buffer_size: int = 10000
    shuffle_buffer_size: int = 10000
    
    # Preprocessing
    num_workers: int = 4
    prefetch_factor: int = 2
    
    # Chat format
    chat_template: Optional[str] = None
    system_prompt: Optional[str] = None


class ChatDataset(Dataset):
    """
    💬 Chat/Text Dataset
    
    For training chat models on conversation data.
    """
    
    def __init__(
        self,
        config: DatasetConfig,
        tokenizer: Any,
        split: str = "train",
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        
        # Load data
        files = config.train_files if split == "train" else config.eval_files
        self.examples = self._load_data(files)
        
        logger.info(f"Loaded {len(self.examples)} examples for {split}")
    
    def _load_data(self, files: List[str]) -> List[Dict]:
        """Load data from files"""
        examples = []
        
        for file_path in files:
            path = Path(file_path)
            
            if path.suffix == ".jsonl":
                with open(path) as f:
                    for line in f:
                        examples.append(json.loads(line))
                        
            elif path.suffix == ".json":
                with open(path) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        examples.extend(data)
                    else:
                        examples.append(data)
                        
            elif path.suffix == ".parquet":
                try:
                    import pandas as pd
                    df = pd.read_parquet(path)
                    examples.extend(df.to_dict('records'))
                except ImportError:
                    logger.error("pandas required for parquet files")
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Get text
        text = example.get(self.config.text_column, "")
        
        # Handle conversation format
        if "messages" in example:
            text = self._format_conversation(example["messages"])
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.config.max_seq_length,
            truncation=self.config.truncation,
            padding=self.config.padding,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0).clone(),
        }
    
    def _format_conversation(self, messages: List[Dict]) -> str:
        """Format conversation messages"""
        formatted = []
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                formatted.append(f"<|system|>\n{content}")
            elif role == "user":
                formatted.append(f"<|user|>\n{content}")
            elif role == "assistant":
                formatted.append(f"<|assistant|>\n{content}")
        
        return "\n".join(formatted)


class StreamingChatDataset(IterableDataset):
    """
    🌊 Streaming Chat Dataset
    
    Memory-efficient streaming for large datasets.
    """
    
    def __init__(
        self,
        config: DatasetConfig,
        tokenizer: Any,
        split: str = "train",
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.split = split
        self.files = config.train_files if split == "train" else config.eval_files
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Single process
            files = self.files
        else:
            # Multi-process: split files
            per_worker = len(self.files) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(self.files)
            files = self.files[start:end]
        
        for file_path in files:
            yield from self._stream_file(file_path)
    
    def _stream_file(self, file_path: str) -> Iterator[Dict[str, torch.Tensor]]:
        """Stream examples from a file"""
        path = Path(file_path)
        
        if path.suffix == ".jsonl":
            with open(path) as f:
                for line in f:
                    example = json.loads(line)
                    yield self._process_example(example)
    
    def _process_example(self, example: Dict) -> Dict[str, torch.Tensor]:
        """Process a single example"""
        text = example.get(self.config.text_column, "")
        
        if "messages" in example:
            text = self._format_conversation(example["messages"])
        
        encoding = self.tokenizer(
            text,
            max_length=self.config.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0).clone(),
        }
    
    def _format_conversation(self, messages: List[Dict]) -> str:
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted.append(f"<|{role}|>\n{content}")
        return "\n".join(formatted)


class ImageDataset(Dataset):
    """
    🎨 Image Dataset
    
    For training image generation models.
    """
    
    def __init__(
        self,
        config: DatasetConfig,
        image_size: int = 512,
        split: str = "train",
    ):
        self.config = config
        self.image_size = image_size
        self.split = split
        
        # Load metadata
        files = config.train_files if split == "train" else config.eval_files
        self.examples = self._load_data(files)
        
        # Transforms
        self.transform = self._get_transforms()
    
    def _load_data(self, files: List[str]) -> List[Dict]:
        examples = []
        for file_path in files:
            with open(file_path) as f:
                for line in f:
                    examples.append(json.loads(line))
        return examples
    
    def _get_transforms(self):
        """Get image transforms"""
        try:
            from torchvision import transforms
            
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        except ImportError:
            return None
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]
        
        # Load image
        image_path = example.get(self.config.image_column or "image", "")
        caption = example.get(self.config.text_column, "")
        
        try:
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
            
            if self.transform:
                image = self.transform(image)
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            image = torch.zeros(3, self.image_size, self.image_size)
        
        return {
            "image": image,
            "caption": caption,
        }


class AudioDataset(Dataset):
    """
    🎵 Audio Dataset
    
    For training TTS models.
    """
    
    def __init__(
        self,
        config: DatasetConfig,
        sample_rate: int = 24000,
        split: str = "train",
    ):
        self.config = config
        self.sample_rate = sample_rate
        self.split = split
        
        files = config.train_files if split == "train" else config.eval_files
        self.examples = self._load_data(files)
    
    def _load_data(self, files: List[str]) -> List[Dict]:
        examples = []
        for file_path in files:
            with open(file_path) as f:
                for line in f:
                    examples.append(json.loads(line))
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        example = self.examples[idx]
        
        audio_path = example.get(self.config.audio_column or "audio", "")
        text = example.get(self.config.text_column, "")
        
        try:
            import torchaudio
            
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
                
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            waveform = torch.zeros(1, self.sample_rate * 5)  # 5 second silence
        
        return {
            "audio": waveform.squeeze(0),
            "text": text,
        }


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> TorchDataLoader:
    """Create a DataLoader with optimal settings"""
    
    return TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch of examples"""
    result = {}
    
    for key in batch[0].keys():
        values = [example[key] for example in batch]
        
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        elif isinstance(values[0], str):
            result[key] = values
        else:
            result[key] = values
    
    return result
