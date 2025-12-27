"""
📊 Data Pipeline - Smart Training Data Management

Auto-detects and prepares data for each model type.
"""

import os
import json
import hashlib
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataStats:
    """Statistics about training data"""
    total_files: int = 0
    total_samples: int = 0
    total_tokens: int = 0
    total_size_gb: float = 0.0
    by_type: Dict[str, int] = None
    
    def __post_init__(self):
        if self.by_type is None:
            self.by_type = {}


class ChatDataProcessor:
    """
    💬 Process chat/text data for language model training
    
    Supports formats:
    - ShareGPT: {"conversations": [{"from": "human/gpt", "value": "..."}]}
    - Alpaca: {"instruction": "...", "input": "...", "output": "..."}
    - OpenAI: {"messages": [{"role": "user/assistant", "content": "..."}]}
    - Simple: {"text": "..."} or {"prompt": "...", "response": "..."}
    """
    
    SUPPORTED_FORMATS = ['sharegpt', 'alpaca', 'openai', 'simple']
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
    
    def detect_format(self, sample: Dict) -> str:
        """Detect the format of a data sample"""
        keys = set(sample.keys())
        
        if 'conversations' in keys:
            return 'sharegpt'
        elif 'messages' in keys:
            return 'openai'
        elif 'instruction' in keys:
            return 'alpaca'
        elif 'text' in keys or ('prompt' in keys and 'response' in keys):
            return 'simple'
        else:
            return 'unknown'
    
    def convert_to_chat(self, sample: Dict, format_type: str = None) -> List[Dict[str, str]]:
        """Convert any format to unified chat format"""
        if format_type is None:
            format_type = self.detect_format(sample)
        
        messages = []
        
        if format_type == 'sharegpt':
            for turn in sample.get('conversations', []):
                role = 'user' if turn['from'] == 'human' else 'assistant'
                messages.append({'role': role, 'content': turn['value']})
        
        elif format_type == 'openai':
            for msg in sample.get('messages', []):
                if msg['role'] in ['user', 'assistant', 'system']:
                    messages.append({'role': msg['role'], 'content': msg['content']})
        
        elif format_type == 'alpaca':
            # System prompt (optional)
            if sample.get('system'):
                messages.append({'role': 'system', 'content': sample['system']})
            
            # User instruction + input
            user_content = sample.get('instruction', '')
            if sample.get('input'):
                user_content += f"\n\n{sample['input']}"
            messages.append({'role': 'user', 'content': user_content})
            
            # Assistant output
            messages.append({'role': 'assistant', 'content': sample.get('output', '')})
        
        elif format_type == 'simple':
            if 'text' in sample:
                # Just raw text
                messages.append({'role': 'text', 'content': sample['text']})
            else:
                # Prompt/response
                messages.append({'role': 'user', 'content': sample.get('prompt', '')})
                messages.append({'role': 'assistant', 'content': sample.get('response', '')})
        
        return messages
    
    def process_file(self, file_path: str) -> Generator[List[Dict], None, None]:
        """Process a single file, yielding chat samples"""
        path = Path(file_path)
        
        try:
            if path.suffix == '.jsonl':
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            sample = json.loads(line)
                            yield self.convert_to_chat(sample)
            
            elif path.suffix == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for sample in data:
                            yield self.convert_to_chat(sample)
                    else:
                        yield self.convert_to_chat(data)
            
            elif path.suffix == '.txt':
                # Plain text - treat as training text
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    yield [{'role': 'text', 'content': text}]
                    
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    def format_for_training(
        self,
        messages: List[Dict],
        add_generation_prompt: bool = False,
    ) -> str:
        """Format messages for training"""
        # ChatML format
        formatted = ""
        
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'text':
                formatted += content
            elif role == 'system':
                formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == 'user':
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == 'assistant':
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        
        if add_generation_prompt:
            formatted += "<|im_start|>assistant\n"
        
        return formatted


class ImageDataProcessor:
    """
    🎨 Process image data for diffusion model training
    
    Supports:
    - Image + caption pairs
    - Image + JSON metadata
    - Folder of images with txt captions
    """
    
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    
    def __init__(self, image_size: int = 1024):
        self.image_size = image_size
    
    def find_caption(self, image_path: Path) -> Optional[str]:
        """Find caption for an image"""
        # Check for .txt with same name
        txt_path = image_path.with_suffix('.txt')
        if txt_path.exists():
            return txt_path.read_text(encoding='utf-8').strip()
        
        # Check for .json with same name
        json_path = image_path.with_suffix('.json')
        if json_path.exists():
            data = json.loads(json_path.read_text())
            return data.get('caption', data.get('description', data.get('text', '')))
        
        # Check for _caption.txt
        caption_path = image_path.parent / f"{image_path.stem}_caption.txt"
        if caption_path.exists():
            return caption_path.read_text(encoding='utf-8').strip()
        
        return None
    
    def process_folder(
        self,
        folder: str,
        recursive: bool = True,
    ) -> Generator[Tuple[Path, str], None, None]:
        """Process folder of images"""
        folder_path = Path(folder)
        
        pattern = '**/*' if recursive else '*'
        
        for file_path in folder_path.glob(pattern):
            if file_path.suffix.lower() in self.IMAGE_EXTENSIONS:
                caption = self.find_caption(file_path)
                if caption:
                    yield (file_path, caption)
                else:
                    logger.warning(f"No caption found for {file_path}")
    
    def process_metadata_file(
        self,
        metadata_file: str,
        image_folder: str = None,
    ) -> Generator[Tuple[Path, str], None, None]:
        """Process metadata file (JSON/JSONL with image paths and captions)"""
        path = Path(metadata_file)
        base_folder = Path(image_folder) if image_folder else path.parent
        
        try:
            if path.suffix == '.jsonl':
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            image_path = base_folder / item.get('image', item.get('file', ''))
                            caption = item.get('caption', item.get('text', ''))
                            if image_path.exists() and caption:
                                yield (image_path, caption)
            
            elif path.suffix == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    items = data if isinstance(data, list) else data.get('data', [])
                    for item in items:
                        image_path = base_folder / item.get('image', item.get('file', ''))
                        caption = item.get('caption', item.get('text', ''))
                        if image_path.exists() and caption:
                            yield (image_path, caption)
                            
        except Exception as e:
            logger.error(f"Error processing {metadata_file}: {e}")


class AudioDataProcessor:
    """
    🎵 Process audio data for TTS training
    
    Supports:
    - Audio + transcript pairs
    - LJSpeech format
    - Common Voice format
    """
    
    AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    
    def find_transcript(self, audio_path: Path) -> Optional[str]:
        """Find transcript for audio file"""
        # Check .txt with same name
        txt_path = audio_path.with_suffix('.txt')
        if txt_path.exists():
            return txt_path.read_text(encoding='utf-8').strip()
        
        # Check .lab (HTK label file)
        lab_path = audio_path.with_suffix('.lab')
        if lab_path.exists():
            return lab_path.read_text(encoding='utf-8').strip()
        
        return None
    
    def process_folder(
        self,
        folder: str,
        recursive: bool = True,
    ) -> Generator[Tuple[Path, str], None, None]:
        """Process folder of audio files"""
        folder_path = Path(folder)
        pattern = '**/*' if recursive else '*'
        
        for file_path in folder_path.glob(pattern):
            if file_path.suffix.lower() in self.AUDIO_EXTENSIONS:
                transcript = self.find_transcript(file_path)
                if transcript:
                    yield (file_path, transcript)
    
    def process_ljspeech(
        self,
        folder: str,
    ) -> Generator[Tuple[Path, str], None, None]:
        """Process LJSpeech format (metadata.csv)"""
        folder_path = Path(folder)
        metadata_path = folder_path / 'metadata.csv'
        wavs_folder = folder_path / 'wavs'
        
        if not metadata_path.exists():
            logger.error(f"metadata.csv not found in {folder}")
            return
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) >= 2:
                    file_id = parts[0]
                    transcript = parts[-1]  # Use normalized text if available
                    
                    audio_path = wavs_folder / f"{file_id}.wav"
                    if audio_path.exists():
                        yield (audio_path, transcript)


class SmartDataPipeline:
    """
    🧠 Smart Data Pipeline
    
    Automatically processes and routes data to correct model.
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.chat_processor = ChatDataProcessor()
        self.image_processor = ImageDataProcessor()
        self.audio_processor = AudioDataProcessor()
        
        self.stats = DataStats()
    
    def process_all(
        self,
        data_folders: List[str],
        num_workers: int = 4,
    ) -> Dict[str, str]:
        """
        Process all data from folders.
        
        Returns dict of output file paths per model type.
        """
        from bagley.training.monitor import SmartDataSorter, DataType
        
        # Sort data
        sorter = SmartDataSorter()
        sorted_data = sorter.scan_folders(data_folders)
        
        outputs = {}
        
        # Process chat data
        if sorted_data[DataType.CHAT]:
            chat_output = self.output_dir / 'chat_data.jsonl'
            self._process_chat_files(sorted_data[DataType.CHAT], chat_output)
            outputs['chat'] = str(chat_output)
        
        # Process image data
        if sorted_data[DataType.IMAGE]:
            image_output = self.output_dir / 'image_data.jsonl'
            self._process_image_files(sorted_data[DataType.IMAGE], image_output)
            outputs['image'] = str(image_output)
        
        # Process audio data
        if sorted_data[DataType.AUDIO]:
            audio_output = self.output_dir / 'audio_data.jsonl'
            self._process_audio_files(sorted_data[DataType.AUDIO], audio_output)
            outputs['tts'] = str(audio_output)
        
        # Process video data
        if sorted_data[DataType.VIDEO]:
            video_output = self.output_dir / 'video_data.jsonl'
            self._process_video_files(sorted_data[DataType.VIDEO], video_output)
            outputs['video'] = str(video_output)
        
        return outputs
    
    def _process_chat_files(self, files: List[Path], output_path: Path):
        """Process chat data files"""
        logger.info(f"Processing {len(files)} chat files...")
        
        count = 0
        with open(output_path, 'w', encoding='utf-8') as out:
            for file_path in files:
                for messages in self.chat_processor.process_file(str(file_path)):
                    formatted = self.chat_processor.format_for_training(messages)
                    if formatted.strip():
                        out.write(json.dumps({'text': formatted}) + '\n')
                        count += 1
        
        logger.info(f"Wrote {count} chat samples to {output_path}")
        self.stats.by_type['chat'] = count
    
    def _process_image_files(self, files: List[Path], output_path: Path):
        """Process image data files"""
        logger.info(f"Processing {len(files)} image files...")
        
        count = 0
        with open(output_path, 'w', encoding='utf-8') as out:
            for file_path in files:
                if file_path.suffix.lower() in {'.json', '.jsonl'}:
                    # Metadata file
                    for img_path, caption in self.image_processor.process_metadata_file(str(file_path)):
                        out.write(json.dumps({
                            'image': str(img_path),
                            'caption': caption,
                        }) + '\n')
                        count += 1
                else:
                    # Direct image
                    caption = self.image_processor.find_caption(file_path)
                    if caption:
                        out.write(json.dumps({
                            'image': str(file_path),
                            'caption': caption,
                        }) + '\n')
                        count += 1
        
        logger.info(f"Wrote {count} image samples to {output_path}")
        self.stats.by_type['image'] = count
    
    def _process_audio_files(self, files: List[Path], output_path: Path):
        """Process audio data files"""
        logger.info(f"Processing {len(files)} audio files...")
        
        count = 0
        with open(output_path, 'w', encoding='utf-8') as out:
            for file_path in files:
                if file_path.suffix.lower() in self.audio_processor.AUDIO_EXTENSIONS:
                    transcript = self.audio_processor.find_transcript(file_path)
                    if transcript:
                        out.write(json.dumps({
                            'audio': str(file_path),
                            'transcript': transcript,
                        }) + '\n')
                        count += 1
        
        logger.info(f"Wrote {count} audio samples to {output_path}")
        self.stats.by_type['audio'] = count
    
    def _process_video_files(self, files: List[Path], output_path: Path):
        """Process video data files"""
        logger.info(f"Processing {len(files)} video files...")
        
        count = 0
        with open(output_path, 'w', encoding='utf-8') as out:
            for file_path in files:
                # Look for caption file
                caption_path = file_path.with_suffix('.txt')
                if caption_path.exists():
                    caption = caption_path.read_text(encoding='utf-8').strip()
                    out.write(json.dumps({
                        'video': str(file_path),
                        'caption': caption,
                    }) + '\n')
                    count += 1
        
        logger.info(f"Wrote {count} video samples to {output_path}")
        self.stats.by_type['video'] = count
    
    def get_stats(self) -> DataStats:
        """Get processing statistics"""
        return self.stats
