"""
📁 Multimodal Processor - Universal File Handler
Drag-and-drop handling for all file types
"""

import os
import mimetypes
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from pathlib import Path
import logging
import base64

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class ProcessedFile:
    """Container for processed file data"""
    file_path: str
    file_type: str
    mime_type: str
    content: Any  # Tensor for media, str for text
    metadata: Dict[str, Any]
    embedding: Optional[Tensor] = None  # Multimodal embedding


class MultimodalProcessor:
    """
    📁 Universal File Processor
    
    Handles all file types:
    - Images: PNG, JPG, WEBP, GIF, BMP, TIFF
    - Audio: WAV, MP3, FLAC, OGG, M4A
    - Video: MP4, AVI, MOV, MKV, WEBM
    - Documents: PDF, DOCX, TXT, MD, HTML
    - Code: PY, JS, TS, C, CPP, JAVA, RS, GO, etc.
    - Data: JSON, CSV, XML, YAML
    
    Extracts content and generates embeddings for Bagley to understand.
    """
    
    # Supported file extensions by category
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff', '.svg'}
    AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma'}
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv', '.flv'}
    DOCUMENT_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt', '.md', '.html', '.rtf', '.odt'}
    CODE_EXTENSIONS = {'.py', '.js', '.ts', '.jsx', '.tsx', '.c', '.cpp', '.h', '.hpp',
                      '.java', '.rs', '.go', '.rb', '.php', '.swift', '.kt', '.scala',
                      '.cs', '.fs', '.hs', '.ml', '.r', '.jl', '.lua', '.sh', '.bash',
                      '.ps1', '.sql', '.css', '.scss', '.less', '.vue', '.svelte'}
    DATA_EXTENSIONS = {'.json', '.csv', '.xml', '.yaml', '.yml', '.toml', '.ini', '.cfg'}
    
    def __init__(
        self,
        max_image_size: int = 4096,
        max_audio_duration: float = 600.0,  # 10 minutes
        max_video_duration: float = 300.0,  # 5 minutes
        device: str = "cuda",
    ):
        self.max_image_size = max_image_size
        self.max_audio_duration = max_audio_duration
        self.max_video_duration = max_video_duration
        self.device = device
        
        # Initialize encoders lazily
        self._image_encoder = None
        self._audio_encoder = None
        self._video_encoder = None
        
        logger.info("Initialized MultimodalProcessor")
    
    def detect_file_type(self, file_path: str) -> str:
        """Detect file type from extension"""
        ext = Path(file_path).suffix.lower()
        
        if ext in self.IMAGE_EXTENSIONS:
            return "image"
        elif ext in self.AUDIO_EXTENSIONS:
            return "audio"
        elif ext in self.VIDEO_EXTENSIONS:
            return "video"
        elif ext in self.DOCUMENT_EXTENSIONS:
            return "document"
        elif ext in self.CODE_EXTENSIONS:
            return "code"
        elif ext in self.DATA_EXTENSIONS:
            return "data"
        else:
            return "unknown"
    
    async def process(self, file_path: str) -> ProcessedFile:
        """
        Process any file type.
        
        Args:
            file_path: Path to the file
            
        Returns:
            ProcessedFile with content and optional embedding
        """
        file_path = str(file_path)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_type = self.detect_file_type(file_path)
        mime_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        
        # Get file metadata
        stat = os.stat(file_path)
        metadata = {
            "size_bytes": stat.st_size,
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "extension": Path(file_path).suffix,
        }
        
        # Process based on type
        if file_type == "image":
            content, embedding = await self._process_image(file_path)
        elif file_type == "audio":
            content, embedding = await self._process_audio(file_path)
        elif file_type == "video":
            content, embedding = await self._process_video(file_path)
        elif file_type == "document":
            content, embedding = await self._process_document(file_path)
        elif file_type == "code":
            content, embedding = await self._process_code(file_path)
        elif file_type == "data":
            content, embedding = await self._process_data(file_path)
        else:
            content, embedding = await self._process_binary(file_path)
        
        return ProcessedFile(
            file_path=file_path,
            file_type=file_type,
            mime_type=mime_type,
            content=content,
            metadata=metadata,
            embedding=embedding,
        )
    
    async def process_batch(self, file_paths: List[str]) -> List[ProcessedFile]:
        """Process multiple files"""
        results = []
        for path in file_paths:
            try:
                result = await self.process(path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
        return results
    
    # ==================== Type-Specific Processors ====================
    
    async def _process_image(self, file_path: str) -> tuple:
        """Process image file"""
        try:
            from PIL import Image
            import numpy as np
            
            img = Image.open(file_path)
            
            # Resize if too large
            if max(img.size) > self.max_image_size:
                ratio = self.max_image_size / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            
            # Convert to tensor
            img_array = np.array(img.convert('RGB'))
            content = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
            
            # Generate embedding (placeholder)
            embedding = torch.randn(1, 768)  # Would use CLIP/SigLIP
            
            return content, embedding
            
        except ImportError:
            logger.warning("PIL not available, returning path only")
            return file_path, None
    
    async def _process_audio(self, file_path: str) -> tuple:
        """Process audio file"""
        try:
            import torchaudio
            
            waveform, sample_rate = torchaudio.load(file_path)
            
            # Truncate if too long
            max_samples = int(self.max_audio_duration * sample_rate)
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]
            
            # Resample to 16kHz for processing
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Generate embedding (placeholder - would use Whisper/CLAP)
            embedding = torch.randn(1, 768)
            
            return waveform, embedding
            
        except ImportError:
            logger.warning("torchaudio not available, returning path only")
            return file_path, None
    
    async def _process_video(self, file_path: str) -> tuple:
        """Process video file"""
        try:
            import torchvision.io as io
            
            # Read video
            video, audio, info = io.read_video(file_path, pts_unit='sec')
            
            # Sample frames if too long
            fps = info.get('video_fps', 30)
            max_frames = int(self.max_video_duration * fps)
            
            if video.shape[0] > max_frames:
                # Sample uniformly
                indices = torch.linspace(0, video.shape[0] - 1, max_frames).long()
                video = video[indices]
            
            # Normalize
            video = video.permute(0, 3, 1, 2).float() / 255.0
            
            # Generate embedding (placeholder)
            embedding = torch.randn(1, 768)
            
            return {"video": video, "audio": audio, "info": info}, embedding
            
        except ImportError:
            logger.warning("torchvision not available, returning path only")
            return file_path, None
    
    async def _process_document(self, file_path: str) -> tuple:
        """Process document file"""
        ext = Path(file_path).suffix.lower()
        
        if ext == '.pdf':
            content = await self._read_pdf(file_path)
        elif ext in {'.docx', '.doc'}:
            content = await self._read_docx(file_path)
        elif ext == '.html':
            content = await self._read_html(file_path)
        else:
            # Plain text
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        
        # Generate text embedding (placeholder)
        embedding = torch.randn(1, 768)
        
        return content, embedding
    
    async def _read_pdf(self, file_path: str) -> str:
        """Extract text from PDF"""
        try:
            import pypdf
            
            reader = pypdf.PdfReader(file_path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return text
            
        except ImportError:
            return f"[PDF file: {file_path}]"
    
    async def _read_docx(self, file_path: str) -> str:
        """Extract text from DOCX"""
        try:
            from docx import Document
            
            doc = Document(file_path)
            text = "\n".join(para.text for para in doc.paragraphs)
            return text
            
        except ImportError:
            return f"[DOCX file: {file_path}]"
    
    async def _read_html(self, file_path: str) -> str:
        """Extract text from HTML"""
        try:
            from bs4 import BeautifulSoup
            
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            return soup.get_text(separator='\n', strip=True)
            
        except ImportError:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    async def _process_code(self, file_path: str) -> tuple:
        """Process code file"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Detect language from extension
        ext = Path(file_path).suffix.lower()
        language = self._detect_code_language(ext)
        
        # Format with language tag
        formatted_content = f"```{language}\n{content}\n```"
        
        # Generate embedding (placeholder)
        embedding = torch.randn(1, 768)
        
        return formatted_content, embedding
    
    def _detect_code_language(self, ext: str) -> str:
        """Map extension to language name"""
        lang_map = {
            '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
            '.jsx': 'jsx', '.tsx': 'tsx', '.c': 'c', '.cpp': 'cpp',
            '.h': 'c', '.hpp': 'cpp', '.java': 'java', '.rs': 'rust',
            '.go': 'go', '.rb': 'ruby', '.php': 'php', '.swift': 'swift',
            '.kt': 'kotlin', '.scala': 'scala', '.cs': 'csharp',
            '.fs': 'fsharp', '.hs': 'haskell', '.ml': 'ocaml',
            '.r': 'r', '.jl': 'julia', '.lua': 'lua', '.sh': 'bash',
            '.bash': 'bash', '.ps1': 'powershell', '.sql': 'sql',
            '.css': 'css', '.scss': 'scss', '.less': 'less',
            '.vue': 'vue', '.svelte': 'svelte',
        }
        return lang_map.get(ext, 'text')
    
    async def _process_data(self, file_path: str) -> tuple:
        """Process data file (JSON, CSV, XML, etc.)"""
        import json
        
        ext = Path(file_path).suffix.lower()
        
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()
        
        try:
            if ext == '.json':
                content = json.loads(raw_content)
            elif ext == '.csv':
                content = raw_content  # Would parse with pandas
            elif ext in {'.yaml', '.yml'}:
                try:
                    import yaml
                    content = yaml.safe_load(raw_content)
                except ImportError:
                    content = raw_content
            else:
                content = raw_content
        except Exception as e:
            logger.warning(f"Error parsing {ext}: {e}")
            content = raw_content
        
        # Generate embedding
        embedding = torch.randn(1, 768)
        
        return content, embedding
    
    async def _process_binary(self, file_path: str) -> tuple:
        """Process unknown binary file"""
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Just return file info
        return {
            "type": "binary",
            "size": len(content),
            "path": file_path,
        }, None
