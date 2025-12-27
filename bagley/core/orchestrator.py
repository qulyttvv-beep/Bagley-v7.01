"""
🧠 Bagley Orchestrator - The Central Brain
Coordinates all subsystems: chat, image, video, TTS, agents
"""

import asyncio
from typing import AsyncGenerator, Optional, Any, Dict, List, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import logging

from bagley.core.memory import BagleyMemory
from bagley.core.personality import BagleyPersonality
from bagley.core.router import IntentRouter, Intent

logger = logging.getLogger(__name__)


class BagleyState(Enum):
    """Current state of Bagley"""
    IDLE = "idle"
    THINKING = "thinking"
    GENERATING = "generating"
    SPEAKING = "speaking"
    EXECUTING = "executing"
    ERROR = "error"


@dataclass
class BagleyConfig:
    """Configuration for Bagley orchestrator"""
    # Model paths (for inference)
    chat_model_path: Optional[str] = None
    image_model_path: Optional[str] = None
    video_model_path: Optional[str] = None
    tts_model_path: Optional[str] = None
    
    # Performance settings
    max_context_length: int = 131072  # 128K context
    enable_streaming: bool = True
    enable_tts_narration: bool = True
    enable_personality: bool = True
    
    # Hardware settings
    device: str = "cuda"
    dtype: str = "bfloat16"
    offload_to_cpu: bool = False
    quantization: Optional[str] = None  # "int4", "int8", "fp8"
    
    # Memory settings
    memory_summarization_threshold: int = 50000  # tokens
    persistent_memory_path: Optional[str] = None
    
    # UI settings
    enable_ui_server: bool = True
    ui_port: int = 8765


@dataclass 
class BagleyResponse:
    """Structured response from Bagley"""
    text: str
    intent: Intent
    metadata: Dict[str, Any] = field(default_factory=dict)
    images: List[Any] = field(default_factory=list)
    videos: List[Any] = field(default_factory=list)
    audio: Optional[bytes] = None
    actions_taken: List[str] = field(default_factory=list)
    thinking_time: float = 0.0
    generation_time: float = 0.0


class BagleyOrchestrator:
    """
    🔥 The Bagley Orchestrator - Central Brain
    
    Coordinates all subsystems and provides a unified interface for:
    - Chat/conversation with the custom MoE model
    - Image generation with the custom DiT model
    - Video generation with the custom AsymmDiT model
    - TTS/voice with the custom DualAR model
    - Agent actions (VS Code, file system, browser, PC control)
    
    All responses are streamed and can be narrated via TTS.
    """
    
    def __init__(self, config: Optional[BagleyConfig] = None):
        self.config = config or BagleyConfig()
        self.state = BagleyState.IDLE
        
        # Core subsystems
        self.memory = BagleyMemory(
            max_context=self.config.max_context_length,
            summarization_threshold=self.config.memory_summarization_threshold,
            persistent_path=self.config.persistent_memory_path
        )
        self.personality = BagleyPersonality(enabled=self.config.enable_personality)
        self.router = IntentRouter()
        
        # Model instances (lazy loaded)
        self._chat_model = None
        self._image_model = None
        self._video_model = None
        self._tts_model = None
        
        # Agent instances (lazy loaded)
        self._vscode_agent = None
        self._file_agent = None
        self._browser_agent = None
        self._system_agent = None
        
        # Callbacks for UI updates
        self._state_callbacks: List[callable] = []
        self._stream_callbacks: List[callable] = []
        
        logger.info("🔥 Bagley Orchestrator initialized")
    
    # ==================== Model Loading ====================
    
    async def load_chat_model(self) -> None:
        """Lazy load the chat model"""
        if self._chat_model is not None:
            return
            
        logger.info("Loading BagleyMoE chat model...")
        from bagley.models.chat import BagleyMoEForCausalLM, BagleyMoEConfig
        
        if self.config.chat_model_path:
            # Load from checkpoint
            self._chat_model = BagleyMoEForCausalLM.from_pretrained(
                self.config.chat_model_path,
                device=self.config.device,
                dtype=self.config.dtype,
                quantization=self.config.quantization
            )
        else:
            # Initialize fresh model (for training)
            config = BagleyMoEConfig()
            self._chat_model = BagleyMoEForCausalLM(config)
            
        logger.info("✅ Chat model loaded")
    
    async def load_image_model(self) -> None:
        """Lazy load the image generation model"""
        if self._image_model is not None:
            return
            
        logger.info("Loading BagleyDiT image model...")
        from bagley.models.image import BagleyDiT, BagleyDiTConfig
        
        if self.config.image_model_path:
            self._image_model = BagleyDiT.from_pretrained(
                self.config.image_model_path,
                device=self.config.device,
                dtype=self.config.dtype
            )
        else:
            config = BagleyDiTConfig()
            self._image_model = BagleyDiT(config)
            
        logger.info("✅ Image model loaded")
    
    async def load_video_model(self) -> None:
        """Lazy load the video generation model"""
        if self._video_model is not None:
            return
            
        logger.info("Loading BagleyVideoMoE video model...")
        from bagley.models.video import BagleyVideoMoE, BagleyVideoConfig
        
        if self.config.video_model_path:
            self._video_model = BagleyVideoMoE.from_pretrained(
                self.config.video_model_path,
                device=self.config.device,
                dtype=self.config.dtype
            )
        else:
            config = BagleyVideoConfig()
            self._video_model = BagleyVideoMoE(config)
            
        logger.info("✅ Video model loaded")
    
    async def load_tts_model(self) -> None:
        """Lazy load the TTS model"""
        if self._tts_model is not None:
            return
            
        logger.info("Loading BagleyVoice TTS model...")
        from bagley.models.tts import BagleyVoice, BagleyVoiceConfig
        
        if self.config.tts_model_path:
            self._tts_model = BagleyVoice.from_pretrained(
                self.config.tts_model_path,
                device=self.config.device
            )
        else:
            config = BagleyVoiceConfig()
            self._tts_model = BagleyVoice(config)
            
        logger.info("✅ TTS model loaded")
    
    # ==================== Main Interface ====================
    
    async def process(
        self,
        user_input: str,
        files: Optional[List[str]] = None,
        images: Optional[List[Any]] = None,
        audio: Optional[bytes] = None,
        stream: bool = True
    ) -> Union[BagleyResponse, AsyncGenerator[str, None]]:
        """
        Main entry point for all Bagley interactions.
        
        Args:
            user_input: Text input from user
            files: Optional list of file paths
            images: Optional list of images (PIL Images or paths)
            audio: Optional audio input (for voice commands)
            stream: Whether to stream the response
            
        Returns:
            BagleyResponse or async generator of text chunks if streaming
        """
        start_time = time.time()
        self._set_state(BagleyState.THINKING)
        
        try:
            # 1. Detect intent
            intent = await self.router.detect_intent(
                user_input, 
                has_files=bool(files),
                has_images=bool(images)
            )
            logger.info(f"Detected intent: {intent}")
            
            # 2. Update memory with user input
            await self.memory.add_message("user", user_input, files=files, images=images)
            
            # 3. Route to appropriate handler
            if stream:
                return self._stream_response(intent, user_input, files, images, start_time)
            else:
                return await self._generate_response(intent, user_input, files, images, start_time)
                
        except Exception as e:
            self._set_state(BagleyState.ERROR)
            logger.error(f"Error processing request: {e}")
            raise
    
    async def _generate_response(
        self,
        intent: Intent,
        user_input: str,
        files: Optional[List[str]],
        images: Optional[List[Any]],
        start_time: float
    ) -> BagleyResponse:
        """Generate a complete response (non-streaming)"""
        self._set_state(BagleyState.GENERATING)
        
        response = BagleyResponse(text="", intent=intent)
        thinking_time = time.time() - start_time
        
        gen_start = time.time()
        
        # Route based on intent
        if intent == Intent.CHAT:
            await self.load_chat_model()
            context = await self.memory.get_context()
            personality_prompt = self.personality.get_system_prompt()
            
            response.text = await self._chat_model.generate(
                context=context,
                system_prompt=personality_prompt,
                max_tokens=4096
            )
            
        elif intent == Intent.IMAGE_GENERATION:
            await self.load_image_model()
            response.images = await self._image_model.generate(
                prompt=user_input,
                reference_images=images
            )
            response.text = self.personality.image_response()
            
        elif intent == Intent.VIDEO_GENERATION:
            await self.load_video_model()
            response.videos = await self._video_model.generate(
                prompt=user_input,
                reference_images=images
            )
            response.text = self.personality.video_response()
            
        elif intent == Intent.CODE:
            await self._ensure_vscode_agent()
            result = await self._vscode_agent.execute(user_input)
            response.text = result
            response.actions_taken = self._vscode_agent.get_actions()
            
        elif intent == Intent.FILE_OPERATION:
            await self._ensure_file_agent()
            result = await self._file_agent.execute(user_input, files)
            response.text = result
            response.actions_taken = self._file_agent.get_actions()
            
        elif intent == Intent.WEB_SEARCH:
            await self._ensure_browser_agent()
            result = await self._browser_agent.search(user_input)
            response.text = result
            
        elif intent == Intent.SYSTEM_CONTROL:
            await self._ensure_system_agent()
            result = await self._system_agent.execute(user_input)
            response.text = result
            response.actions_taken = self._system_agent.get_actions()
        
        # Apply personality post-processing
        response.text = self.personality.post_process(response.text)
        
        # Update memory
        await self.memory.add_message("assistant", response.text)
        
        response.thinking_time = thinking_time
        response.generation_time = time.time() - gen_start
        
        # TTS narration if enabled
        if self.config.enable_tts_narration:
            self._set_state(BagleyState.SPEAKING)
            await self.load_tts_model()
            response.audio = await self._tts_model.synthesize(
                response.text,
                voice="bagley"
            )
        
        self._set_state(BagleyState.IDLE)
        return response
    
    async def _stream_response(
        self,
        intent: Intent,
        user_input: str,
        files: Optional[List[str]],
        images: Optional[List[Any]],
        start_time: float
    ) -> AsyncGenerator[str, None]:
        """Stream response tokens as they're generated"""
        self._set_state(BagleyState.GENERATING)
        
        full_response = ""
        
        if intent == Intent.CHAT:
            await self.load_chat_model()
            context = await self.memory.get_context()
            personality_prompt = self.personality.get_system_prompt()
            
            async for token in self._chat_model.stream_generate(
                context=context,
                system_prompt=personality_prompt,
                max_tokens=4096
            ):
                full_response += token
                for callback in self._stream_callbacks:
                    await callback(token)
                yield token
                
        elif intent == Intent.IMAGE_GENERATION:
            await self.load_image_model()
            
            # Stream status updates while generating
            yield "🎨 Alright, generating your image... "
            
            images = await self._image_model.generate(prompt=user_input)
            
            yield f"\n\n✅ Done! Created {len(images)} image(s) with pure chaos energy 🔥"
            full_response = f"Generated {len(images)} images for: {user_input}"
            
        # ... handle other intents similarly
        
        # Update memory
        await self.memory.add_message("assistant", full_response)
        
        # Optionally narrate
        if self.config.enable_tts_narration and full_response:
            self._set_state(BagleyState.SPEAKING)
            await self.load_tts_model()
            await self._tts_model.stream_synthesize(full_response, voice="bagley")
        
        self._set_state(BagleyState.IDLE)
    
    # ==================== Agent Loading ====================
    
    async def _ensure_vscode_agent(self):
        if self._vscode_agent is None:
            from bagley.agents.vscode import VSCodeAgent
            self._vscode_agent = VSCodeAgent()
    
    async def _ensure_file_agent(self):
        if self._file_agent is None:
            from bagley.agents.filesystem import FileSystemAgent
            self._file_agent = FileSystemAgent()
    
    async def _ensure_browser_agent(self):
        if self._browser_agent is None:
            from bagley.agents.browser import BrowserAgent
            self._browser_agent = BrowserAgent()
    
    async def _ensure_system_agent(self):
        if self._system_agent is None:
            from bagley.agents.system import SystemAgent
            self._system_agent = SystemAgent()
    
    # ==================== State Management ====================
    
    def _set_state(self, state: BagleyState):
        """Update state and notify callbacks"""
        self.state = state
        for callback in self._state_callbacks:
            callback(state)
    
    def on_state_change(self, callback: callable):
        """Register a state change callback"""
        self._state_callbacks.append(callback)
    
    def on_stream(self, callback: callable):
        """Register a streaming callback"""
        self._stream_callbacks.append(callback)
    
    # ==================== Utility Methods ====================
    
    async def speak(self, text: str, voice: str = "bagley") -> bytes:
        """Generate speech from text"""
        await self.load_tts_model()
        return await self._tts_model.synthesize(text, voice=voice)
    
    async def generate_image(
        self, 
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int = 28,
        guidance_scale: float = 7.5
    ) -> List[Any]:
        """Direct image generation interface"""
        await self.load_image_model()
        return await self._image_model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale
        )
    
    async def generate_video(
        self,
        prompt: str,
        duration: float = 5.0,
        fps: int = 24,
        width: int = 1280,
        height: int = 720
    ) -> Any:
        """Direct video generation interface"""
        await self.load_video_model()
        return await self._video_model.generate(
            prompt=prompt,
            duration=duration,
            fps=fps,
            width=width,
            height=height
        )
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        return self.memory.get_stats()
    
    async def clear_memory(self, keep_personality: bool = True):
        """Clear conversation memory"""
        await self.memory.clear(keep_personality=keep_personality)
    
    async def save_state(self, path: str):
        """Save current state to disk"""
        await self.memory.save(path)
    
    async def load_state(self, path: str):
        """Load state from disk"""
        await self.memory.load(path)


# Convenience function for quick setup
def create_bagley(
    chat_model: Optional[str] = None,
    image_model: Optional[str] = None,
    video_model: Optional[str] = None,
    tts_model: Optional[str] = None,
    **kwargs
) -> BagleyOrchestrator:
    """
    Create a Bagley instance with optional model paths.
    
    Example:
        bagley = create_bagley(
            chat_model="./models/bagley-moe-70b",
            image_model="./models/bagley-dit-12b"
        )
    """
    config = BagleyConfig(
        chat_model_path=chat_model,
        image_model_path=image_model,
        video_model_path=video_model,
        tts_model_path=tts_model,
        **kwargs
    )
    return BagleyOrchestrator(config)
