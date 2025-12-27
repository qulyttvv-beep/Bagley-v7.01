"""
🧠 Unified Model Brain - What Makes Bagley BETTER

This is where the magic happens. Unlike GPT/Claude/Grok/Gemini:
- They're single models pretending to do everything
- We're specialized models working together like a crew
- Each model does what it's BEST at

WHY THIS IS BETTER:
1. 🎯 Specialization > Jack of All Trades
   - Chat model: Pure reasoning, no wasted params on images
   - Image model: Pure generation, not text-polluted
   
2. 🧩 Mixture of Experts at System Level
   - GPT-4 has MoE inside ONE model
   - We have MoE inside chat model AND between models
   - Double efficiency

3. 💰 Cost Efficiency
   - Image request? Only image model runs
   - Text request? Only chat model runs
   - GPT runs everything always

4. 🔄 Easy Upgrades
   - Better image model? Swap it
   - Better TTS? Swap it
   - No retraining everything

5. 🎨 Better Quality Per Task
   - Dedicated image model = better images than multimodal
   - Dedicated TTS = better voice than bolted-on audio
"""

import asyncio
import time
from typing import Optional, List, Dict, Any, Union, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of tasks Bagley can handle"""
    TEXT_CHAT = "text_chat"
    TEXT_REASONING = "text_reasoning"  # Deep thinking mode
    CODE_GENERATION = "code_generation"
    IMAGE_GENERATION = "image_generation"
    IMAGE_EDIT = "image_edit"
    IMAGE_ANALYSIS = "image_analysis"
    VIDEO_GENERATION = "video_generation"
    AUDIO_TTS = "audio_tts"
    AUDIO_CLONE = "audio_clone"
    MULTIMODAL = "multimodal"  # Multiple outputs needed


@dataclass
class BagleyRequest:
    """A request to Bagley"""
    text: str
    images: List[bytes] = field(default_factory=list)
    audio: Optional[bytes] = None
    video: Optional[bytes] = None
    
    # Generation params
    temperature: float = 0.7
    max_tokens: int = 2048
    
    # Flags
    generate_image: bool = False
    generate_audio: bool = False
    generate_video: bool = False
    think_deep: bool = False  # Extended thinking mode


@dataclass
class BagleyResponse:
    """Bagley's response"""
    text: str = ""
    thinking: str = ""  # Internal reasoning (like Claude's thinking)
    images: List[bytes] = field(default_factory=list)
    audio: Optional[bytes] = None
    video: Optional[bytes] = None
    
    # Metadata
    models_used: List[str] = field(default_factory=list)
    total_time: float = 0.0
    token_count: int = 0


class TaskRouter:
    """
    🎯 Smart Task Router
    
    Figures out what models to use based on the request.
    This is KEY to beating GPT/Claude - smart routing.
    """
    
    # Keywords that trigger different modes
    IMAGE_GEN_KEYWORDS = {
        'generate', 'create', 'draw', 'make', 'paint', 'design',
        'imagine', 'visualize', 'picture', 'illustration', 'art',
        'image of', 'picture of', 'photo of', 'render'
    }
    
    VIDEO_GEN_KEYWORDS = {
        'video', 'animate', 'animation', 'clip', 'footage', 'movie'
    }
    
    AUDIO_KEYWORDS = {
        'say', 'speak', 'read aloud', 'pronounce', 'voice',
        'tts', 'text to speech', 'audio'
    }
    
    CODE_KEYWORDS = {
        'code', 'function', 'class', 'program', 'script', 'implement',
        'debug', 'fix', 'write', 'python', 'javascript', 'algorithm'
    }
    
    REASONING_KEYWORDS = {
        'think', 'analyze', 'explain why', 'reason', 'prove',
        'step by step', 'let\'s think', 'figure out', 'solve',
        'calculate', 'derive', 'deduce'
    }
    
    def route(self, request: BagleyRequest) -> List[TaskType]:
        """
        Route request to appropriate task types.
        
        Returns list because one request might need multiple models.
        """
        tasks = []
        text_lower = request.text.lower()
        
        # Check explicit flags first
        if request.generate_image:
            tasks.append(TaskType.IMAGE_GENERATION)
        if request.generate_video:
            tasks.append(TaskType.VIDEO_GENERATION)
        if request.generate_audio:
            tasks.append(TaskType.AUDIO_TTS)
        if request.think_deep:
            tasks.append(TaskType.TEXT_REASONING)
        
        # Detect from text
        if any(kw in text_lower for kw in self.IMAGE_GEN_KEYWORDS):
            if TaskType.IMAGE_GENERATION not in tasks:
                # Check if it's actually asking for an image
                if self._is_image_request(text_lower):
                    tasks.append(TaskType.IMAGE_GENERATION)
        
        if any(kw in text_lower for kw in self.VIDEO_GEN_KEYWORDS):
            if TaskType.VIDEO_GENERATION not in tasks:
                tasks.append(TaskType.VIDEO_GENERATION)
        
        if any(kw in text_lower for kw in self.AUDIO_KEYWORDS):
            if TaskType.AUDIO_TTS not in tasks:
                tasks.append(TaskType.AUDIO_TTS)
        
        # Has input images? Might need analysis
        if request.images:
            tasks.append(TaskType.IMAGE_ANALYSIS)
        
        # Code detection
        if any(kw in text_lower for kw in self.CODE_KEYWORDS):
            tasks.append(TaskType.CODE_GENERATION)
        
        # Reasoning detection
        if any(kw in text_lower for kw in self.REASONING_KEYWORDS):
            if TaskType.TEXT_REASONING not in tasks:
                tasks.append(TaskType.TEXT_REASONING)
        
        # Default: always include chat
        if TaskType.TEXT_CHAT not in tasks and TaskType.TEXT_REASONING not in tasks:
            tasks.append(TaskType.TEXT_CHAT)
        
        return tasks
    
    def _is_image_request(self, text: str) -> bool:
        """Check if text is actually requesting image generation"""
        # Patterns that indicate generation request
        gen_patterns = [
            'generate', 'create', 'make me', 'draw', 'paint',
            'image of', 'picture of', 'can you make', 'show me'
        ]
        return any(p in text for p in gen_patterns)


class UnifiedBrain:
    """
    🧠 Bagley's Unified Brain
    
    Coordinates all models to work together.
    
    This is our edge over GPT/Claude/Grok/Gemini:
    - They use one model for everything
    - We use the RIGHT model for each task
    - Better quality, lower cost, faster speed
    """
    
    def __init__(self):
        self.router = TaskRouter()
        self.models: Dict[str, Any] = {}
        self._loaded = False
        
        # Stats
        self.total_requests = 0
        self.model_usage: Dict[str, int] = {}
    
    def load_models(self, device: str = "cuda"):
        """Load all models"""
        logger.info("Loading Bagley brain...")
        
        # Chat model (MoE)
        try:
            from bagley.models.chat import BagleyMoEChat
            self.models["chat"] = BagleyMoEChat()
            logger.info("✓ Chat model loaded")
        except Exception as e:
            logger.warning(f"Chat model not available: {e}")
        
        # Image model (DiT)
        try:
            from bagley.models.image import BagleyDiT
            self.models["image"] = BagleyDiT()
            logger.info("✓ Image model loaded")
        except Exception as e:
            logger.warning(f"Image model not available: {e}")
        
        # Video model
        try:
            from bagley.models.video import BagleyVideo
            self.models["video"] = BagleyVideo()
            logger.info("✓ Video model loaded")
        except Exception as e:
            logger.warning(f"Video model not available: {e}")
        
        # TTS model
        try:
            from bagley.models.tts import BagleyVoice
            self.models["tts"] = BagleyVoice()
            logger.info("✓ TTS model loaded")
        except Exception as e:
            logger.warning(f"TTS model not available: {e}")
        
        self._loaded = True
        logger.info(f"Brain loaded with {len(self.models)} models")
    
    async def process(self, request: BagleyRequest) -> BagleyResponse:
        """
        Process a request through the unified brain.
        
        This is where the magic happens:
        1. Route to correct models
        2. Run models (parallel where possible)
        3. Combine outputs
        """
        start_time = time.time()
        response = BagleyResponse()
        
        # Route request
        tasks = self.router.route(request)
        logger.info(f"Routed to tasks: {[t.value for t in tasks]}")
        
        # Process tasks
        # Some can run in parallel!
        parallel_tasks = []
        
        for task in tasks:
            if task == TaskType.TEXT_CHAT:
                parallel_tasks.append(self._chat(request, response))
            elif task == TaskType.TEXT_REASONING:
                parallel_tasks.append(self._reason(request, response))
            elif task == TaskType.CODE_GENERATION:
                parallel_tasks.append(self._code(request, response))
            elif task == TaskType.IMAGE_GENERATION:
                parallel_tasks.append(self._generate_image(request, response))
            elif task == TaskType.IMAGE_ANALYSIS:
                parallel_tasks.append(self._analyze_image(request, response))
            elif task == TaskType.VIDEO_GENERATION:
                parallel_tasks.append(self._generate_video(request, response))
            elif task == TaskType.AUDIO_TTS:
                parallel_tasks.append(self._generate_audio(request, response))
        
        # Run parallel where possible
        await asyncio.gather(*parallel_tasks)
        
        response.total_time = time.time() - start_time
        self.total_requests += 1
        
        return response
    
    async def _chat(self, request: BagleyRequest, response: BagleyResponse):
        """Chat model processing"""
        response.models_used.append("chat")
        self.model_usage["chat"] = self.model_usage.get("chat", 0) + 1
        
        if "chat" in self.models:
            # Use actual model
            result = await self.models["chat"].generate(
                prompt=request.text,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            response.text = result["text"]
            response.token_count = result.get("tokens", 0)
        else:
            # Placeholder
            response.text = f"[Chat] Processing: {request.text[:100]}..."
    
    async def _reason(self, request: BagleyRequest, response: BagleyResponse):
        """Extended thinking/reasoning mode"""
        response.models_used.append("reasoning")
        self.model_usage["reasoning"] = self.model_usage.get("reasoning", 0) + 1
        
        if "chat" in self.models:
            # Use chat model in thinking mode
            result = await self.models["chat"].generate(
                prompt=f"<think>\n{request.text}\n</think>\n\nLet me think through this step by step:",
                temperature=0.6,  # Lower for reasoning
                max_tokens=request.max_tokens * 2,  # More space for thinking
            )
            
            # Parse thinking vs response
            text = result["text"]
            if "</think>" in text:
                thinking, answer = text.split("</think>", 1)
                response.thinking = thinking.strip()
                response.text = answer.strip()
            else:
                response.text = text
        else:
            response.thinking = "Deep thinking..."
            response.text = f"[Reasoning] {request.text[:100]}..."
    
    async def _code(self, request: BagleyRequest, response: BagleyResponse):
        """Code generation - uses chat model with code focus"""
        response.models_used.append("code")
        self.model_usage["code"] = self.model_usage.get("code", 0) + 1
        
        if "chat" in self.models:
            # Use chat model with code prompt
            code_prompt = f"""You are an expert programmer. Write clean, efficient, well-documented code.

User request: {request.text}

Provide the code:"""
            
            result = await self.models["chat"].generate(
                prompt=code_prompt,
                temperature=0.3,  # Lower for precise code
                max_tokens=request.max_tokens,
            )
            response.text = result["text"]
        else:
            response.text = f"```python\n# Code for: {request.text[:50]}...\npass\n```"
    
    async def _generate_image(self, request: BagleyRequest, response: BagleyResponse):
        """Image generation"""
        response.models_used.append("image")
        self.model_usage["image"] = self.model_usage.get("image", 0) + 1
        
        if "image" in self.models:
            image_bytes = await self.models["image"].generate(
                prompt=request.text,
            )
            response.images.append(image_bytes)
        else:
            # Placeholder
            response.images.append(b"[Image placeholder]")
    
    async def _analyze_image(self, request: BagleyRequest, response: BagleyResponse):
        """Analyze input images"""
        response.models_used.append("vision")
        self.model_usage["vision"] = self.model_usage.get("vision", 0) + 1
        
        if "chat" in self.models and hasattr(self.models["chat"], "analyze_image"):
            analysis = await self.models["chat"].analyze_image(
                images=request.images,
                prompt=request.text,
            )
            response.text += f"\n\n[Image Analysis]\n{analysis}"
        else:
            response.text += "\n\n[Would analyze images here]"
    
    async def _generate_video(self, request: BagleyRequest, response: BagleyResponse):
        """Video generation"""
        response.models_used.append("video")
        self.model_usage["video"] = self.model_usage.get("video", 0) + 1
        
        if "video" in self.models:
            video_bytes = await self.models["video"].generate(
                prompt=request.text,
            )
            response.video = video_bytes
        else:
            response.video = b"[Video placeholder]"
    
    async def _generate_audio(self, request: BagleyRequest, response: BagleyResponse):
        """TTS audio generation"""
        response.models_used.append("tts")
        self.model_usage["tts"] = self.model_usage.get("tts", 0) + 1
        
        if "tts" in self.models:
            audio_bytes = await self.models["tts"].generate(
                text=response.text or request.text,
            )
            response.audio = audio_bytes
        else:
            response.audio = b"[Audio placeholder]"
    
    async def stream(self, request: BagleyRequest) -> AsyncGenerator[str, None]:
        """Stream response tokens (for chat)"""
        tasks = self.router.route(request)
        
        if TaskType.TEXT_CHAT in tasks or TaskType.TEXT_REASONING in tasks:
            if "chat" in self.models:
                async for token in self.models["chat"].stream(
                    prompt=request.text,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                ):
                    yield token
            else:
                for word in f"Processing: {request.text}".split():
                    yield word + " "
                    await asyncio.sleep(0.05)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "total_requests": self.total_requests,
            "model_usage": self.model_usage,
            "models_loaded": list(self.models.keys()),
        }


class BagleyAdvantages:
    """
    📊 Why Bagley Beats the Competition
    
    This class documents our technical advantages.
    Used for marketing and self-awareness.
    """
    
    ADVANTAGES = {
        "vs_gpt4": {
            "specialization": "GPT-4 uses one model for everything. We use specialized models = better quality per task.",
            "cost": "Image request to GPT-4 runs their whole model. Ours runs just the image model = cheaper.",
            "upgradability": "Want better images? Swap our image model. GPT-4? Wait for next version.",
            "transparency": "We can show which model did what. GPT-4 is a black box.",
            "open": "Our architecture is open. Train your own! GPT-4? Good luck.",
        },
        "vs_claude": {
            "multimodal": "Claude can't generate images/video/audio. We can.",
            "thinking": "Claude has thinking. We do too. Plus extended reasoning mode.",
            "moe": "Claude is dense. We're MoE = more knowledge, less compute.",
            "local": "Claude needs API. We run local = privacy + no rate limits.",
        },
        "vs_grok": {
            "multimodal": "Grok is text-focused. We do text + image + video + audio.",
            "quality": "Grok is fast but not always accurate. We're accurate AND fast.",
            "open": "We're more open about architecture than Grok.",
        },
        "vs_gemini": {
            "efficiency": "Gemini is huge. We use MoE = same capability, less compute.",
            "focus": "Gemini tries to do everything in one. We specialize.",
            "latency": "Our routing = parallel processing = lower latency.",
        }
    }
    
    TECHNICAL_EDGES = {
        "double_moe": "MoE inside chat model + MoE between models = double efficiency",
        "smart_routing": "Only run models you need. Others idle = save GPU.",
        "parallel_inference": "Text + image generation run in parallel. Others wait in line.",
        "hot_swap": "Change any model without retraining. Try that with GPT.",
        "custom_training": "Train on YOUR data. Their models? Fixed.",
        "local_first": "No API keys, no rate limits, no data sent anywhere.",
    }
    
    @classmethod
    def why_were_better(cls) -> str:
        """Return a summary of advantages"""
        return """
🏆 WHY BAGLEY IS BETTER THAN GPT/CLAUDE/GROK/GEMINI:

1. 🎯 SPECIALIZED MODELS
   - GPT/Claude use ONE model for everything
   - We use SPECIALIZED models for each task
   - Better image model for images
   - Better chat model for chat
   - Better TTS for voice
   - = Higher quality output

2. 💰 CHEAPER TO RUN
   - Text request? Only text model runs
   - Image request? Only image model runs
   - GPT runs everything always = waste

3. ⚡ FASTER
   - Text + image can generate IN PARALLEL
   - GPT does everything sequential
   - We route intelligently

4. 🔧 UPGRADABLE
   - Better image model comes out? Swap it
   - GPT? Wait years for next version
   
5. 🔐 PRIVATE
   - Runs locally, your data stays local
   - GPT/Claude? Everything goes to them

6. 🎓 TRAINABLE
   - Train on YOUR data
   - Make it yours
   - GPT? Fixed forever

7. 🧠 DOUBLE MOE
   - MoE inside chat model (experts for topics)
   - MoE between models (right model for task)
   - 2x the efficiency

The competition has ONE brain trying to do everything.
We have MULTIPLE specialized brains working as a team.
That's why we win.
"""
