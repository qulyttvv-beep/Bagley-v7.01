"""
🌐 Bagley API Server
====================
Host Bagley as an API - local or worldwide
FastAPI-based with authentication and rate limiting
"""

import os
import sys
import json
import time
import asyncio
import logging
import secrets
import hashlib
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


# ==================== Configuration ====================

class HostMode(Enum):
    LOCAL = "local"  # 127.0.0.1 only
    LAN = "lan"  # Local network
    WORLDWIDE = "worldwide"  # 0.0.0.0 (internet)


@dataclass
class APIConfig:
    """API Server Configuration"""
    
    # Hosting
    host_mode: HostMode = HostMode.LOCAL
    port: int = 8000
    
    # Security
    require_auth: bool = True
    api_keys: List[str] = field(default_factory=list)
    rate_limit_per_minute: int = 60
    max_request_size_mb: int = 100
    
    # SSL for worldwide hosting
    ssl_enabled: bool = False
    ssl_cert_path: str = ""
    ssl_key_path: str = ""
    
    # CORS
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    # Logging
    log_requests: bool = True
    log_path: str = "logs/api"
    
    # Limits
    max_concurrent_requests: int = 10
    request_timeout: int = 300  # seconds
    
    def get_host(self) -> str:
        """Get host based on mode"""
        if self.host_mode == HostMode.LOCAL:
            return "127.0.0.1"
        elif self.host_mode == HostMode.LAN:
            return "0.0.0.0"  # Binds to all interfaces but only LAN can reach
        else:  # WORLDWIDE
            return "0.0.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'host_mode': self.host_mode.value,
            'port': self.port,
            'require_auth': self.require_auth,
            'api_keys': self.api_keys,
            'rate_limit_per_minute': self.rate_limit_per_minute,
            'ssl_enabled': self.ssl_enabled,
            'cors_enabled': self.cors_enabled,
            'max_concurrent_requests': self.max_concurrent_requests,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'APIConfig':
        config = cls()
        if 'host_mode' in data:
            config.host_mode = HostMode(data['host_mode'])
        for key in ['port', 'require_auth', 'api_keys', 'rate_limit_per_minute',
                    'ssl_enabled', 'ssl_cert_path', 'ssl_key_path', 'cors_enabled',
                    'max_concurrent_requests', 'request_timeout']:
            if key in data:
                setattr(config, key, data[key])
        return config


# ==================== API Key Management ====================

class APIKeyManager:
    """Manage API keys for authentication"""
    
    def __init__(self, keys_file: str = "api_keys.json"):
        self.keys_file = keys_file
        self.keys: Dict[str, Dict[str, Any]] = {}
        self._load_keys()
    
    def _load_keys(self):
        """Load keys from file"""
        if os.path.exists(self.keys_file):
            try:
                with open(self.keys_file, 'r') as f:
                    self.keys = json.load(f)
            except:
                self.keys = {}
    
    def _save_keys(self):
        """Save keys to file"""
        with open(self.keys_file, 'w') as f:
            json.dump(self.keys, f, indent=2)
    
    def generate_key(self, name: str = "default", permissions: List[str] = None) -> str:
        """Generate a new API key"""
        key = f"bg_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        self.keys[key_hash] = {
            'name': name,
            'created': datetime.now().isoformat(),
            'permissions': permissions or ['chat', 'image', 'video', 'tts', '3d'],
            'requests_total': 0,
            'last_used': None,
            'active': True
        }
        
        self._save_keys()
        logger.info(f"Generated new API key: {name}")
        
        return key  # Return unhashed key (only time it's visible)
    
    def validate_key(self, key: str) -> Optional[Dict[str, Any]]:
        """Validate an API key"""
        if not key:
            return None
        
        # Remove 'Bearer ' prefix if present
        if key.startswith('Bearer '):
            key = key[7:]
        
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        if key_hash in self.keys:
            key_data = self.keys[key_hash]
            if key_data.get('active', True):
                # Update usage stats
                key_data['requests_total'] = key_data.get('requests_total', 0) + 1
                key_data['last_used'] = datetime.now().isoformat()
                self._save_keys()
                return key_data
        
        return None
    
    def revoke_key(self, key_hash: str):
        """Revoke an API key"""
        if key_hash in self.keys:
            self.keys[key_hash]['active'] = False
            self._save_keys()
            logger.info(f"Revoked API key: {self.keys[key_hash]['name']}")
    
    def list_keys(self) -> List[Dict[str, Any]]:
        """List all keys (without revealing the actual keys)"""
        return [
            {
                'hash': h[:16] + '...',
                **{k: v for k, v in data.items() if k != 'key'}
            }
            for h, data in self.keys.items()
        ]


# ==================== Rate Limiter ====================

class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, List[float]] = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed"""
        now = time.time()
        minute_ago = now - 60
        
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Remove old requests
        self.requests[client_id] = [
            t for t in self.requests[client_id] if t > minute_ago
        ]
        
        # Check limit
        if len(self.requests[client_id]) >= self.requests_per_minute:
            return False
        
        # Record this request
        self.requests[client_id].append(now)
        return True
    
    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for client"""
        if client_id not in self.requests:
            return self.requests_per_minute
        
        now = time.time()
        minute_ago = now - 60
        recent = [t for t in self.requests[client_id] if t > minute_ago]
        
        return max(0, self.requests_per_minute - len(recent))


# ==================== API Server ====================

class BagleyAPIServer:
    """
    🌐 Main API Server
    Hosts Bagley capabilities via REST API
    """
    
    def __init__(self, config: Optional[APIConfig] = None):
        self.config = config or APIConfig()
        self.key_manager = APIKeyManager()
        self.rate_limiter = RateLimiter(self.config.rate_limit_per_minute)
        self.app = None
        self.server = None
        self._running = False
        
        # Request stats
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'start_time': None,
        }
    
    def create_app(self):
        """Create FastAPI application"""
        try:
            from fastapi import FastAPI, HTTPException, Depends, Header, Request, BackgroundTasks
            from fastapi.middleware.cors import CORSMiddleware
            from fastapi.responses import JSONResponse, StreamingResponse
            from pydantic import BaseModel
            
        except ImportError:
            logger.error("FastAPI not installed. Run: pip install fastapi uvicorn")
            return None
        
        app = FastAPI(
            title="Bagley API",
            description="AI Assistant API - Chat, Image, Video, TTS, 3D Generation",
            version="7.01",
            docs_url="/docs" if self.config.host_mode != HostMode.WORLDWIDE else None,
        )
        
        # CORS
        if self.config.cors_enabled:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config.cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # ===== Request Models =====
        
        class ChatRequest(BaseModel):
            messages: List[Dict[str, str]]
            max_tokens: int = 2048
            temperature: float = 0.7
            stream: bool = False
        
        class ImageRequest(BaseModel):
            prompt: str
            negative_prompt: str = ""
            width: int = 1024
            height: int = 1024
            steps: int = 30
        
        class VideoRequest(BaseModel):
            prompt: str
            frames: int = 49
            fps: int = 24
            width: int = 1280
            height: int = 720
            generate_audio: bool = True
        
        class TTSRequest(BaseModel):
            text: str
            voice: str = "default"
            speed: float = 1.0
        
        class Model3DRequest(BaseModel):
            prompt: str
            format: str = "glb"
            texture: bool = True
        
        # ===== Auth Dependency =====
        
        async def verify_api_key(authorization: str = Header(None)):
            if not self.config.require_auth:
                return {'permissions': ['chat', 'image', 'video', 'tts', '3d']}
            
            if not authorization:
                raise HTTPException(status_code=401, detail="API key required")
            
            key_data = self.key_manager.validate_key(authorization)
            if not key_data:
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            return key_data
        
        # ===== Rate Limit Middleware =====
        
        @app.middleware("http")
        async def rate_limit_middleware(request: Request, call_next):
            client_ip = request.client.host
            
            if not self.rate_limiter.is_allowed(client_ip):
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Rate limit exceeded"}
                )
            
            response = await call_next(request)
            return response
        
        # ===== Endpoints =====
        
        @app.get("/")
        async def root():
            return {
                "name": "Bagley API",
                "version": "7.01",
                "status": "online",
                "endpoints": ["/chat", "/image", "/video", "/tts", "/3d"]
            }
        
        @app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "uptime": str(datetime.now() - self.stats['start_time']) if self.stats['start_time'] else "0",
                "total_requests": self.stats['total_requests']
            }
        
        @app.post("/chat")
        async def chat(request: ChatRequest, key_data: dict = Depends(verify_api_key)):
            if 'chat' not in key_data.get('permissions', []):
                raise HTTPException(status_code=403, detail="No chat permission")
            
            self.stats['total_requests'] += 1
            
            try:
                # TODO: Connect to actual chat model
                response = {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": f"[Bagley API] Received {len(request.messages)} messages. Model not loaded."
                        }
                    }],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0
                    }
                }
                self.stats['successful_requests'] += 1
                return response
                
            except Exception as e:
                self.stats['failed_requests'] += 1
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/image")
        async def generate_image(request: ImageRequest, key_data: dict = Depends(verify_api_key)):
            if 'image' not in key_data.get('permissions', []):
                raise HTTPException(status_code=403, detail="No image permission")
            
            self.stats['total_requests'] += 1
            
            try:
                # TODO: Connect to actual image model
                response = {
                    "status": "queued",
                    "message": f"Image generation queued: {request.prompt[:50]}..."
                }
                self.stats['successful_requests'] += 1
                return response
                
            except Exception as e:
                self.stats['failed_requests'] += 1
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/video")
        async def generate_video(request: VideoRequest, key_data: dict = Depends(verify_api_key)):
            if 'video' not in key_data.get('permissions', []):
                raise HTTPException(status_code=403, detail="No video permission")
            
            self.stats['total_requests'] += 1
            
            try:
                # TODO: Connect to actual video model
                response = {
                    "status": "queued",
                    "message": f"Video generation queued: {request.prompt[:50]}..."
                }
                self.stats['successful_requests'] += 1
                return response
                
            except Exception as e:
                self.stats['failed_requests'] += 1
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/tts")
        async def text_to_speech(request: TTSRequest, key_data: dict = Depends(verify_api_key)):
            if 'tts' not in key_data.get('permissions', []):
                raise HTTPException(status_code=403, detail="No TTS permission")
            
            self.stats['total_requests'] += 1
            
            try:
                # TODO: Connect to actual TTS model
                response = {
                    "status": "queued",
                    "message": f"TTS queued: {request.text[:50]}..."
                }
                self.stats['successful_requests'] += 1
                return response
                
            except Exception as e:
                self.stats['failed_requests'] += 1
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/3d")
        async def generate_3d(request: Model3DRequest, key_data: dict = Depends(verify_api_key)):
            if '3d' not in key_data.get('permissions', []):
                raise HTTPException(status_code=403, detail="No 3D permission")
            
            self.stats['total_requests'] += 1
            
            try:
                # TODO: Connect to actual 3D model
                response = {
                    "status": "queued",
                    "message": f"3D generation queued: {request.prompt[:50]}..."
                }
                self.stats['successful_requests'] += 1
                return response
                
            except Exception as e:
                self.stats['failed_requests'] += 1
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/stats")
        async def get_stats(key_data: dict = Depends(verify_api_key)):
            return {
                **self.stats,
                'start_time': self.stats['start_time'].isoformat() if self.stats['start_time'] else None
            }
        
        self.app = app
        return app
    
    async def start_async(self):
        """Start server asynchronously"""
        if self.app is None:
            self.create_app()
        
        if self.app is None:
            logger.error("Failed to create app")
            return
        
        try:
            import uvicorn
            
            self.stats['start_time'] = datetime.now()
            self._running = True
            
            config = uvicorn.Config(
                app=self.app,
                host=self.config.get_host(),
                port=self.config.port,
                ssl_keyfile=self.config.ssl_key_path if self.config.ssl_enabled else None,
                ssl_certfile=self.config.ssl_cert_path if self.config.ssl_enabled else None,
                log_level="info"
            )
            
            self.server = uvicorn.Server(config)
            
            logger.info(f"Starting API server on {self.config.get_host()}:{self.config.port}")
            await self.server.serve()
            
        except ImportError:
            logger.error("uvicorn not installed. Run: pip install uvicorn")
    
    def start(self):
        """Start server (blocking)"""
        asyncio.run(self.start_async())
    
    def start_background(self):
        """Start server in background thread"""
        import threading
        
        def run():
            asyncio.run(self.start_async())
        
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        logger.info("API server started in background")
        return thread
    
    def stop(self):
        """Stop the server"""
        self._running = False
        if self.server:
            self.server.should_exit = True
        logger.info("API server stopped")
    
    def get_url(self) -> str:
        """Get the server URL"""
        protocol = "https" if self.config.ssl_enabled else "http"
        host = "localhost" if self.config.host_mode == HostMode.LOCAL else self.config.get_host()
        return f"{protocol}://{host}:{self.config.port}"


# ==================== Factory ====================

def create_api_server(
    host_mode: str = "local",
    port: int = 8000,
    require_auth: bool = True
) -> BagleyAPIServer:
    """Create an API server"""
    config = APIConfig(
        host_mode=HostMode(host_mode),
        port=port,
        require_auth=require_auth
    )
    return BagleyAPIServer(config)


# ==================== Exports ====================

__all__ = [
    'HostMode',
    'APIConfig',
    'APIKeyManager',
    'RateLimiter',
    'BagleyAPIServer',
    'create_api_server',
]
