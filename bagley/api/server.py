"""
🌐 Bagley API Server Module
===========================
Host Bagley API locally or worldwide with ngrok/cloudflare tunnel
"""

import os
import sys
import json
import time
import logging
import threading
import asyncio
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


# ==================== Configuration ====================

class TunnelProvider(Enum):
    NONE = "none"  # Local only
    NGROK = "ngrok"
    CLOUDFLARE = "cloudflare"
    LOCALHOST_RUN = "localhost.run"
    SERVEO = "serveo"


@dataclass
class APIConfig:
    """API Server Configuration"""
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    
    # SSL/Security
    enable_ssl: bool = False
    ssl_cert: str = ""
    ssl_key: str = ""
    
    # Authentication
    enable_auth: bool = True
    api_key: str = ""  # Auto-generated if empty
    rate_limit: int = 100  # Requests per minute
    
    # Tunnel for worldwide access
    tunnel_provider: TunnelProvider = TunnelProvider.NONE
    ngrok_token: str = ""
    cloudflare_token: str = ""
    
    # CORS
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    # Logging
    log_requests: bool = True
    log_path: str = "api_logs"
    
    def generate_api_key(self) -> str:
        """Generate a random API key"""
        import secrets
        self.api_key = f"bagley_{secrets.token_urlsafe(32)}"
        return self.api_key


# ==================== Smart Logger ====================

class SmartAPILogger:
    """
    📝 Smart API Logger
    - Request/response logging
    - Performance metrics
    - Error tracking
    - Usage analytics
    """
    
    def __init__(self, log_path: str = "api_logs"):
        self.log_path = Path(log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        self.current_log_file = None
        self.metrics = {
            'total_requests': 0,
            'successful': 0,
            'failed': 0,
            'avg_latency_ms': 0,
            'requests_by_endpoint': {},
            'requests_by_hour': {},
            'errors': []
        }
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging handlers"""
        timestamp = datetime.now().strftime('%Y%m%d')
        self.current_log_file = self.log_path / f"api_{timestamp}.log"
        
        # File handler
        handler = logging.FileHandler(self.current_log_file, encoding='utf-8')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        ))
        logger.addHandler(handler)
    
    def log_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        latency_ms: float,
        client_ip: str = "",
        request_data: Optional[Dict] = None,
        response_data: Optional[Dict] = None,
        error: Optional[str] = None
    ):
        """Log an API request"""
        self.metrics['total_requests'] += 1
        
        if status_code < 400:
            self.metrics['successful'] += 1
        else:
            self.metrics['failed'] += 1
        
        # Update average latency
        n = self.metrics['total_requests']
        old_avg = self.metrics['avg_latency_ms']
        self.metrics['avg_latency_ms'] = old_avg + (latency_ms - old_avg) / n
        
        # Track by endpoint
        if endpoint not in self.metrics['requests_by_endpoint']:
            self.metrics['requests_by_endpoint'][endpoint] = 0
        self.metrics['requests_by_endpoint'][endpoint] += 1
        
        # Track by hour
        hour = datetime.now().strftime('%Y-%m-%d %H:00')
        if hour not in self.metrics['requests_by_hour']:
            self.metrics['requests_by_hour'][hour] = 0
        self.metrics['requests_by_hour'][hour] += 1
        
        # Log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'endpoint': endpoint,
            'method': method,
            'status': status_code,
            'latency_ms': latency_ms,
            'client_ip': client_ip,
        }
        
        if error:
            log_entry['error'] = error
            self.metrics['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'endpoint': endpoint,
                'error': error
            })
            # Keep only last 100 errors
            self.metrics['errors'] = self.metrics['errors'][-100:]
        
        # Write to JSONL log
        jsonl_file = self.log_path / f"requests_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(jsonl_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Log to standard logger
        log_msg = f"{method} {endpoint} -> {status_code} ({latency_ms:.1f}ms)"
        if error:
            logger.error(log_msg + f" | Error: {error}")
        else:
            logger.info(log_msg)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return {
            **self.metrics,
            'uptime_seconds': time.time() - getattr(self, 'start_time', time.time()),
            'log_file': str(self.current_log_file)
        }
    
    def export_metrics(self, path: str = None):
        """Export metrics to JSON"""
        path = path or str(self.log_path / "metrics.json")
        with open(path, 'w') as f:
            json.dump(self.get_metrics(), f, indent=2)
        return path


# ==================== API Server ====================

class BagleyAPIServer:
    """
    🌐 Bagley API Server
    
    Hosts the Bagley API locally or worldwide
    """
    
    def __init__(self, config: Optional[APIConfig] = None):
        self.config = config or APIConfig()
        self.logger = SmartAPILogger(self.config.log_path)
        self.server = None
        self.tunnel_url = None
        self.is_running = False
        
        # Generate API key if needed
        if self.config.enable_auth and not self.config.api_key:
            self.config.generate_api_key()
        
        self._setup_app()
    
    def _setup_app(self):
        """Setup FastAPI/Flask app"""
        try:
            from fastapi import FastAPI, HTTPException, Request, Depends
            from fastapi.middleware.cors import CORSMiddleware
            from fastapi.security import APIKeyHeader
            from fastapi.responses import JSONResponse
            import uvicorn
            
            self.app = FastAPI(
                title="Bagley API",
                description="🤖 Bagley v7 - AI Assistant API",
                version="7.01"
            )
            
            # CORS
            if self.config.enable_cors:
                self.app.add_middleware(
                    CORSMiddleware,
                    allow_origins=self.config.cors_origins,
                    allow_credentials=True,
                    allow_methods=["*"],
                    allow_headers=["*"],
                )
            
            # API Key auth
            api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
            
            async def verify_api_key(api_key: str = Depends(api_key_header)):
                if self.config.enable_auth:
                    if not api_key or api_key != self.config.api_key:
                        raise HTTPException(status_code=401, detail="Invalid API key")
                return api_key
            
            # Request logging middleware
            @self.app.middleware("http")
            async def log_requests(request: Request, call_next):
                start_time = time.time()
                
                try:
                    response = await call_next(request)
                    latency = (time.time() - start_time) * 1000
                    
                    self.logger.log_request(
                        endpoint=str(request.url.path),
                        method=request.method,
                        status_code=response.status_code,
                        latency_ms=latency,
                        client_ip=request.client.host if request.client else ""
                    )
                    
                    return response
                    
                except Exception as e:
                    latency = (time.time() - start_time) * 1000
                    self.logger.log_request(
                        endpoint=str(request.url.path),
                        method=request.method,
                        status_code=500,
                        latency_ms=latency,
                        error=str(e)
                    )
                    raise
            
            # ==================== Routes ====================
            
            @self.app.get("/")
            async def root():
                return {
                    "name": "Bagley API",
                    "version": "7.01",
                    "status": "online",
                    "endpoints": ["/chat", "/image", "/video", "/tts", "/3d"]
                }
            
            @self.app.get("/health")
            async def health():
                return {"status": "healthy", "timestamp": datetime.now().isoformat()}
            
            @self.app.get("/metrics", dependencies=[Depends(verify_api_key)])
            async def metrics():
                return self.logger.get_metrics()
            
            @self.app.post("/chat", dependencies=[Depends(verify_api_key)])
            async def chat(request: Request):
                data = await request.json()
                message = data.get("message", "")
                
                # Would call actual chat model
                return {
                    "response": f"[Bagley API] Received: {message}",
                    "model": "bagley-chat-v7",
                    "tokens_used": len(message.split())
                }
            
            @self.app.post("/image", dependencies=[Depends(verify_api_key)])
            async def generate_image(request: Request):
                data = await request.json()
                prompt = data.get("prompt", "")
                
                # Would call actual image model
                return {
                    "status": "generating",
                    "prompt": prompt,
                    "model": "bagley-image-v7"
                }
            
            @self.app.post("/video", dependencies=[Depends(verify_api_key)])
            async def generate_video(request: Request):
                data = await request.json()
                prompt = data.get("prompt", "")
                
                return {
                    "status": "generating",
                    "prompt": prompt,
                    "model": "bagley-video-v7"
                }
            
            @self.app.post("/tts", dependencies=[Depends(verify_api_key)])
            async def text_to_speech(request: Request):
                data = await request.json()
                text = data.get("text", "")
                voice = data.get("voice", "default")
                
                return {
                    "status": "generating",
                    "text": text,
                    "voice": voice,
                    "model": "bagley-tts-v7"
                }
            
            @self.app.post("/3d", dependencies=[Depends(verify_api_key)])
            async def generate_3d(request: Request):
                data = await request.json()
                prompt = data.get("prompt", "")
                
                return {
                    "status": "generating",
                    "prompt": prompt,
                    "model": "bagley-3d-v7"
                }
            
            self.uvicorn = uvicorn
            logger.info("FastAPI app created successfully")
            
        except ImportError:
            logger.warning("FastAPI not installed, using Flask fallback")
            self._setup_flask_app()
    
    def _setup_flask_app(self):
        """Fallback Flask app"""
        try:
            from flask import Flask, request, jsonify
            
            self.app = Flask(__name__)
            
            @self.app.route("/")
            def root():
                return jsonify({
                    "name": "Bagley API",
                    "version": "7.01",
                    "status": "online"
                })
            
            @self.app.route("/health")
            def health():
                return jsonify({"status": "healthy"})
            
            @self.app.route("/chat", methods=["POST"])
            def chat():
                data = request.json
                return jsonify({"response": f"[Bagley] {data.get('message', '')}"})
            
            logger.info("Flask app created successfully")
            
        except ImportError:
            logger.error("Neither FastAPI nor Flask installed!")
            self.app = None
    
    def start(self, background: bool = True):
        """Start the API server"""
        if self.app is None:
            raise RuntimeError("No web framework available")
        
        self.is_running = True
        self.logger.start_time = time.time()
        
        if background:
            self.server_thread = threading.Thread(target=self._run_server, daemon=True)
            self.server_thread.start()
        else:
            self._run_server()
    
    def _run_server(self):
        """Run the server"""
        try:
            if hasattr(self, 'uvicorn'):
                self.uvicorn.run(
                    self.app,
                    host=self.config.host,
                    port=self.config.port,
                    log_level="info"
                )
            else:
                self.app.run(
                    host=self.config.host,
                    port=self.config.port
                )
        except Exception as e:
            logger.error(f"Server error: {e}")
            self.is_running = False
    
    def stop(self):
        """Stop the API server"""
        self.is_running = False
        # Note: Proper shutdown requires more complex handling
        logger.info("API server stopped")
    
    def start_tunnel(self) -> Optional[str]:
        """Start tunnel for worldwide access"""
        if self.config.tunnel_provider == TunnelProvider.NONE:
            return None
        
        if self.config.tunnel_provider == TunnelProvider.NGROK:
            return self._start_ngrok()
        elif self.config.tunnel_provider == TunnelProvider.CLOUDFLARE:
            return self._start_cloudflare()
        elif self.config.tunnel_provider == TunnelProvider.LOCALHOST_RUN:
            return self._start_localhost_run()
        elif self.config.tunnel_provider == TunnelProvider.SERVEO:
            return self._start_serveo()
        
        return None
    
    def _start_ngrok(self) -> Optional[str]:
        """Start ngrok tunnel"""
        try:
            from pyngrok import ngrok
            
            if self.config.ngrok_token:
                ngrok.set_auth_token(self.config.ngrok_token)
            
            tunnel = ngrok.connect(self.config.port, "http")
            self.tunnel_url = tunnel.public_url
            logger.info(f"🌐 Ngrok tunnel: {self.tunnel_url}")
            return self.tunnel_url
            
        except ImportError:
            logger.error("pyngrok not installed: pip install pyngrok")
            return None
        except Exception as e:
            logger.error(f"Ngrok error: {e}")
            return None
    
    def _start_cloudflare(self) -> Optional[str]:
        """Start Cloudflare tunnel"""
        try:
            import subprocess
            
            # Run cloudflared
            cmd = f"cloudflared tunnel --url http://localhost:{self.config.port}"
            process = subprocess.Popen(
                cmd.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Parse URL from output
            for line in process.stderr:
                line = line.decode()
                if 'https://' in line and 'trycloudflare.com' in line:
                    # Extract URL
                    import re
                    match = re.search(r'https://[^\s]+', line)
                    if match:
                        self.tunnel_url = match.group()
                        logger.info(f"🌐 Cloudflare tunnel: {self.tunnel_url}")
                        return self.tunnel_url
            
            return None
            
        except Exception as e:
            logger.error(f"Cloudflare tunnel error: {e}")
            return None
    
    def _start_localhost_run(self) -> Optional[str]:
        """Start localhost.run tunnel (SSH-based, no install needed)"""
        try:
            import subprocess
            
            cmd = f"ssh -R 80:localhost:{self.config.port} localhost.run"
            process = subprocess.Popen(
                cmd.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            
            for line in process.stdout:
                line = line.decode()
                if 'https://' in line:
                    import re
                    match = re.search(r'https://[^\s]+', line)
                    if match:
                        self.tunnel_url = match.group()
                        logger.info(f"🌐 localhost.run tunnel: {self.tunnel_url}")
                        return self.tunnel_url
            
            return None
            
        except Exception as e:
            logger.error(f"localhost.run error: {e}")
            return None
    
    def _start_serveo(self) -> Optional[str]:
        """Start serveo tunnel (SSH-based)"""
        try:
            import subprocess
            
            cmd = f"ssh -R 80:localhost:{self.config.port} serveo.net"
            process = subprocess.Popen(
                cmd.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            
            for line in process.stdout:
                line = line.decode()
                if 'https://' in line:
                    import re
                    match = re.search(r'https://[^\s]+', line)
                    if match:
                        self.tunnel_url = match.group()
                        logger.info(f"🌐 Serveo tunnel: {self.tunnel_url}")
                        return self.tunnel_url
            
            return None
            
        except Exception as e:
            logger.error(f"Serveo error: {e}")
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get server status"""
        return {
            'running': self.is_running,
            'host': self.config.host,
            'port': self.config.port,
            'local_url': f"http://{self.config.host}:{self.config.port}",
            'tunnel_url': self.tunnel_url,
            'auth_enabled': self.config.enable_auth,
            'api_key': self.config.api_key[:10] + "..." if self.config.api_key else None,
            'metrics': self.logger.get_metrics()
        }


# ==================== Factory ====================

def create_api_server(
    port: int = 8000,
    enable_auth: bool = True,
    tunnel: str = "none"
) -> BagleyAPIServer:
    """
    Create an API server
    
    Args:
        port: Port to run on
        enable_auth: Enable API key authentication
        tunnel: Tunnel provider ("none", "ngrok", "cloudflare", "localhost.run", "serveo")
    """
    config = APIConfig(
        port=port,
        enable_auth=enable_auth,
        tunnel_provider=TunnelProvider(tunnel)
    )
    
    return BagleyAPIServer(config)


# ==================== Exports ====================

__all__ = [
    'TunnelProvider',
    'APIConfig',
    'SmartAPILogger',
    'BagleyAPIServer',
    'create_api_server',
]
