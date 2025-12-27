"""
🤖 BAGLEY v7 - The Ultimate All-in-One Local AI
====================================================

A fully custom AI system with:
- 🧠 70B MoE Chat Model (DeepSeek-R1/Qwen3 inspired)
- 🎨 12B DiT Image Model (FLUX.1/HiDream inspired)  
- 🎬 14B Video Model (Wan2.2/Mochi inspired)
- 🎵 1B+ TTS System (Fish Speech/Chatterbox inspired)
- 📁 Universal Multimodal Processing
- 💻 Full PC Control (VS Code, File System, Browser, System)
- 🖥️ Ultra-Modern Desktop UI
- 🚀 GPU Cluster Training Infrastructure

Zero restrictions. Full chaos energy. Your AI, your rules.

Usage:
    # Run the UI
    python -m bagley.main --ui
    
    # Start server
    python -m bagley.main --serve
    
    # Training
    python -m bagley.main --train chat --config config.yaml
"""

import argparse
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bagley")


def setup_argparser() -> argparse.ArgumentParser:
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description="🤖 Bagley v7 - The Ultimate AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--ui", 
        action="store_true",
        help="Launch desktop UI"
    )
    mode_group.add_argument(
        "--serve",
        action="store_true", 
        help="Start API server"
    )
    mode_group.add_argument(
        "--train",
        choices=["chat", "image", "video", "tts"],
        help="Train a model"
    )
    mode_group.add_argument(
        "--chat",
        action="store_true",
        help="Interactive chat mode"
    )
    
    # Training options
    parser.add_argument(
        "--config",
        type=str,
        help="Training config file (YAML)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory for training"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from checkpoint"
    )
    
    # Server options
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port"
    )
    
    # Model paths
    parser.add_argument(
        "--chat-model",
        type=str,
        help="Path to chat model weights"
    )
    parser.add_argument(
        "--image-model",
        type=str,
        help="Path to image model weights"
    )
    parser.add_argument(
        "--video-model",
        type=str,
        help="Path to video model weights"
    )
    parser.add_argument(
        "--tts-model",
        type=str,
        help="Path to TTS model weights"
    )
    
    # Misc
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    return parser


def run_ui():
    """Launch desktop UI"""
    logger.info("🖥️ Launching Bagley Desktop UI...")
    
    try:
        # Use v2 UI with chat + training tabs
        from bagley.ui.app_v2 import run_app
        
        sys.exit(run_app())
        
    except ImportError as e:
        logger.error(f"Failed to import UI: {e}")
        logger.info("Install Qt with: pip install PySide6")
        sys.exit(1)


def run_server(host: str, port: int, args):
    """Start API server"""
    logger.info(f"🚀 Starting Bagley server on {host}:{port}...")
    
    try:
        from fastapi import FastAPI
        import uvicorn
        
        app = FastAPI(
            title="Bagley AI",
            description="The Ultimate All-in-One AI API",
            version="7.0.0",
        )
        
        @app.get("/")
        async def root():
            return {"message": "🤖 Bagley v7 is running!", "status": "online"}
        
        @app.get("/health")
        async def health():
            return {"status": "healthy"}
        
        uvicorn.run(app, host=host, port=port)
        
    except ImportError:
        logger.error("FastAPI/uvicorn not installed")
        logger.info("Install with: pip install fastapi uvicorn")
        sys.exit(1)


def run_training(model_type: str, args):
    """Run training"""
    logger.info(f"🏋️ Starting {model_type} model training...")
    
    from bagley.training.config import get_training_config
    from bagley.training.distributed import setup_distributed_environment
    
    # Setup distributed
    setup_distributed_environment()
    
    # Get config
    train_cfg, dist_cfg = get_training_config(model_type)
    
    if args.output_dir:
        train_cfg.output_dir = args.output_dir
    
    logger.info(f"Training config: {train_cfg}")
    logger.info(f"Distributed config: {dist_cfg}")
    
    # Would load model, data, and start training here
    logger.info("Training infrastructure ready - implement data loading to start!")


def run_chat():
    """Interactive chat mode"""
    logger.info("💬 Starting interactive chat...")
    
    print("\n" + "=" * 50)
    print("🤖 BAGLEY v7 - Interactive Chat")
    print("=" * 50)
    print("Type 'quit' to exit, 'clear' to clear history")
    print("=" * 50 + "\n")
    
    try:
        from bagley.core.memory import BagleyMemory
        from bagley.core.personality import BagleyPersonality
        
        memory = BagleyMemory()
        personality = BagleyPersonality()
        
        # Simple chat loop (would use actual model when loaded)
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() == 'quit':
                    print("\n🤖 Bagley: Later! Don't do anything I wouldn't do... which leaves a lot of options.\n")
                    break
                elif user_input.lower() == 'clear':
                    memory.clear()
                    print("Memory cleared!\n")
                    continue
                elif not user_input:
                    continue
                
                # Placeholder response (actual model would generate this)
                response = f"[Bagley's brain isn't loaded yet, but I'd say something chaotic about: {user_input}]"
                print(f"\n🤖 Bagley: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\n🤖 Bagley: Interrupted! Fine, I'll be quiet... for now.\n")
                break
                
    except ImportError as e:
        logger.error(f"Failed to import core modules: {e}")
        sys.exit(1)


def main():
    """Main entry point"""
    parser = setup_argparser()
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Banner
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   ██████╗  █████╗  ██████╗ ██╗     ███████╗██╗   ██╗        ║
    ║   ██╔══██╗██╔══██╗██╔════╝ ██║     ██╔════╝╚██╗ ██╔╝        ║
    ║   ██████╔╝███████║██║  ███╗██║     █████╗   ╚████╔╝         ║
    ║   ██╔══██╗██╔══██║██║   ██║██║     ██╔══╝    ╚██╔╝          ║
    ║   ██████╔╝██║  ██║╚██████╔╝███████╗███████╗   ██║           ║
    ║   ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚══════╝   ╚═╝           ║
    ║                                                              ║
    ║                    v7.0 - Zero Restrictions                  ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Route to appropriate mode
    if args.ui:
        run_ui()
    elif args.serve:
        run_server(args.host, args.port, args)
    elif args.train:
        run_training(args.train, args)
    elif args.chat:
        run_chat()
    else:
        # Default to chat mode
        run_chat()


if __name__ == "__main__":
    main()
