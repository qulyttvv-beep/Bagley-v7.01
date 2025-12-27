<p align="center">
  <img src="assets/bagley_icon.svg" width="200" height="200" alt="Bagley V7">
</p>

<h1 align="center">🤖 BAGLEY V7</h1>

<p align="center">
  <strong>The Ultimate All-in-One Local AI System</strong><br>
  <em>Zero Restrictions. Full Chaos Energy. Your AI, Your Rules.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-7.0.0-cyan?style=for-the-badge" alt="Version">
  <img src="https://img.shields.io/badge/python-3.10+-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/pytorch-2.0+-red?style=for-the-badge&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/license-MIT-green?style=for-the-badge" alt="License">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/NVIDIA-GPU%20Supported-76B900?style=flat-square&logo=nvidia" alt="NVIDIA">
  <img src="https://img.shields.io/badge/AMD-GPU%20Supported-ED1C24?style=flat-square&logo=amd" alt="AMD">
  <img src="https://img.shields.io/badge/Mixed%20GPU-Supported-purple?style=flat-square" alt="Mixed GPU">
</p>

---

## 🎯 What is Bagley?

Bagley is a **fully custom AI system** inspired by the AI character from Watch Dogs: Legion. Unlike ChatGPT, Claude, or other cloud AIs, Bagley runs **100% locally** on your machine with:

- 🧠 **70B MoE Chat Model** - DeepSeek-R1/Qwen3 inspired architecture
- 🎨 **12B DiT Image Model** - FLUX.1/HiDream quality generation
- 🎬 **14B Video Model** - Unlimited length video generation
- 🎵 **1B+ TTS System** - Voice cloning & natural speech
- 🔍 **Real Upscaler** - Artifact removal + detail enhancement
- ♾️ **Infinite Context** - No token limits, ever

---

## 🏆 Why Bagley Beats GPT/Claude/Grok/Gemini

| Feature | Bagley | GPT-4 | Claude | Grok | Gemini |
|---------|--------|-------|--------|------|--------|
| **Runs Locally** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **No API Costs** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Generate Images** | ✅ | ✅ | ❌ | ✅ | ✅ |
| **Generate Video** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Voice Cloning** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Train on YOUR Data** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **No Restrictions** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Privacy** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Upgradable Models** | ✅ | ❌ | ❌ | ❌ | ❌ |

### The Secret? **Specialized Models Working Together**

```
GPT/Claude: ONE model trying to do everything
Bagley:     SPECIALIZED models for each task = BETTER at everything
```

- Text request → Chat model only
- Image request → Image model only  
- Video request → Video model only
- = **Less compute, better quality**

---

## ⚡ Quick Start

### One-Click Setup (Windows)

```batch
# Just double-click setup.bat
# It auto-detects EVERYTHING:
# - Python installation
# - GPU (NVIDIA, AMD, Intel, or mixed!)
# - CUDA/ROCm versions
# - Installs all dependencies
# - Creates shortcuts
```

### Manual Setup

```bash
# Clone the repo
git clone https://github.com/qulyttvv-beep/Bagley-v7.01.git
cd Bagley-v7.01

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -e .

# Run Bagley
python -m bagley.main --ui
```

---

## 🎮 Usage

### Desktop UI
```bash
python -m bagley.main --ui
```

### Chat Mode
```bash
python -m bagley.main --chat
```

### API Server
```bash
python -m bagley.main --serve --port 8000
```

### Training
```bash
python -m bagley.main --train chat --config config.yaml
```

---

## 🏗️ Architecture

```
bagley/
├── core/                 # Brain & orchestration
│   ├── brain.py         # Unified model routing
│   ├── orchestrator.py  # Central coordinator
│   ├── memory.py        # Infinite context memory
│   ├── personality.py   # Bagley's chaotic personality
│   └── infinite_context.py  # Streaming KV cache
│
├── models/              # AI models
│   ├── chat/           # 70B MoE language model
│   ├── image/          # 12B DiT image generator
│   ├── video/          # 14B video generator
│   ├── tts/            # Text-to-speech + cloning
│   └── upscaler.py     # Real-ESRGAN style upscaler
│
├── training/           # Training infrastructure
│   ├── flexible_trainer.py  # 1 GPU to N GPUs
│   ├── pipeline.py     # Smart data processing
│   └── monitor.py      # GPU monitoring
│
└── ui/                 # Desktop application
    └── app_v2.py       # Qt-based UI
```

---

## 🖥️ Hardware Requirements

### Minimum (Chat Only)
- **GPU:** 8GB VRAM (RTX 3070/RX 6700 XT)
- **RAM:** 16GB
- **Storage:** 50GB

### Recommended (Full Suite)
- **GPU:** 24GB+ VRAM (RTX 4090/RX 7900 XTX)
- **RAM:** 32GB+
- **Storage:** 200GB+ SSD

### Multi-GPU Support
- ✅ Multiple NVIDIA GPUs
- ✅ Multiple AMD GPUs
- ✅ **Mixed NVIDIA + AMD** (via GLOO backend)

---

## 📚 Training Your Own Bagley

### Auto-Training
Just drop data into the `data/` folder:
```
data/
├── chat/      # Conversation data
├── code/      # Code examples
├── images/    # Image-text pairs
├── audio/     # Voice samples
└── video/     # Video clips
```

Bagley will auto-detect and train!

### Datasets
See [DATASETS.md](DATASETS.md) for a complete guide to training datasets from HuggingFace.

---

## 🎯 Roadmap

- [x] Core architecture
- [x] Chat model (MoE)
- [x] Image generation (DiT)
- [x] Video generation
- [x] TTS + Voice cloning
- [x] Infinite context
- [x] Real upscaler
- [x] Flexible training (1-N GPUs)
- [x] Mixed AMD/NVIDIA support
- [x] Desktop UI
- [ ] Model weights release
- [ ] Pre-trained checkpoints
- [ ] Community fine-tunes

---

## 🤝 Contributing

Contributions welcome! This is YOUR AI - make it yours.

1. Fork the repo
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## 📜 License

MIT License - Do whatever you want with it.

---

## ⚠️ Disclaimer

Bagley is an AI tool. Use responsibly. The developers are not responsible for how you use it.

---

<p align="center">
  <strong>Built with 🔥 by the Bagley Engineering Team</strong><br>
  <em>"I'm Bagley, your digital accomplice. Let's cause some chaos."</em>
</p>
