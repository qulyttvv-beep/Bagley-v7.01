# 📚 Bagley Training Datasets Guide

All the datasets you need from Hugging Face to train a killer AI.

## 🎯 Quick Start

```bash
# Install huggingface_hub
pip install huggingface_hub datasets

# Login (for gated datasets)
huggingface-cli login
```

---

## 💬 Chat/Language Model Datasets

### Core Conversation Datasets

| Dataset | Size | Description | Command |
|---------|------|-------------|---------|
| [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | 1M+ | High quality instruction data | `datasets.load_dataset("teknium/OpenHermes-2.5")` |
| [SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca) | 500K | Filtered Orca dataset | `datasets.load_dataset("Open-Orca/SlimOrca")` |
| [UltraChat-200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | 200K | Multi-turn conversations | `datasets.load_dataset("HuggingFaceH4/ultrachat_200k")` |
| [Capybara](https://huggingface.co/datasets/LDJnr/Capybara) | 16K | Multi-turn, diverse | `datasets.load_dataset("LDJnr/Capybara")` |

### Reasoning & Math

| Dataset | Size | Description | Command |
|---------|------|-------------|---------|
| [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) | 395K | Math reasoning | `datasets.load_dataset("meta-math/MetaMathQA")` |
| [OpenMathInstruct](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1) | 1.8M | NVIDIA math data | `datasets.load_dataset("nvidia/OpenMathInstruct-1")` |
| [Orca-Math](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) | 200K | Word problems | `datasets.load_dataset("microsoft/orca-math-word-problems-200k")` |
| [GSM8K](https://huggingface.co/datasets/gsm8k) | 8.5K | Grade school math | `datasets.load_dataset("gsm8k", "main")` |

### Code

| Dataset | Size | Description | Command |
|---------|------|-------------|---------|
| [StarCoder-Data](https://huggingface.co/datasets/bigcode/starcoderdata) | 250GB | Source code | `datasets.load_dataset("bigcode/starcoderdata")` |
| [CodeFeedback](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction) | 150K | Code instructions | `datasets.load_dataset("m-a-p/CodeFeedback-Filtered-Instruction")` |
| [Evol-CodeAlpaca](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1) | 111K | Evolved code data | `datasets.load_dataset("theblackcat102/evol-codealpaca-v1")` |
| [CodeExercises](https://huggingface.co/datasets/jinaai/code_exercises) | 1M+ | Code exercises | `datasets.load_dataset("jinaai/code_exercises")` |

### General Knowledge

| Dataset | Size | Description | Command |
|---------|------|-------------|---------|
| [Wikipedia](https://huggingface.co/datasets/wikipedia) | 6M+ articles | Encyclopedia | `datasets.load_dataset("wikipedia", "20220301.en")` |
| [RedPajama](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) | 1.2T tokens | Web text | `datasets.load_dataset("togethercomputer/RedPajama-Data-1T")` |
| [Pile](https://huggingface.co/datasets/EleutherAI/pile) | 800GB | Diverse text | `datasets.load_dataset("EleutherAI/pile")` |
| [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | 15T tokens | Web data | `datasets.load_dataset("HuggingFaceFW/fineweb")` |

### Roleplay & Creative

| Dataset | Size | Description | Command |
|---------|------|-------------|---------|
| [PIPPA](https://huggingface.co/datasets/PygmalionAI/PIPPA) | 1M+ | Character roleplay | `datasets.load_dataset("PygmalionAI/PIPPA")` |
| [Bluemoon](https://huggingface.co/datasets/Norquinal/claude_multiround_chat_30k) | 30K | Multi-round creative | `datasets.load_dataset("Norquinal/claude_multiround_chat_30k")` |

### Function Calling & Tools

| Dataset | Size | Description | Command |
|---------|------|-------------|---------|
| [Glaive-Function-Calling](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | 113K | Function calls | `datasets.load_dataset("glaiveai/glaive-function-calling-v2")` |
| [ToolBench](https://huggingface.co/datasets/ToolBench/ToolBench) | 126K | Tool usage | `datasets.load_dataset("ToolBench/ToolBench")` |

---

## 🎨 Image Model Datasets

### Image-Text Pairs

| Dataset | Size | Description | Command |
|---------|------|-------------|---------|
| [LAION-5B](https://huggingface.co/datasets/laion/laion5B-index) | 5B pairs | Massive web scrape | See LAION docs |
| [LAION-Aesthetics](https://huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_6.5plus) | 600K | High quality images | `datasets.load_dataset("ChristophSchuhmann/improved_aesthetics_6.5plus")` |
| [JourneyDB](https://huggingface.co/datasets/JourneyDB/JourneyDB) | 4M | Midjourney images | `datasets.load_dataset("JourneyDB/JourneyDB")` |
| [COYO-700M](https://huggingface.co/datasets/kakaobrain/coyo-700m) | 700M | Korean web images | `datasets.load_dataset("kakaobrain/coyo-700m")` |
| [SAM](https://huggingface.co/datasets/facebook/sam) | 11M | Segmentation masks | `datasets.load_dataset("facebook/sam")` |

### Art & Style

| Dataset | Size | Description | Command |
|---------|------|-------------|---------|
| [WikiArt](https://huggingface.co/datasets/huggan/wikiart) | 80K | Art paintings | `datasets.load_dataset("huggan/wikiart")` |
| [Danbooru](https://huggingface.co/datasets/animelover/danbooru2021) | 4.9M | Anime images | `datasets.load_dataset("animelover/danbooru2021")` |

### High Resolution

| Dataset | Size | Description | Command |
|---------|------|-------------|---------|
| [ImageNet-1K](https://huggingface.co/datasets/imagenet-1k) | 1.28M | Classification | `datasets.load_dataset("imagenet-1k")` |
| [FFHQ](https://huggingface.co/datasets/datasets-examples/FFHQ) | 70K | Faces 1024x1024 | `datasets.load_dataset("datasets-examples/FFHQ")` |

---

## 🎬 Video Model Datasets

| Dataset | Size | Description | Command |
|---------|------|-------------|---------|
| [WebVid-10M](https://huggingface.co/datasets/TempoFunk/webvid-10M) | 10M clips | Web videos | `datasets.load_dataset("TempoFunk/webvid-10M")` |
| [Panda-70M](https://huggingface.co/datasets/Leymore/Panda-70M) | 70M | Video captions | See dataset docs |
| [InternVid](https://huggingface.co/datasets/OpenGVLab/InternVid) | 7M | High quality | `datasets.load_dataset("OpenGVLab/InternVid")` |
| [HD-VG-130M](https://huggingface.co/datasets/Vchitect/HD-VG-130M) | 130M | HD videos | See dataset docs |

---

## 🎵 TTS/Audio Datasets

### Speech

| Dataset | Size | Description | Command |
|---------|------|-------------|---------|
| [LibriSpeech](https://huggingface.co/datasets/librispeech_asr) | 1000hrs | Clean speech | `datasets.load_dataset("librispeech_asr")` |
| [LibriTTS](https://huggingface.co/datasets/cdminix/libritts-r-aligned) | 585hrs | TTS quality | `datasets.load_dataset("cdminix/libritts-r-aligned")` |
| [GigaSpeech](https://huggingface.co/datasets/speechcolab/gigaspeech) | 10000hrs | Diverse speech | `datasets.load_dataset("speechcolab/gigaspeech")` |
| [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0) | 17000hrs | Multilingual | `datasets.load_dataset("mozilla-foundation/common_voice_13_0", "en")` |
| [VoxCeleb](https://huggingface.co/datasets/ProgramComputer/voxceleb) | 2000hrs | Speaker ID | `datasets.load_dataset("ProgramComputer/voxceleb")` |

### Music

| Dataset | Size | Description | Command |
|---------|------|-------------|---------|
| [MusicCaps](https://huggingface.co/datasets/google/MusicCaps) | 5.5K | Music + captions | `datasets.load_dataset("google/MusicCaps")` |
| [MTG-Jamendo](https://huggingface.co/datasets/mtg-jamendo/mtg-jamendo-dataset) | 55K | Music tagging | `datasets.load_dataset("mtg-jamendo/mtg-jamendo-dataset")` |

---

## 🔍 Upscaler Training Datasets

| Dataset | Size | Description | Command |
|---------|------|-------------|---------|
| [DIV2K](https://huggingface.co/datasets/eugenesiow/Div2k) | 1000 | High-res images | `datasets.load_dataset("eugenesiow/Div2k")` |
| [Flickr2K](https://huggingface.co/datasets/goodfellowliu/Flickr2K) | 2650 | High-res photos | `datasets.load_dataset("goodfellowliu/Flickr2K")` |
| [RealSR](https://huggingface.co/datasets/eugenesiow/RealSR) | 559 | Real-world SR | `datasets.load_dataset("eugenesiow/RealSR")` |

---

## 📥 Download Script

Save this as `download_datasets.py`:

```python
"""Download all training datasets for Bagley"""

from datasets import load_dataset
from pathlib import Path
import os

# Where to save
DATA_DIR = Path("./training_data")
DATA_DIR.mkdir(exist_ok=True)

# ======== CHAT DATASETS ========
print("📥 Downloading chat datasets...")

chat_datasets = [
    ("teknium/OpenHermes-2.5", "chat/openhermes"),
    ("Open-Orca/SlimOrca", "chat/slimorca"),
    ("HuggingFaceH4/ultrachat_200k", "chat/ultrachat"),
    ("meta-math/MetaMathQA", "chat/metamath"),
    ("glaiveai/glaive-function-calling-v2", "chat/function_calling"),
]

for dataset_name, save_path in chat_datasets:
    try:
        print(f"  Downloading {dataset_name}...")
        ds = load_dataset(dataset_name)
        ds.save_to_disk(DATA_DIR / save_path)
        print(f"  ✅ Saved to {save_path}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")

# ======== IMAGE DATASETS ========
print("\n📥 Downloading image datasets...")

image_datasets = [
    ("ChristophSchuhmann/improved_aesthetics_6.5plus", "image/laion_aesthetics"),
    ("huggan/wikiart", "image/wikiart"),
]

for dataset_name, save_path in image_datasets:
    try:
        print(f"  Downloading {dataset_name}...")
        ds = load_dataset(dataset_name)
        ds.save_to_disk(DATA_DIR / save_path)
        print(f"  ✅ Saved to {save_path}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")

# ======== AUDIO DATASETS ========
print("\n📥 Downloading audio datasets...")

audio_datasets = [
    ("librispeech_asr", "audio/librispeech"),
    ("google/MusicCaps", "audio/musiccaps"),
]

for dataset_name, save_path in audio_datasets:
    try:
        print(f"  Downloading {dataset_name}...")
        ds = load_dataset(dataset_name)
        ds.save_to_disk(DATA_DIR / save_path)
        print(f"  ✅ Saved to {save_path}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")

print("\n✅ Download complete!")
print(f"Data saved to: {DATA_DIR.absolute()}")
```

---

## 🎯 Recommended Training Order

### Phase 1 - Foundation (Chat Model)

1. **Wikipedia** - General knowledge base
2. **RedPajama/FineWeb** - Web text understanding
3. **OpenHermes** - Instruction following
4. **SlimOrca** - Reasoning

### Phase 2 - Specialization (Chat Model)

5. **MetaMathQA** - Math abilities
6. **CodeFeedback** - Coding skills
7. **Glaive-Function-Calling** - Tool use
8. **UltraChat** - Conversation flow

### Phase 3 - Image Model

9. **LAION-Aesthetics** - Quality images
10. **JourneyDB** - Artistic generation

### Phase 4 - Video Model

11. **WebVid-10M** - Video understanding
12. **InternVid** - Higher quality

### Phase 5 - TTS Model

13. **LibriTTS** - Clean speech
14. **GigaSpeech** - Diverse speakers

### Phase 6 - Upscaler

15. **DIV2K + Flickr2K** - Super resolution

---

## 💾 Storage Requirements

| Model | Dataset | Approx Size |
|-------|---------|-------------|
| Chat | Full stack | ~500GB - 2TB |
| Image | LAION subset | ~200GB |
| Video | WebVid-10M | ~1TB |
| TTS | LibriTTS + GigaSpeech | ~100GB |
| Upscaler | DIV2K + Flickr2K | ~5GB |

**Total: ~2-4TB** (depending on what you download)

---

## ⚡ Quick Download Commands

```bash
# Chat data (essential)
huggingface-cli download teknium/OpenHermes-2.5 --local-dir ./data/chat

# Image data (essential)
huggingface-cli download ChristophSchuhmann/improved_aesthetics_6.5plus --local-dir ./data/images

# Audio data
huggingface-cli download cdminix/libritts-r-aligned --local-dir ./data/audio

# Video data (large!)
huggingface-cli download TempoFunk/webvid-10M --local-dir ./data/video
```

---

## 🔥 Pro Tips

1. **Start Small**: Train on OpenHermes first, then scale up
2. **Quality > Quantity**: LAION-Aesthetics beats raw LAION
3. **Mix Datasets**: Combine multiple for better generalization
4. **Clean Your Data**: Remove duplicates and garbage
5. **Use Streaming**: For huge datasets, stream don't download:

```python
ds = load_dataset("dataset_name", streaming=True)
for sample in ds:
    # Process without downloading everything
    pass
```

---

## 📋 Data Format Examples

### Chat Format (what Bagley expects)

```json
{"messages": [
  {"role": "system", "content": "You are Bagley, a chaotic AI assistant."},
  {"role": "user", "content": "Hello!"},
  {"role": "assistant", "content": "Yo! What's good?"}
]}
```

### Image Format

```json
{"image": "path/to/image.jpg", "caption": "A description of the image"}
```

### Audio Format

```json
{"audio": "path/to/audio.wav", "text": "The transcript of what was said"}
```

---

Now go train that beast! 🚀
