# 🚀 Bagley v7 - Setup Guide (For Dummies)

## Step 1: Copy Files to Cluster

```bash
# From your PC, copy the whole folder to your cluster
scp -r bagley-v7/ user@cluster:/home/user/
```

Or just use a USB drive, whatever works.

---

## Step 2: SSH into Cluster

```bash
ssh user@your-cluster-address
cd bagley-v7
```

---

## Step 3: Create Virtual Environment

```bash
# Create venv
python3 -m venv venv

# Activate it
source venv/bin/activate   # Linux/Mac
```

---

## Step 4: Install PyTorch (IMPORTANT - AMD vs NVIDIA)

### For AMD GPUs (ROCm)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.0
```

### For NVIDIA GPUs (CUDA)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Step 5: Install Everything Else

```bash
pip install -r requirements.txt
```

If some packages fail (like flash-attn on AMD), that's fine - they're optional.

---

## Step 6: Test It Works

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'GPUs: {torch.cuda.device_count()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

You should see your GPUs listed.

---

## Step 7: Prepare Training Data

Put your training data somewhere accessible. Format depends on what you're training:

### For Chat Model

JSONL file with conversations:

```json
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hey!"}]}
{"messages": [{"role": "user", "content": "What's 2+2?"}, {"role": "assistant", "content": "4, obviously"}]}
```

### For Image Model

JSONL with image paths and captions:

```json
{"image": "/path/to/image1.jpg", "caption": "A cat sitting on a couch"}
{"image": "/path/to/image2.jpg", "caption": "Mountain landscape at sunset"}
```

### For TTS

JSONL with audio paths and transcripts:

```json
{"audio": "/path/to/audio1.wav", "text": "Hello, how are you today?"}
{"audio": "/path/to/audio2.wav", "text": "This is a test sentence."}
```

---

## Step 8: Start Training

### Single GPU

```bash
python -m bagley.main --train chat --output-dir ./outputs
```

### Multi-GPU (Single Node)

```bash
torchrun --nproc_per_node=8 -m bagley.main --train chat --output-dir ./outputs
```

### Multi-Node Cluster (with SLURM)

```bash
# Create a job script: train.slurm
sbatch train.slurm
```

Example `train.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=bagley-train
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=72:00:00

srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=8 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:29500 \
    -m bagley.main --train chat --output-dir ./outputs
```

---

## Step 9: Monitor Training

```bash
# Watch GPU usage
watch -n 1 rocm-smi   # AMD
watch -n 1 nvidia-smi  # NVIDIA

# Check logs
tail -f outputs/training.log

# If using wandb
# Go to wandb.ai and check your project
```

---

## Step 10: Export Trained Model

After training, your model is in `./outputs/final/`:

```bash
# Copy back to your PC
scp -r user@cluster:/home/user/bagley-v7/outputs/final/ ./my-trained-model/
```

---

## Common Problems & Fixes

### "CUDA out of memory"

- Reduce batch size in training config
- Enable gradient checkpointing
- Use DeepSpeed ZeRO-3

### "No module named X"

```bash
pip install X
```

### "ROCm not detected"

```bash
# Check ROCm is installed
rocm-smi

# If not, install ROCm first (see AMD docs)
```

### "NCCL timeout" (multi-node)

- Check network connectivity between nodes
- Try setting: `export NCCL_DEBUG=INFO`

### Training is slow

- Make sure you're using all GPUs
- Check if data loading is the bottleneck (increase num_workers)
- Enable torch.compile for inference

---

## Quick Reference

| What | Command |
| ---- | ------- |
| Activate venv | `source venv/bin/activate` |
| Train chat model | `python -m bagley.main --train chat` |
| Train image model | `python -m bagley.main --train image` |
| Train video model | `python -m bagley.main --train video` |
| Train TTS model | `python -m bagley.main --train tts` |
| Check GPU | `rocm-smi` or `nvidia-smi` |
| Kill training | `Ctrl+C` or `scancel <job_id>` |

---

## That's It

1. Copy files
2. Make venv
3. Install PyTorch (AMD or NVIDIA)
4. Install requirements
5. Add training data
6. Run training
7. Wait (a lot)
8. Get model

Questions? You'll figure it out. You're Balkan. 💪

---

*"If it doesn't work, try turning it off and on again. If that doesn't work, blame the hardware."* - Ancient Balkan Proverb
