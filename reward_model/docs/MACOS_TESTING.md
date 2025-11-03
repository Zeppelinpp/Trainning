# macOS Training Guide

## Overview

The reward model training script has been modified to support macOS with the following changes:

### Key Modifications

1. **Device Detection**: Auto-detects CUDA → MPS → CPU
2. **Backend Support**: Uses `gloo` backend (instead of `nccl`) for CPU/macOS
3. **Small Model**: Uses `bert-base-chinese` for faster testing

## Quick Start

### 1. Run with test configuration

```bash
cd reward_model
uv run python train.py --config config_macos.yaml
```

Or use the provided test script:

```bash
cd reward_model
./test_train_macos.sh
```

### 2. Configuration Details

The `config_macos.yaml` file has the following optimizations for macOS:

- **Model**: `bert-base-chinese` (108M parameters vs 7B)
- **Max Length**: 512 tokens (vs 4096)
- **Batch Size**: 2 (vs 4)
- **Backend**: `gloo` (vs `nccl`)
- **WandB**: Disabled for quick testing
- **Training**: 2 epochs with reduced steps

### 3. Expected Behavior

On Apple Silicon Macs:
- Will automatically use **MPS** (Metal Performance Shaders) for GPU acceleration
- Training should take 5-10 minutes depending on your Mac model

On Intel Macs:
- Will use **CPU**
- Training will be slower (15-30 minutes)

### 4. Monitor Training

The script will show:
- Device being used (MPS/CPU)
- Training progress with loss and learning rate
- Validation loss at evaluation steps
- Model checkpoints in `./output/` directory

## Troubleshooting

### MPS Issues

If you encounter MPS-related errors, disable MPS by setting:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
uv run python train.py --config config_macos.yaml
```

Or force CPU-only:

```python
# Add to train.py before device detection
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
```

### Memory Issues

If you run out of memory, reduce batch size in `config_macos.yaml`:

```yaml
training:
  batch_size: 1  # Reduce from 2 to 1
  gradient_accumulation_steps: 4  # Increase to maintain effective batch size
```

## Switching to Production

To train with the full model on a GPU server:

```bash
# Use the original config
uv run python train.py --config config.yaml

# Or with distributed training
torchrun --nproc_per_node=4 train.py --config config.yaml
```

