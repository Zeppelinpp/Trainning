#!/bin/bash
# Test training script for macOS

echo "Testing Reward Model Training on macOS..."
echo "Device: $(python -c 'import torch; print("MPS" if torch.backends.mps.is_available() else "CPU")')"
echo ""

cd "$(dirname "$0")"

# Run training with macOS-specific config
uv run python train.py --config config_macos.yaml

