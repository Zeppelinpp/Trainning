#!/bin/bash

# Reward Model Training Script
# 使用PyTorch分布式训练

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 指定使用的GPU
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# 训练参数
NUM_GPUS=4
CONFIG_FILE="config.yaml"

# 单卡训练
if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Starting single GPU training..."
    python train.py --config $CONFIG_FILE
else
    # 多卡分布式训练
    echo "Starting distributed training on $NUM_GPUS GPUs..."
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=29500 \
        train.py \
        --config $CONFIG_FILE
fi

