# Financial Reward Model Training

这是一个用于训练财务分析报告Reward Model的完整框架，支持在准确度(Accuracy)、专业度(Professionalism)和分析深度(Depth of Analysis)三个维度上评估财务分析报告质量。

## 项目结构

```
Train/
├── config.yaml              # 训练配置文件
├── requirements.txt         # Python依赖
├── model.py                 # Reward Model定义
├── dataset.py               # 数据集加载
├── train.py                 # 训练脚本
├── evaluate.py              # 评估脚本
├── inference.py             # 推理脚本
├── run_train.sh            # 训练启动脚本
├── data/
│   ├── train.jsonl         # 训练数据
│   ├── val.jsonl           # 验证数据
│   └── test.jsonl          # 测试数据
└── README.md
```

## 推荐框架

本项目基于以下框架实现：

1. **PyTorch 2.0+**: 核心深度学习框架，支持原生分布式训练
2. **Transformers**: 使用预训练语言模型作为基座
3. **DeepSpeed** (可选): 用于大规模模型的高效训练
4. **Accelerate**: 简化分布式训练配置

### 为什么选择这些框架？

- **灵活性**: PyTorch提供了灵活的模型定义和训练流程
- **效率**: 原生DDP支持和DeepSpeed优化保证训练效率
- **易用性**: Transformers提供了丰富的预训练模型
- **可扩展性**: 支持从单卡到多机多卡的无缝扩展

## 安装依赖

```bash
pip install -r requirements.txt
```

或使用uv (推荐):

```bash
uv pip install -r requirements.txt
```

## 数据格式

### 点式评分数据格式 (Pointwise)

训练数据为JSONL格式，每行一个JSON对象：

```json
{
  "system_prompt": "请根据以下财务数据生成分析报告",
  "input_data": {
    "revenue": 12500000,
    "net_income": 1406250,
    ...
  },
  "report": "# 财务分析报告\n\n...",
  "scores": {
    "accuracy": 4.5,
    "professionalism": 4.7,
    "depth_of_analysis": 4.3
  }
}
```

### 成对比较数据格式 (Pairwise)

```json
{
  "chosen": {
    "system_prompt": "...",
    "input_data": {...},
    "report": "更好的报告"
  },
  "rejected": {
    "system_prompt": "...",
    "input_data": {...},
    "report": "较差的报告"
  }
}
```

## 模型架构

### FinancialRewardModel

点式评分模型，输出三个维度的分数：

```
Input: [System Prompt + Input Data + Report]
  ↓
Tokenizer
  ↓
Base Model (e.g., Qwen2.5-7B)
  ↓
Pooling (Last Token / Mean / CLS)
  ↓
Score Head(s)
  ↓
Output: [accuracy_score, professionalism_score, depth_score]
```

### PairwiseRewardModel

成对比较模型，判断哪个报告更好：

```
Chosen Report → Score₁
Rejected Report → Score₂
Loss = -log(σ(Score₁ - Score₂))
```

## 训练

### 单卡训练

```bash
uv run train.py --config config.yaml
```

### 多卡分布式训练

```bash
# 方式1: 使用torchrun (推荐)
torchrun --nproc_per_node=4 train.py --config config.yaml

# 方式2: 使用提供的脚本
bash run_train.sh
```

### 使用DeepSpeed加速

修改配置文件启用DeepSpeed:

```yaml
deepspeed:
  enabled: true
  config: "deepspeed_config.json"
```

运行训练:

```bash
deepspeed train.py --config config.yaml
```

## 评估

```bash
uv run evaluate.py \
    --checkpoint output/best_model \
    --config config.yaml \
    --test_file data/test.jsonl \
    --output evaluation_results.json
```

评估指标包括：
- **MSE/MAE**: 预测分数与真实分数的误差
- **Pearson/Spearman相关系数**: 预测分数与真实分数的相关性
- 每个维度独立评估 + 综合评估

## 推理

### 命令行推理

```bash
uv run inference.py \
    --checkpoint output/best_model \
    --config config.yaml \
    --input sample_report.json
```

### Python API

```python
from inference import RewardModelInference

# 初始化模型
inference = RewardModelInference(
    checkpoint_path="output/best_model",
    config_path="config.yaml"
)

# 评分
result = inference.score_report(
    system_prompt="请生成财务分析报告",
    input_data={"revenue": 12500000, ...},
    report="# 财务分析报告\n\n..."
)

print(result)
# {
#   'scores': {
#     'accuracy': 4.5,
#     'professionalism': 4.7,
#     'depth_of_analysis': 4.3
#   },
#   'overall_score': 4.5,
#   'grade': 'A+ (优秀)'
# }
```

## 配置说明

### 关键配置项

```yaml
model:
  base_model: "Qwen/Qwen2.5-7B"  # 基座模型
  max_length: 4096                # 最大序列长度
  
training:
  num_epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4  # 有效batch_size = 4 * 4 = 16
  learning_rate: 1e-5
  
reward_dimensions:
  weight_strategy: "average"      # average, weighted, multi_head
  dimension_weights:               # weighted策略下的权重
    accuracy: 0.4
    professionalism: 0.3
    depth_of_analysis: 0.3
```

### 评分策略

1. **average**: 三个维度独立预测，取平均值作为总分
2. **weighted**: 三个维度独立预测，按权重加权求和
3. **multi_head**: 每个维度独立的评分头

## 分布式训练详解

### PyTorch DDP

PyTorch的DistributedDataParallel (DDP)是推荐的分布式训练方式：

**优势：**
- 原生支持，无需额外依赖
- 支持梯度累积
- 通信开销小
- 易于调试

**使用方式：**

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化进程组
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])

# 模型分布式包装
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])
```

**启动命令：**

```bash
# 单机4卡
torchrun --nproc_per_node=4 train.py

# 多机训练 (2台机器，每台4卡)
# 机器1 (主节点)
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr=192.168.1.1 --master_port=29500 train.py

# 机器2
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
    --master_addr=192.168.1.1 --master_port=29500 train.py
```

### 环境变量

训练脚本会自动读取以下环境变量：

- `RANK`: 全局进程序号 (0 到 world_size-1)
- `LOCAL_RANK`: 本地进程序号 (0 到 nproc_per_node-1)
- `WORLD_SIZE`: 总进程数
- `MASTER_ADDR`: 主节点地址
- `MASTER_PORT`: 主节点端口

## 性能优化建议

### 内存优化

1. **梯度累积**: 增加`gradient_accumulation_steps`以使用更大的有效batch size
2. **混合精度训练**: 启用FP16/BF16
3. **梯度检查点**: 对于大模型，牺牲计算换内存

```python
model.gradient_checkpointing_enable()
```

### 速度优化

1. **增加num_workers**: 数据加载并行化
2. **pin_memory**: 加速CPU到GPU的数据传输
3. **编译模型** (PyTorch 2.0+):

```python
model = torch.compile(model)
```

## 后续使用

训练好的Reward Model可用于：

1. **RLHF训练**: 作为奖励信号指导语言模型生成更好的财务分析报告
2. **质量评估**: 自动评估生成报告的质量
3. **数据筛选**: 从大量生成的报告中筛选高质量样本
4. **在线学习**: 持续收集用户反馈，迭代优化模型

## 常见问题

### Q: 如何选择基座模型？

A: 建议选择在中文和数值理解上表现较好的模型，如：
- Qwen系列 (推荐)
- ChatGLM系列
- Baichuan系列

### Q: 训练数据需要多少？

A: 建议至少准备：
- 训练集: 1000+ 样本
- 验证集: 200+ 样本
- 测试集: 200+ 样本

质量比数量更重要，确保标注准确。

### Q: 如何标注训练数据？

A: 建议方式：
1. 多位专业分析师独立评分
2. 取平均分作为最终标签
3. 制定详细的评分标准
4. 定期校准评分一致性

### Q: 训练需要多长时间？

A: 取决于：
- 数据量: 1000样本约需1-2小时 (4×V100)
- 模型大小: 7B模型推荐使用4-8卡
- Epoch数: 通常3-5个epoch即可

## 许可证

MIT License

## 联系方式

如有问题，请提Issue或联系项目维护者。

