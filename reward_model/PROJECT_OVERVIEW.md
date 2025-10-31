# 财务分析报告Reward Model训练项目总览

## 🎯 项目目标

训练一个专门用于评估财务分析报告质量的Reward Model，在以下三个维度进行打分：
1. **准确度 (Accuracy)**: 财务数据计算和引用的准确性
2. **专业度 (Professionalism)**: 专业术语使用、报告结构、语言规范性
3. **分析深度 (Depth of Analysis)**: 洞察深度、因果分析、建议质量

该模型可用于：
- RLHF/RLAIF训练的奖励信号
- 自动评估生成报告质量
- 高质量数据筛选
- 持续学习和模型迭代

---

## 📋 推荐训练框架

### 核心框架组合

| 框架 | 版本 | 用途 | 优势 |
|------|------|------|------|
| **PyTorch** | 2.0+ | 深度学习框架 | 灵活、高效、原生分布式支持 |
| **Transformers** | 4.35+ | 预训练模型库 | 丰富的模型、统一接口 |
| **DeepSpeed** | 0.12+ | 大模型训练优化 | ZeRO优化、显存效率高 |
| **Accelerate** | 0.24+ | 分布式训练抽象 | 简化配置、易于使用 |
| **WandB** | 0.15+ | 实验跟踪 | 可视化训练过程 |

### 框架对比

| 特性 | PyTorch DDP | DeepSpeed | Megatron-LM |
|------|------------|-----------|-------------|
| 易用性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 显存效率 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 训练速度 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 灵活性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 适用场景 | < 13B模型 | 所有规模 | > 20B模型 |

**推荐选择**：
- **7B以下模型**: PyTorch DDP (原生，简单高效)
- **7B-30B模型**: DeepSpeed ZeRO Stage 2
- **30B以上模型**: DeepSpeed ZeRO Stage 3 或 Megatron-LM

---

## 🏗️ 项目结构

```
Train/
├── README.md                    # 英文文档（详细说明）
├── 快速开始.md                  # 中文快速上手指南
├── PROJECT_OVERVIEW.md          # 本文件：项目总览
├── requirements.txt             # Python依赖
├── .gitignore                   # Git忽略文件
│
├── config.yaml                  # 训练配置（核心）
├── deepspeed_config.json        # DeepSpeed配置
│
├── model.py                     # Reward Model定义
│   ├── FinancialRewardModel         (点式评分模型)
│   └── PairwiseRewardModel          (成对比较模型)
│
├── dataset.py                   # 数据集加载
│   ├── FinancialRewardDataset
│   ├── collate_fn_pointwise
│   └── collate_fn_pairwise
│
├── train.py                     # 训练脚本（支持分布式）
│   └── RewardModelTrainer
│
├── evaluate.py                  # 评估脚本
│   └── RewardModelEvaluator
│
├── inference.py                 # 推理脚本
│   └── RewardModelInference
│
├── visualize_data.py            # 数据分析工具
│
├── run_train.sh                 # 训练启动脚本
│
├── sample_report.json           # 推理样例输入
│
└── data/                        # 数据目录
    ├── train.jsonl              # 训练集 (5条样例)
    ├── val.jsonl                # 验证集 (1条样例)
    └── test.jsonl               # 测试集 (1条样例)
```

---

## 📊 样例数据说明

### 训练集样例 (5条)

| 序号 | 主题 | 报告类型 | 准确度 | 专业度 | 深度 |
|------|------|---------|--------|--------|------|
| 1 | Q3季度分析 | 盈利能力分析 | 4.5 | 4.7 | 4.3 |
| 2 | 全年分析 | 资产效率&偿债能力 | 4.8 | 4.9 | 4.7 |
| 3 | 风险评估 | 财务风险分析 | 4.6 | 4.8 | 4.9 |
| 4 | Q2环比分析 | 财务变化分析 | 4.4 | 4.5 | 4.2 |
| 5 | 投资回报 | ROE/ROIC分析 | 4.7 | 4.8 | 4.8 |

**特点**：
- 覆盖多种财务分析场景
- 评分标准清晰
- 报告结构完整
- 包含实际财务数据

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
# 或使用uv（更快）
uv pip install -r requirements.txt
```

### 2. 查看数据

```bash
uv run visualize_data.py --data data/train.jsonl --check
```

### 3. 训练模型

```bash
# 单卡训练
uv run train.py --config config.yaml

# 4卡分布式训练
bash run_train.sh
```

### 4. 评估模型

```bash
uv run evaluate.py \
    --checkpoint output/best_model \
    --config config.yaml \
    --test_file data/test.jsonl
```

### 5. 推理测试

```bash
uv run inference.py \
    --checkpoint output/best_model \
    --config config.yaml \
    --input sample_report.json
```

---

## 🔧 PyTorch分布式训练详解

### 原理架构

```
┌─────────────────────────────────────────────────────┐
│                 Master Process                       │
│              (RANK=0, 负责同步)                      │
└─────────────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
    ┌───▼───┐        ┌───▼───┐      ┌───▼───┐
    │ GPU 0 │        │ GPU 1 │      │ GPU 2 │
    │Model₀ │        │Model₁ │      │Model₂ │
    │Batch₀ │        │Batch₁ │      │Batch₂ │
    └───┬───┘        └───┬───┘      └───┬───┘
        │                │                │
        └────────────────┼────────────────┘
                         │
                   All-Reduce梯度
                         │
              ┌──────────▼──────────┐
              │   同步更新所有模型   │
              └─────────────────────┘
```

### 关键概念

| 概念 | 说明 | 示例 |
|------|------|------|
| **RANK** | 全局进程ID | 4卡训练: 0,1,2,3 |
| **LOCAL_RANK** | 本地GPU ID | 单机4卡: 0,1,2,3 |
| **WORLD_SIZE** | 总进程数 | 4卡训练: 4 |
| **Backend** | 通信后端 | NCCL (GPU), Gloo (CPU) |

### 启动方式对比

```bash
# 方式1: torchrun (推荐，PyTorch 1.10+)
torchrun --nproc_per_node=4 train.py --config config.yaml

# 方式2: torch.distributed.launch (旧版)
python -m torch.distributed.launch --nproc_per_node=4 train.py

# 方式3: 多机训练
# 主节点 (192.168.1.100)
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
    --master_addr=192.168.1.100 --master_port=29500 train.py

# 从节点 (192.168.1.101)
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
    --master_addr=192.168.1.100 --master_port=29500 train.py
```

### 有效批次大小

```
真实batch_size = batch_size × gradient_accumulation_steps × num_gpus

示例配置:
  batch_size: 4
  gradient_accumulation_steps: 4
  num_gpus: 4
  
有效batch_size = 4 × 4 × 4 = 64
```

---

## 📈 训练配置建议

### 根据GPU显存选择配置

| GPU显存 | 基座模型 | Batch Size | Grad Accum | 策略 |
|---------|---------|-----------|------------|------|
| 24GB (3090/4090) | Qwen-7B | 2 | 8 | DDP |
| 40GB (A100) | Qwen-7B | 4 | 4 | DDP |
| 80GB (A100-80GB) | Qwen-7B | 8 | 2 | DDP |
| 24GB × 4 | Qwen-7B | 4 | 4 | DDP |
| 24GB × 4 | Qwen-14B | 2 | 8 | DeepSpeed ZeRO-2 |

### 超参数调优建议

```yaml
# 初始配置（保守）
learning_rate: 1e-5
warmup_ratio: 0.1
weight_decay: 0.01
num_epochs: 3

# 如果欠拟合
learning_rate: 2e-5  # 提高学习率
num_epochs: 5        # 增加训练轮数

# 如果过拟合
weight_decay: 0.05   # 增加正则化
dropout: 0.2         # 增加dropout
```

---

## 📊 评估指标说明

### 核心指标

| 指标 | 含义 | 目标值 | 说明 |
|------|------|--------|------|
| **MSE** | 均方误差 | < 0.1 | 预测分数与真实分数的平方误差 |
| **MAE** | 平均绝对误差 | < 0.3 | 约1.5分的误差（5分制） |
| **Pearson相关系数** | 线性相关性 | > 0.7 | 预测与真实的线性相关 |
| **Spearman相关系数** | 秩相关性 | > 0.7 | 预测与真实的排序一致性 |

### 各维度独立评估

```python
# 评估输出示例
{
  "accuracy_mse": 0.085,
  "accuracy_pearson": 0.82,
  "professionalism_mse": 0.072,
  "professionalism_pearson": 0.85,
  "depth_of_analysis_mse": 0.095,
  "depth_of_analysis_pearson": 0.78,
  "overall_mse": 0.084,
  "overall_mae": 0.25
}
```

---

## 🎓 后续应用场景

### 1. RLHF训练

```python
# 使用Reward Model作为奖励信号
from transformers import AutoModelForCausalLM
from ppo_trainer import PPOTrainer

policy_model = AutoModelForCausalLM.from_pretrained("your-llm")
reward_model = load_reward_model("output/best_model")

trainer = PPOTrainer(
    model=policy_model,
    reward_model=reward_model,
    ...
)

trainer.train()
```

### 2. 质量评估系统

```python
# 批量评估生成报告
evaluator = RewardModelInference("output/best_model", "config.yaml")

for report in generated_reports:
    scores = evaluator.score_report(
        report['system_prompt'],
        report['input_data'],
        report['report']
    )
    
    # 根据分数进行分级
    if scores['overall_score'] >= 4.5:
        tier = 'A+'  # 优秀
    elif scores['overall_score'] >= 4.0:
        tier = 'A'   # 良好
    else:
        tier = 'B'   # 需改进
```

### 3. 数据筛选

```python
# 从大量弱标注数据中筛选高质量样本
high_quality_samples = []

for sample in weak_labeled_data:
    score = reward_model.score_report(...)
    
    if score['overall_score'] >= 4.5 and \
       score['scores']['accuracy'] >= 4.3:
        high_quality_samples.append(sample)

# 用于后续SFT训练
train_sft_model(high_quality_samples)
```

### 4. 在线学习

```python
# 持续收集用户反馈，定期重训练
def collect_feedback():
    user_ratings = get_user_ratings()
    return user_ratings

# 每周重训练
feedback_data = collect_feedback()
finetune_reward_model(feedback_data)
```

---

## 🛠️ 常见问题与解决方案

### 训练问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| OOM（显存不足） | Batch size太大 | 减小batch_size，增加grad_accum |
| 训练慢 | 单卡训练 | 使用多卡DDP或DeepSpeed |
| Loss不下降 | 学习率问题 | 调整学习率或warmup_ratio |
| 梯度爆炸 | 梯度裁剪失效 | 检查max_grad_norm设置 |
| 验证集不收敛 | 过拟合 | 增加weight_decay或dropout |

### 数据问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 数据不平衡 | 某个分数段样本过多 | 数据重采样或加权损失 |
| 标注不一致 | 标注员标准不统一 | 制定详细标注指南 |
| 数据量不足 | 样本太少 | 数据增强或使用预训练模型 |

---

## 📚 参考资源

### 论文

- **InstructGPT**: Training language models to follow instructions with human feedback
- **RLHF**: Fine-Tuning Language Models from Human Preferences
- **Constitutional AI**: Training a Helpful and Harmless Assistant with RLHF

### 框架文档

- [PyTorch Distributed Training](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

### 相关项目

- [trl](https://github.com/huggingface/trl): Transformer Reinforcement Learning
- [OpenRLHF](https://github.com/OpenLLMAI/OpenRLHF): 开源RLHF框架
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory): 易用的LLM微调框架

---

## 📝 检查清单

### 训练前

- [ ] 已安装所有依赖
- [ ] GPU驱动和CUDA正常
- [ ] 数据格式正确（使用visualize_data.py检查）
- [ ] 配置文件已根据硬件调整
- [ ] 有足够的存储空间（模型+日志 > 20GB）

### 训练中

- [ ] Loss正常下降
- [ ] GPU利用率 > 80%
- [ ] 没有NaN或Inf
- [ ] 验证集指标稳步提升
- [ ] 定期保存checkpoint

### 训练后

- [ ] 测试集评估通过
- [ ] 模型可以正常推理
- [ ] 推理速度满足需求
- [ ] 保存了最佳模型和配置
- [ ] 记录了实验结果

---

## 🤝 贡献与支持

如有问题或建议，欢迎：
- 提Issue
- 提Pull Request
- 参与讨论

## 📄 许可证

MIT License

---

**祝训练顺利！🚀**

如需更详细的说明，请参考：
- `README.md` - 完整英文文档
- `快速开始.md` - 中文快速上手指南

