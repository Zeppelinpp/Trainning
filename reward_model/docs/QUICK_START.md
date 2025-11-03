# 快速开始 - 多维度奖励模型

## 🎯 目标

训练一个多维度评分模型，在以下3个维度上评估财务分析报告（每维度0-4分）：
1. **分析深度** (depth)
2. **专业度** (professionalism)
3. **数值计算准确性** (accuracy)

## ⚡ 三步走

### 第一步：生成数据（约30分钟，取决于样本数）

```bash
uv run reward_model/data/synthetic_gen_v2.py
```

这会：
1. ✅ 生成黄金响应（高质量）
2. ✅ 受控降级生成缺陷响应
3. ✅ AI裁判多维度打分（0-4分）
4. ✅ 输出 `comparison_pairs_scored.jsonl`

### 第二步：验证数据质量

```bash
uv run reward_model/data/validate_multidim_pairs.py \
    reward_model/data/comparison_pairs_scored.jsonl
```

查看报告，确保：
- ✅ 平均分差 > 1.0
- ✅ 正差异率 > 90%
- ✅ 各维度分布合理

### 第三步：训练模型（约1-2小时，取决于GPU）

```bash
uv run reward_model/train_multidim.py
```

模型保存在 `reward_model/outputs/multidim/best_model.pt`

## 📊 数据格式说明

### 生成的数据格式
```json
{
  "prompt": "用户需求",
  "chosen": "黄金响应",
  "rejected": "缺陷响应",
  "scores": {
    "chosen": {"depth": 4, "professionalism": 3, "accuracy": 4},
    "rejected": {"depth": 1, "professionalism": 2, "accuracy": 3}
  }
}
```

### 为什么是0-4分？
- **4分**：优秀（完全符合高质量标准）
- **3分**：良好（基本符合质量标准）
- **2分**：中等（存在明显不足）
- **1分**：较差（有严重问题）
- **0分**：很差（基本不可用）

## 🔧 核心配置

### 数据生成配置（synthetic_gen_v2.py）

```python
generate_comparison_dataset(
    fields=["制造业", "服务业", "金融业", "房地产", "科技业"],
    n_pairs_per_field=10,  # 每个行业生成10个对比对
    model_configs=[...],    # 生成模型列表
)
```

**调整建议**：
- 测试阶段：`n_pairs_per_field=2`（快速验证）
- 正式训练：`n_pairs_per_field=20-50`（充分数据）

### 训练配置（train_multidim.py）

```python
config = {
    "base_model_name": "hfl/chinese-roberta-wwm-ext",
    "batch_size": 4,        # 根据显存调整
    "learning_rate": 2e-5,
    "num_epochs": 10,
    "max_length": 2048,     # 根据报告长度调整
}
```

**显存优化**：
- 4GB: `batch_size=1`, `max_length=1024`
- 8GB: `batch_size=2`, `max_length=1536`
- 16GB+: `batch_size=4`, `max_length=2048`

## 🎨 核心创新：黄金标准 + 受控降级

### 传统方法的问题
```
生成模型A（优质提示词） → 响应A
生成模型B（劣质提示词） → 响应B
```
❌ 问题：响应A和B的差异不可控，可能完全不同

### 我们的方法
```
生成模型（优质提示词） → 黄金响应A
                           ↓
               黄金响应A + 降级提示词 → 缺陷响应B
```
✅ 优势：
- 响应B是从A降级而来，结构类似
- 差异是**受控的**（如只降低"深度"）
- AI裁判打分更准确

### 5种质量维度 × 5种降级方式

**质量维度**（用于生成黄金响应）：
1. 数据准确性导向
2. 深度分析导向
3. 行业专业性导向
4. 全面性导向
5. 数据完整性导向

**降级方式**（用于生成缺陷响应）：
1. 浅化深度
2. 简化计算
3. 泛化通用
4. 模糊精确
5. 片面覆盖

随机组合 → 多样化数据集！

## 📈 训练效果预期

### 良好的训练曲线
```
Epoch 1: Loss=1.2, Acc=45%
Epoch 3: Loss=0.8, Acc=65%
Epoch 5: Loss=0.5, Acc=75%
Epoch 10: Loss=0.3, Acc=82%
```

### 各维度预期表现
- **Depth**: 80-85% (较难，需要理解分析逻辑)
- **Professionalism**: 75-80% (中等)
- **Accuracy**: 85-90% (较易，数值计算可验证)

### 如果效果不好？

1. **准确率 < 60%**
   - 数据质量问题：检查分数差异是否明显
   - 模型容量不足：换更大的基础模型
   - 数据量不足：增加`n_pairs_per_field`

2. **某维度准确率很低**
   - 标签不平衡：使用类别权重
   - 降级不明显：调整降级提示词
   - 数据不足：增加该维度的样本

3. **训练不收敛**
   - 降低学习率：`learning_rate=1e-5`
   - 增加warmup：`warmup_ratio=0.2`
   - 检查数据：是否有异常样本

## 🎓 模型使用

### 推理示例

```python
import torch
from transformers import AutoTokenizer
from model import FinancialRewardModel

# 加载模型
model = FinancialRewardModel(
    base_model_name="hfl/chinese-roberta-wwm-ext",
    num_dimensions=3,
    num_classes=5,
)
checkpoint = torch.load("outputs/multidim/best_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

# 准备输入
text = "财务分析报告全文..."
inputs = tokenizer(text, return_tensors="pt", max_length=2048, truncation=True)

# 推理
with torch.no_grad():
    outputs = model(**inputs)
    predicted_labels = outputs["predicted_labels"][0]  # [3]
    
print(f"分析深度: {predicted_labels[0]}分")
print(f"专业度: {predicted_labels[1]}分")
print(f"数值准确性: {predicted_labels[2]}分")
```

### 输出logits（用于softmax得到概率分布）

```python
logits = outputs["logits"][0]  # [3, 5]

# 转换为概率
import torch.nn.functional as F
probs = F.softmax(logits, dim=-1)

print("分析深度分数分布:")
for score, prob in enumerate(probs[0]):
    print(f"  {score}分: {prob:.2%}")
```

## 📚 下一步

1. **调优模型**
   - 尝试不同的基础模型
   - 调整学习率和训练轮数
   - 使用更多数据

2. **应用场景**
   - 财务报告质量评估
   - 自动反馈生成
   - 报告改进建议

3. **扩展维度**
   - 增加新的评分维度
   - 调整评分档位（如0-9分）
   - 针对特定行业定制

## ❓ 遇到问题？

查看详细文档：
- 📘 [完整指南](README_MULTIDIM.md)
- 📊 [数据生成说明](data/README_V2.md)
- 🐛 [常见问题](README_MULTIDIM.md#常见问题)

---

**开始你的多维度奖励模型训练之旅吧！** 🚀

