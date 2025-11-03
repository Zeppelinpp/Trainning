# 数据生成说明

## 📋 流程概览

### 步骤0：准备阶段（使用synthetic_gen.py）

生成多样化的分析框架和系统提示词：

```bash
uv run reward_model/data/synthetic_gen.py
```

这会生成：
- `analysis_framework/` - 多个不同的分析框架（不同行业、不同视角）
- `system_prompt/` - 多个系统提示词（正面/负面）

### 步骤1-2：生成对比对并打分（使用synthetic_gen_v2.py）

```bash
uv run reward_model/data/synthetic_gen_v2.py
```

#### 完整流程

1. **加载资源**
   - 从 `analysis_framework/` 加载所有分析框架
   - 从 `system_prompt/` 加载所有系统提示词

2. **构建输入Prompt**
   ```
   <角色定义>
   
   指引
   {{ system_prompt }}  # 随机选择
   
   分析框架
   {{ analysis_framework }}  # 随机选择
   
   参考数据
   {{ data }}
   ```

3. **生成黄金响应**
   - 使用完整的输入prompt
   - 添加优质输出指示
   - 低温度（0.3）保证质量

4. **受控降级**
   - 将黄金响应作为输入
   - 使用降级提示词（5种随机选择）
   - 生成缺陷响应

5. **AI裁判打分**
   - 三维度评分（0-4分）：
     - depth: 分析深度
     - professionalism: 专业度
     - accuracy: 数值准确性

## 📁 输出结构

```
data/
├── dataset/                          # 最终数据集输出目录
│   ├── comparison_pairs.jsonl       # 未打分的对比对
│   └── comparison_pairs_scored.jsonl # AI裁判打分后的数据
│
├── analysis_framework/               # 分析框架库
│   ├── qwen-turbo_制造业_framework_1.md
│   ├── qwen-turbo_制造业_framework_2.md
│   └── ...
│
└── system_prompt/                    # 系统提示词库
    ├── positive_qwen-turbo_制造业_sys_prompt_1.md
    ├── negative_qwen-turbo_制造业_sys_prompt_2.md
    └── ...
```

## 📊 数据格式

### comparison_pairs_scored.jsonl

```json
{
  "prompt": "<角色定义>\n\n指引\n\n{{ system_prompt }}\n\n分析框架\n\n{{ analysis_framework }}\n\n参考数据\n\n{{ data }}",
  "chosen": "黄金响应全文...",
  "rejected": "缺陷响应全文...",
  "metadata": {
    "field": "制造业",
    "model": "qwen-plus",
    "gold_metadata": {...},
    "defect_metadata": {...}
  },
  "scores": {
    "chosen": {
      "depth": 4,
      "professionalism": 3,
      "accuracy": 4
    },
    "rejected": {
      "depth": 1,
      "professionalism": 2,
      "accuracy": 3
    },
    "reasoning": {
      "depth": "黄金响应有深入的归因分析...",
      "professionalism": "黄金响应使用专业术语...",
      "accuracy": "黄金响应计算精确..."
    },
    "overall_assessment": "黄金响应在所有维度都明显优于缺陷响应..."
  }
}
```

## ⚙️ 配置说明

### synthetic_gen.py 配置

```python
prompt_pipeline(
    fields=["制造业", "服务业", "金融业", "房地产", "科技业"],
    model_configs=[...],  # 使用多个模型增加多样性
    samples_per_field=20,  # 每个行业生成20组
    samples_per_model=5,   # 每组生成5个变体
)
```

### synthetic_gen_v2.py 配置

```python
generate_comparison_dataset(
    fields=["制造业", "服务业", "金融业", "房地产", "科技业"],
    model_configs=[...],  # 用于生成响应的模型
    n_pairs_per_field=10,  # 每个行业生成10个对比对
    framework_dir="./reward_model/data/analysis_framework/",
    system_prompt_dir="./reward_model/data/system_prompt/",
    output_dir="./reward_model/data/dataset/",
)
```

## 🎯 核心特点

1. **多样性最大化**
   - 多个分析框架 × 多个系统提示词 × 多个生成模型
   - 随机组合确保每个样本都不同

2. **受控降级**
   - 黄金响应确实高质量
   - 缺陷响应有明确的降级类型
   - 对比对差异可控且明显

3. **三维度评分**
   - 不是简单的chosen/rejected
   - 每个维度独立评分（0-4分）
   - AI裁判提供详细理由

## 🔧 调优建议

### 增加数据量

```python
# synthetic_gen.py
samples_per_field=50  # 增加到50组
samples_per_model=10  # 每组10个变体

# synthetic_gen_v2.py  
n_pairs_per_field=50  # 每个行业50个对比对
```

### 使用真实数据

修改 `SAMPLE_DATA` 变量，替换为真实的财务数据。

### 调整评分标准

修改 `add_multidim_scores()` 函数中的AI裁判提示词，调整评分档位或维度。

## ❓ 常见问题

**Q: 为什么要分两步生成？**

A: 第一步(`synthetic_gen.py`)生成多样化的框架和提示词库，第二步(`synthetic_gen_v2.py`)从库中随机组合，确保每个对比对都不同。

**Q: 可以只运行一步吗？**

A: 可以，但需要确保 `analysis_framework/` 和 `system_prompt/` 目录已有足够的文件。

**Q: 如何验证数据质量？**

A: 使用验证脚本：
```bash
uv run reward_model/data/validate_multidim_pairs.py \
    reward_model/data/dataset/comparison_pairs_scored.jsonl
```

**Q: 生成的数据保存在哪里？**

A: 保存在 `reward_model/data/dataset/` 目录下。

