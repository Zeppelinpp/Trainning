import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, Optional, Tuple


class FinancialRewardModel(nn.Module):
    """
    Financial Report Reward Model
    评估财务分析报告的准确度、专业度和分析深度
    
    使用多维度分类：每个维度5个档位（0-4分）
    维度：
    1. 分析深度 (depth)
    2. 专业度 (professionalism)  
    3. 数值计算准确性 (accuracy)
    """

    def __init__(
        self,
        base_model_name: str,
        num_dimensions: int = 3,
        num_classes: int = 5,  # 0-4分，共5个档位
        use_multi_head: bool = True,  # 推荐使用多头
        pooling_strategy: str = "last",
    ):
        super().__init__()

        self.config = AutoConfig.from_pretrained(base_model_name)
        self.base_model = AutoModel.from_pretrained(base_model_name)
        self.hidden_size = self.config.hidden_size
        self.num_dimensions = num_dimensions
        self.num_classes = num_classes
        self.use_multi_head = use_multi_head
        self.pooling_strategy = pooling_strategy

        if use_multi_head:
            # 每个维度独立的分类头（推荐方式）
            self.score_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.hidden_size, self.hidden_size // 2),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(self.hidden_size // 2, num_classes),  # 输出5个类别的logits
                    )
                    for _ in range(num_dimensions)
                ]
            )
        else:
            # 共享的分类头（输出 num_dimensions * num_classes）
            self.score_head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_size // 2, num_dimensions * num_classes),
            )

    def pool_hidden_states(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """从隐藏状态提取表示向量"""
        if self.pooling_strategy == "last":
            # 使用最后一个非padding token
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.shape[0]
            pooled = hidden_states[torch.arange(batch_size), sequence_lengths]
        elif self.pooling_strategy == "mean":
            # 平均池化
            mask_expanded = (
                attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            )
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_hidden / sum_mask
        elif self.pooling_strategy == "cls":
            # 使用CLS token
            pooled = hidden_states[:, 0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")

        return pooled

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, num_dimensions] 每个维度的类别标签 (0-4)
        Returns:
            Dict containing loss, logits, and predicted scores
        """
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )

        hidden_states = outputs.last_hidden_state
        pooled_output = self.pool_hidden_states(hidden_states, attention_mask)

        if self.use_multi_head:
            # 多头输出：每个维度输出 num_classes 个logits
            # logits_list: List of [batch_size, num_classes]
            logits_list = [head(pooled_output) for head in self.score_heads]
            # Stack to [batch_size, num_dimensions, num_classes]
            logits = torch.stack(logits_list, dim=1)
        else:
            # 共享头输出：[batch_size, num_dimensions * num_classes]
            logits_flat = self.score_head(pooled_output)
            # Reshape to [batch_size, num_dimensions, num_classes]
            logits = logits_flat.view(-1, self.num_dimensions, self.num_classes)

        # 预测类别（argmax）
        predicted_labels = torch.argmax(logits, dim=-1)  # [batch_size, num_dimensions]

        result = {
            "logits": logits,  # [batch_size, num_dimensions, num_classes]
            "predicted_labels": predicted_labels,  # [batch_size, num_dimensions]
        }

        if labels is not None:
            # labels: [batch_size, num_dimensions]，值为0-4的整数
            # 计算交叉熵损失
            batch_size = labels.shape[0]
            
            # Reshape logits for cross_entropy: [batch_size * num_dimensions, num_classes]
            logits_flat = logits.view(-1, self.num_classes)
            # Reshape labels: [batch_size * num_dimensions]
            labels_flat = labels.view(-1)
            
            # 计算交叉熵损失
            loss = nn.functional.cross_entropy(logits_flat, labels_flat)
            
            # 计算准确率
            correct = (predicted_labels == labels).float()
            accuracy = correct.mean()
            per_dim_accuracy = correct.mean(dim=0)  # [num_dimensions]
            
            result["loss"] = loss
            result["accuracy"] = accuracy
            result["per_dim_accuracy"] = per_dim_accuracy

        return result


class PairwiseRewardModel(nn.Module):
    """
    成对比较的Reward Model
    给定两个报告，判断哪个更好
    """

    def __init__(self, base_model_name: str):
        super().__init__()

        self.config = AutoConfig.from_pretrained(base_model_name)
        self.base_model = AutoModel.from_pretrained(base_model_name)
        self.hidden_size = self.config.hidden_size

        self.score_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, 1),
        )

    def get_score(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """获取单个样本的分数"""
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )

        # 使用最后一个token的隐藏状态
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = input_ids.shape[0]
        last_hidden = outputs.last_hidden_state[
            torch.arange(batch_size), sequence_lengths
        ]

        score = self.score_head(last_hidden)
        return score

    def forward(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            chosen_input_ids: 更好的报告
            rejected_input_ids: 较差的报告
        Returns:
            Dict containing loss and scores
        """
        chosen_score = self.get_score(chosen_input_ids, chosen_attention_mask)
        rejected_score = self.get_score(rejected_input_ids, rejected_attention_mask)

        # 使用ranking loss: chosen应该得分更高
        loss = -nn.functional.logsigmoid(chosen_score - rejected_score).mean()

        return {
            "loss": loss,
            "chosen_score": chosen_score,
            "rejected_score": rejected_score,
            "accuracy": (chosen_score > rejected_score).float().mean(),
        }
