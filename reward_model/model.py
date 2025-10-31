import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, Optional, Tuple


class FinancialRewardModel(nn.Module):
    """
    Financial Report Reward Model
    评估财务分析报告的准确度、专业度和分析深度
    """

    def __init__(
        self,
        base_model_name: str,
        num_dimensions: int = 3,
        use_multi_head: bool = False,
        pooling_strategy: str = "last",
    ):
        super().__init__()

        self.config = AutoConfig.from_pretrained(base_model_name)
        self.base_model = AutoModel.from_pretrained(base_model_name)
        self.hidden_size = self.config.hidden_size
        self.num_dimensions = num_dimensions
        self.use_multi_head = use_multi_head
        self.pooling_strategy = pooling_strategy

        if use_multi_head:
            # 每个维度独立的评分头
            self.score_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.hidden_size, self.hidden_size // 2),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(self.hidden_size // 2, 1),
                    )
                    for _ in range(num_dimensions)
                ]
            )
        else:
            # 单个评分头输出综合分数
            self.score_head = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_size // 2, num_dimensions),
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
            labels: [batch_size, num_dimensions] 每个维度的得分
        Returns:
            Dict containing loss and scores
        """
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )

        hidden_states = outputs.last_hidden_state
        pooled_output = self.pool_hidden_states(hidden_states, attention_mask)

        if self.use_multi_head:
            # 多头输出
            scores = torch.cat(
                [head(pooled_output) for head in self.score_heads], dim=1
            )
        else:
            scores = self.score_head(pooled_output)

        result = {"scores": scores}

        if labels is not None:
            # 计算MSE损失
            loss = nn.functional.mse_loss(scores, labels)
            result["loss"] = loss

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
