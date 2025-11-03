"""
多维度评分数据集
用于训练多维度分类的奖励模型
"""

import json
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import Dict, List, Any, Optional


class MultiDimRewardDataset(Dataset):
    """
    多维度奖励模型数据集
    
    数据格式：
    {
        "prompt": "用户输入",
        "chosen": "黄金响应",
        "rejected": "缺陷响应",
        "scores": {
            "chosen": {
                "depth": 4,
                "professionalism": 3,
                "accuracy": 4
            },
            "rejected": {
                "depth": 1,
                "professionalism": 2,
                "accuracy": 2
            }
        }
    }
    """

    def __init__(
        self,
        data_file: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        include_prompt: bool = True,
        dimensions: List[str] = None,
    ):
        """
        Args:
            data_file: JSONL格式的数据文件
            tokenizer: 分词器
            max_length: 最大序列长度
            include_prompt: 是否在输入中包含prompt
            dimensions: 评分维度列表，默认为 ["depth", "professionalism", "accuracy"]
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_prompt = include_prompt
        self.dimensions = dimensions or ["depth", "professionalism", "accuracy"]

        # Load data
        self.data = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if "scores" in item:  # Only include scored items
                    self.data.append(item)

        print(f"Loaded {len(self.data)} samples from {data_file}")

        # Create both chosen and rejected examples
        self.examples = []
        for item in self.data:
            # Add chosen example
            self.examples.append(
                {
                    "text": self._format_text(item["prompt"], item["chosen"]),
                    "labels": self._extract_labels(item["scores"]["chosen"]),
                    "type": "chosen",
                    "metadata": item.get("metadata", {}),
                }
            )
            # Add rejected example
            self.examples.append(
                {
                    "text": self._format_text(item["prompt"], item["rejected"]),
                    "labels": self._extract_labels(item["scores"]["rejected"]),
                    "type": "rejected",
                    "metadata": item.get("metadata", {}),
                }
            )

        print(f"Created {len(self.examples)} training examples (chosen + rejected)")

    def _format_text(self, prompt: str, response: str) -> str:
        """格式化输入文本"""
        if self.include_prompt:
            return f"{prompt}\n\n{response}"
        else:
            return response

    def _extract_labels(self, scores: Dict[str, int]) -> List[int]:
        """提取标签向量"""
        return [scores[dim] for dim in self.dimensions]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]

        # Tokenize
        encoding = self.tokenizer(
            example["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Convert labels to tensor
        labels = torch.tensor(example["labels"], dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
        }


class PairwiseMultiDimDataset(Dataset):
    """
    成对比较的多维度数据集
    每个样本包含一个对比对（chosen vs rejected）
    
    用于对比学习训练
    """

    def __init__(
        self,
        data_file: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        include_prompt: bool = True,
        dimensions: List[str] = None,
        min_score_diff: float = 0.0,  # 最小分数差异过滤
    ):
        """
        Args:
            data_file: JSONL格式的数据文件
            tokenizer: 分词器
            max_length: 最大序列长度
            include_prompt: 是否在输入中包含prompt
            dimensions: 评分维度列表
            min_score_diff: 过滤掉chosen和rejected平均分差异小于此值的样本
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_prompt = include_prompt
        self.dimensions = dimensions or ["depth", "professionalism", "accuracy"]
        self.min_score_diff = min_score_diff

        # Load and filter data
        self.data = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if "scores" not in item:
                    continue

                # Calculate average score difference
                chosen_avg = sum(item["scores"]["chosen"].values()) / len(
                    item["scores"]["chosen"]
                )
                rejected_avg = sum(item["scores"]["rejected"].values()) / len(
                    item["scores"]["rejected"]
                )
                score_diff = chosen_avg - rejected_avg

                # Filter by score difference
                if score_diff >= min_score_diff:
                    self.data.append(item)

        print(
            f"Loaded {len(self.data)} pairs from {data_file} (min_score_diff={min_score_diff})"
        )

    def _format_text(self, prompt: str, response: str) -> str:
        """格式化输入文本"""
        if self.include_prompt:
            return f"{prompt}\n\n{response}"
        else:
            return response

    def _extract_labels(self, scores: Dict[str, int]) -> List[int]:
        """提取标签向量"""
        return [scores[dim] for dim in self.dimensions]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        # Tokenize chosen
        chosen_text = self._format_text(item["prompt"], item["chosen"])
        chosen_encoding = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize rejected
        rejected_text = self._format_text(item["prompt"], item["rejected"])
        rejected_encoding = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Extract labels
        chosen_labels = torch.tensor(
            self._extract_labels(item["scores"]["chosen"]), dtype=torch.long
        )
        rejected_labels = torch.tensor(
            self._extract_labels(item["scores"]["rejected"]), dtype=torch.long
        )

        return {
            "chosen_input_ids": chosen_encoding["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_encoding["attention_mask"].squeeze(0),
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_encoding["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_encoding["attention_mask"].squeeze(0),
            "rejected_labels": rejected_labels,
        }


def get_label_distribution(data_file: str, dimensions: List[str] = None) -> Dict:
    """
    分析数据集中各维度的标签分布
    
    Args:
        data_file: JSONL数据文件
        dimensions: 评分维度列表
    
    Returns:
        包含统计信息的字典
    """
    dimensions = dimensions or ["depth", "professionalism", "accuracy"]

    # Initialize counters
    chosen_counts = {dim: {i: 0 for i in range(5)} for dim in dimensions}
    rejected_counts = {dim: {i: 0 for i in range(5)} for dim in dimensions}

    total_pairs = 0
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if "scores" not in item:
                continue

            total_pairs += 1

            # Count chosen labels
            for dim in dimensions:
                label = item["scores"]["chosen"][dim]
                chosen_counts[dim][label] += 1

            # Count rejected labels
            for dim in dimensions:
                label = item["scores"]["rejected"][dim]
                rejected_counts[dim][label] += 1

    # Calculate statistics
    stats = {
        "total_pairs": total_pairs,
        "total_samples": total_pairs * 2,  # chosen + rejected
        "chosen_distribution": {},
        "rejected_distribution": {},
        "overall_distribution": {},
    }

    for dim in dimensions:
        # Chosen distribution
        stats["chosen_distribution"][dim] = {
            label: count / total_pairs for label, count in chosen_counts[dim].items()
        }

        # Rejected distribution
        stats["rejected_distribution"][dim] = {
            label: count / total_pairs for label, count in rejected_counts[dim].items()
        }

        # Overall distribution
        overall_counts = {
            label: chosen_counts[dim][label] + rejected_counts[dim][label]
            for label in range(5)
        }
        stats["overall_distribution"][dim] = {
            label: count / (total_pairs * 2)
            for label, count in overall_counts.items()
        }

    return stats


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("用法: python dataset_multidim.py <data_file>")
        sys.exit(1)

    data_file = sys.argv[1]

    # Analyze label distribution
    print("=" * 70)
    print("标签分布分析")
    print("=" * 70)

    stats = get_label_distribution(data_file)

    print(f"\n总对比对数: {stats['total_pairs']}")
    print(f"总样本数: {stats['total_samples']}")

    dimension_names = {
        "depth": "分析深度",
        "professionalism": "专业度",
        "accuracy": "数值准确性",
    }

    for dim in ["depth", "professionalism", "accuracy"]:
        print(f"\n{'=' * 70}")
        print(f"{dimension_names[dim]} ({dim})")
        print(f"{'=' * 70}")

        print("\nChosen样本分布:")
        for label in range(5):
            pct = stats["chosen_distribution"][dim][label] * 100
            print(f"  {label}分: {pct:5.1f}%")

        print("\nRejected样本分布:")
        for label in range(5):
            pct = stats["rejected_distribution"][dim][label] * 100
            print(f"  {label}分: {pct:5.1f}%")

        print("\n整体分布:")
        for label in range(5):
            pct = stats["overall_distribution"][dim][label] * 100
            print(f"  {label}分: {pct:5.1f}%")

