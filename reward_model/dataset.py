import json
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import Dict, List, Optional


class FinancialRewardDataset(Dataset):
    """
    财务分析报告Reward Model数据集
    支持点式评分和成对比较两种模式
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 4096,
        mode: str = "pointwise"  # pointwise or pairwise
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.data = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
    
    def __len__(self) -> int:
        return len(self.data)
    
    def format_input(self, item: Dict) -> str:
        """格式化输入文本"""
        prompt = f"""### 系统提示
{item['system_prompt']}

### 输入数据
{json.dumps(item['input_data'], ensure_ascii=False, indent=2)}

### 生成的财务分析报告
{item['report']}"""
        return prompt
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        if self.mode == "pointwise":
            # 点式评分模式
            text = self.format_input(item)
            
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            scores = item['scores']
            # 标准化到0-1范围
            score_tensor = torch.tensor([
                scores['accuracy'] / 5.0,
                scores['professionalism'] / 5.0,
                scores['depth_of_analysis'] / 5.0
            ], dtype=torch.float32)
            
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0),
                "labels": score_tensor
            }
        
        elif self.mode == "pairwise":
            # 成对比较模式
            chosen_text = self.format_input(item['chosen'])
            rejected_text = self.format_input(item['rejected'])
            
            chosen_encoding = self.tokenizer(
                chosen_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            rejected_encoding = self.tokenizer(
                rejected_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            
            return {
                "chosen_input_ids": chosen_encoding["input_ids"].squeeze(0),
                "chosen_attention_mask": chosen_encoding["attention_mask"].squeeze(0),
                "rejected_input_ids": rejected_encoding["input_ids"].squeeze(0),
                "rejected_attention_mask": rejected_encoding["attention_mask"].squeeze(0)
            }
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


def collate_fn_pointwise(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """点式评分的collate函数"""
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch])
    }


def collate_fn_pairwise(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """成对比较的collate函数"""
    return {
        "chosen_input_ids": torch.stack([item["chosen_input_ids"] for item in batch]),
        "chosen_attention_mask": torch.stack([item["chosen_attention_mask"] for item in batch]),
        "rejected_input_ids": torch.stack([item["rejected_input_ids"] for item in batch]),
        "rejected_attention_mask": torch.stack([item["rejected_attention_mask"] for item in batch])
    }

