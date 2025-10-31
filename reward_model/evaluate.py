import torch
import yaml
import json
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

from model import FinancialRewardModel
from dataset import FinancialRewardDataset


class RewardModelEvaluator:
    """Reward Model评估器"""
    
    def __init__(self, checkpoint_path: str, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        
        # 加载模型
        checkpoint = torch.load(
            Path(checkpoint_path) / 'pytorch_model.bin',
            map_location=self.device
        )
        
        self.model = FinancialRewardModel(
            base_model_name=self.config['model']['base_model'],
            num_dimensions=3,
            use_multi_head=False,
            pooling_strategy="last"
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def evaluate_dataset(self, data_path: str) -> dict:
        """评估整个数据集"""
        dataset = FinancialRewardDataset(
            data_path=data_path,
            tokenizer=self.tokenizer,
            max_length=self.config['model']['max_length'],
            mode="pointwise"
        )
        
        predictions = []
        labels = []
        
        for item in tqdm(dataset, desc="Evaluating"):
            input_ids = item['input_ids'].unsqueeze(0).to(self.device)
            attention_mask = item['attention_mask'].unsqueeze(0).to(self.device)
            label = item['labels'].numpy()
            
            output = self.model(input_ids, attention_mask)
            pred = output['scores'].cpu().numpy()[0]
            
            predictions.append(pred)
            labels.append(label)
        
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        # 计算各维度的评估指标
        metrics = {}
        dimensions = ['accuracy', 'professionalism', 'depth_of_analysis']
        
        for i, dim in enumerate(dimensions):
            pred_dim = predictions[:, i]
            label_dim = labels[:, i]
            
            mse = mean_squared_error(label_dim, pred_dim)
            mae = mean_absolute_error(label_dim, pred_dim)
            
            # 转换回原始分数范围(0-5)
            pred_dim_scaled = pred_dim * 5.0
            label_dim_scaled = label_dim * 5.0
            
            pearson_corr, _ = pearsonr(label_dim_scaled, pred_dim_scaled)
            spearman_corr, _ = spearmanr(label_dim_scaled, pred_dim_scaled)
            
            metrics[f'{dim}_mse'] = float(mse)
            metrics[f'{dim}_mae'] = float(mae)
            metrics[f'{dim}_pearson'] = float(pearson_corr)
            metrics[f'{dim}_spearman'] = float(spearman_corr)
        
        # 总体指标
        overall_mse = mean_squared_error(labels.flatten(), predictions.flatten())
        overall_mae = mean_absolute_error(labels.flatten(), predictions.flatten())
        
        metrics['overall_mse'] = float(overall_mse)
        metrics['overall_mae'] = float(overall_mae)
        
        return metrics
    
    @torch.no_grad()
    def predict_single(self, system_prompt: str, input_data: dict, report: str) -> dict:
        """对单个样本进行预测"""
        text = f"""### 系统提示
{system_prompt}

### 输入数据
{json.dumps(input_data, ensure_ascii=False, indent=2)}

### 生成的财务分析报告
{report}"""
        
        encoding = self.tokenizer(
            text,
            max_length=self.config['model']['max_length'],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        output = self.model(input_ids, attention_mask)
        scores = output['scores'].cpu().numpy()[0]
        
        # 转换回原始分数范围
        return {
            'accuracy': float(scores[0] * 5.0),
            'professionalism': float(scores[1] * 5.0),
            'depth_of_analysis': float(scores[2] * 5.0),
            'overall_score': float(scores.mean() * 5.0)
        }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--test_file', type=str, default='data/test.jsonl')
    parser.add_argument('--output', type=str, default='evaluation_results.json')
    args = parser.parse_args()
    
    evaluator = RewardModelEvaluator(args.checkpoint, args.config)
    
    metrics = evaluator.evaluate_dataset(args.test_file)
    
    print("\n=== Evaluation Results ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()

