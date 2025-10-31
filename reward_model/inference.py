import torch
import yaml
import json
import argparse
from pathlib import Path
from transformers import AutoTokenizer

from model import FinancialRewardModel


class RewardModelInference:
    """Reward Model推理服务"""
    
    def __init__(self, checkpoint_path: str, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        
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
    def score_report(
        self,
        system_prompt: str,
        input_data: dict,
        report: str
    ) -> dict:
        """对财务分析报告进行评分"""
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
        
        # 转换回原始分数范围(0-5)
        scores_scaled = scores * 5.0
        
        result = {
            'scores': {
                'accuracy': float(scores_scaled[0]),
                'professionalism': float(scores_scaled[1]),
                'depth_of_analysis': float(scores_scaled[2])
            },
            'overall_score': float(scores_scaled.mean()),
            'grade': self._get_grade(scores_scaled.mean())
        }
        
        return result
    
    def _get_grade(self, score: float) -> str:
        """根据分数给出等级"""
        if score >= 4.5:
            return "A+ (优秀)"
        elif score >= 4.0:
            return "A (良好)"
        elif score >= 3.5:
            return "B+ (中上)"
        elif score >= 3.0:
            return "B (中等)"
        elif score >= 2.5:
            return "C (及格)"
        else:
            return "D (需改进)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--input', type=str, required=True, help='Input JSON file')
    args = parser.parse_args()
    
    # 加载推理服务
    inference = RewardModelInference(args.checkpoint, args.config)
    
    # 读取输入
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 评分
    result = inference.score_report(
        system_prompt=data['system_prompt'],
        input_data=data['input_data'],
        report=data['report']
    )
    
    # 输出结果
    print("\n=== 财务分析报告评分结果 ===\n")
    print(f"准确度 (Accuracy): {result['scores']['accuracy']:.2f}/5.0")
    print(f"专业度 (Professionalism): {result['scores']['professionalism']:.2f}/5.0")
    print(f"分析深度 (Depth of Analysis): {result['scores']['depth_of_analysis']:.2f}/5.0")
    print(f"\n综合评分: {result['overall_score']:.2f}/5.0")
    print(f"等级: {result['grade']}")
    
    # 保存结果
    output_file = args.input.replace('.json', '_scored.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({**data, 'model_scores': result}, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存至: {output_file}")


if __name__ == '__main__':
    main()

