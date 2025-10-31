import json
import argparse
from pathlib import Path
from collections import defaultdict


def analyze_dataset(data_path: str):
    """分析数据集统计信息"""
    
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    print(f"\n{'='*60}")
    print(f"数据集分析: {data_path}")
    print(f"{'='*60}\n")
    
    # 基础统计
    print(f"样本数量: {len(data)}")
    
    # 评分统计
    scores_by_dimension = defaultdict(list)
    for item in data:
        if 'scores' in item:
            for dim, score in item['scores'].items():
                scores_by_dimension[dim].append(score)
    
    print(f"\n评分统计 (0-5分):")
    print(f"{'-'*60}")
    for dim, scores in scores_by_dimension.items():
        avg = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        print(f"  {dim:20s}: 平均={avg:.2f}, 最小={min_score:.1f}, 最大={max_score:.1f}")
    
    # 报告长度统计
    report_lengths = [len(item['report']) for item in data if 'report' in item]
    if report_lengths:
        avg_length = sum(report_lengths) / len(report_lengths)
        min_length = min(report_lengths)
        max_length = max(report_lengths)
        
        print(f"\n报告长度统计 (字符数):")
        print(f"{'-'*60}")
        print(f"  平均长度: {avg_length:.0f}")
        print(f"  最短报告: {min_length}")
        print(f"  最长报告: {max_length}")
    
    # 数据分布可视化
    print(f"\n评分分布:")
    print(f"{'-'*60}")
    
    for dim, scores in scores_by_dimension.items():
        print(f"\n{dim}:")
        
        # 创建简单的直方图
        bins = [0, 3.0, 3.5, 4.0, 4.5, 5.0]
        bin_labels = ['< 3.0', '3.0-3.5', '3.5-4.0', '4.0-4.5', '4.5-5.0']
        bin_counts = [0] * len(bin_labels)
        
        for score in scores:
            for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
                if low <= score < high or (i == len(bins)-2 and score == high):
                    bin_counts[i] += 1
                    break
        
        max_count = max(bin_counts) if bin_counts else 1
        for label, count in zip(bin_labels, bin_counts):
            bar = '█' * int(20 * count / max_count) if max_count > 0 else ''
            print(f"  {label:10s} |{bar:20s} {count}")
    
    # 样本预览
    print(f"\n{'='*60}")
    print(f"样本预览 (第1个样本):")
    print(f"{'='*60}\n")
    
    if data:
        sample = data[0]
        print(f"System Prompt:")
        print(f"  {sample.get('system_prompt', 'N/A')[:100]}...")
        
        print(f"\nInput Data Keys:")
        if 'input_data' in sample:
            print(f"  {', '.join(sample['input_data'].keys())}")
        
        print(f"\nReport Preview:")
        report = sample.get('report', 'N/A')
        preview = report[:300] + '...' if len(report) > 300 else report
        print(f"  {preview}")
        
        if 'scores' in sample:
            print(f"\nScores:")
            for dim, score in sample['scores'].items():
                print(f"  {dim}: {score}")
    
    print(f"\n{'='*60}\n")


def check_data_quality(data_path: str):
    """检查数据质量"""
    
    issues = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            try:
                item = json.loads(line)
                
                # 检查必需字段
                required_fields = ['system_prompt', 'input_data', 'report', 'scores']
                for field in required_fields:
                    if field not in item:
                        issues.append(f"行 {i}: 缺少字段 '{field}'")
                
                # 检查评分范围
                if 'scores' in item:
                    for dim, score in item['scores'].items():
                        if not (0 <= score <= 5):
                            issues.append(f"行 {i}: {dim}分数超出范围 [0, 5]: {score}")
                
                # 检查报告长度
                if 'report' in item and len(item['report']) < 100:
                    issues.append(f"行 {i}: 报告过短 ({len(item['report'])}字符)")
                
            except json.JSONDecodeError as e:
                issues.append(f"行 {i}: JSON解析错误: {e}")
    
    print(f"\n{'='*60}")
    print(f"数据质量检查: {data_path}")
    print(f"{'='*60}\n")
    
    if issues:
        print(f"发现 {len(issues)} 个问题:\n")
        for issue in issues[:10]:  # 只显示前10个
            print(f"  ⚠ {issue}")
        if len(issues) > 10:
            print(f"\n  ... 还有 {len(issues) - 10} 个问题")
    else:
        print("✓ 数据质量检查通过！")
    
    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='数据集分析和可视化工具')
    parser.add_argument('--data', type=str, required=True, help='数据文件路径')
    parser.add_argument('--check', action='store_true', help='执行数据质量检查')
    args = parser.parse_args()
    
    if not Path(args.data).exists():
        print(f"错误: 文件不存在: {args.data}")
        return
    
    # 分析数据集
    analyze_dataset(args.data)
    
    # 可选：质量检查
    if args.check:
        check_data_quality(args.data)


if __name__ == '__main__':
    main()

