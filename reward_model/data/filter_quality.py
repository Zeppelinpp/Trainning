"""
数据质量过滤脚本
过滤掉差异不明显的样本，确保训练数据质量
"""
import json
import argparse
from typing import List, Dict


def calculate_score_diff(sample: Dict) -> int:
    """计算总分差异"""
    scores = sample.get('scores', {})
    chosen = scores.get('chosen', {})
    rejected = scores.get('rejected', {})
    
    if not chosen or not rejected:
        return -999  # 标记为无效
    
    chosen_total = chosen.get('depth', 0) + chosen.get('professionalism', 0) + chosen.get('accuracy', 0)
    rejected_total = rejected.get('depth', 0) + rejected.get('professionalism', 0) + rejected.get('accuracy', 0)
    
    return chosen_total - rejected_total


def calculate_length_diff(sample: Dict) -> int:
    """计算长度差异"""
    chosen_len = len(sample.get('chosen', ''))
    rejected_len = len(sample.get('rejected', ''))
    return abs(chosen_len - rejected_len)


def filter_dataset(
    input_file: str,
    output_file: str,
    min_score_diff: int = 5,
    min_length_diff: int = 50,
    remove_reverse: bool = True,
    verbose: bool = True
):
    """
    过滤数据集
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        min_score_diff: 最小总分差异（默认5分，基于0-9分x3维度=总分0-27）
        min_length_diff: 最小长度差异（默认50字符）
        remove_reverse: 是否移除反向样本（rejected得分更高）
        verbose: 是否打印详细信息
    """
    samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except:
                    pass
    
    print(f"读取了 {len(samples)} 个样本")
    
    # 统计信息
    stats = {
        'total': len(samples),
        'no_score': 0,
        'reverse': 0,
        'low_score_diff': 0,
        'low_length_diff': 0,
        'passed': 0
    }
    
    filtered_samples = []
    
    for sample in samples:
        # 检查是否有评分
        if 'scores' not in sample or sample.get('scores') is None:
            stats['no_score'] += 1
            continue
        
        scores = sample.get('scores', {})
        if not scores.get('chosen') or not scores.get('rejected'):
            stats['no_score'] += 1
            continue
        
        # 计算差异
        score_diff = calculate_score_diff(sample)
        length_diff = calculate_length_diff(sample)
        
        # 过滤反向样本
        if remove_reverse and score_diff < 0:
            stats['reverse'] += 1
            if verbose:
                print(f"[WARN] 过滤反向样本: 总分差异={score_diff}")
            continue
        
        # 过滤评分差异过小的样本
        if score_diff < min_score_diff:
            stats['low_score_diff'] += 1
            if verbose:
                print(f"[WARN] 过滤低评分差异样本: 总分差异={score_diff}")
            continue
        
        # 过滤长度差异过小的样本
        if length_diff < min_length_diff:
            stats['low_length_diff'] += 1
            if verbose:
                print(f"[WARN] 过滤低长度差异样本: 长度差异={length_diff}")
            continue
        
        # 通过所有检查
        stats['passed'] += 1
        filtered_samples.append(sample)
    
    # 保存过滤后的数据
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in filtered_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    # 打印统计信息
    print("\n" + "=" * 80)
    print("数据质量过滤统计")
    print("=" * 80)
    print(f"原始样本数: {stats['total']}")
    print(f"\n被过滤的样本:")
    print(f"  - 无评分: {stats['no_score']} ({stats['no_score']/stats['total']*100:.1f}%)")
    print(f"  - 反向样本 (rejected得分更高): {stats['reverse']} ({stats['reverse']/stats['total']*100:.1f}%)")
    print(f"  - 评分差异<{min_score_diff}分: {stats['low_score_diff']} ({stats['low_score_diff']/stats['total']*100:.1f}%)")
    print(f"  - 长度差异<{min_length_diff}字符: {stats['low_length_diff']} ({stats['low_length_diff']/stats['total']*100:.1f}%)")
    print(f"\n[PASS] 通过过滤的样本: {stats['passed']} ({stats['passed']/stats['total']*100:.1f}%)")
    print(f"\n已保存到: {output_file}")
    
    # 分析过滤后的数据质量
    if filtered_samples:
        score_diffs = [calculate_score_diff(s) for s in filtered_samples]
        length_diffs = [calculate_length_diff(s) for s in filtered_samples]
        
        print("\n" + "=" * 80)
        print("过滤后数据质量分析")
        print("=" * 80)
        print(f"评分差异:")
        print(f"  - 平均: {sum(score_diffs)/len(score_diffs):.2f}")
        print(f"  - 最小: {min(score_diffs)}")
        print(f"  - 最大: {max(score_diffs)}")
        print(f"  - 中位数: {sorted(score_diffs)[len(score_diffs)//2]}")
        
        print(f"\n长度差异:")
        print(f"  - 平均: {sum(length_diffs)/len(length_diffs):.1f} 字符")
        print(f"  - 最小: {min(length_diffs)} 字符")
        print(f"  - 最大: {max(length_diffs)} 字符")
        
        # 评分差异分布
        diff_ranges = {
            "差异<=5": sum(1 for d in score_diffs if d <= 5),
            "差异6-8": sum(1 for d in score_diffs if 6 <= d <= 8),
            "差异9-12": sum(1 for d in score_diffs if 9 <= d <= 12),
            "差异13-15": sum(1 for d in score_diffs if 13 <= d <= 15),
            "差异>=16": sum(1 for d in score_diffs if d >= 16),
        }
        print(f"\n评分差异分布:")
        for label, count in diff_ranges.items():
            print(f"  - {label}: {count} ({count/len(score_diffs)*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="过滤低质量数据样本")
    parser.add_argument("--input", type=str, default="./reward_model/data/dataset/train.jsonl",
                        help="输入文件路径")
    parser.add_argument("--output", type=str, default="./reward_model/data/dataset/train_filtered.jsonl",
                        help="输出文件路径")
    parser.add_argument("--min-score-diff", type=int, default=5,
                        help="最小总分差异阈值（默认5分，基于0-9分x3维度）")
    parser.add_argument("--min-length-diff", type=int, default=50,
                        help="最小长度差异阈值（默认50字符）")
    parser.add_argument("--keep-reverse", action="store_true",
                        help="保留反向样本（不推荐）")
    parser.add_argument("--quiet", action="store_true",
                        help="静默模式，不打印详细信息")
    
    args = parser.parse_args()
    
    filter_dataset(
        input_file=args.input,
        output_file=args.output,
        min_score_diff=args.min_score_diff,
        min_length_diff=args.min_length_diff,
        remove_reverse=not args.keep_reverse,
        verbose=not args.quiet
    )

