"""
验证多维度评分对比对数据集的质量
"""

import json
from collections import Counter
from typing import List, Dict, Any
import statistics


def load_pairs(file_path: str) -> List[Dict[str, Any]]:
    """加载对比对数据"""
    pairs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def validate_multidim_scores(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """验证多维度分数差异"""
    pairs_with_scores = [p for p in pairs if "scores" in p and "chosen" in p["scores"]]

    if not pairs_with_scores:
        return {
            "error": "没有找到打分数据",
            "total_pairs": len(pairs),
            "scored_pairs": 0,
        }

    # Get dimensions from first pair
    dimensions = list(pairs_with_scores[0]["scores"]["chosen"].keys())
    
    # Calculate per-dimension differences
    dim_diffs = {dim: [] for dim in dimensions}
    avg_diffs = []

    for p in pairs_with_scores:
        for dim in dimensions:
            chosen_score = p["scores"]["chosen"][dim]
            rejected_score = p["scores"]["rejected"][dim]
            dim_diffs[dim].append(chosen_score - rejected_score)

        # Calculate average difference across all dimensions
        chosen_avg = sum(p["scores"]["chosen"].values()) / len(p["scores"]["chosen"])
        rejected_avg = sum(p["scores"]["rejected"].values()) / len(
            p["scores"]["rejected"]
        )
        avg_diffs.append(chosen_avg - rejected_avg)

    positive_diffs = [d for d in avg_diffs if d > 0]
    negative_diffs = [d for d in avg_diffs if d < 0]
    zero_diffs = [d for d in avg_diffs if d == 0]

    result = {
        "total_pairs": len(pairs),
        "scored_pairs": len(pairs_with_scores),
        "coverage": f"{len(pairs_with_scores) / len(pairs) * 100:.1f}%",
        "average_diff": statistics.mean(avg_diffs),
        "median_diff": statistics.median(avg_diffs),
        "min_diff": min(avg_diffs),
        "max_diff": max(avg_diffs),
        "std_dev": statistics.stdev(avg_diffs) if len(avg_diffs) > 1 else 0,
        "positive_diffs": len(positive_diffs),
        "negative_diffs": len(negative_diffs),
        "zero_diffs": len(zero_diffs),
        "quality_ratio": f"{len(positive_diffs) / len(avg_diffs) * 100:.1f}%",
        "dimensions": {},
    }

    # Add per-dimension statistics
    for dim in dimensions:
        diffs = dim_diffs[dim]
        result["dimensions"][dim] = {
            "average_diff": statistics.mean(diffs),
            "median_diff": statistics.median(diffs),
            "min_diff": min(diffs),
            "max_diff": max(diffs),
            "positive_rate": f"{len([d for d in diffs if d > 0]) / len(diffs) * 100:.1f}%",
        }

    return result


def analyze_label_distribution(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分析标签分布"""
    pairs_with_scores = [p for p in pairs if "scores" in p and "chosen" in p["scores"]]

    if not pairs_with_scores:
        return {"error": "没有找到打分数据"}

    dimensions = list(pairs_with_scores[0]["scores"]["chosen"].keys())

    # Count labels
    chosen_counts = {dim: Counter() for dim in dimensions}
    rejected_counts = {dim: Counter() for dim in dimensions}

    for p in pairs_with_scores:
        for dim in dimensions:
            chosen_counts[dim][p["scores"]["chosen"][dim]] += 1
            rejected_counts[dim][p["scores"]["rejected"][dim]] += 1

    # Calculate distribution
    total = len(pairs_with_scores)
    result = {
        "total_pairs": total,
        "dimensions": {},
    }

    for dim in dimensions:
        result["dimensions"][dim] = {
            "chosen": {
                label: count / total for label, count in chosen_counts[dim].items()
            },
            "rejected": {
                label: count / total for label, count in rejected_counts[dim].items()
            },
        }

    return result


def find_problematic_pairs(
    pairs: List[Dict[str, Any]], min_avg_diff: float = 0.5
) -> List[Dict[str, Any]]:
    """找出有问题的对比对"""
    problematic = []

    for i, p in enumerate(pairs):
        if "scores" not in p:
            problematic.append({
                "index": i,
                "reason": "缺少评分",
                "metadata": p.get("metadata", {}),
            })
            continue

        if "chosen" not in p["scores"]:
            problematic.append({
                "index": i,
                "reason": "评分格式错误",
                "metadata": p.get("metadata", {}),
            })
            continue

        # Calculate average difference
        chosen_avg = sum(p["scores"]["chosen"].values()) / len(p["scores"]["chosen"])
        rejected_avg = sum(p["scores"]["rejected"].values()) / len(
            p["scores"]["rejected"]
        )
        avg_diff = chosen_avg - rejected_avg

        if avg_diff < min_avg_diff:
            problematic.append({
                "index": i,
                "reason": f"平均分差异过小 ({avg_diff:.2f} < {min_avg_diff})",
                "chosen_avg": chosen_avg,
                "rejected_avg": rejected_avg,
                "chosen_scores": p["scores"]["chosen"],
                "rejected_scores": p["scores"]["rejected"],
                "metadata": p.get("metadata", {}),
            })

    return problematic


def print_report(pairs: List[Dict[str, Any]]):
    """打印验证报告"""
    print("=" * 70)
    print("多维度对比对数据集验证报告")
    print("=" * 70)

    # 1. Multi-dimensional score differences
    print("\n📊 1. 多维度分数差异统计")
    print("-" * 70)
    score_stats = validate_multidim_scores(pairs)
    
    if "error" not in score_stats:
        print(f"总对比对数: {score_stats['total_pairs']}")
        print(f"已评分数: {score_stats['scored_pairs']} ({score_stats['coverage']})")
        print(f"\n整体平均分差: {score_stats['average_diff']:.2f}")
        print(f"整体中位分差: {score_stats['median_diff']:.2f}")
        print(f"分差范围: [{score_stats['min_diff']:.2f}, {score_stats['max_diff']:.2f}]")
        print(f"标准差: {score_stats['std_dev']:.2f}")
        print(f"\n分差分布:")
        print(f"  正差异 (chosen > rejected): {score_stats['positive_diffs']} ({score_stats['quality_ratio']})")
        print(f"  负差异 (chosen < rejected): {score_stats['negative_diffs']}")
        print(f"  零差异 (chosen = rejected): {score_stats['zero_diffs']}")

        # Per-dimension statistics
        dimension_names = {
            "depth": "分析深度",
            "professionalism": "专业度",
            "accuracy": "数值准确性",
        }

        print(f"\n各维度分差统计:")
        for dim, stats in score_stats["dimensions"].items():
            dim_name = dimension_names.get(dim, dim)
            print(f"\n  {dim_name} ({dim}):")
            print(f"    平均分差: {stats['average_diff']:.2f}")
            print(f"    中位分差: {stats['median_diff']:.2f}")
            print(f"    分差范围: [{stats['min_diff']}, {stats['max_diff']}]")
            print(f"    正差异率: {stats['positive_rate']}")

        # Quality assessment
        avg_diff = score_stats["average_diff"]
        quality_ratio = float(score_stats["quality_ratio"].rstrip("%"))

        print(f"\n✅ 质量评估:")
        if avg_diff >= 1.5 and quality_ratio >= 95:
            print("  🌟 优秀 - 对比对质量非常好，chosen显著优于rejected")
        elif avg_diff >= 1.0 and quality_ratio >= 90:
            print("  ✅ 良好 - 对比对质量符合预期")
        elif avg_diff >= 0.5 and quality_ratio >= 85:
            print("  ⚠️  一般 - 对比对质量可接受，但建议优化")
        else:
            print("  ❌ 较差 - 对比对质量需要改进")
    else:
        print(f"❌ {score_stats['error']}")

    # 2. Label distribution
    print("\n📈 2. 标签分布统计")
    print("-" * 70)
    dist_stats = analyze_label_distribution(pairs)

    if "error" not in dist_stats:
        dimension_names = {
            "depth": "分析深度",
            "professionalism": "专业度",
            "accuracy": "数值准确性",
        }

        for dim, stats in dist_stats["dimensions"].items():
            dim_name = dimension_names.get(dim, dim)
            print(f"\n{dim_name} ({dim}):")

            print("  Chosen样本分布:")
            for label in sorted(stats["chosen"].keys()):
                pct = stats["chosen"][label] * 100
                bar = "█" * int(pct / 2)
                print(f"    {label}分: {pct:5.1f}% {bar}")

            print("  Rejected样本分布:")
            for label in sorted(stats["rejected"].keys()):
                pct = stats["rejected"][label] * 100
                bar = "█" * int(pct / 2)
                print(f"    {label}分: {pct:5.1f}% {bar}")
    else:
        print(f"❌ {dist_stats['error']}")

    # 3. Problematic pairs
    print("\n⚠️  3. 问题对比对检测")
    print("-" * 70)
    problematic = find_problematic_pairs(pairs, min_avg_diff=0.5)

    if problematic:
        print(f"发现 {len(problematic)} 个需要关注的对比对:")
        for p in problematic[:5]:  # Only show first 5
            print(f"\n  索引 {p['index']}:")
            print(f"    原因: {p['reason']}")
            if "chosen_scores" in p:
                print(f"    Chosen分数: {p['chosen_scores']}")
                print(f"    Rejected分数: {p['rejected_scores']}")
        if len(problematic) > 5:
            print(f"\n  ... 还有 {len(problematic) - 5} 个问题对比对")

        print(f"\n  问题占比: {len(problematic) / len(pairs) * 100:.1f}%")
    else:
        print("✅ 所有对比对质量良好")

    # 4. Recommendations
    print("\n💡 4. 改进建议")
    print("-" * 70)
    recommendations = []

    if "error" not in score_stats:
        if score_stats["average_diff"] < 1.0:
            recommendations.append("• 整体分数差异偏小，建议增强降级提示词的强度")

        # Check per-dimension differences
        for dim, stats in score_stats["dimensions"].items():
            if stats["average_diff"] < 0.5:
                dim_name = dimension_names.get(dim, dim)
                recommendations.append(
                    f"• {dim_name}维度差异过小 ({stats['average_diff']:.2f})，建议针对性优化降级策略"
                )

        if float(score_stats["quality_ratio"].rstrip("%")) < 90:
            recommendations.append("• 存在较多负差异，建议检查黄金响应和降级逻辑")

        if score_stats["negative_diffs"] > 0:
            recommendations.append(
                f"• 发现 {score_stats['negative_diffs']} 个负差异对比对，建议人工检查或过滤"
            )

    if problematic:
        problem_ratio = len(problematic) / len(pairs)
        if problem_ratio > 0.1:
            recommendations.append(
                f"• 问题对比对占比 {problem_ratio * 100:.1f}%，建议过滤或重新生成"
            )

    # Check label distribution balance
    if "error" not in dist_stats:
        for dim, stats in dist_stats["dimensions"].items():
            if stats["chosen"]:
                max_pct = max(stats["chosen"].values())
                min_pct = min(stats["chosen"].values())
                if max_pct > min_pct * 3:
                    dim_name = dimension_names.get(dim, dim)
                    recommendations.append(
                        f"• {dim_name}维度标签分布不均衡，建议调整生成策略或使用类别权重"
                    )

    if recommendations:
        for rec in recommendations:
            print(rec)
    else:
        print("✅ 数据集质量良好，无需改进")

    print("\n" + "=" * 70)


def export_filtered_dataset(
    input_file: str,
    output_file: str,
    min_avg_diff: float = 0.5,
    require_positive_diff: bool = True,
):
    """导出过滤后的高质量数据集"""
    pairs = load_pairs(input_file)

    filtered = []
    for p in pairs:
        if "scores" not in p or "chosen" not in p["scores"]:
            continue

        # Calculate average difference
        chosen_avg = sum(p["scores"]["chosen"].values()) / len(p["scores"]["chosen"])
        rejected_avg = sum(p["scores"]["rejected"].values()) / len(
            p["scores"]["rejected"]
        )
        avg_diff = chosen_avg - rejected_avg

        if require_positive_diff and avg_diff <= 0:
            continue

        if avg_diff < min_avg_diff:
            continue

        filtered.append(p)

    with open(output_file, "w", encoding="utf-8") as f:
        for p in filtered:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n已导出 {len(filtered)}/{len(pairs)} 个高质量对比对到 {output_file}")
    print(f"过滤率: {(1 - len(filtered) / len(pairs)) * 100:.1f}%")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("用法: python validate_multidim_pairs.py <input_file> [output_file]")
        print("示例: python validate_multidim_pairs.py comparison_pairs_scored.jsonl")
        print(
            "      python validate_multidim_pairs.py comparison_pairs_scored.jsonl filtered_pairs.jsonl"
        )
        sys.exit(1)

    input_file = sys.argv[1]
    pairs = load_pairs(input_file)

    # Print validation report
    print_report(pairs)

    # Export filtered dataset if output file specified
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
        print("\n" + "=" * 70)
        print("导出过滤后的数据集")
        print("=" * 70)
        export_filtered_dataset(input_file, output_file, min_avg_diff=0.5)

