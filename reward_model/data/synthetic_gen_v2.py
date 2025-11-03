import os
import json
import random
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from pydantic import BaseModel

load_dotenv()


class ModelConfig(BaseModel):
    model: str
    base_url: str
    api_key: str


class ComparisonPair(BaseModel):
    """对比对数据结构"""

    prompt: str  # 用户输入（分析框架 + 数据要求）
    chosen: str  # 黄金响应
    rejected: str  # 缺陷响应
    metadata: Dict[str, Any]  # 元数据：行业、模型、质量维度等


def generate_quality_prompt_template(field: str, quality_type: str = "high") -> str:
    """生成质量导向的提示词模板"""

    if quality_type == "high":
        # High-quality prompt templates with different focuses
        templates = [
            {
                "name": "数据准确性导向",
                "template": f"""你是一位专业的{field}财务分析师，请生成一份高质量的财务分析报告。

**核心质量要求**：
1. **数值计算精确**：所有财务指标、比率、增长率的计算必须精确到小数点后2位，严格遵循会计准则
2. **特殊情况处理**：对负值、零值、扭亏为盈等特殊情况，使用专门的处理逻辑，禁止除零错误
3. **数据可追溯**：每个结论都要明确数据来源，确保可以追溯到原始财务报表

请按照给定的分析框架生成报告。""",
            },
            {
                "name": "深度分析导向",
                "template": f"""你是一位资深的{field}行业财务分析专家，请生成一份具有深度洞察的财务分析报告。

**核心质量要求**：
1. **多层次归因**：不止于现象描述，要深入业务层面进行根本原因分析
2. **关联分析**：识别财务指标之间的内在联系和因果关系
3. **前瞻性洞察**：结合行业特点和公司战略，提供前瞻性的分析和建议

请按照给定的分析框架生成报告。""",
            },
            {
                "name": "行业专业性导向",
                "template": f"""你是{field}领域的顶级财务分析师，请生成一份体现行业专业性的财务分析报告。

**核心质量要求**：
1. **行业特征理解**：深入理解{field}的行业特征、关键成功因素和业务逻辑
2. **专业术语使用**：准确使用{field}行业的专业术语和分析框架
3. **标杆对比**：与行业标杆企业进行对比分析，识别行业周期和竞争格局影响

请按照给定的分析框架生成报告。""",
            },
            {
                "name": "全面性导向",
                "template": f"""你是一位严谨的{field}财务分析师，请生成一份全面完整的财务分析报告。

**核心质量要求**：
1. **全维度覆盖**：必须覆盖盈利能力、营运能力、偿债能力、成长能力等全部维度
2. **立体分析**：从时间维度（趋势）、空间维度（对比）、结构维度（拆解）进行立体分析
3. **逻辑严密**：建立完整的分析链条，确保各部分之间相互呼应，结论有充分数据支撑

请按照给定的分析框架生成报告。""",
            },
        ]
        return random.choice(templates)

    else:  # low quality - for degradation instructions
        # Degradation templates that will transform gold standard into defective version
        templates = [
            {
                "name": "浅化深度",
                "template": """请将以下财务分析报告改写得更加浅显和表面化：

**改写要求**：
1. 移除所有的根本原因分析和深层归因，只保留现象描述
2. 将具体的业务洞察替换为通用的、表面的观察
3. 删除所有前瞻性的预测和建议，只保留基本的数据罗列
4. 保持报告结构和基本数据，但降低分析深度

原始报告：
""",
            },
            {
                "name": "简化计算",
                "template": """请将以下财务分析报告中的计算方法简化：

**改写要求**：
1. 对于复杂的财务指标计算，使用简化方法，不考虑特殊情况
2. 将处理负值、零值的专门逻辑替换为简单的统一处理
3. 可以适当降低数值精度（如从2位小数改为1位或整数）
4. 对于计算异常的地方，可以跳过或使用近似值

原始报告：
""",
            },
            {
                "name": "泛化通用",
                "template": """请将以下财务分析报告改写得更加通用化，降低行业针对性：

**改写要求**：
1. 移除所有行业特定的专业术语和分析框架
2. 删除与行业标杆企业的对比分析
3. 用通用的财务分析表述替换行业特色的洞察
4. 保持基本的财务数据，但使用标准化的分析模板

原始报告：
""",
            },
            {
                "name": "模糊精确",
                "template": """请将以下财务分析报告改写得更加模糊和不够精确：

**改写要求**：
1. 将具体的数值和百分比替换为"基本符合预期"、"略有增长"等模糊表述
2. 删除数据来源和可追溯性的说明
3. 将明确的结论改为含糊的、有保留的表述
4. 对数据异常和特殊情况不做详细解释，简单带过

原始报告：
""",
            },
            {
                "name": "片面覆盖",
                "template": """请将以下财务分析报告简化，使其覆盖面更加片面：

**改写要求**：
1. 只保留最核心的2-3个财务维度分析，其他维度简要提及或删除
2. 删除多角度的立体分析（时间、空间、结构维度），只保留最简单的对比
3. 对数据缺失的部分直接跳过，不做说明
4. 将详细的表格和多维度分析简化为概括性描述

原始报告：
""",
            },
        ]
        return random.choice(templates)


def generate_gold_response(
    client: OpenAI,
    model: str,
    field: str,
    analysis_framework: str,
    sample_data_description: str,
) -> tuple[str, Dict[str, Any]]:
    """
    生成黄金标准响应

    Returns:
        (response_text, metadata)
    """
    # Get high-quality prompt template
    quality_template = generate_quality_prompt_template(field, quality_type="high")

    full_prompt = f"""{quality_template['template']}

# 分析框架
{analysis_framework}

# 数据说明
{sample_data_description}

请严格按照上述要求生成高质量的财务分析报告。
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.3,  # Lower temperature for gold standard
    )

    metadata = {
        "quality_focus": quality_template["name"],
        "model": model,
        "field": field,
        "temperature": 0.3,
    }

    return response.choices[0].message.content, metadata


def generate_defect_response(
    client: OpenAI,
    model: str,
    gold_response: str,
    field: str,
) -> tuple[str, Dict[str, Any]]:
    """
    通过受控降级生成缺陷响应

    Returns:
        (response_text, metadata)
    """
    # Get degradation template
    degradation_template = generate_quality_prompt_template(
        field, quality_type="low"
    )

    full_prompt = f"""{degradation_template['template']}

{gold_response}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.5,  # Moderate temperature for controlled variation
    )

    metadata = {
        "degradation_type": degradation_template["name"],
        "model": model,
        "field": field,
        "temperature": 0.5,
    }

    return response.choices[0].message.content, metadata


def generate_comparison_pair(
    client: OpenAI,
    model: str,
    field: str,
    analysis_framework: str,
    sample_data_description: str,
) -> ComparisonPair:
    """
    生成一个完整的对比对

    流程：
    1. 生成黄金标准响应
    2. 对黄金响应进行受控降级，生成缺陷响应
    3. 返回对比对
    """
    # Step 1: Generate gold standard
    gold_response, gold_metadata = generate_gold_response(
        client, model, field, analysis_framework, sample_data_description
    )

    # Step 2: Generate defect through controlled degradation
    defect_response, defect_metadata = generate_defect_response(
        client, model, gold_response, field
    )

    # Step 3: Construct the user prompt (what the model saw)
    user_prompt = f"""请根据以下分析框架生成{field}的财务分析报告：

{analysis_framework}

{sample_data_description}
"""

    # Step 4: Create comparison pair
    pair = ComparisonPair(
        prompt=user_prompt,
        chosen=gold_response,
        rejected=defect_response,
        metadata={
            "field": field,
            "model": model,
            "gold_metadata": gold_metadata,
            "defect_metadata": defect_metadata,
        },
    )

    return pair


def load_frameworks(framework_dir: str) -> List[str]:
    """加载所有分析框架"""
    frameworks = []
    for file in os.listdir(framework_dir):
        if file.endswith(".md"):
            with open(os.path.join(framework_dir, file), "r", encoding="utf-8") as f:
                frameworks.append(f.read())
    return frameworks


def generate_comparison_dataset(
    fields: List[str],
    model_configs: List[ModelConfig],
    framework_dir: str,
    sample_data_description: str,
    n_pairs_per_field: int,
    output_file: str,
):
    """
    生成对比对数据集

    Args:
        fields: 行业列表
        model_configs: 模型配置列表
        framework_dir: 分析框架目录
        sample_data_description: 样例数据描述（或实际数据）
        n_pairs_per_field: 每个行业生成的对比对数量
        output_file: 输出文件路径
    """
    # Load frameworks
    frameworks = load_frameworks(framework_dir)
    if not frameworks:
        raise ValueError(f"No frameworks found in {framework_dir}")

    total_pairs = len(fields) * n_pairs_per_field
    comparison_pairs = []

    with tqdm(total=total_pairs, desc="生成对比对") as pbar:
        for field in fields:
            for _ in range(n_pairs_per_field):
                # Randomly select framework and model
                framework = random.choice(frameworks)
                model_config = random.choice(model_configs)

                # Create OpenAI client
                client = OpenAI(
                    api_key=model_config.api_key, base_url=model_config.base_url
                )

                try:
                    # Generate comparison pair
                    pair = generate_comparison_pair(
                        client=client,
                        model=model_config.model,
                        field=field,
                        analysis_framework=framework,
                        sample_data_description=sample_data_description,
                    )

                    comparison_pairs.append(pair.model_dump())

                except Exception as e:
                    print(f"\n生成对比对时出错: {e}")
                    continue

                pbar.update(1)

    # Save to JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for pair in comparison_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n成功生成 {len(comparison_pairs)} 个对比对，保存至 {output_file}")


def add_multidim_scores(
    input_file: str,
    output_file: str,
    judge_client: OpenAI,
    judge_model: str,
):
    """
    使用AI裁判为报告打多维度分数 (0-4档)
    
    三个核心维度：
    1. 分析深度 (depth): 0-4分
    2. 专业度 (professionalism): 0-4分  
    3. 数值计算准确性 (accuracy): 0-4分
    
    评分标准：
    - 4分：优秀
    - 3分：良好
    - 2分：中等
    - 1分：较差
    - 0分：很差
    """
    pairs = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))

    scored_pairs = []

    with tqdm(total=len(pairs), desc="AI裁判多维度打分") as pbar:
        for pair in pairs:
            try:
                # Create judge prompt for multi-dimensional scoring
                judge_prompt = f"""你是一位资深的财务分析专家，负责评估财务分析报告的质量。

请对以下两份报告在三个核心维度上进行评分。每个维度使用0-4分的5档评分制：
- **4分（优秀）**：该维度表现卓越，完全符合高质量标准
- **3分（良好）**：该维度表现较好，基本符合质量标准
- **2分（中等）**：该维度表现一般，存在明显不足
- **1分（较差）**：该维度表现较差，有严重问题
- **0分（很差）**：该维度表现很差，基本不可用

**评分维度定义**：

1. **分析深度 (depth)**
   - 4分：多层次归因分析，深入业务层面，识别根本原因，有前瞻性洞察
   - 3分：有一定的原因分析，能识别部分关联关系
   - 2分：基本的现象描述，分析较为表面
   - 1分：仅罗列数据，缺乏分析
   - 0分：没有任何分析，纯数据堆砌

2. **专业度 (professionalism)**
   - 4分：深入理解行业特征，准确使用专业术语，有标杆对比，体现行业洞察
   - 3分：了解行业基本特点，术语使用基本准确
   - 2分：通用财务分析，缺乏行业针对性
   - 1分：术语使用不当，缺乏专业性
   - 0分：完全外行，错误百出

3. **数值计算准确性 (accuracy)**
   - 4分：所有计算精确无误，正确处理特殊情况（负值、零值等），数据可追溯
   - 3分：主要计算正确，个别小误差
   - 2分：存在明显计算错误或精度问题
   - 1分：多处计算错误，严重影响结论
   - 0分：计算完全错误或缺失

**用户需求**：
{pair['prompt'][:500]}...

**报告A（黄金响应）**：
{pair['chosen'][:3000]}...

**报告B（缺陷响应）**：
{pair['rejected'][:3000]}...

请以JSON格式输出评分结果：
{{
    "chosen_scores": {{
        "depth": <0-4的整数>,
        "professionalism": <0-4的整数>,
        "accuracy": <0-4的整数>
    }},
    "rejected_scores": {{
        "depth": <0-4的整数>,
        "professionalism": <0-4的整数>,
        "accuracy": <0-4的整数>
    }},
    "reasoning": {{
        "depth": "<对比两份报告在分析深度上的差异>",
        "professionalism": "<对比两份报告在专业度上的差异>",
        "accuracy": "<对比两份报告在数值准确性上的差异>"
    }},
    "overall_assessment": "<总体评价，说明哪份报告更优及原因>"
}}

注意：
1. 分数必须是0-4之间的整数
2. 黄金响应通常应该在各维度上得分更高（除非降级失败）
3. 请根据实际内容客观评分，不要因为标注为"黄金"就自动给高分
"""

                response = judge_client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": judge_prompt}],
                    temperature=0.3,
                    response_format={"type": "json_object"},
                )

                score_result = json.loads(response.choices[0].message.content)

                # Validate scores are in 0-4 range
                for report_type in ["chosen_scores", "rejected_scores"]:
                    for dim in ["depth", "professionalism", "accuracy"]:
                        score = score_result[report_type][dim]
                        if not isinstance(score, int) or not (0 <= score <= 4):
                            print(f"\n警告: {report_type}.{dim} 分数异常: {score}，使用默认值2")
                            score_result[report_type][dim] = 2

                # Add scores to pair
                pair["scores"] = {
                    "chosen": score_result["chosen_scores"],
                    "rejected": score_result["rejected_scores"],
                    "reasoning": score_result["reasoning"],
                    "overall_assessment": score_result.get("overall_assessment", ""),
                    "judge_model": judge_model,
                }

                scored_pairs.append(pair)

            except Exception as e:
                print(f"\n打分时出错: {e}")
                import traceback
                traceback.print_exc()
                # Keep pair without scores
                scored_pairs.append(pair)

            pbar.update(1)

    # Save scored pairs
    with open(output_file, "w", encoding="utf-8") as f:
        for pair in scored_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n完成打分，保存至 {output_file}")


if __name__ == "__main__":
    # Sample data description (you can replace with actual data)
    SAMPLE_DATA = """
# 数据说明
本次分析基于某公司2024年3月的财务数据：
- 营业收入：2,380.79万元
- 净利润：588.22万元
- 毛利率：69.08%
- 总资产：5,280.45万元
- 净资产：2,892.17万元

请基于这些数据进行详细的财务分析。
"""

    # Step 1: Generate comparison pairs
    print("=" * 50)
    print("第一步：生成对比对数据集")
    print("=" * 50)

    generate_comparison_dataset(
        fields=["制造业", "服务业", "金融业", "房地产", "科技业"],
        model_configs=[
            ModelConfig(
                model="qwen-plus",
                base_url=os.getenv("QWEN_URL"),
                api_key=os.getenv("QWEN_KEY"),
            ),
            ModelConfig(
                model="qwen3-max",
                base_url=os.getenv("QWEN_URL"),
                api_key=os.getenv("QWEN_KEY"),
            ),
            ModelConfig(
                model="deepseek-chat",
                base_url=os.getenv("DS_URL"),
                api_key=os.getenv("DS_KEY"),
            ),
            ModelConfig(
                model="qwen-turbo",
                base_url=os.getenv("QWEN_URL"),
                api_key=os.getenv("QWEN_KEY"),
            )
        ],
        framework_dir="./reward_model/data/analysis_framework/",
        sample_data_description=SAMPLE_DATA,
        n_pairs_per_field=50,  # Each field generates 10 pairs
        output_file="./reward_model/data/comparison_pairs.jsonl",
    )

    # Step 2: Add multi-dimensional scores
    print("\n" + "=" * 50)
    print("第二步：AI裁判多维度打分")
    print("=" * 50)

    judge_client = OpenAI(
        api_key=os.getenv("QWEN_KEY"), base_url=os.getenv("QWEN_URL")
    )

    add_multidim_scores(
        input_file="./reward_model/data/comparison_pairs.jsonl",
        output_file="./reward_model/data/comparison_pairs_scored.jsonl",
        judge_client=judge_client,
        judge_model="qwen-max",  # Use strongest model as judge
    )

    print("\n" + "=" * 50)
    print("数据生成完成！")
    print("=" * 50)

