import os
import json
import random
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from pydantic import BaseModel
from multiprocessing import Pool, cpu_count
from functools import partial

load_dotenv()

# qwen3-max 支持 128K tokens (约250K字符)
# 使用100K作为安全限制，留有足够余量
MAX_INPUT_LENGTH = 100000


class ModelConfig(BaseModel):
    model: str
    base_url: str
    api_key: str


class ScoresDimension(BaseModel):
    """单个响应的三维度评分（10档评分制）"""

    depth: int  # 0-9分
    professionalism: int  # 0-9分
    accuracy: int  # 0-9分


class ScoresData(BaseModel):
    """完整的评分数据"""

    chosen: ScoresDimension
    rejected: ScoresDimension
    reasoning: Dict[str, str]  # 每个维度的对比理由
    overall_assessment: str


class ComparisonPair(BaseModel):
    """对比对数据结构"""

    prompt: (
        str  # 完整的用户输入（角色定义 + system_prompt + analysis_framework + 数据）
    )
    chosen: str  # 黄金响应
    rejected: str  # 缺陷响应
    metadata: Dict[str, Any]  # 元数据：行业、模型、质量维度等
    scores: Optional[ScoresData] = None  # AI裁判评分（可选，打分后添加）


def generate_quality_prompt_template(field: str, quality_type: str = "low") -> Dict[str, str]:
    """
    生成降级提示词模板（用于将黄金响应降级为缺陷响应）
    
    Args:
        field: 行业名称
        quality_type: 固定为 "low"（只用于降级）
    
    Returns:
        包含 name 和 template 的字典
    """
    if quality_type != "low":
        raise ValueError("该函数仅支持 quality_type='low' (降级模板)")
    
    # Degradation templates that will transform gold standard into defective version
    templates = [
        {
            "name": "浅化深度",
            "template": """请将以下财务分析报告改写得更加浅显和表面化，必须显著降低分析质量：

**强制改写要求**（必须严格执行）：
1. **完全移除**所有的根本原因分析、深层归因和业务洞察，只保留最表面的数据描述
2. **删除**所有业务层面的解释，如"由于原材料价格上涨"等归因分析，改为简单的"XX指标变化了X%"
3. **移除**所有前瞻性预测、趋势判断和管理建议，只保留当前数据的基本描述
4. **简化**所有分析段落，将深入分析改为"数据显示XX"、"XX指标为XX"等简单陈述
5. **保持**报告的基本结构和数据表格，但大幅降低分析深度和洞察力

**重要**：改写后的报告应该明显缺乏深度分析，仅停留在数据表面，让读者感觉分析不够深入。

原始报告：
""",
        },
        {
            "name": "简化计算",
            "template": """请将以下财务分析报告中的计算方法简化，引入明显的计算错误和不准确：

**强制改写要求**（必须严格执行）：
1. **引入计算错误**：对于复杂的财务指标，使用错误或过于简化的计算方法
2. **忽略特殊情况**：完全不处理负值、零值等特殊情况，使用简单的统一公式导致错误
3. **降低精度**：将精确的小数计算改为整数或1位小数，丢失精度
4. **跳过异常**：对于计算异常的数值，直接跳过不处理或使用明显错误的近似值
5. **错误公式**：使用错误的财务指标计算公式（如ROE、周转率等）

**重要**：改写后的报告应该包含明显的计算错误，数值准确性与原报告有明显差距。

原始报告：
""",
        },
        {
            "name": "泛化通用",
            "template": """请将以下财务分析报告改写得更加通用化，显著降低行业专业性：

**强制改写要求**（必须严格执行）：
1. **完全移除**所有行业特定的专业术语（如"产能利用率"、"供应链管理"等），替换为通用词汇
2. **删除**所有行业标杆对比、行业平均值对比和行业特色指标分析
3. **替换**所有行业特色的业务洞察为通用的财务分析套话，如"需要关注"、"有所改善"等
4. **移除**行业特定的分析框架和视角，只保留最基础的财务指标罗列
5. **通用化**所有表述，让报告看起来不针对任何特定行业

**重要**：改写后的报告应该明显缺乏行业专业性，看不出是针对哪个行业的分析。

原始报告：
""",
        },
        {
            "name": "模糊精确",
            "template": """请将以下财务分析报告改写得更加模糊和不够精确，显著降低数据准确性：

**强制改写要求**（必须严格执行）：
1. **大幅模糊化数值**：将具体的数值和百分比替换为"基本符合预期"、"略有增长"、"有所改善"等模糊表述，删除精确数字
2. **删除数据来源**：完全移除数据来源说明、可追溯性标注和计算过程
3. **模糊化结论**：将明确的结论改为含糊的、有保留的表述，如"可能"、"似乎"、"大概"等
4. **跳过异常处理**：对数据异常和特殊情况不做任何详细解释，简单带过或完全忽略
5. **降低精度**：将所有精确的小数、百分比改为模糊的定性描述

**重要**：改写后的报告应该明显缺乏精确数据，大量使用模糊表述，无法进行精确的数据分析。

原始报告：
""",
        },
        {
            "name": "片面覆盖",
            "template": """请将以下财务分析报告大幅简化，使其覆盖面非常片面，明显不完整：

**强制改写要求**（必须严格执行）：
1. **大幅删除内容**：只保留最核心的1-2个财务维度分析（如只保留盈利分析），其他维度（偿债、营运、现金流等）直接删除或仅一句话带过
2. **删除多维度分析**：完全移除多角度的立体分析（时间、空间、结构维度），只保留最简单的单维度对比
3. **跳过缺失数据**：对数据缺失的部分直接跳过，不做任何说明或补充
4. **简化表格**：将详细的表格和多维度分析简化为概括性的一句话描述，删除具体数据
5. **删除关键章节**：可以删除"管理建议"、"问题发现"等重要章节，或仅保留标题

**重要**：改写后的报告应该明显不完整，覆盖面很窄，读者会感觉分析不够全面。

原始报告：
""",
        },
    ]
    return random.choice(templates)




def build_complete_prompt(
    system_prompt: str,
    analysis_framework: str,
    data: str,
) -> str:
    """构建完整的prompt（无需截断，qwen3-max支持长上下文）"""
    return f"""你是一个专业的财务分析师

指引

{system_prompt}

分析框架

{analysis_framework}

参考数据

{data}
"""


def generate_gold_response(
    client: OpenAI,
    model: str,
    field: str,
    system_prompt: str,
    analysis_framework: str,
    sample_data: str,
) -> tuple[str, str, Dict[str, Any]]:
    """
    生成黄金标准响应

    Returns:
        (complete_prompt, response_text, metadata)
    """
    # 构建完整的输入prompt
    complete_prompt = build_complete_prompt(
        system_prompt=system_prompt,
        analysis_framework=analysis_framework,
        data=sample_data,
    )

    # 添加明确的优质输出指示
    instruction = """
请严格按照上述指引和分析框架生成高质量的财务分析报告。

**质量要求**：
1. 数据计算必须精确，正确处理特殊情况
2. 分析深度充分，进行多层次归因
3. 体现行业专业性，使用专业术语
4. 全面覆盖分析框架中的所有要点
"""

    full_prompt = complete_prompt + instruction

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.3,  # Lower temperature for gold standard
    )

    metadata = {
        "model": model,
        "field": field,
        "temperature": 0.3,
        "type": "gold",
    }

    return complete_prompt, response.choices[0].message.content, metadata


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
    degradation_template = generate_quality_prompt_template(field)

    template_text = degradation_template["template"]
    full_prompt = f"""{template_text}

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


def _generate_single_pair(args):
    """单个对比对生成任务（用于并发）"""
    (
        model_config,
        field,
        system_prompt,
        analysis_framework,
        sample_data_template,
        data_sample_dir,
    ) = args

    try:
        client = OpenAI(
            api_key=model_config["api_key"], base_url=model_config["base_url"]
        )
        model = model_config["model"]
        
        # 按行业加载数据（只加载当前行业的数据）
        if data_sample_dir and os.path.exists(data_sample_dir):
            data_dict = load_sample_data(data_sample_dir, field=field)
            # 替换模板变量
            sample_data = replace_data_template(sample_data_template, data_dict)
        else:
            sample_data = sample_data_template

        # Generate gold standard
        complete_prompt, gold_response, gold_metadata = generate_gold_response(
            client=client,
            model=model,
            field=field,
            system_prompt=system_prompt,
            analysis_framework=analysis_framework,
            sample_data=sample_data,
        )

        # Generate defect through controlled degradation
        defect_response, defect_metadata = generate_defect_response(
            client, model, gold_response, field
        )

        # Create comparison pair
        pair = ComparisonPair(
            prompt=complete_prompt,
            chosen=gold_response,
            rejected=defect_response,
            metadata={
                "field": field,
                "model": model,
                "gold_metadata": gold_metadata,
                "defect_metadata": defect_metadata,
            },
        )

        return pair.model_dump()
    except Exception as e:
        # Return error info as dict to avoid serialization issues
        return {"error": str(e), "field": field}




def load_frameworks(framework_dir: str) -> List[str]:
    """加载所有分析框架"""
    frameworks = []
    if not os.path.exists(framework_dir):
        raise ValueError(f"Framework directory not found: {framework_dir}")

    for file in os.listdir(framework_dir):
        if file.endswith(".md"):
            file_path = os.path.join(framework_dir, file)
            with open(file_path, "r", encoding="utf-8") as f:
                frameworks.append(f.read())

    if not frameworks:
        raise ValueError(f"No framework files found in {framework_dir}")

    return frameworks


def load_system_prompts(system_prompt_dir: str) -> List[str]:
    """加载所有系统提示词"""
    system_prompts = []
    if not os.path.exists(system_prompt_dir):
        raise ValueError(f"System prompt directory not found: {system_prompt_dir}")

    for file in os.listdir(system_prompt_dir):
        if file.endswith(".md") and file.startswith("positive"):
            file_path = os.path.join(system_prompt_dir, file)
            with open(file_path, "r", encoding="utf-8") as f:
                system_prompts.append(f.read())

    if not system_prompts:
        raise ValueError(f"No system prompt files found in {system_prompt_dir}")

    return system_prompts


def extract_industry_indicators(content: str, field: str) -> str:
    """从 industry_indicators.md 中提取特定行业的数据"""
    # 行业名称映射（匹配文件中的章节标题）
    industry_mapping = {
        "制造业": "一、制造业",
        "服务业": "二、服务业",
        "金融业": "三、金融业",
        "房地产": "四、房地产",
        "科技业": "五、科技业",
    }
    
    target_section = industry_mapping.get(field)
    if not target_section:
        return ""
    
    lines = content.split('\n')
    start_idx = None
    end_idx = None
    
    # 查找目标行业的起始位置（匹配 "## 一、制造业行业指标" 这样的格式）
    for i, line in enumerate(lines):
        if target_section in line and line.startswith('##'):
            start_idx = i
            break
    
    if start_idx is None:
        return ""
    
    # 查找下一个行业的起始位置（查找下一个以 "##" 开头且包含 "行业指标" 的行）
    for i in range(start_idx + 1, len(lines)):
        if lines[i].startswith('##') and '行业指标' in lines[i]:
            # 确保不是当前行业
            if target_section not in lines[i]:
                end_idx = i
                break
    
    # 如果没有找到下一个行业章节，查找其他主要章节（"## 六" 或 "## 七"）
    if end_idx is None:
        for i in range(start_idx + 1, len(lines)):
            if lines[i].startswith('## 六') or lines[i].startswith('## 七'):
                end_idx = i
                break
    
    # 如果还是没找到，提取到文件结尾
    if end_idx is None:
        end_idx = len(lines)
    
    # 提取行业数据
    industry_data = '\n'.join(lines[start_idx:end_idx])
    
    # 添加头部说明
    header = f"# {field}行业特色指标 & 均值数据\n\n**数据期间**: 2025年3月\n\n---\n\n"
    
    return header + industry_data


def load_sample_data(data_sample_dir: str, field: str = None) -> Dict[str, str]:
    """加载样例数据文件
    
    Args:
        data_sample_dir: 数据文件目录
        field: 行业名称，如果提供则只加载该行业的指标数据
    """
    data_files = {
        "profit_analysis_data": "profit_analysis_data.md",
        "dimension_analysis_data": "dimension_analysis_data.md",
        "industry_indicators": "industry_indicators.md",
        "budget_data": "budget_data.md",
    }
    
    loaded_data = {}
    for key, filename in data_files.items():
        file_path = os.path.join(data_sample_dir, filename)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            # 如果是行业指标且提供了行业参数，则提取特定行业的数据
            if key == "industry_indicators" and field:
                loaded_data[key] = extract_industry_indicators(content, field)
            else:
                loaded_data[key] = content
        else:
            print(f"警告: 数据文件 {file_path} 不存在，将使用空字符串")
            loaded_data[key] = ""
    
    return loaded_data


def replace_data_template(template: str, data_dict: Dict[str, str]) -> str:
    """替换数据模板中的占位符"""
    result = template
    for key, value in data_dict.items():
        placeholder = f"{{{{{key}}}}}"
        result = result.replace(placeholder, value)
    return result


def generate_comparison_dataset(
    fields: List[str],
    model_configs: List[ModelConfig],
    framework_dir: str,
    system_prompt_dir: str,
    sample_data: str,
    n_pairs_per_field: int,
    output_dir: str,
    data_sample_dir: str = None,
    use_parallel: bool = True,
    max_workers: int = None,
):
    """
    生成对比对数据集

    Args:
        fields: 行业列表
        model_configs: 模型配置列表
        framework_dir: 分析框架目录
        system_prompt_dir: 系统提示词目录
        sample_data: 样例数据描述（或实际数据，支持模板变量）
        n_pairs_per_field: 每个行业生成的对比对数量
        output_dir: 输出目录
        data_sample_dir: 样例数据目录（用于加载实际数据文件）
        use_parallel: 是否使用并发（默认True）
        max_workers: 最大并发数（默认为CPU核心数）
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "comparison_pairs.jsonl")

    # Load frameworks and system prompts
    print("加载分析框架和系统提示词...")
    frameworks = load_frameworks(framework_dir)
    system_prompts = load_system_prompts(system_prompt_dir)

    print(f"已加载 {len(frameworks)} 个分析框架")
    print(f"已加载 {len(system_prompts)} 个系统提示词")
    
    # 注意：这里不再统一加载数据，而是在每个任务中按行业加载

    total_pairs = len(fields) * n_pairs_per_field

    # 准备所有任务参数
    tasks = []
    for field in fields:
        for i in range(n_pairs_per_field):
            # Randomly select framework, system_prompt and model
            framework = random.choice(frameworks)
            system_prompt = random.choice(system_prompts)
            model_config = random.choice(model_configs)

            model_config_dict = {
                "model": model_config.model,
                "api_key": model_config.api_key,
                "base_url": model_config.base_url,
            }

            tasks.append(
                (
                    model_config_dict,
                    field,
                    system_prompt,
                    framework,
                    sample_data,  # 这是模板，不是实际数据
                    data_sample_dir,  # 传递数据目录，让每个任务按行业加载
                )
            )

    comparison_pairs = []

    # 并发执行
    if use_parallel and total_pairs > 1:
        # 生成阶段可以使用较高并发数（未触发速率限制）
        workers = max_workers or min(18, total_pairs)
        print(f"使用 {workers} 个并发进程生成对比对...")

        with Pool(processes=workers) as pool:
            with tqdm(total=total_pairs, desc="生成对比对") as pbar:
                for result in pool.imap_unordered(_generate_single_pair, tasks):
                    if result is not None:
                        if "error" in result:
                            print(
                                f"\n生成对比对时出错 ({result.get('field', 'unknown')}): {result['error']}"
                            )
                        else:
                            comparison_pairs.append(result)
                    pbar.update(1)
    else:
        # 串行执行（用于调试）
        print("串行模式生成对比对...")
        with tqdm(total=total_pairs, desc="生成对比对") as pbar:
            for task in tasks:
                result = _generate_single_pair(task)
                if result is not None:
                    comparison_pairs.append(result)
                pbar.update(1)

    # Save to JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for pair in comparison_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n成功生成 {len(comparison_pairs)} 个对比对，保存至 {output_file}")
    return output_file


def _score_single_pair(args):
    """单个对比对打分任务（用于并发）"""
    (pair, judge_config, judge_model) = args

    try:
        time.sleep(random.uniform(0.3, 0.5))
        
        judge_client = OpenAI(
            api_key=judge_config["api_key"], base_url=judge_config["base_url"]
        )
        # Create judge prompt for multi-dimensional scoring
        # 基础模板（不含动态内容）
        base_template = """你是一位严格且资深的财务分析专家，负责客观评估财务分析报告的质量。请严格、公正地评分，确保能清晰区分报告质量的差异。

请对以下两份报告在三个核心维度上进行严格评分。每个维度使用0-9分的10档评分制：

**评分标准**：
- **9分（卓越）**：该维度表现完美，远超高质量标准
- **8分（优秀+）**：该维度表现优秀，完全符合高质量标准
- **7分（优秀）**：该维度表现很好，基本符合高质量标准
- **6分（良好+）**：该维度表现良好，有小的提升空间
- **5分（良好）**：该维度表现中上，符合基本标准
- **4分（中等+）**：该维度表现一般偏上，有明显不足
- **3分（中等）**：该维度表现一般，存在较多不足
- **2分（较差）**：该维度表现较差，有严重问题
- **1分（很差）**：该维度表现很差，基本不可用
- **0分（极差）**：该维度完全不合格，毫无价值

**评分维度定义（请严格对照）**：

1. **分析深度 (depth)** [0-9分]
   - 9分：多层次深度归因，完整的业务逻辑链，根本原因识别精准，前瞻性洞察深刻，指标关联分析完整
   - 8分：深层次归因分析，深入业务层面，识别根本原因，有前瞻性洞察，能建立指标间逻辑关联
   - 7分：较好的归因分析，能深入业务层面，识别主要原因，部分前瞻性洞察
   - 6分：有归因分析，能识别部分原因，分析有一定深度但不够系统
   - 5分：有一定的原因分析，能识别部分关联关系，但缺乏深层次洞察
   - 4分：简单的原因分析，较为表面，缺乏系统性归因
   - 3分：基本的现象描述为主，分析较为表面，缺乏归因分析
   - 2分：主要是数据罗列，极少分析，仅有简单描述
   - 1分：仅罗列数据，几乎没有分析
   - 0分：完全没有分析，纯数据堆砌

2. **专业度 (professionalism)** [0-9分]
   - 9分：深刻理解行业本质，专业术语使用精准到位，多维度标杆对比，体现顶级行业洞察
   - 8分：深入理解行业特征，准确使用专业术语，有行业标杆对比，体现深刻行业洞察
   - 7分：很好地理解行业特点，专业术语使用准确，有部分标杆对比
   - 6分：理解行业基本特征，专业术语使用较准确，有行业针对性
   - 5分：了解行业基本特点，术语使用基本准确，但行业特色不够突出
   - 4分：部分体现行业特点，术语使用一般，行业针对性较弱
   - 3分：主要是通用财务分析，缺乏明显行业针对性，使用通用术语
   - 2分：术语使用不当，专业性较差，几乎看不出行业特征
   - 1分：缺乏基本专业性，术语错误较多
   - 0分：完全外行，错误百出，毫无专业性

3. **数值计算准确性 (accuracy)** [0-9分]
   - 9分：所有计算完美精确，特殊情况处理完美，数据完全可追溯，格式非常规范
   - 8分：所有计算精确无误，正确处理特殊情况（负值、零值等），数据可追溯，格式规范
   - 7分：主要计算精确，特殊情况处理正确，数据基本可追溯
   - 6分：大部分计算正确，有极个别小误差，基本符合规范
   - 5分：主要计算正确，个别小误差，基本符合规范
   - 4分：部分计算正确，存在一些误差，影响较小
   - 3分：存在明显计算错误或精度问题，影响部分结论
   - 2分：较多计算错误，明显影响结论的可信度
   - 1分：多处严重计算错误，严重影响结论
   - 0分：计算完全错误或缺失，数据完全不可信

**评分原则**：
1. **严格区分**：充分利用10档评分，两份报告必须在三个维度上都有明显差异（理想情况下总分差异应≥6分）
2. **客观公正**：完全基于报告实际内容评分，不受"黄金"或"缺陷"标签影响
3. **细粒度评估**：仔细对比细节差异，使用完整的0-9分区间，不要集中在某几个分数上
4. **差异放大**：如果发现两份报告差异不明显，请更严格地扣rejected的分，确保能区分质量差异

**用户需求**：
{user_prompt}

**报告A（黄金响应）**：
{chosen_report}

**报告B（缺陷响应）**：
{rejected_report}

请以JSON格式输出评分结果：
{{
    "chosen_scores": {{
        "depth": <0-9的整数>,
        "professionalism": <0-9的整数>,
        "accuracy": <0-9的整数>
    }},
    "rejected_scores": {{
        "depth": <0-9的整数>,
        "professionalism": <0-9的整数>,
        "accuracy": <0-9的整数>
    }},
    "reasoning": {{
        "depth": "<详细对比两份报告在分析深度上的具体差异，解释为什么chosen得X分、rejected得Y分>",
        "professionalism": "<详细对比两份报告在专业度上的具体差异，解释为什么chosen得X分、rejected得Y分>",
        "accuracy": "<详细对比两份报告在数值准确性上的具体差异，解释为什么chosen得X分、rejected得Y分>"
    }},
    "overall_assessment": "<总体评价，明确说明哪份报告更优，总分差异是多少，以及主要原因>"
}}

**重要提醒**：
1. 分数必须是0-9之间的整数
2. 确保两份报告在各维度上有明显差异（总分差异通常应≥6分）
3. 充分利用10档评分的细粒度，不要只用几个固定分数
4. 详细说明评分的具体依据，解释每个分数的理由
"""

        
        user_prompt = pair["prompt"]
        chosen_report = pair["chosen"]
        rejected_report = pair["rejected"]
        
        judge_prompt = base_template.format(
            user_prompt=user_prompt,
            chosen_report=chosen_report,
            rejected_report=rejected_report
        )

        response = judge_client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        score_result = json.loads(response.choices[0].message.content)

        # Validate scores are in 0-9 range (10档评分制)
        for report_type in ["chosen_scores", "rejected_scores"]:
            for dim in ["depth", "professionalism", "accuracy"]:
                score = score_result[report_type][dim]
                if not isinstance(score, int) or not (0 <= score <= 9):
                    print(f"\n警告: {report_type}.{dim} 分数异常: {score}，使用默认值5")
                    score_result[report_type][dim] = 5

        # 构建符合ScoresData模型的数据结构
        scores_data = {
            "chosen": score_result["chosen_scores"],
            "rejected": score_result["rejected_scores"],
            "reasoning": score_result["reasoning"],
            "overall_assessment": score_result.get("overall_assessment", ""),
        }

        # Add scores to pair
        pair["scores"] = scores_data
        return pair

    except Exception as e:
        # Return error info to avoid serialization issues
        pair["score_error"] = str(e)
        return pair


def add_multidim_scores(
    input_file: str,
    output_file: str,
    judge_client: OpenAI,
    judge_model: str,
    use_parallel: bool = True,
    max_workers: int = None,
):
    """
    使用AI裁判为报告打多维度分数（10档评分制：0-9分）

    三个核心维度：
    1. 分析深度 (depth): 0-9分
    2. 专业度 (professionalism): 0-9分
    3. 数值计算准确性 (accuracy): 0-9分

    评分标准（10档）：
    - 9分：卓越 - 8分：优秀+ - 7分：优秀
    - 6分：良好+ - 5分：良好 - 4分：中等+
    - 3分：中等 - 2分：较差 - 1分：很差 - 0分：极差
    """
    pairs = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))

    judge_config = {
        "api_key": judge_client.api_key,
        "base_url": judge_client.base_url,
    }

    # 准备所有任务参数
    tasks = [(pair, judge_config, judge_model) for pair in pairs]

    scored_pairs = []

    # 并发执行
    if use_parallel and len(pairs) > 1:
        workers = max_workers or min(4, len(pairs))
        print(f"使用 {workers} 个并发进程进行打分（已限制并发数避免速率限制）...")

        with Pool(processes=workers) as pool:
            with tqdm(total=len(pairs), desc="AI裁判多维度打分") as pbar:
                for result in pool.imap_unordered(_score_single_pair, tasks):
                    if "score_error" in result:
                        print(f"\n打分时出错: {result['score_error']}")
                    scored_pairs.append(result)
                    pbar.update(1)
    else:
        # 串行执行（用于调试）
        print("串行模式进行打分...")
        with tqdm(total=len(pairs), desc="AI裁判多维度打分") as pbar:
            for task in tasks:
                result = _score_single_pair(task)
                scored_pairs.append(result)
                pbar.update(1)

    # Save scored pairs
    with open(output_file, "w", encoding="utf-8") as f:
        for pair in scored_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"\n完成打分，保存至 {output_file}")

if __name__ == "__main__":
    # Sample data description (you can replace with actual data)
    SAMPLE_DATA = """
    利润分析数据:
    {{profit_analysis_data}}

    维度穿透分析数据:
    {{dimension_analysis_data}}

    行业特色指标 & 均值:
    {{industry_indicators}}

    预算数据:
    {{budget_data}}
    """
    # 配置
    fields = ["制造业", "服务业", "金融业", "房地产", "科技业"]
    framework_dir = "./reward_model/data/analysis_framework/"
    system_prompt_dir = "./reward_model/data/system_prompt/"
    output_dir = "./reward_model/data/dataset/"
    data_sample_dir = "./reward_model/data/data_sample/"

    model_configs = [
        ModelConfig(
            model="qwen-long",
            base_url=os.getenv("QWEN_URL"),
            api_key=os.getenv("QWEN_KEY"),
        ),
        ModelConfig(
            model="qwen3-max",
            base_url=os.getenv("QWEN_URL"),
            api_key=os.getenv("QWEN_KEY"),
        ),
    ]

    # Step 1: Generate comparison pairs
    print("=" * 70)
    print("第一步：生成对比对数据集")
    print("=" * 70)

    output_file = generate_comparison_dataset(
        fields=fields,
        model_configs=model_configs,
        framework_dir=framework_dir,
        system_prompt_dir=system_prompt_dir,
        sample_data=SAMPLE_DATA,
        n_pairs_per_field=3,  # 每个行业生成10个对比对
        output_dir=output_dir,
        data_sample_dir=data_sample_dir,  # 传入数据样例目录
        use_parallel=True,  # 启用并发
        max_workers=None,  # 自动选择worker数量
    )

    # Step 2: Add multi-dimensional scores
    print("\n" + "=" * 70)
    print("第二步：AI裁判多维度打分")
    print("=" * 70)

    judge_client = OpenAI(api_key=os.getenv("QWEN_KEY"), base_url=os.getenv("QWEN_URL"))

    scored_output_file = os.path.join(output_dir, "comparison_pairs_scored.jsonl")
    add_multidim_scores(
        input_file=output_file,
        output_file=scored_output_file,
        judge_client=judge_client,
        judge_model="qwen3-max",  # Use strongest model as judge
        use_parallel=True,  # 启用并发
        max_workers=None,  # 自动选择worker数量
    )

    print("\n" + "=" * 70)
    print("数据生成完成！")
    print("=" * 70)
    print(f"未打分数据: {output_file}")
    print(f"已打分数据: {scored_output_file}")
