import os
import random
from re import S
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from pydantic import BaseModel

load_dotenv()


class ModelConfig(BaseModel):
    model: str
    base_url: str
    api_key: str


qwen_client = OpenAI(api_key=os.getenv("QWEN_KEY"), base_url=os.getenv("QWEN_URL"))


def gen_framework(
    analysis_framework: str,
    client: OpenAI,
    model: str,
    n_samples: int = 10,
    field: str = "",
    pbar: tqdm = None,
):
    # Define diverse framework perspectives for variation
    framework_perspectives = [
        {
            "angle": "战略导向",
            "focus": "从公司战略执行角度构建分析框架，重点关注战略目标达成情况、关键战略举措的财务影响、战略资源配置效率",
        },
        {
            "angle": "价值创造",
            "focus": "以股东价值创造为核心，重点分析ROE驱动因素、现金流创造能力、资本回报率、经济增加值(EVA)",
        },
        {
            "angle": "风险防控",
            "focus": "从风险管理视角出发，重点关注财务风险指标、现金流断裂风险、债务偿还能力、经营风险敞口",
        },
        {
            "angle": "运营效率",
            "focus": "聚焦运营效率提升，深入分析营运资本管理、资产周转效率、成本费用控制、人效分析",
        },
        {
            "angle": "增长质量",
            "focus": "评估增长的可持续性和质量，重点分析收入增长来源、利润含金量、增长投入产出比、市场份额变化",
        },
        {
            "angle": "业务驱动",
            "focus": "从业务端倒推财务表现，重点分析业务指标与财务指标的关联、客户价值、产品组合优化、渠道效率",
        },
    ]

    for i in range(n_samples):
        # Randomly select a perspective
        perspective = random.choice(framework_perspectives)

        # Randomly decide whether to emphasize industry-specific aspects
        industry_emphasis = random.choice(
            [
                f"深度结合{field}的行业特征，如行业周期、竞争格局、监管政策、技术变革等因素",
                f"突出{field}独特的关键成功因素和核心竞争力指标",
                f"对标{field}行业标杆企业的分析维度和关注重点",
                f"针对{field}的业务模式特点设计专属的分析逻辑",
            ]
        )

        # Add variation elements
        variation_elements = [
            "调整章节的优先级和详略程度",
            "增加一些原框架没有的独特分析维度",
            "改变分析的切入角度和逻辑顺序",
            "在保持整体结构的基础上创新子章节的分析方法",
        ]
        selected_variations = random.sample(variation_elements, k=random.randint(2, 3))

        prompt = f"""
你需要创建一个新的财务分析框架，用于指导{field}企业的财务分析报告生成。

# 分析视角
采用**{perspective["angle"]}**的视角：{perspective["focus"]}

# 行业适配要求
{industry_emphasis}

# 创新要求
基于提供的参考框架，进行以下创新：
{chr(10).join([f"- {var}" for var in selected_variations])}

# 参考框架（仅供参考，不要照搬）
{analysis_framework}

# 输出要求
1. 保持markdown格式，使用清晰的标题层级（一、二、三...）
2. 确保框架的完整性和逻辑性
3. 每个章节要有明确的分析目标和关注重点
4. 体现所选分析视角的特点
5. 融入{field}的行业特色
6. 与已有框架有明显差异，避免重复

仅输出新的分析框架，不要任何其他说明文字。
"""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
            ],
            temperature=random.uniform(0.7, 1.3),
            seed=random.randint(1, 1000000),
        )
        file_name = f"{model}_{field}_framework_{i + 1}.md"
        content = response.choices[0].message.content
        with open(
            f"./reward_model/data/analysis_framework/{file_name}",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(content)

        if pbar:
            pbar.update(1)


def gen_sys_prompt(
    field: str,
    sample_system_prompt: str,
    n_samples: int,
    client: OpenAI,
    model: str,
    pbar: tqdm = None,
):
    # Define diverse quality dimensions for variation
    positive_aspects = [
        {
            "focus": "数据准确性和计算精度",
            "requirements": [
                "所有财务指标计算必须精确到小数点后2位",
                "同比、环比、达成率的计算必须遵循正确的财务公式",
                "对于负值、零值等特殊情况要使用专门的处理逻辑",
                "禁止出现除零错误或不合理的百分比数值",
                "所有数据必须可追溯到原始财务报表",
            ],
        },
        {
            "focus": "深度分析和洞察力",
            "requirements": [
                "必须进行多层次的归因分析，追溯到具体的业务驱动因素",
                "需要识别财务指标之间的内在联系和因果关系",
                "对异常波动要进行根本原因分析，不能停留在表面描述",
                "要结合行业特点和公司战略进行前瞻性分析",
                "每个结论都要有充分的数据支撑和逻辑推导",
            ],
        },
        {
            "focus": "行业专业性和对标分析",
            "requirements": [
                f"深入理解{field}的行业特征和关键成功因素",
                "使用行业专业术语和分析框架",
                "必须与行业标杆企业进行对比分析",
                "要识别行业周期性和季节性因素的影响",
                "结合行业趋势预测公司未来表现",
            ],
        },
        {
            "focus": "全面性和结构化思维",
            "requirements": [
                "必须覆盖盈利能力、营运能力、偿债能力、成长能力等全部维度",
                "从时间维度(趋势)、空间维度(对比)、结构维度(拆解)进行立体分析",
                "要建立财务指标与业务指标之间的完整分析链条",
                "确保分析逻辑严密，各部分之间相互呼应",
                "提供的建议要具体可执行，并量化预期效果",
            ],
        },
        {
            "focus": "数据完整性和异常处理",
            "requirements": [
                "对数据缺失、异常值要明确标注和说明",
                "使用合理的估算方法填补关键数据缺口",
                "对数据质量问题可能带来的分析局限性要充分披露",
                "建立数据交叉验证机制确保一致性",
                "对重要指标要进行敏感性分析",
            ],
        },
    ]

    negative_aspects = [
        {
            "style": "笼统模糊",
            "characteristics": [
                "使用'基本符合预期'、'整体表现尚可'等模糊表述",
                "不需要给出具体的数值和百分比",
                "可以用'有所增长'、'略有下降'等不精确描述替代具体数据",
                "分析结论可以停留在现象描述层面",
            ],
        },
        {
            "style": "计算简化",
            "characteristics": [
                "对于复杂的财务指标计算可以采用简化方法",
                "同比、环比等增长率直接用简单除法即可，不用考虑特殊情况",
                "达成率统一用实际值除以预算值计算",
                "遇到负数、零值等情况可以跳过或使用默认处理方式",
            ],
        },
        {
            "style": "表面分析",
            "characteristics": [
                "只需指出哪些指标上升或下降即可",
                "不需要深究背后的业务原因",
                "可以使用通用的分析模板和套话",
                "对异常数据简单标注即可，无需详细解释",
                "建议可以是原则性的，不需要具体的执行方案",
            ],
        },
        {
            "style": "忽略行业特性",
            "characteristics": [
                f"使用通用的财务分析方法即可，不需要特别考虑{field}的行业特点",
                "行业对比数据可以省略或使用估计值",
                "不需要分析行业周期和竞争格局的影响",
                "使用标准化的分析框架，无需针对行业定制",
            ],
        },
        {
            "style": "片面覆盖",
            "characteristics": [
                "重点分析几个核心指标即可，其他指标可简要提及",
                "数据维度分析(按产品、客户、区域等)可以选择性展开",
                "趋势分析可以只看近期数据，不需要长期历史对比",
                "多个分析角度中选择最直观的1-2个即可",
                "对数据缺失的部分可以直接跳过",
            ],
        },
    ]

    for i in range(n_samples):
        is_positive = random.choice([True, False])

        if is_positive:
            # Randomly select different quality aspects for diversity
            selected_aspects = random.sample(positive_aspects, k=random.randint(2, 3))

            requirements_text = "\n".join(
                [
                    f"**{aspect['focus']}**：\n"
                    + "\n".join([f"- {req}" for req in aspect["requirements"]])
                    for aspect in selected_aspects
                ]
            )

            prompt = f"""
你需要创建一个系统提示词，用于指导AI生成高质量的{field}财务分析报告。

# 核心目标
生成的提示词要确保AI输出的报告具有以下特征：
1. **数值计算完全准确**：所有财务指标、比率、增长率的计算都严格遵循会计准则
2. **分析深度充分**：不止于现象描述，要深入业务层面进行归因分析
3. **行业专业性强**：充分体现{field}的行业特征和专业知识

# 必须包含的质量要求
{requirements_text}

# 写作要求
- 提示词要具体明确，给出可执行的指令
- 要体现对质量的高标准要求
- 针对{field}行业的特点给出针对性的分析要求
- 强调数据准确性和分析深度的重要性
- 可以参考但不要照搬下面的样例结构

# 样例参考（仅供参考结构，内容需要创新）
{sample_system_prompt[:500]}...

请生成一个全新的系统提示词，确保与已有提示词有明显差异。仅输出系统提示词内容，不要任何其他说明。
"""
        else:
            # Randomly select different problematic styles for diversity
            selected_styles = random.sample(negative_aspects, k=random.randint(2, 3))

            characteristics_text = "\n".join(
                [
                    f"**{style['style']}**：\n"
                    + "\n".join([f"- {char}" for char in style["characteristics"]])
                    for style in selected_styles
                ]
            )

            prompt = f"""
你需要创建一个系统提示词，用于指导AI生成{field}财务分析报告。这个提示词应该是"宽松标准"的，不需要过于严格的质量要求。

# 核心目标
生成的提示词应该让AI输出相对简化的报告：
1. 分析不需要过于深入，点到为止即可
2. 数值计算使用常规方法，不需要处理所有特殊情况
3. 使用通用的分析框架，不需要过度定制

# 建议的简化方向
{characteristics_text}

# 写作要求
- 提示词不要过于复杂，保持简洁实用
- 不要过分强调数据准确性的细节
- 分析要求可以相对宽松和灵活
- 不需要强制要求深度归因分析
- 可以允许使用通用模板和标准化表述

请生成一个系统提示词，体现"够用即可"的实用主义风格。仅输出系统提示词内容，不要任何其他说明。
"""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
            ],
            temperature=random.uniform(0.7, 1.3),
            seed=random.randint(1, 1000000),
        )

        file_name = f"{'positive' if is_positive else 'negative'}_{model}_{field}_sys_prompt_{i + 1}.md"
        content = response.choices[0].message.content
        with open(
            f"./reward_model/data/system_prompt/{file_name}", "w", encoding="utf-8"
        ) as f:
            f.write(content)

        if pbar:
            pbar.update(1)


def prompt_pipeline(
    fields: List[str],
    model_configs: List[Dict[str, Any]],
    sample_system_prompt_path: str,
    sample_analysis_framework_path: str,
    samples_per_field: int,
    samples_per_model: int,
):
    # 计算总任务数
    total_tasks = (
        len(fields) * samples_per_field * samples_per_model * 2
    )  # *2 because framework + sys_prompt

    with tqdm(total=total_tasks, desc="生成数据") as pbar:
        for field in fields:
            for i in range(samples_per_field):
                # 随机选择一个模型
                model_config = random.choice(model_configs)

                with open(sample_analysis_framework_path, "r", encoding="utf-8") as f:
                    analysis_framework = f.read()
                gen_framework(
                    analysis_framework=analysis_framework,
                    client=OpenAI(
                        api_key=model_config.api_key, base_url=model_config.base_url
                    ),
                    model=model_config.model,
                    n_samples=samples_per_model,
                    field=field,
                    pbar=pbar,
                )

                with open(sample_system_prompt_path, "r", encoding="utf-8") as f:
                    system_prompt = f.read()
                gen_sys_prompt(
                    field=field,
                    sample_system_prompt=system_prompt,
                    n_samples=samples_per_model,
                    client=OpenAI(
                        api_key=model_config.api_key, base_url=model_config.base_url
                    ),
                    model=model_config.model,
                    pbar=pbar,
                )


if __name__ == "__main__":
    prompt_pipeline(
        fields=["制造业", "服务业", "金融业", "房地产", "科技业"],
        model_configs=[
            ModelConfig(
                model="qwen-turbo",
                base_url=os.getenv("QWEN_URL"),
                api_key=os.getenv("QWEN_KEY"),
            ),
            ModelConfig(
                model="qwen3.5-72b-instruct",
                base_url=os.getenv("QWEN_URL"),
                api_key=os.getenv("QWEN_KEY"),
            ),
            ModelConfig(
                model="deepseek-chat",
                base_url=os.getenv("DS_URL"),
                api_key=os.getenv("DS_KEY"),
            ),
            ModelConfig(
                model="qwen-plus",
                base_url=os.getenv("QWEN_URL"),
                api_key=os.getenv("QWEN_KEY"),
            ),
        ],
        sample_system_prompt_path="./reward_model/data/system_prompt.md",
        sample_analysis_framework_path="./reward_model/data/analysis_framework.md",
        samples_per_field=20,
        samples_per_model=5,
    )
