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
    
    prompt = f"""
    按照给出的财务分析框架, 尝试根据行业特质写出关注点不同, 逻辑不同但是结构类似的分析框架。 给出的框架都可以指示生成高质量的分析报告。

    # 财务分析框架
    {analysis_framework}

    # 所处行业
    {field}

    仅输出新的分析框架, 不要任何其他内容
    """
    for i in range(n_samples):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
            ],
            temperature=random.uniform(0.3, 1.5),
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
    for i in range(n_samples):
        # 每个样本随机选择正负类型
        is_positive = random.choice([True, False])
        if is_positive:
            demand = "优质"
        else:
            demand = "瑕疵"
        prompt = f"""
        根据提供的系统提示词和当前需求什么样的提示词, 生成对应的新的系统提示词

        # 指引
        - 如果需要优质报告: 提示词中需要明确要求数据准确性，分析深度和针对对应行业的分析专业性
        - 如果需要瑕疵报告: 提示词中不需要明确的质量要求，只需要模糊的分析要点和描述，对文本的专业性和深度不做指示与要求

        # 样例系统提示词
        {sample_system_prompt}

        # 所处行业
        {field}

        # 需求
        {demand}提示词

        请注意区分是要优质提示词还是瑕疵提示词，仅输出新的系统提示词, 不要任何其他内容
        """
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
            ],
            temperature=random.uniform(0.3, 1.5),
            seed=random.randint(1, 1000000),
        )

        file_name = (
            f"{'positive' if is_positive else 'negative'}_{model}_{field}_sys_prompt_{i + 1}.md"
        )
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
    total_tasks = len(fields) * samples_per_field * samples_per_model * 2  # *2 because framework + sys_prompt
    
    with tqdm(total=total_tasks, desc="生成数据") as pbar:
        for field in fields:
            for i in range(samples_per_field):
                # 随机选择一个模型
                model_config = random.choice(model_configs)
                
                with open(sample_analysis_framework_path, "r", encoding="utf-8") as f:
                    analysis_framework = f.read()
                gen_framework(
                    analysis_framework=analysis_framework,
                    client=OpenAI(api_key=model_config.api_key, base_url=model_config.base_url),
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
                    client=OpenAI(api_key=model_config.api_key, base_url=model_config.base_url),
                    model=model_config.model,
                    pbar=pbar,
                )
        
        
if __name__ == "__main__":
    prompt_pipeline(
        fields=["制造业", "服务业", "金融业", "房地产", "科技业"],
        model_configs=[ModelConfig(model="qwen-turbo", base_url=os.getenv("QWEN_URL"), api_key=os.getenv("QWEN_KEY"))],
        sample_system_prompt_path="./reward_model/data/system_prompt.md",
        sample_analysis_framework_path="./reward_model/data/analysis_framework.md",
        samples_per_field=5,
        samples_per_model=2,
    )
