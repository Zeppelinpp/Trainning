import os
import random
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

qwen_client = OpenAI(
    api_key=os.getenv("QWEN_KEY"),
    base_url=os.getenv("QWEN_URL")
)

def alter_framework(analysis_framework: str, client: OpenAI, model: str, n_samples: int = 10):
    prompt = f"""
    按照给出的财务分析框架, 尝试写出关注点不同, 逻辑不同但是结构类似的分析框架。 给出的框架都可以指示生成高质量的分析报告。

    # 财务分析框架
    {analysis_framework}

    仅输出新的分析框架, 不要任何其他内容
    """
    for i in tqdm(range(n_samples)):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
            ],
            temperature=1,
            seed=random.randint(1, 1000000)
        )
        file_name = f"{model}_framework_{i + 1}.md"
        content = response.choices[0].message.content
        with open(f"./reward_model/data/analysis_framework_synthetic/{file_name}", "w", encoding="utf-8") as f:
            f.write(content)
    

# def prepare_prompt(analysis_framework: str):



# def generate_synthetic_data(n_samples: int = 1000, prompt: str = ""):
#     pass

if __name__ == "__main__":
    with open("./reward_model/data/analysis_framework.md", "r", encoding="utf-8") as f:
        analysis_framework = f.read()
    alter_framework(
        analysis_framework=analysis_framework,
        client=OpenAI(
            api_key=os.getenv("QWEN_KEY"),
            base_url=os.getenv("QWEN_URL")
        ),
        model="qwen-turbo",
        n_samples=5
    )