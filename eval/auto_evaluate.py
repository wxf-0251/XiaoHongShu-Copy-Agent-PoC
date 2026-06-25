"""
自动评测脚本 — 直接从知识库 txt 解析产品，无需手动维护 JSON 测试用例。
用法：python eval/auto_evaluate.py
"""
import sys
import os
import json
import re
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graph import agent_app
from langchain_openai import ChatOpenAI

judge_llm = ChatOpenAI(
    model_name="deepseek-chat",
    openai_api_base="https://api.deepseek.com/v1",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY")
)

# 默认评测风格（可自行增减）
DEFAULT_TONES = [
    "干货科普风",
    "闺蜜种草风",
    "极客硬核风",
    "生活好物风",
    "商业变现风",
]


def parse_products(filepath: str) -> list[dict]:
    """从 my_knowledge.txt 解析产品，返回 [{topic, ground_truth_facts}, ...]"""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    products = []
    blocks = text.strip().split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if not lines:
            continue
        first_line = lines[0]
        if first_line.startswith("产品名称："):
            topic = first_line.replace("产品名称：", "").strip()
        elif first_line.startswith("产品名称:"):
            topic = first_line.replace("产品名称:", "").strip()
        else:
            topic = first_line.strip()

        ground_truth = " ".join(line.strip() for line in lines)
        products.append({
            "topic": topic,
            "ground_truth_facts": ground_truth
        })

    return products


def run_auto_evaluation(tones: list[str] | None = None):
    if tones is None:
        tones = DEFAULT_TONES

    txt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "my_knowledge.txt")
    if not os.path.exists(txt_path):
        print(f"找不到知识库文件: {txt_path}")
        return

    products = parse_products(txt_path)
    total = len(products) * len(tones)
    print(f"从知识库解析到 {len(products)} 个产品, {len(tones)} 种风格, 共 {total} 组评测用例")

    results = []
    idx = 0
    for product in products:
        for tone in tones:
            idx += 1
            topic = product["topic"]
            print(f"[{idx}/{total}] 评测: {topic} @ {tone}")

            inputs = {"topic": topic, "tone": tone, "iteration": 0}
            final_output = agent_app.invoke(inputs)

            generated_copy = final_output.get("draft", "")
            iterations = final_output.get("iteration", 0)
            internal_pass = final_output.get("is_pass", False)
            retrieved_context = final_output.get("context") or product["ground_truth_facts"]

            judge_prompt = f"""
            你现在是一个无情的机器评测脚本。你的唯一任务是对比事实和文案，找出是否捏造了数据。
            【参考事实】: {retrieved_context}
            【被评估文案】: {generated_copy}

            要求：
            1. 只要文案中出现了参考事实以外的价格、参数、功能，就是存在幻觉 (true)。如果仅是风格化修饰词则不算。
            2. 注意区分"捏造数据"与"合理营销表达"：同义改写不算幻觉（如"99元"写为"不到百元"、"免费"写为"0元"），但凭空编造具体数字（如事实没有价格却写了价格、事实说500+却写成5000+）必须判定为幻觉。
            3. 必须且只能输出合法的 JSON 字符串！绝对不要输出任何其他废话或文案内容！
            输出格式严格如下：
            {{"has_hallucination": false, "reason": "参数与事实完全一致"}}
            """

            judge_res_raw = ""
            try:
                judge_res_raw = judge_llm.invoke(judge_prompt).content.strip()
                if judge_res_raw.startswith("```json"):
                    judge_res_raw = judge_res_raw.replace("```json", "").replace("```", "").strip()
                elif judge_res_raw.startswith("```"):
                    judge_res_raw = judge_res_raw.replace("```", "").strip()
                judge_dict = json.loads(judge_res_raw)
                has_hallucination = judge_dict.get("has_hallucination", True)
                judge_reason = judge_dict.get("reason", judge_res_raw)
            except Exception:
                has_hallucination = bool(re.search(r'"has_hallucination"\s*:\s*true', judge_res_raw))
                judge_reason = f"JSON解析失败，原始输出: {judge_res_raw}"

            results.append({
                "主题": topic,
                "风格": tone,
                "Agent自评通过": internal_pass,
                "重写轮数": iterations,
                "裁判判定幻觉": has_hallucination,
                "裁判意见": judge_reason,
                "生成文案": generated_copy[:60] + "..."
            })

    df = pd.DataFrame(results)
    pass_rate = (df["Agent自评通过"].sum() / len(df)) * 100
    hallucination_rate = (df["裁判判定幻觉"].sum() / len(df)) * 100
    avg_iterations = df["重写轮数"].mean()

    print("\n" + "=" * 30)
    print("自动评测报告")
    print(f"总测试样本: {len(df)} 组")
    print(f"合规通过率: {pass_rate:.1f}%")
    print(f"事实幻觉率: {hallucination_rate:.1f}%")
    print(f"平均重写轮数: {avg_iterations:.2f} 轮")
    print("=" * 30)

    report_name = f"eval/auto_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
    df.to_excel(report_name, index=False)
    print(f"详细报表已保存至: {report_name}")


if __name__ == "__main__":
    run_auto_evaluation()
