import sys
import os
import json
import re
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# 将项目根目录加入路径，确保能导入 core 里的代码
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graph import agent_app # 导入你的 Agent 实例
from langchain_openai import ChatOpenAI

# 1. 初始化裁判模型（建议用性能最强的模型，如 DeepSeek-V3 或 GPT-4o）
judge_llm = ChatOpenAI(
    model_name="deepseek-chat",
    openai_api_base="https://api.deepseek.com/v1",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY")
)

def run_evaluation():
    # 加载测试用例
    with open("eval/test_cases.json", "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    results = []
    print(f"🚀 开始自动化评测，共 {len(test_cases)} 组用例...")

    for case in test_cases:
        print(f"正在测试主题: {case['topic']}...")
        
        # A. 运行你的 Agent
        inputs = {"topic": case['topic'], "tone": case['tone'], "iteration": 0}
        final_output = agent_app.invoke(inputs)
        
        generated_copy = final_output.get("draft", "")
        iterations = final_output.get("iteration", 0)
        internal_pass = final_output.get("is_pass", False)

        # 提取 Agent 真正查到的知识库内容
        retrieved_context = final_output.get("context") or case['ground_truth_facts']

        # B. 裁判打分：幻觉检测 (JSON 强约束版)
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
        
        try:
            judge_res_raw = judge_llm.invoke(judge_prompt).content.strip()
            # 清洗掉大模型可能自带的 markdown 代码块符号
            if judge_res_raw.startswith("```json"):
                judge_res_raw = judge_res_raw.replace("```json", "").replace("```", "").strip()
            elif judge_res_raw.startswith("```"):
                judge_res_raw = judge_res_raw.replace("```", "").strip()
                
            # 解析 JSON
            judge_dict = json.loads(judge_res_raw)
            has_hallucination = judge_dict.get("has_hallucination", True)
            judge_reason = judge_dict.get("reason", judge_res_raw)
        except Exception as e:
            # 万一解析失败的暴力兜底，用正则精确匹配 has_hallucination 值
            has_hallucination = bool(re.search(r'"has_hallucination"\s*:\s*true', judge_res_raw))
            judge_reason = f"JSON解析失败，原始输出: {judge_res_raw}"

        # C. 记录结果
        results.append({
            "主题": case['topic'],
            "Agent自评通过": internal_pass,
            "重写轮数": iterations,
            "裁判判定幻觉": has_hallucination,
            "裁判意见": judge_reason,
            "生成文案": generated_copy[:50] + "..." # 缩略显示即可
        })

    # 3. 计算统计指标
    df = pd.DataFrame(results)
    pass_rate = (df["Agent自评通过"].sum() / len(df)) * 100
    hallucination_rate = (df["裁判判定幻觉"].sum() / len(df)) * 100
    avg_iterations = df["重写轮数"].mean()

    print("\n" + "="*30)
    print("📊 最终评测报告")
    print(f"总测试样本: {len(df)} 组")
    print(f"合规通过率: {pass_rate:.1f}%")
    print(f"事实幻觉率: {hallucination_rate:.1f}%")
    print(f"平均重写轮数: {avg_iterations:.2f} 轮")
    print("="*30)

    # 导出详细 Excel 报表供查看
    report_name = f"eval/report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
    df.to_excel(report_name, index=False)
    print(f"详细报表已保存至: {report_name}")

if __name__ == "__main__":
    run_evaluation()