import json
import os
from functools import lru_cache

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

from init_db import init_database

LLM = ChatOpenAI(
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    openai_api_base="https://api.deepseek.com/v1",
    model_name="deepseek-chat",
)

PERSIST_DIRECTORY = "./chroma_db_ollama"


@lru_cache(maxsize=1)
def _get_embeddings():
    return HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")


@lru_cache(maxsize=1)
def _get_vector_db():
    if not os.path.exists(os.path.join(PERSIST_DIRECTORY, "chroma.sqlite3")):
        init_database(force=True)
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=_get_embeddings(),
    )


def retrieve_knowledge(state) -> dict:
    vector_db = _get_vector_db()
    results = vector_db.similarity_search_with_score(state["topic"], k=5)
    context_text = ""

    if results:
        best_doc = results[0][0]
        for doc, _score in results:
            for line in doc.page_content.split("\n"):
                if "产品名称" in line and state["topic"][:6] in line:
                    best_doc = doc
                    break
        context_text = best_doc.page_content

    return {"context": context_text}


def generate_draft(state) -> dict:
    if state.get("iteration", 0) == 0:
        prompt = f"""请以「{state['tone']}」写关于「{state['topic']}」的小红书文案。

参考以下事实（不可捏造）：
{state['context']}

【隔离指令】请仔细甄别参考事实。如果资料中包含其他无关产品的描述，请绝对无视；只提取与「{state['topic']}」完全一致的功能和参数，绝不允许张冠李戴。

【强制指令】直接输出文案正文，绝对不允许包含任何类似「好的」这样的寒暄语或前言。"""
        response = LLM.bind(temperature=0.7).invoke(prompt)
    else:
        prompt = f"""请严格根据以下修改意见，重新修改文案。
【修改示例】：
- 意见指出：删除捏造的).replace(", 原价1299元。
- 错误原文：这款耳机原价1299元，现在只要399。
- 正确修改：这款耳机限时特惠只要399。

【不可捏造的事实基准】：
{state['context']}

【原稿】：
{state['draft']}

【修改意见】：
{state['feedback']}

【思维链指令】请先在心里思考后再直接输出修改后的最终文案正文。"""
        response = LLM.bind(temperature=0.0).invoke(prompt)

    return {"draft": response.content, "iteration": 1}


def review_draft(state) -> dict:
    print("[Reviewer] 正在审核文案...")
    prompt = f"""你是一个严格的企业宣发总监。请审核以下文案草稿。

【企业知识基准】：
{state['context']}

【待审文案】：
{state['draft']}

【要求风格】：
{state['tone']}

审核标准：
1. 文案是否涉及了企业知识基准之外的虚假参数。
2. 是否符合要求风格。

请务必严格按照以下 JSON 格式输出你的审核结果，不要输出任何额外解释文字：
{{
    "is_pass": false,
    "feedback": "如果不合格，给出具体的修改指导。如果完全合格，输出'无'"
}}"""

    response = LLM.bind(temperature=0.0).invoke(prompt)

    try:
        clean_text = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean_text)
        return {
            "is_pass": result.get("is_pass", False),
            "feedback": result.get("feedback", "格式错误重写"),
        }
    except Exception:
        return {
            "is_pass": False,
            "feedback": "JSON解析失败，请严格输出合规的JSON格式。",
        }
