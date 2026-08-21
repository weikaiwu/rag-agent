"""
生成 RAG 评估集答案（answer）脚本
==================================
对 eval_dataset_v2.json 中 44 条新问题，用项目自己的 LLM 客户端
基于每条问题的标注上下文（contexts）生成 answer，模拟 RAG 生成口径，
回填 answer 字段后落盘。

用法：
  cd 项目根目录
  python -m app.eval.gen_answers
"""
import json
import sys
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from app.lm.lm_utils import get_llm_client
from app.core.logger import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "app" / "eval" / "eval_dataset_v2.json"

SYSTEM_PROMPT = (
    "你是企业知识库 RAG 问答系统的答案生成模块。"
    "你必须严格基于给定的上下文片段回答问题，不得编造上下文之外的信息。"
    "回答要简洁、准确、直接，控制在 2-4 句话。"
)


def build_user_prompt(row: dict) -> str:
    ctx = "\n\n".join(f"[片段{i+1}] {c}" for i, c in enumerate(row["contexts"]))
    return f"上下文片段：\n{ctx}\n\n问题：{row['question']}"


def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        rows = json.load(f)

    llm = get_llm_client()
    from langchain_core.messages import HumanMessage, SystemMessage

    generated = 0
    for row in rows:
        if row.get("answer"):
            continue
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=build_user_prompt(row)),
        ]
        try:
            resp = llm.invoke(messages)
            row["answer"] = resp.content.strip()
            generated += 1
            logger.info(f"[{generated}/{len(rows)}] 已生成: {row['question'][:30]}...")
        except Exception as e:
            logger.warning(f"生成失败: {row['question'][:30]}... -> {e}")

    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"\n完成：{len(rows)} 条中生成 answer {generated} 条，剩余 {len(rows) - sum(1 for r in rows if r['answer'])} 条未生成")
    print(f"已保存: {DATA_PATH}")


if __name__ == "__main__":
    main()
