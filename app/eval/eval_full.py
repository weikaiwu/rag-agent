"""
全量 RAGAS 评估脚本（54 条）
============================
合并 sample_dataset.json（10 条已有）与 eval_dataset_v2.json（44 条新增），
共 54 条评估集，复用项目 LLM/BGE 客户端跑 RAGAS 四指标，
结果落盘 ragas_report_full.json。

用法：
  cd 项目根目录
  python -m app.eval.eval_full
"""
import json
import sys
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from app.eval.ragas_eval import (
    _ensure_ragas_importable,
    ProjectBGEEmbeddings,
)
from app.lm.lm_utils import get_llm_client
from app.core.logger import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT_ROOT / "app" / "eval"

_ensure_ragas_importable()
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset


def load_merged_dataset():
    with open(EVAL_DIR / "sample_dataset.json", "r", encoding="utf-8") as f:
        old = json.load(f)
    with open(EVAL_DIR / "eval_dataset_v2.json", "r", encoding="utf-8") as f:
        new = json.load(f)
    merged = old + new
    # 校验字段完整
    for i, r in enumerate(merged):
        missing = [k for k in ("question", "contexts", "answer", "ground_truth") if not r.get(k)]
        if missing:
            logger.warning(f"第 {i} 条缺字段 {missing}: {r.get('question', '')[:30]}")
    logger.info(f"合并后评估集: {len(merged)} 条（旧 {len(old)} + 新 {len(new)}）")
    return merged


def main():
    rows = load_merged_dataset()
    dataset = Dataset.from_list(rows)

    llm = LangchainLLMWrapper(get_llm_client())
    embeddings = LangchainEmbeddingsWrapper(ProjectBGEEmbeddings())

    logger.info("开始 RAGAS 评估（54 条，预计 10-20 分钟）...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    print("\n===== RAGAS 全量评估结果（54 条）=====")
    print(result)

    try:
        df = result.to_pandas()
        report_path = EVAL_DIR / "ragas_report_full.json"
        df.to_json(report_path, orient="records", force_ascii=False, indent=2)
        print(f"\n报告已写入: {report_path}")
    except Exception as e:
        logger.warning(f"结果保存失败（不影响评估）：{e}")


if __name__ == "__main__":
    main()
