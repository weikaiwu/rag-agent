"""
重排分数分布分析脚本（P4 阈值校准）

目的：用数据回答"置信度阈值定多少"——不能拍脑袋。
方法：对比两组问题的 BGE-Reranker top1 分数分布：
  A 组（能答）：评估集里 policy 类问题（FAQ 知识库有答案）
  B 组（不能答）：知识库完全没有的无关问题
分界点即推荐阈值。

用法：python scripts/analyze_rerank_scores.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from app.conf.milvus_config import milvus_config
from app.lm.embedding_utils import generate_embeddings
from app.lm.reranker_utils import get_reranker_model
from app.core.logger import logger

# 不能答的问题（知识库完全没覆盖的领域）
UNANSWERABLE_QUESTIONS = [
    "量子纠缠的物理原理是什么？",
    "推荐一下深圳好吃的餐厅",
    "今天A股大盘走势怎么样？",
    "怎么修复Linux内核的调度器bug？",
    "特斯拉最新车型的续航是多少？",
    "Python 3.13 的 GIL 移除细节",
    "怎么治疗腰椎间盘突出？",
    "React 19 的新特性有哪些？",
    "比特币的挖矿算法是什么？",
    "如何训练一个扩散模型？",
]

logger.disable("app")  # 关掉业务日志，只看结果


def get_top1_rerank_score(query: str) -> float | None:
    """对问题跑 混合检索 + rerank，返回 top1 分数"""
    client = get_milvus_client()
    vec = generate_embeddings([query])
    reqs = create_hybrid_search_requests(vec["dense"][0], vec["sparse"][0], limit=5)
    res = hybrid_search(
        client, milvus_config.chunks_collection, reqs,
        ranker_weights=(0.5, 0.5), norm_score=True, limit=5,
        output_fields=["content", "title"],
    )
    if not res or not res[0]:
        return None

    # 收集候选文本做 rerank
    candidates = []
    for hit in res[0]:
        entity = hit.get("entity", {})
        text = entity.get("content", "")
        if text:
            candidates.append(text)
    if not candidates:
        return None

    rerank = get_reranker_model()
    pairs = [[query, t] for t in candidates[:5]]
    scores = rerank.compute_score(pairs, normalize=True)
    return float(max(scores)) if scores else None


def main() -> None:
    eval_set = json.loads(Path("app/eval/cs_eval_set.json").read_text(encoding="utf-8"))
    answerable_questions = [x["question"] for x in eval_set if x["intent"] == "policy"][:10]

    print("\n===== A 组：能答的问题（policy 类，FAQ 有答案）=====")
    a_scores = []
    for q in answerable_questions:
        s = get_top1_rerank_score(q)
        a_scores.append(s)
        print(f"  {s:.3f}  {q[:35]}" if s is not None else f"  None  {q[:35]}")

    print("\n===== B 组：不能答的问题（知识库没覆盖）=====")
    b_scores = []
    for q in UNANSWERABLE_QUESTIONS:
        s = get_top1_rerank_score(q)
        b_scores.append(s)
        print(f"  {s:.3f}  {q[:35]}" if s is not None else f"  None  {q[:35]}")

    a_valid = [s for s in a_scores if s is not None]
    b_valid = [s for s in b_scores if s is not None]
    print(f"\nA 组 top1 分数: min={min(a_valid):.3f} max={max(a_valid):.3f} avg={sum(a_valid)/len(a_valid):.3f}")
    print(f"B 组 top1 分数: min={min(b_valid):.3f} max={max(b_valid):.3f} avg={sum(b_valid)/len(b_valid):.3f}")
    if a_valid and b_valid:
        print(f"\n建议阈值区间: B组max({max(b_valid):.3f}) ~ A组min({min(a_valid):.3f})")


if __name__ == "__main__":
    main()
