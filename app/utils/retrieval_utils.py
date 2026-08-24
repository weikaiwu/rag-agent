"""检索链路纯逻辑工具（无外部 IO 依赖，可单元测试）。

从 LangGraph 查询节点中抽离的纯函数：
- reciprocal_rank_fusion: 加权 RRF 多路召回融合
- dynamic_topk_with_gap: 动态 TopK + 断崖检测截断
- build_item_name_expr: Milvus 主题过滤表达式构造
"""
from app.core.logger import logger

# -----------------------------
# 动态 TopK 全局常量
# -----------------------------
# 动态 TopK 硬上限：最多取前 N 条（<=10）
RERANK_MAX_TOPK: int = 10
# 最小 TopK：至少保留前 N 条（>=1，且 <= RERANK_MAX_TOPK）
RERANK_MIN_TOPK: int = 1
# 断崖阈值（相对）：相邻分数落差超过该比例即截断
RERANK_GAP_RATIO: float = 0.25
# 断崖阈值（绝对）：相邻分数落差超过该分值即截断
RERANK_GAP_ABS: float = 0.5


def reciprocal_rank_fusion(source_with_weight, top_k: int = 5):
    """加权 RRF 融合：多路召回按 1/(60+rank)*weight 累加分数，取 top_k。

    Args:
        source_with_weight: [(chunks, weight), ...]，chunk 需含 "id" 或 "entity.chunk_id"。
        top_k: 融合后保留的最大条数。

    Returns:
        按融合分数降序的 chunk 列表。
    """
    # 累加各路召回的 RRF 分数；k=60 为经典 RRF 常数，抑制高分项的边际影响
    score_dict = {}
    chunk_dict = {}
    for source, weight in source_with_weight:
        for rank, chunk in enumerate(source, start=1):
            chunk_id = chunk.get("id") or chunk.get("entity").get("chunk_id")
            score_dict[chunk_id] = score_dict.get(chunk_id, 0.0) + (1.0 / (60 + rank)) * weight
            chunk_dict.setdefault(chunk_id, chunk)

    merged = [(chunk_dict[cid], score) for cid, score in score_dict.items()]
    merged.sort(key=lambda x: x[1], reverse=True)
    rank_chunks = [chunk for chunk, _ in merged[:top_k]]
    logger.info(f"完成了rrf的排序处理，结果为{rank_chunks}")
    return rank_chunks


def dynamic_topk_with_gap(rerank_score_list):
    """动态 TopK：结合硬上下限与断崖检测，避免把质量骤降的尾部片段带入上下文。

    Args:
        rerank_score_list: 已按分数降序的 [{"score": float, ...}, ...]。

    Returns:
        截断后的片段列表（长度 <= RERANK_MAX_TOPK，>= RERANK_MIN_TOPK）。
    """
    topk = min(RERANK_MAX_TOPK, len(rerank_score_list))
    if topk > RERANK_MIN_TOPK:
        # 双指针扫描相邻分差：触发绝对/相对断崖即截断
        for index in range(RERANK_MIN_TOPK - 1, topk - 1):
            score_1 = rerank_score_list[index].get("score", 0.0)
            score_2 = rerank_score_list[index + 1].get("score", 0.0)
            gap = score_1 - score_2
            # abs(score_1)+1e-6 防止分母为 0（含负值场景）
            rel = gap / (abs(score_1) + 1e-6)
            if gap >= RERANK_GAP_ABS or rel >= RERANK_GAP_RATIO:
                logger.info(f"数据集合{index}和{index + 1}的位置发生了断崖，结束循环！！")
                topk = index + 1
                break

    topk_doc_list = rerank_score_list[:topk]
    logger.info(f"最终截取的长度：{topk},截取的内容:{topk_doc_list}")
    return topk_doc_list


def build_item_name_expr(item_names):
    """构造 Milvus 主题过滤表达式；空列表返回 None（全库语义检索）。

    Args:
        item_names: 已确认的主题名列表，或 None/空列表。

    Returns:
        'item_name in ["a","b"]' 形式的过滤表达式；无主题时返回 None。
    """
    if item_names:
        item_name_str = ",".join(f'"{item}"' for item in item_names)
        return f"item_name in [{item_name_str}]"
    return None
