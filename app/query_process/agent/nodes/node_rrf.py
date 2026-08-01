from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger


def step_3_reciprocal_rank_fusion(source_with_weight, top_k: int = 5):
    # 累加各路召回的 RRF 分数；k=60 为经典 RRF 常数，抑制高分项的边际影响
    score_dict = {}
    chunk_dict = {}
    for source, wight in source_with_weight:
        for rank, chunk in enumerate(source, start=1):
            chunk_id = chunk.get("id") or chunk.get("entity").get("chunk_id")
            score_dict[chunk_id] = score_dict.get(chunk_id, 0.0) + (1.0 / (60 + rank)) * wight
            chunk_dict.setdefault(chunk_id, chunk)

    merged = [(chunk_dict[cid], score) for cid, score in score_dict.items()]
    merged.sort(key=lambda x: x[1], reverse=True)
    rank_chunks = [chunk for chunk, _ in merged[:top_k]]
    logger.info(f"完成了rrf的排序处理，结果为{rank_chunks}")
    return rank_chunks


def node_rrf(state):
    """
    节点功能：Reciprocal Rank Fusion
    将多路召回的结果（向量、HyDE、Web、KG）进行加权融合排序。
    """
    add_running_task(state["session_id"], "node_rrf", state.get("is_stream"))

    embedding_chunks = state.get("embedding_chunks")
    hyde_embedding_chunks = state.get("hyde_embedding_chunks")
    kg_chunks = state.get("kg_chunks") or []

    # 多路加权融合：向量 / HyDE / 知识图谱；Web 在 rerank 阶段并入
    source_with_weight = [
        (embedding_chunks, 1.0),
        (hyde_embedding_chunks, 1.0),
    ]
    if kg_chunks:
        source_with_weight.append((kg_chunks, 1.0))

    state["rrf_chunks"] = step_3_reciprocal_rank_fusion(source_with_weight)

    add_done_task(state['session_id'], "node_rrf", state.get("is_stream"))
    return state
