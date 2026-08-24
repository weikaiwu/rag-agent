from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger
from app.utils.retrieval_utils import reciprocal_rank_fusion


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

    state["rrf_chunks"] = reciprocal_rank_fusion(source_with_weight)

    add_done_task(state['session_id'], "node_rrf", state.get("is_stream"))
    return state
