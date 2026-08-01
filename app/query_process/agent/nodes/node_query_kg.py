from typing import List, Dict, Any

from app.utils.task_utils import add_running_task, add_done_task
from app.clients.neo4j_utils import query_related_entities
from app.core.logger import logger


def node_query_kg(state):
    """
    节点功能：在 Neo4j 知识图谱中检索与文档主题相关的实体/关系，
    作为一路补充召回（与向量检索、HyDE、Web 搜索并行），最终汇入 RRF 融合。
    未配置 Neo4j 时返回空列表，安全降级。
    """
    add_running_task(state["session_id"], "node_query_kg", state.get("is_stream"))

    kg_docs: List[Dict[str, Any]] = []
    try:
        item_names = state.get("item_names") or []
        if item_names:
            related = query_related_entities(item_names)
            for i, doc in enumerate(related):
                kg_docs.append(
                    {
                        "id": f"kg_{i}",
                        "distance": 1.0,
                        "entity": {
                            "chunk_id": f"kg_{i}",
                            "content": doc.get("content", ""),
                            "title": doc.get("title", ""),
                        },
                    }
                )
        logger.info(f"知识图谱召回 {len(kg_docs)} 条")
    except Exception as e:
        logger.error(f"知识图谱查询异常：{e}", exc_info=True)
    finally:
        add_done_task(state["session_id"], "node_query_kg", state.get("is_stream"))

    return {"kg_chunks": kg_docs}
