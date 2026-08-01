import json
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage

from app.import_process.agent.state import ImportGraphState
from app.lm.lm_utils import get_llm_client
from app.core.load_prompt import load_prompt
from app.clients.neo4j_utils import is_neo4j_enabled, write_graph
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger


def _extract_entities_relations(item_name: str, chunks: List[Dict[str, Any]]):
    """调用大模型从切片文本中抽取实体与关系。"""
    texts = [c.get("item_content") or c.get("content") or "" for c in chunks[:8]]
    context = "\n\n".join(texts)
    prompt = load_prompt("kg_extraction", item_name=item_name, context=context)
    llm = get_llm_client(json_mode=True)
    resp = llm.invoke([HumanMessage(content=prompt)])
    content = resp.content
    if content.startswith("```json"):
        content = content.replace("```json", "").replace("```", "")
    data = json.loads(content)
    return data.get("entities", []), data.get("relations", [])


def node_build_kg(state: ImportGraphState) -> ImportGraphState:
    """
    节点：构建知识图谱（可选环节）
    从切分后的文本中抽取实体与关系，写入 Neo4j。
    若未配置 Neo4j（NEO4J_URI 为空），本节点安全跳过，不影响主流程。
    """
    current_node = "node_build_kg"
    add_running_task(state.get("task_id", ""), current_node)

    try:
        if not is_neo4j_enabled():
            logger.info("Neo4j 未启用，跳过知识图谱构建节点。")
            return state

        item_name = state.get("item_name") or (state.get("chunks") or [{}])[0].get(
            "item_name"
        )
        chunks = state.get("chunks") or []
        if not item_name or not chunks:
            logger.info("缺少 item_name 或 chunks，跳过知识图谱构建。")
            return state

        entities, relations = _extract_entities_relations(item_name, chunks)
        write_graph(item_name, entities, relations)
    except Exception as e:
        logger.error(f"知识图谱构建节点异常：{e}", exc_info=True)
    finally:
        add_done_task(state.get("task_id", ""), current_node)
    return state
