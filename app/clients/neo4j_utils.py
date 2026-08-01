import os
from typing import List, Dict, Any, Optional

from app.core.logger import logger

_neo4j_driver = None


def is_neo4j_enabled() -> bool:
    """是否配置了 Neo4j。未配置时知识图谱模块安全跳过，不影响主流程。"""
    return bool(os.getenv("NEO4J_URI"))


def get_neo4j_driver():
    """获取 Neo4j 驱动单例；未配置或初始化失败时返回 None。"""
    global _neo4j_driver
    if not is_neo4j_enabled():
        return None
    if _neo4j_driver is None:
        try:
            from neo4j import GraphDatabase

            _neo4j_driver = GraphDatabase.driver(
                os.getenv("NEO4J_URI"),
                auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
            )
        except Exception as e:
            logger.error(f"Neo4j 驱动初始化失败：{e}")
            return None
    return _neo4j_driver


def write_graph(
    item_name: str,
    entities: List[str],
    relations: List[Dict[str, str]],
) -> None:
    """
    把一份文档抽取出的实体/关系写入 Neo4j 知识图谱。
    :param item_name: 所属文档主题名称（作为图谱根节点）
    :param entities: 实体名称列表（部件、参数、操作等）
    :param relations: 关系列表，每项 {head, relation, tail}
    """
    driver = get_neo4j_driver()
    if driver is None:
        logger.info("Neo4j 未启用，跳过知识图谱写入。")
        return
    try:
        with driver.session() as session:
            # 文档主题根节点
            session.run("MERGE (d:Device {name:$name})", name=item_name)
            # 实体节点 + 与文档主题的归属关系
            for ent in entities:
                if not ent:
                    continue
                session.run(
                    "MERGE (e:Entity {name:$name}) "
                    "MERGE (d:Device {name:$dev}) "
                    "MERGE (d)-[:HAS_ENTITY]->(e)",
                    name=ent,
                    dev=item_name,
                )
            # 实体之间的关系
            for r in relations:
                h, rel, t = r.get("head"), r.get("relation"), r.get("tail")
                if not (h and rel and t):
                    continue
                session.run(
                    "MERGE (h:Entity {name:$h}) "
                    "MERGE (t:Entity {name:$t}) "
                    "MERGE (h)-[:REL {type:$rel}]->(t)",
                    h=h,
                    t=t,
                    rel=rel,
                )
        logger.info(
            f"已写入知识图谱：主题={item_name}, 实体数={len(entities)}, 关系数={len(relations)}"
        )
    except Exception as e:
        logger.error(f"知识图谱写入失败：{e}", exc_info=True)


def query_related_entities(
    item_names: List[str], limit: int = 20
) -> List[Dict[str, str]]:
    """
    根据文档主题查询其关联实体与关系，返回文本片段供 RAG 作为一路召回使用。
    :return: [{"title": 关系描述, "content": 关系文本}]
    """
    driver = get_neo4j_driver()
    if driver is None:
        return []
    results: List[Dict[str, str]] = []
    try:
        with driver.session() as session:
            for name in item_names:
                recs = session.run(
                    "MATCH (d:Device {name:$name})-[*1..2]-(n) "
                    "RETURN labels(n)[0] AS kind, n.name AS nm LIMIT $limit",
                    name=name,
                    limit=limit,
                )
                for rec in recs:
                    kind = rec.get("kind")
                    nm = rec.get("nm")
                    if nm:
                        results.append(
                            {
                                "title": f"{name} 关联{kind}",
                                "content": f"{name} 的{kind}：{nm}",
                            }
                        )
    except Exception as e:
        logger.error(f"知识图谱查询失败：{e}", exc_info=True)
    return results
