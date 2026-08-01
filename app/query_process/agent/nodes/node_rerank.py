from dotenv import load_dotenv

from app.lm.reranker_utils import get_reranker_model
from app.core.logger import logger
from app.utils.task_utils import add_running_task, add_done_task

load_dotenv()

# -----------------------------
# Rerank / TopK 全局常量（不从 state 读取）
# -----------------------------
# 动态 TopK 硬上限：最多取前 N 条（<=10）
RERANK_MAX_TOPK: int = 10
# 最小 TopK：至少保留前 N 条（>=1，且 <= RERANK_MAX_TOPK）
RERANK_MIN_TOPK: int = 1
# 断崖阈值（相对）：相邻分数落差超过该比例即截断
RERANK_GAP_RATIO: float = 0.25
# 断崖阈值（绝对）：相邻分数落差超过该分值即截断
RERANK_GAP_ABS: float = 0.5


def step_1_merge_rrf_mcp(state):
    """将 RRF 本地召回与 MCP 联网结果统一为 {text, chunk_id, title, source, url} 结构。"""
    rrf_chunks = state.get("rrf_chunks", [])
    web_search_docs = state.get("web_search_docs", [])

    chunks_list = []
    # 本地召回片段（source=local）
    for chunk in rrf_chunks:
        entity = chunk.get('entity')
        chunks_list.append({
            "chunk_id": entity.get('chunk_id'),
            "text": entity.get('content'),
            "title": entity.get('title'),
            "source": "local",
            "url": ""
        })
    # 联网检索片段（source=web）
    for doc in web_search_docs:
        chunks_list.append({
            "chunk_id": "",
            "text": doc.get("snippet"),
            "title": doc.get("title"),
            "source": "web",
            "url": doc.get("url")
        })

    logger.info(f"多路数据融合，最终结果为:{chunks_list}")
    return chunks_list


def step_2_rerank_doc_list(doc_list, state):
    """用 Cross-Encoder 对候选片段与问题做相关性打分并降序排列。"""
    rewritten_query = state.get("rewritten_query") or state.get("original_query")
    text_list = [doc['text'] for doc in doc_list]

    rerank = get_reranker_model()
    # 构造 (query, doc) 配对；normalize=True 将分压缩到 0~1，避免不同 query 间分差尺度不一致
    question_pairs = [[rewritten_query, text] for text in text_list]
    scores = rerank.compute_score(question_pairs, normalize=True)

    doc_list_with_score = []
    for score, item in zip(scores, doc_list):
        item['score'] = score
        doc_list_with_score.append(item)

    doc_list_with_score.sort(key=lambda x: x['score'], reverse=True)
    logger.info(f"已经完成排序和打分！最终结果为：{doc_list_with_score}")
    return doc_list_with_score


def step_3_topk_and_gap(rerank_score_list):
    """动态 TopK：结合硬上下限与断崖检测，避免把质量骤降的尾部片段带入上下文。"""
    max_topk = RERANK_MAX_TOPK
    min_topk = RERANK_MIN_TOPK
    gap_abs = RERANK_GAP_ABS
    gap_ratio = RERANK_GAP_RATIO

    topk = min(max_topk, len(rerank_score_list))
    if topk > min_topk:
        # 双指针扫描相邻分差：触发绝对/相对断崖即截断
        for index in range(min_topk - 1, topk - 1):
            score_1 = rerank_score_list[index].get("score", 0.0)
            score_2 = rerank_score_list[index + 1].get("score", 0.0)
            gap = score_1 - score_2
            # abs(score_1)+1e-6 防止分母为 0（含负值场景）
            rel = gap / (abs(score_1) + 1e-6)
            if gap >= gap_abs or rel >= gap_ratio:
                logger.info(f"数据集合{index}和{index + 1}的位置发生了断崖，结束循环！！")
                topk = index + 1
                break

    topk_doc_list = rerank_score_list[:topk]
    logger.info(f"最终截取的长度：{topk},截取的内容:{topk_doc_list}")
    return topk_doc_list


def node_rerank(state):
    """
    节点功能：使用 Cross-Encoder 模型对 RRF 后的结果进行精确打分重排。
    """
    add_running_task(state["session_id"], "node_rerank", state.get("is_stream"))

    doc_list = step_1_merge_rrf_mcp(state)
    rerank_score_list = step_2_rerank_doc_list(doc_list, state)
    final_doc_list = step_3_topk_and_gap(rerank_score_list)

    state["reranked_docs"] = final_doc_list
    add_done_task(state['session_id'], "node_rerank", state.get("is_stream"))
    return state
