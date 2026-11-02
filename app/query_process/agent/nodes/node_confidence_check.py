"""
置信度检查节点（node_confidence_check）—— 客服业务化 P4 核心

作用：FAQ 检索链路里，答案生成前先判断"能不能答"。不能答就拒答/转人工，绝不硬答。

5 个转人工触发条件（对齐真实客服系统）：
1. user_request_human    用户主动要求人工（"转人工/人工客服/找客服"）
2. empty_retrieval       检索结果为空（知识库没覆盖）
3. low_confidence        检索 top1 重排分数低于阈值（低置信不硬答）
4. multi_turn_unresolved 多轮未解决（上一轮助手已拒答/未找到）
5. compensation_related  涉及赔付/复杂售后（complex_aftersale 意图 + 赔付关键词）

设计要点：
- 阈值不是拍脑袋：先用评估脚本分析"能答问题 vs 不能答问题"的分数分布再定（scripts/analyze_rerank_scores.py）
- complex_aftersale 意图阈值更宽松（-0.1），宁可早转人工，不冒险硬答
"""
from app.core.logger import logger
from app.utils.task_utils import add_running_task, add_done_task

# 用户主动要求人工的关键词
ESCAPE_WORDS = ("人工", "转人工", "人工客服", "找客服", "客服接")
# 涉及赔付/复杂售后关键词
COMPENSATION_WORDS = ("赔", "赔偿", "补偿", "投诉", "坏了", "损坏", "质量问题", "换新")
# 上一轮助手未解决的标记（用于多轮未解决判定）
UNRESOLVED_MARKERS = ("未找到", "未查询到", "转人工", "无法处理", "抱歉，我只能")

# 检索置信度阈值（BGE-Reranker normalize 后 0~1）
# 校准依据（scripts/analyze_rerank_scores.py 实测 20 条分布）：
#   - 知识库外问题（不能答）: max=0.082, avg=0.011 -> 阈值 0.2 可 100% 拦截
#   - 能答问题: 分布 0.001~0.998，少数 FAQ 有答案但检索召回不足分数极低（如"耳机质保多久"=0.001）
# 取舍：阈值定 0.2，宁可把"检索召回不足但可能有答案"的问题转人工，也不硬答（低分硬答=幻觉）
SCORE_THRESHOLD = 0.2
# 复杂售后意图的阈值放宽量：宁可早转人工
COMPLEX_AFTERSALE_RELAX = 0.1


def check_escalation(state: dict) -> tuple[bool, str]:
    """
    返回 (是否转人工, 触发原因)
    """
    query = state.get("original_query", "") or ""
    intent = state.get("intent", "")
    reranked_docs = state.get("reranked_docs", []) or []
    history = state.get("history", []) or []

    # 1. 用户主动要求人工
    if any(w in query for w in ESCAPE_WORDS):
        logger.info("[confidence_check] 用户主动要求人工")
        return True, "user_request_human"

    # 2. 检索结果为空
    if not reranked_docs:
        logger.info("[confidence_check] 检索结果为空，知识库未覆盖")
        return True, "empty_retrieval"

    # 3. 检索 top1 分数低于阈值
    top1_score = float(reranked_docs[0].get("score", 0) or 0)
    threshold = SCORE_THRESHOLD - (COMPLEX_AFTERSALE_RELAX if intent == "complex_aftersale" else 0)
    if top1_score < threshold:
        logger.info(f"[confidence_check] top1 分数 {top1_score:.3f} < 阈值 {threshold:.3f}，低置信")
        return True, f"low_confidence_{top1_score:.3f}"

    # 4. 多轮未解决：上一轮助手已拒答/未找到
    if history:
        last = history[-1]
        if last.get("role") == "assistant":
            last_text = last.get("text", "") or ""
            if any(m in last_text for m in UNRESOLVED_MARKERS):
                logger.info("[confidence_check] 多轮未解决（上一轮已拒答/未找到）")
                return True, "multi_turn_unresolved"

    # 5. 涉及赔付/复杂售后
    if intent == "complex_aftersale" and any(w in query for w in COMPENSATION_WORDS):
        logger.info("[confidence_check] 涉及赔付/复杂售后，直接转人工")
        return True, "compensation_related"

    return False, ""


def node_confidence_check(state: dict) -> dict:
    """
    节点：置信度检查
    输出：state['escalate']（是否转人工）+ state['escalate_reason']（原因）
         + state['retrieval_confidence']（top1 重排分数，供 KPI 统计用）
    """
    function_name = "node_confidence_check"
    session_id = state.get("session_id", "")
    add_running_task(session_id, function_name, state.get("is_stream"))

    # 记录检索置信度（top1 分数），无论是否转人工都落 state
    reranked_docs = state.get("reranked_docs", []) or []
    top1_score = float(reranked_docs[0].get("score", 0) or 0) if reranked_docs else 0.0
    state["retrieval_confidence"] = round(top1_score, 3)

    escalate, reason = check_escalation(state)
    state["escalate"] = escalate
    state["escalate_reason"] = reason
    logger.info(
        f"[{function_name}] escalate={escalate}, reason={reason or '无'}, "
        f"top1_score={state['retrieval_confidence']}"
    )

    add_done_task(session_id, function_name, state.get("is_stream"))
    return state
