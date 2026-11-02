"""
转人工节点（node_escalation）—— 客服业务化 P4 核心

作用：触发转人工时，生成结构化工单并落库（模拟"人工坐席工作台"），
且**携带完整上下文**——客户不用重新描述问题（这是公司客服最在意的体验点）。

工单内容：
- 用户问了什么（user_query）
- AI 已答了什么（ai_answer，如有）
- 意图分类（intent）
- 涉及订单号（order_id）
- 触发原因（escalate_reason）
- 检索置信度（retrieval_top1_score）
- 会话 ID / 时间

落库：MongoDB kb002 库 escalation_tickets 集合（与 chat_message 同库，独立集合）。
"""
import uuid
from datetime import datetime

from pymongo import MongoClient

from app.core.logger import logger
from app.utils.task_utils import add_running_task, add_done_task

# 转人工话术（固定，不用 LLM 生成，保证稳定）
ESCALATION_REPLY = (
    "非常抱歉没能帮您解决问题，已为您转接人工客服。"
    "您的对话上下文（问题、已尝试的回答、订单信息）已经同步给客服，"
    "无需重复描述，请稍候。"
)

_ESCALATION_COLLECTION = "escalation_tickets"


def save_escalation_ticket(ticket: dict) -> None:
    """工单落库 MongoDB（失败不抛异常，降级为日志记录）"""
    try:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        client = MongoClient(os.getenv("MONGO_URL"), serverSelectionTimeoutMS=3000)
        db = client[os.getenv("MONGO_DB_NAME", "kb002")]
        db[_ESCALATION_COLLECTION].insert_one(ticket)
        client.close()
        logger.info(f"[escalation] 工单已落库：{ticket.get('ticket_id')}")
    except Exception as e:
        logger.error(f"[escalation] 工单落库失败（降级，不影响主流程）：{e}")


def node_escalation(state: dict) -> dict:
    """
    节点：转人工
    输入：state 中的 original_query / answer / intent / order_id / escalate_reason / retrieval_confidence
    输出：state['answer']（转人工话术）+ state['escalate_ticket']（工单对象）
    """
    function_name = "node_escalation"
    session_id = state.get("session_id", "")
    add_running_task(session_id, function_name, state.get("is_stream"))

    ticket = {
        "ticket_id": f"TCK{uuid.uuid4().hex[:12].upper()}",
        "session_id": session_id,
        "user_query": state.get("original_query", ""),
        "ai_answer": state.get("answer", ""),  # 若上游已尝试回答过，带上
        "intent": state.get("intent", ""),
        "order_id": state.get("order_id", ""),
        "escalate_reason": state.get("escalate_reason", ""),
        "retrieval_top1_score": state.get("retrieval_confidence"),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    save_escalation_ticket(ticket)
    # 同步记录"未命中问题"（知识更新闭环的数据源，P6）
    save_unresolved_question(state, ticket)

    state["escalate_ticket"] = ticket
    state["answer"] = ESCALATION_REPLY
    logger.info(
        f"[{function_name}] 工单 {ticket['ticket_id']} 已生成，"
        f"原因：{ticket['escalate_reason']}"
    )

    add_done_task(session_id, function_name, state.get("is_stream"))
    return state


# ===================== 未命中问题记录（P6 知识闭环数据源） =====================
import json as _json
from pathlib import Path as _Path

_UNRESOLVED_PATH = _Path(__file__).resolve().parent.parent.parent.parent.parent / "app" / "eval" / "unresolved_questions.jsonl"


def save_unresolved_question(state: dict, ticket: dict) -> None:
    """把 AI 答不上来（转人工）的问题追加到未命中清单，供 knowledge_update.py 复盘"""
    try:
        _UNRESOLVED_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": ticket["created_at"],
            "session_id": ticket["session_id"],
            "question": ticket["user_query"],
            "intent": ticket["intent"],
            "reason": ticket["escalate_reason"],
        }
        with open(_UNRESOLVED_PATH, "a", encoding="utf-8") as f:
            f.write(_json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"[escalation] 未命中问题已记录：{record['question'][:30]}")
    except Exception as e:
        logger.warning(f"[escalation] 未命中问题记录失败：{e}")
