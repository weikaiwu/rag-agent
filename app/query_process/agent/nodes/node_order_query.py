"""
订单查询 Agent 节点（node_order_query）—— 客服业务化 P3

作用：order_status / logistics 意图走动态数据链路——查 MySQL 订单库返回真实订单状态。
架构要点：**静态知识（退换货政策/FAQ）走 RAG，动态数据（订单状态/物流）走 API 查库**，
动态数据不进知识库，避免"数据过期"。

处理分支：
1. 无订单号       -> 追问用户提供（不硬查）
2. 查到订单       -> LLM 基于订单数据组装自然语言回复（带"来自订单系统"溯源）
3. 查不到订单     -> 明确告知核对订单号（绝不编造）
4. 数据库异常     -> 降级提示"系统繁忙"（不挂流程）
"""
import json
import re

from langchain_core.messages import HumanMessage

from app.conf.mysql_config import mysql_config
from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.lm.lm_utils import get_llm_client, invoke_llm_with_metrics
from app.utils.task_utils import add_running_task, add_done_task

# 订单号格式：DD + 12 位数字
ORDER_ID_PATTERN = re.compile(r"DD\d{12,}")

# 订单状态 -> 中文（与 data/mock_orders.sql 的 ENUM 保持一致）
ORDER_STATUS_CN = {
    "pending_payment": "待付款",
    "paid": "已付款",
    "shipped": "已发货",
    "delivering": "运输中",
    "completed": "已完成",
    "refunding": "退款中",
    "refunded": "已退款",
    "cancelled": "已取消",
}
# 退款状态 -> 中文
REFUND_STATUS_CN = {
    "none": "无",
    "applied": "退款申请中",
    "approved": "退款审核通过",
    "refunded": "已退款",
    "rejected": "退款被拒绝",
}

# 固定话术（不用 LLM，保证稳定）
ASK_ORDER_REPLY = "您好，查询订单需要订单号，麻烦您提供一下（订单号以 DD 开头）。"
NOT_FOUND_REPLY = "您好，没有查询到订单 {order_id}，麻烦您核对一下订单号是否正确。"
DB_DOWN_REPLY = "抱歉，订单系统暂时繁忙，请您稍后再试，或者我为您转接人工客服。"


def extract_order_id(text: str) -> str | None:
    """从文本中提取订单号（DD+12位数字），没有返回 None"""
    if not text:
        return None
    m = ORDER_ID_PATTERN.search(text)
    return m.group(0) if m else None


def query_order(order_id: str) -> dict | None:
    """
    查询订单库（MySQL order_service.orders）
    :param order_id: 订单号
    :return: 订单 dict（含中文状态映射）；未查到返回 None；DB 异常向上抛
    """
    import pymysql

    conn = pymysql.connect(
        host=mysql_config.host,
        port=mysql_config.port,
        user=mysql_config.user,
        password=mysql_config.password,
        database=mysql_config.db,
        connect_timeout=3,
        cursorclass=pymysql.cursors.DictCursor,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT order_no, product_name, sku, amount, status, logistics_company, "
                "tracking_no, logistics_status, refund_status, created_at, updated_at "
                "FROM orders WHERE order_no = %s",
                (order_id,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        return None
    # 状态码转中文
    row["status_cn"] = ORDER_STATUS_CN.get(row["status"], row["status"])
    row["refund_status_cn"] = REFUND_STATUS_CN.get(row["refund_status"], row["refund_status"])
    return row


def build_answer(order: dict, intent: str, query: str) -> str:
    """
    用 LLM 基于订单系统返回的真实数据组装自然语言回复
    数据为事实，LLM 只做润色——防止模型编造订单信息
    """
    # 传给 LLM 的字段（剔除不需要的原始字段）
    order_for_llm = {
        "order_no": order["order_no"],
        "product_name": order["product_name"],
        "status_cn": order["status_cn"],
        "logistics_company": order.get("logistics_company"),
        "tracking_no": order.get("tracking_no"),
        "logistics_status": order.get("logistics_status"),
        "refund_status_cn": order.get("refund_status_cn"),
    }
    prompt = load_prompt(
        "order_query_answer",
        order_json=json.dumps(order_for_llm, ensure_ascii=False),
        intent="订单状态查询" if intent == "order_status" else "物流查询",
        query=query,
    )
    llm = get_llm_client(json_mode=False)
    response = invoke_llm_with_metrics(llm, [HumanMessage(content=prompt)], scene="order_query")
    return str(response.content or "").strip()


def node_order_query(state: dict) -> dict:
    """
    节点：订单查询 Agent（动态数据链路）
    输入：original_query / intent / order_id（triage 已提取）
    输出：answer（追问订单号 / 订单信息 / 未查到 / 系统繁忙）
    """
    function_name = "node_order_query"
    session_id = state.get("session_id", "")
    add_running_task(session_id, function_name, state.get("is_stream"))
    logger.info(f">>> [{function_name}] 开始执行，意图={state.get('intent')}")

    try:
        query = state.get("original_query", "")
        # 1. 订单号：优先用 triage 提取的，其次从问题里再捞一次
        order_id = (state.get("order_id") or "").strip() or extract_order_id(query)

        # 2. 无订单号 -> 追问
        if not order_id:
            state["answer"] = ASK_ORDER_REPLY
            state["escalate_reason"] = "missing_order_id_ask_user"
            logger.info(f"[{function_name}] 无订单号，追问用户")
            return state

        # 3. 查订单库
        order = query_order(order_id)
        if order is None:
            state["answer"] = NOT_FOUND_REPLY.format(order_id=order_id)
            state["escalate_reason"] = "order_not_found"
            logger.info(f"[{function_name}] 订单 {order_id} 未查到，明确告知")
            return state

        # 4. LLM 组装自然语言回复
        state["answer"] = build_answer(order, state.get("intent", "order_status"), query)
        state["escalate_reason"] = ""
        logger.info(f"[{function_name}] 订单 {order_id} 查询成功，已生成回复")
    except Exception as e:
        # 数据库异常/LLM 异常：降级不挂流程
        logger.error(f"[{function_name}] 查询异常，降级提示：{e}", exc_info=True)
        state["answer"] = DB_DOWN_REPLY
        state["escalate_reason"] = "db_or_llm_error"

    add_done_task(session_id, function_name, state.get("is_stream"))
    return state
