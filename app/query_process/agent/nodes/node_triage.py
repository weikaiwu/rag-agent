"""
意图路由节点（node_triage）—— 客服业务化 P2

作用：用户消息进来先做意图分类（5 类），路由到不同专才链路。
- order_status / logistics   -> 订单/物流查询（P3 接入订单库，当前先走 FAQ 检索链路）
- policy                     -> FAQ 知识库检索（现有链路）
- complex_aftersale          -> 复杂售后（P4 接入转人工，当前先走 FAQ 检索链路）
- chitchat                   -> 直接拒答（不回知识库，节省 token）

设计要点：
1. 分类逻辑独立成 classify_intent()，评估脚本直接复用，避免"能跑但测不了"。
2. 低置信兜底：confidence < 0.5 时强制走 FAQ 主链路（policy）——路由错的代价比答不到更大。
3. chitchat 直接给拒答答案，不进检索链路——这是 guardrails 思想（openai-cs-agents-demo 同款）。
4. JSON 解析失败/模型抽风时兜底为 policy，保证流程不挂。
"""
import json
import re

from langchain_core.messages import HumanMessage

from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.lm.lm_utils import get_llm_client, invoke_llm_with_metrics
from app.utils.task_utils import add_running_task, add_done_task

# 5 类意图白名单（与评估集 cs_eval_set.json 的 intent 标注保持一致）
INTENTS = ("order_status", "logistics", "policy", "complex_aftersale", "chitchat")

# 低置信兜底阈值：低于该值强制走 FAQ 主链路（宁可答不到，不能答错路由）
CONFIDENCE_THRESHOLD = 0.5

# 订单号格式：DD + 12 位数字（如 DD202608150001）
ORDER_ID_PATTERN = re.compile(r"DD\d{12,}")

# chitchat 拒答话术（与 FAQ 边界一致，不进检索）
CHITCHAT_REPLY = (
    "抱歉，我只能解答本店商品相关的订单、物流、退换货和售后问题，"
    "其他问题我无法处理哦。需要人工客服的话，我也可以帮您转接。"
)


def classify_intent(query: str, history_text: str = "") -> dict:
    """
    意图分类核心逻辑（独立函数，评估脚本 scripts/eval_intent.py 直接复用）
    :param query: 用户当前消息
    :param history_text: 对话历史（文本形式），用于指代消解，可传空
    :return: {"intent": str, "order_id": str|None, "confidence": float, "reason": str}
             任何异常都会兜底返回 intent=policy（走 FAQ 主链路，保证流程不挂）
    """
    prompt = load_prompt("triage_intent", history_text=history_text or "（无）", query=query)
    llm = get_llm_client(json_mode=True)
    response = invoke_llm_with_metrics(llm, [HumanMessage(content=prompt)], scene="triage")
    content = str(response.content or "")

    # 去掉模型可能包裹的 ```json ... ``` 标记
    if content.startswith("```json"):
        content = content.replace("```json", "").replace("```", "")
    content = content.strip()

    try:
        parsed = json.loads(content)
    except Exception as e:
        # 模型输出不是合法 JSON：兜底走 FAQ 主链路
        logger.warning(f"[triage] JSON 解析失败，兜底为 policy。原始输出：{content[:200]}，异常：{e}")
        return {"intent": "policy", "order_id": None, "confidence": 0.0, "reason": "parse_fallback"}

    # 意图白名单校验
    intent = str(parsed.get("intent", "")).strip()
    if intent not in INTENTS:
        logger.warning(f"[triage] 模型返回未知意图 [{intent}]，兜底为 policy")
        intent = "policy"

    # 订单号提取 + 格式校验（防止模型编造订单号）
    order_id = parsed.get("order_id")
    if order_id and ORDER_ID_PATTERN.fullmatch(str(order_id).strip()):
        order_id = str(order_id).strip()
    else:
        order_id = None

    # 置信度解析（非法值兜底 0）
    try:
        confidence = float(parsed.get("confidence", 0) or 0)
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    reason = str(parsed.get("reason", "")).strip()
    return {"intent": intent, "order_id": order_id, "confidence": confidence, "reason": reason}


def _build_history_text(history: list) -> str:
    """把历史对话记录转成文本（给 LLM 做指代消解）"""
    lines = []
    for chat in history or []:
        role = chat.get("role", "user")
        text = chat.get("text", "")
        if text:
            lines.append(f"{role}: {text}")
    return "\n".join(lines[-6:])  # 只取最近 6 轮，避免上下文过长


def node_triage(state: dict) -> dict:
    """
    节点：意图路由（Triage）
    输入 state：original_query / session_id / history
    输出 state：intent / order_id / confidence / escalate_reason
                chitchat 时直接写 answer（拒答），后续节点检测到 answer 会跳过检索
    """
    function_name = "node_triage"
    add_running_task(state.get("session_id", ""), function_name, state.get("is_stream"))
    logger.info(f">>> [{function_name}] 开始执行，原始问题：{state.get('original_query')}")

    try:
        history_text = _build_history_text(state.get("history", []))
        result = classify_intent(state.get("original_query", ""), history_text)

        state["intent"] = result["intent"]
        state["order_id"] = result.get("order_id") or ""
        state["confidence"] = result.get("confidence", 0.0)
        state["escalate_reason"] = ""
        logger.info(
            f"[{function_name}] 意图={state['intent']}, 订单号={state['order_id'] or '无'}, "
            f"置信度={state['confidence']:.2f}, 理由={result.get('reason')}"
        )

        # chitchat：直接拒答，不进检索链路（guardrails）
        if state["intent"] == "chitchat":
            state["answer"] = CHITCHAT_REPLY
            logger.info(f"[{function_name}] chitchat 直接拒答，跳过检索")
        else:
            # 非 chitchat 清空可能残留的 answer，走正常链路
            state.pop("answer", None)
    except Exception as e:
        logger.error(f"[{function_name}] 意图分类异常，兜底走 FAQ 主链路：{e}", exc_info=True)
        state["intent"] = "policy"
        state["order_id"] = ""
        state["confidence"] = 0.0
        state.pop("answer", None)

    add_done_task(state.get("session_id", ""), function_name, state.get("is_stream"))
    return state
