from typing_extensions import TypedDict
from typing import List
import copy

class QueryGraphState(TypedDict):
    """
    QueryGraphState 定义了整个查询流程中流转的数据结构。
    """
    session_id: str  # 会话唯一标识
    original_query: str  # 用户原始问题

    # 检索过程中的中间数据
    embedding_chunks: list  # 普通向量检索回来的切片
    hyde_embedding_chunks: list  # HyDE 检索回来的切片
    web_search_docs: list  # 网络搜索回来的文档

    # 排序过程中的数据
    rrf_chunks: list  # RRF 融合排序后的切片
    reranked_docs: list  # 重排序后的最终 Top-K 文档

    # 生成过程中的数据
    prompt: str  # 组装好的 Prompt
    answer: str  # 最终生成的答案

    # 辅助信息
    item_names: List[str]  # 提取出的知识主题/文档主题名称
    rewritten_query: str  # 改写后的问题
    history: list  # 历史对话记录
    is_stream: bool  # 是否流式输出标记

    # 意图路由（客服业务化新增）
    intent: str  # Triage 意图分类结果：order_status/logistics/policy/complex_aftersale/chitchat
    order_id: str  # 从对话中提取的订单号（DD+12位数字），无则为空
    confidence: float  # 意图分类置信度（0-1），低于阈值会兜底到 FAQ 主链路
    escalate_reason: str  # 转人工原因（P4 使用：低置信/多轮未解决/用户要求/涉及赔付）

    # 置信度/转人工（P4 新增，必须定义在 TypedDict 中 LangGraph 才会传递）
    escalate: bool  # 是否触发转人工（confidence_check 输出）
    retrieval_confidence: float  # 检索 top1 重排分数（confidence_check 输出）
    escalate_ticket: dict  # 转人工工单对象（escalation 输出）

    # 图片 URL 列表（answer_output 输出，SSE 配图用）
    image_urls: list


# ========================
# 默认状态（全部为空）
# ========================
query_graph_default_state: QueryGraphState = {
    "session_id": "",
    "original_query": "",
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "web_search_docs": [],
    "rrf_chunks": [],
    "reranked_docs": [],
    "prompt": "",
    "answer": "",
    "item_names": [],
    "rewritten_query": "",
    "history": [],
    "is_stream": False,
    "intent": "",
    "order_id": "",
    "confidence": 0.0,
    "escalate_reason": "",
    "escalate": False,
    "retrieval_confidence": 0.0,
    "escalate_ticket": {},
    "image_urls": []
}


# ========================
# 创建默认状态（可覆盖）
# ========================
def create_query_default_state(**overrides) -> QueryGraphState:
    """
    创建查询流程的默认状态，支持覆盖字段
    """
    state = copy.deepcopy(query_graph_default_state)
    state.update(overrides)
    return state


# ========================
# 获取干净状态
# ========================
def get_query_default_state() -> QueryGraphState:
    return copy.deepcopy(query_graph_default_state)


# ========================
# ✅ 状态复制函数（你要的）
# ========================
def copy_query_state(state: QueryGraphState, **overrides) -> QueryGraphState:
    """
    复制现有状态并可覆盖字段，深拷贝，不污染原数据
    """
    new_state = copy.deepcopy(state)
    new_state.update(overrides)
    return new_state

