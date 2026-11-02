from langgraph.graph import StateGraph, END

from app.query_process.agent.nodes.node_answer_output import node_answer_output
from app.query_process.agent.nodes.node_item_name_confirm import node_item_name_confirm
from app.query_process.agent.nodes.node_rerank import node_rerank
from app.query_process.agent.nodes.node_rrf import node_rrf
from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde
from app.query_process.agent.nodes.node_web_search_mcp import node_web_search_mcp
from app.query_process.agent.nodes.node_query_kg import node_query_kg
from app.query_process.agent.nodes.node_triage import node_triage
from app.query_process.agent.nodes.node_order_query import node_order_query
from app.query_process.agent.nodes.node_confidence_check import node_confidence_check
from app.query_process.agent.nodes.node_escalation import node_escalation
from app.query_process.agent.nodes.node_metrics import node_metrics
from app.query_process.agent.state import QueryGraphState


builder=StateGraph(QueryGraphState)
# 添加节点
builder.add_node("node_triage",node_triage)  # 意图路由（客服业务化入口，P2）
builder.add_node("node_order_query",node_order_query)  # 订单查询 Agent（动态数据链路，P3）
builder.add_node("node_item_name_confirm",node_item_name_confirm)
builder.add_node("node_search_embedding",node_search_embedding)
builder.add_node("node_search_embedding_hyde",node_search_embedding_hyde)
builder.add_node("node_web_search_mcp",node_web_search_mcp)
builder.add_node("node_query_kg",node_query_kg)
builder.add_node("node_rrf",node_rrf)
builder.add_node("node_rerank",node_rerank)
builder.add_node("node_confidence_check",node_confidence_check)  # 置信度检查（低置信转人工，P4）
builder.add_node("node_escalation",node_escalation)  # 转人工（带完整上下文工单，P4）
builder.add_node("node_answer_output",node_answer_output)
# 设置入口：先做意图路由（Triage）
builder.set_entry_point("node_triage")
# Triage 路由：
#   - chitchat 已在节点内直接拒答（answer 非空）-> 直接输出
#   - order_status/logistics -> 订单查询 Agent（动态数据走 MySQL，P3）
#   - policy/complex_aftersale -> FAQ 检索链路（P4 置信度检查决定是否转人工）
def route_after_triage(state: QueryGraphState):
    if state.get("answer"):
        return "node_answer_output"
    if state.get("intent") in ("order_status", "logistics"):
        return "node_order_query"
    return "node_item_name_confirm"
builder.add_conditional_edges("node_triage",
                              route_after_triage,
                              {
                                  "node_answer_output": "node_answer_output",
                                  "node_order_query": "node_order_query",
                                  "node_item_name_confirm": "node_item_name_confirm"
                              })
# 订单查询 Agent 总是产出 answer（查到/追问/未查到/降级），直接输出
builder.add_edge("node_order_query", "node_answer_output")
# 可能会因为没有明确的item_name会提前结束返回提示
# 客服 FAQ 链路：封闭知识域，禁用联网检索（node_web_search_mcp）——
# 实测"分期付款"会被 web 检索成银行知识（rerank 0.527 分）导致 LLM 基于域外信息编造答案
def route_after_node_item_name_confirm(state: QueryGraphState):
    if state.get("answer"):
        return "node_answer_output"
    return "node_search_embedding", "node_search_embedding_hyde", "node_query_kg"
builder.add_conditional_edges("node_item_name_confirm",
                              route_after_node_item_name_confirm,
                                {
                                  "node_answer_output":"node_answer_output",
                                  "node_search_embedding":"node_search_embedding",
                                  "node_search_embedding_hyde":"node_search_embedding_hyde",
                                  "node_query_kg":"node_query_kg"
                               })
builder.add_edge("node_search_embedding", "node_rrf")
builder.add_edge("node_search_embedding_hyde", "node_rrf")
builder.add_edge("node_web_search_mcp", "node_rrf")
builder.add_edge("node_query_kg", "node_rrf")
builder.add_edge("node_rrf", "node_rerank")
# 重排后先做置信度检查：低置信/触发条件 -> 转人工；否则正常生成答案
builder.add_edge("node_rerank", "node_confidence_check")
def route_after_confidence_check(state: QueryGraphState):
    if state.get("escalate"):
        return "node_escalation"
    return "node_answer_output"
builder.add_conditional_edges("node_confidence_check",
                              route_after_confidence_check,
                              {
                                  "node_escalation": "node_escalation",
                                  "node_answer_output": "node_answer_output"
                              })
builder.add_edge("node_escalation", "node_answer_output")
# 会话结束前记录业务 KPI（P5）
builder.add_node("node_metrics", node_metrics)
builder.add_edge("node_answer_output", "node_metrics")
builder.add_edge("node_metrics", END)

query_app=builder.compile()