"""
业务 KPI 记录节点（node_metrics）—— 客服业务化 P5

作用：每个会话结束时，把"业务指标"落盘到 eval/kpi_metrics.jsonl（追加式），
供 scripts/report_kpi.py 统计出周报。

记录字段（对齐公司客服指标体系）：
- 转人工相关：escalate / escalate_reason / retrieval_confidence
  -> 转人工率 = 转人工会话 / 总会话；AI 独立解决率 = 未转人工会话 / 总会话
- 效率相关：answer_chars（答案长度，token 成本的代理指标；精确 token 在 P6 埋点）
- 意图相关：intent（按意图统计转人工分布，定位薄弱环节）

设计要点：
1. 独立文件追加式写入（与 TTFT 落盘同模式），不受日志级别限制
2. 会话级一条记录，统计脚本聚合
3. 不阻塞主流程：落盘失败仅记日志
"""
import json
from datetime import datetime
from pathlib import Path

from app.core.logger import logger
from app.utils.task_utils import add_running_task, add_done_task

# KPI 落盘路径：app/query_process/agent/nodes/ 上溯 4 层到项目根，再进 eval/
_KPI_LOG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "eval" / "kpi_metrics.jsonl"


def _append_kpi(record: dict) -> None:
    """追加写入 KPI 记录（失败不阻塞主流程）"""
    try:
        _KPI_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_KPI_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"[KPI] 指标落盘失败：{e}")


def node_metrics(state: dict) -> dict:
    """
    节点：业务 KPI 记录（每个会话结束执行一次）
    """
    function_name = "node_metrics"
    session_id = state.get("session_id", "")
    add_running_task(session_id, function_name, state.get("is_stream"))

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id,
        "intent": state.get("intent", ""),
        # 转人工指标
        "escalate": bool(state.get("escalate", False)),
        "escalate_reason": state.get("escalate_reason", ""),
        "retrieval_confidence": state.get("retrieval_confidence"),
        # 效率/成本代理指标
        "answer_chars": len(state.get("answer", "") or ""),
        # 业务来源标记：order_query 链路（动态数据）/ FAQ 链路（RAG）
        "channel": "order_query" if state.get("intent") in ("order_status", "logistics") else "faq",
    }
    _append_kpi(record)
    logger.info(f"[{function_name}] KPI 已记录: intent={record['intent']}, escalate={record['escalate']}")

    add_done_task(session_id, function_name, state.get("is_stream"))
    return state
