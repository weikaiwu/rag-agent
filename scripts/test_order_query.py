"""
订单查询 Agent 验证脚本（P3）

覆盖 4 种分支：
1. 有订单号且查到（运输中）       -> 返回物流信息
2. 有订单号且查到（退款中）       -> 返回退款状态
3. 无订单号                       -> 追问订单号
4. 假订单号                       -> 明确告知未查到（不编造）

用法：python scripts/test_order_query.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.query_process.agent.nodes.node_order_query import node_order_query, extract_order_id


def run_case(name: str, query: str, intent: str = "order_status", order_id: str = "") -> None:
    state = {
        "session_id": f"test-{name}",
        "original_query": query,
        "intent": intent,
        "order_id": order_id,
        "is_stream": False,
    }
    result = node_order_query(state)
    print(f"\n=== {name} ===")
    print(f"  问题: {query}")
    print(f"  回答: {result.get('answer', '')}")
    print(f"  escalate_reason: {result.get('escalate_reason', '')}")


def main() -> None:
    # 分支 1：有订单号且查到（运输中，走物流）
    run_case("查到-物流", "我买的椅子物流到哪了？", intent="logistics", order_id="DD202608170006")
    # 分支 2：有订单号且查到（退款中）
    run_case("查到-退款", "我的订单退到哪一步了？", order_id="DD202608160004")
    # 分支 3：无订单号 -> 追问
    run_case("无订单号", "我昨天买的怎么还没发货？", intent="order_status")
    # 分支 4：假订单号 -> 明确未查到
    run_case("假订单号", "查一下订单 DD999999999999", order_id="DD999999999999")

    # 辅助函数单测
    assert extract_order_id("我的订单 DD202608150001 发货了吗") == "DD202608150001"
    assert extract_order_id("能退吗") is None
    assert extract_order_id("DD123456") is None  # 格式不完整
    print("\n✅ extract_order_id 正则单测通过")


if __name__ == "__main__":
    main()
