"""
客服系统冒烟测试（P5 KPI 数据生成）

从评估集抽 20 条（每类 4 条）+ 3 条订单查询 + 2 条闲聊，走完整 query_app 图，
生成 eval/kpi_metrics.jsonl 数据，供 scripts/report_kpi.py 统计周报。

用法：python scripts/eval_cs_smoke.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.query_process.agent.main_graph import query_app

EVAL_SET = Path(__file__).resolve().parent.parent / "app" / "eval" / "cs_eval_set.json"

# 订单查询用例（走动态数据链路）
ORDER_CASES = [
    ("DD202608170006 现在到哪了？", "logistics"),
    ("我的订单 DD202608160004 退款退到哪了？", "order_status"),
    ("我买的键盘发货了吗？", "order_status"),
]
# 闲聊用例（guardrails）
CHITCHAT_CASES = ["你好", "帮我写首诗"]


def main() -> None:
    data = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    # 每类抽 4 条
    by_intent: dict[str, list] = defaultdict(list)
    for item in data:
        by_intent[item["intent"]].append(item)
    selected: list[dict] = []
    for intent, items in by_intent.items():
        selected.extend(items[:4])
    print(f"评估集抽样: {len(selected)} 条（5 类各 4 条）")

    results = {"total": 0, "escalate": 0}
    for idx, item in enumerate(selected):
        state = {
            "session_id": f"smoke-{idx:03d}",
            "original_query": item["question"],
            "is_stream": False,
        }
        result = query_app.invoke(state)
        results["total"] += 1
        if result.get("escalate"):
            results["escalate"] += 1
        print(f"  [{idx+1}/{len(selected)}] {item['intent']:<16} "
              f"escalate={result.get('escalate')} answer={result.get('answer','')[:30]}")

    # 订单查询用例
    for idx, (q, intent) in enumerate(ORDER_CASES):
        state = {"session_id": f"smoke-order-{idx}", "original_query": q, "is_stream": False}
        result = query_app.invoke(state)
        results["total"] += 1
        print(f"  [order-{idx}] {intent:<16} answer={result.get('answer','')[:40]}")

    # 闲聊用例
    for idx, q in enumerate(CHITCHAT_CASES):
        state = {"session_id": f"smoke-chat-{idx}", "original_query": q, "is_stream": False}
        result = query_app.invoke(state)
        results["total"] += 1
        print(f"  [chat-{idx}] chitchat          answer={result.get('answer','')[:30]}")

    print(f"\n完成: {results['total']} 个会话, 转人工 {results['escalate']} 个")
    print("KPI 数据已写入 eval/kpi_metrics.jsonl，运行 scripts/report_kpi.py 出周报")


if __name__ == "__main__":
    main()
