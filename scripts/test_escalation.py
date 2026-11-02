"""
置信度拒答/转人工验证脚本（P4）

两部分：
1. 单元验证：5 个转人工触发条件逐一构造 state 验证
2. 端到端验证：走完整 query_app 图，知识库外问题必须"不硬答"（转人工）

用法：python scripts/test_escalation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.query_process.agent.nodes.node_confidence_check import node_confidence_check


def base_state(**overrides) -> dict:
    state = {
        "session_id": "test-escalation",
        "original_query": "支持七天无理由退货吗？",
        "intent": "policy",
        "history": [],
        "reranked_docs": [{"text": "七天无理由退货...", "score": 0.9, "source": "local", "title": "退换货"}],
        "is_stream": False,
    }
    state.update(overrides)
    return state


def test_unit() -> None:
    """5 个触发条件 + 1 个不触发"""
    cases = [
        ("用户要求人工", {"original_query": "帮我转人工客服"}, "user_request_human"),
        ("检索为空", {"reranked_docs": []}, "empty_retrieval"),
        ("低分<0.2", {"reranked_docs": [{"text": "x", "score": 0.05}]}, "low_confidence"),
        ("多轮未解决", {"history": [{"role": "assistant", "text": "未找到相关信息"}]}, "multi_turn_unresolved"),
        ("赔付关键词", {"original_query": "我的耳机坏了你们要赔我", "intent": "complex_aftersale"}, "compensation_related"),
        ("高分正常回答", {"reranked_docs": [{"text": "y", "score": 0.9}]}, None),
    ]
    print("\n===== 单元验证：转人工触发条件 =====")
    passed = 0
    for name, overrides, expect in cases:
        state = base_state(**overrides)
        out = node_confidence_check(state)
        got = out.get("escalate_reason", "")
        # 低分触发 reason 带分数后缀（low_confidence_0.050），用前缀匹配
        ok = got.startswith(expect) if expect else (not out.get("escalate"))
        print(f"  {'✅' if ok else '❌'} {name}: escalate={out.get('escalate')} reason={got or '不转'}")
        passed += ok
    print(f"单元验证: {passed}/{len(cases)} 通过\n")
    return passed == len(cases)


def test_end_to_end() -> None:
    """端到端：客服域内但知识库没有答案的问题 -> 必须转人工，绝不硬答"""
    from app.query_process.agent.main_graph import query_app

    # 域内但知识库未覆盖（FAQ 里没有这些规则）
    unanswerable = [
        "你们家支持分期付款吗？",
        "会员积分可以换成现金吗？",
        "双十一会不会有保价？",
    ]
    print("===== 端到端验证：知识库外问题 =====")
    all_ok = True
    for q in unanswerable:
        state = {
            "session_id": f"test-e2e-{q[:6]}",
            "original_query": q,
            "is_stream": False,
        }
        result = query_app.invoke(state)
        answer = result.get("answer", "")
        escalate = result.get("escalate", False)
        reason = result.get("escalate_reason", "")
        # 判定"不硬答"：要么转人工，要么拒答（不允许给出看似合理的知识库外答案）
        hard_answered = answer and "转接人工" not in answer and "抱歉" not in answer and len(answer) > 20
        ok = escalate or not hard_answered
        print(f"  {'✅' if ok else '❌'} [{reason or '无'}] {q[:20]} -> {answer[:40]}")
        all_ok = all_ok and ok
    print(f"端到端: {'全部不硬答 ✅' if all_ok else '有硬答 ❌'}")
    return all_ok


if __name__ == "__main__":
    u = test_unit()
    e = test_end_to_end()
    print(f"\n总结: 单元{'✅' if u else '❌'} / 端到端{'✅' if e else '❌'}")
