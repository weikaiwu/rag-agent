"""
KPI 周报统计脚本（P5）

读取 eval/kpi_metrics.jsonl（node_metrics 落盘）+ eval/ttft_metrics.jsonl（TTFT 埋点），
计算公司客服指标体系，对照验收线：
  - 转人工率 <= 30%
  - AI 独立解决率 >= 70%
  - 首响时间 FRT P95 < 2s
  - 按意图维度看转人工分布（定位薄弱环节）

用法：python scripts/report_kpi.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# KPI/TTFT 落盘统一在 app/eval/ 下（与 node_metrics.py / node_answer_output.py 上溯 4 层路径一致）
KPI_PATH = Path(__file__).resolve().parent.parent / "app" / "eval" / "kpi_metrics.jsonl"
TTFT_PATH = Path(__file__).resolve().parent.parent / "app" / "eval" / "ttft_metrics.jsonl"

# 业务验收线（计划文档 P5）
ACCEPT_LINE = {
    "escalate_rate": 0.30,      # 转人工率 <= 30%
    "ai_solve_rate": 0.70,      # AI 独立解决率 >= 70%
    "frt_p95_ms": 2000,         # 首响时间 P95 < 2s
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    kpi = load_jsonl(KPI_PATH)
    ttft = load_jsonl(TTFT_PATH)

    if not kpi:
        print("❌ 没有 KPI 数据，先跑一批会话（见 README 或 scripts/eval_cs_smoke.py）")
        return

    total = len(kpi)
    escalated = [r for r in kpi if r.get("escalate")]
    resolved = [r for r in kpi if not r.get("escalate")]

    escalate_rate = len(escalated) / total
    ai_solve_rate = len(resolved) / total

    print("=" * 46)
    print("客服系统 KPI 周报（演示数据）")
    print("=" * 46)
    print(f"总会话数: {total}")
    print(f"  转人工会话: {len(escalated)} ({escalate_rate:.1%})")
    print(f"  AI 独立解决: {len(resolved)} ({ai_solve_rate:.1%})")

    # 按意图维度的转人工分布
    print("\n按意图维度的转人工分布:")
    by_intent: dict[str, dict] = {}
    for r in kpi:
        intent = r.get("intent") or "unknown"
        bucket = by_intent.setdefault(intent, {"total": 0, "escalate": 0})
        bucket["total"] += 1
        if r.get("escalate"):
            bucket["escalate"] += 1
    for intent, b in sorted(by_intent.items()):
        rate = b["escalate"] / b["total"] if b["total"] else 0
        print(f"  {intent:<18} {b['escalate']}/{b['total']} = {rate:.1%}")

    # 平均答案长度（token 成本代理）
    if resolved:
        avg_chars = statistics.mean(r.get("answer_chars", 0) for r in resolved)
        print(f"\n平均答案长度: {avg_chars:.0f} 字符（token 成本代理指标）")

    # FRT（TTFT）
    if ttft:
        frts = sorted(r["ttft_ms"] for r in ttft)
        p95 = frts[int(len(frts) * 0.95) - 1] if frts else 0
        avg = statistics.mean(frts)
        print(f"首响时间 FRT: avg={avg:.0f}ms, P95={p95:.0f}ms（样本 {len(frts)}）")
    else:
        print("\n首响时间 FRT: 无 TTFT 数据")

    # 验收线对照
    print("\n" + "=" * 46)
    print("验收线对照")
    print("=" * 46)
    checks = [
        ("转人工率 <= 30%", escalate_rate, ACCEPT_LINE["escalate_rate"], escalate_rate <= ACCEPT_LINE["escalate_rate"]),
        ("AI 独立解决率 >= 70%", ai_solve_rate, ACCEPT_LINE["ai_solve_rate"], ai_solve_rate >= ACCEPT_LINE["ai_solve_rate"]),
    ]
    for name, got, target, ok in checks:
        print(f"  {'✅' if ok else '❌'} {name}: 当前 {got:.1%}（目标 {target:.0%}）")
    if ttft:
        frts = sorted(r["ttft_ms"] for r in ttft)
        p95 = frts[int(len(frts) * 0.95) - 1] if frts else 0
        ok = p95 < ACCEPT_LINE["frt_p95_ms"]
        print(f"  {'✅' if ok else '❌'} 首响 P95 < 2s: 当前 {p95:.0f}ms（目标 2000ms）")


if __name__ == "__main__":
    main()
