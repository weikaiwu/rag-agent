"""
知识更新闭环脚本（P6）

读 eval/unresolved_questions.jsonl（node_escalation 自动记录），
按意图 + 关键词聚类，输出"高频未命中问题"复盘清单——
某类问题持续高频出现 = 已从偶发变成新场景，应补进正式知识库（FAQ）。

流程（公司标准）：未命中 -> 复盘 -> 补知识 -> 重测（重跑评估集）
用法：python scripts/knowledge_update.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

UNRESOLVED_PATH = Path(__file__).resolve().parent.parent / "app" / "eval" / "unresolved_questions.jsonl"
FAQ_DIR = Path(__file__).resolve().parent.parent / "data" / "faq"

# 关键词聚类（可扩展）：命中同一关键词的问题归为一类
KEYWORD_GROUPS = {
    "分期付款": ["分期"],
    "积分兑换": ["积分"],
    "保价/降价保护": ["保价", "降价保护"],
    "退款时效": ["退款", "到账"],
    "发票": ["发票", "专票"],
    "优惠券": ["优惠券", "券"],
}


def load_unresolved() -> list[dict]:
    if not UNRESOLVED_PATH.exists():
        return []
    return [json.loads(line) for line in UNRESOLVED_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def group_questions(records: list[dict]) -> list[dict]:
    """按关键词聚类，输出 (类名, 数量, 示例问题)"""
    groups: dict[str, list[str]] = defaultdict(list)
    for r in records:
        q = r.get("question", "")
        matched = False
        for label, kws in KEYWORD_GROUPS.items():
            if any(k in q for k in kws):
                groups[label].append(q)
                matched = True
                break
        if not matched:
            groups["其他"].append(q)
    result = []
    for label, questions in groups.items():
        result.append({"label": label, "count": len(questions), "samples": questions[:3]})
    result.sort(key=lambda x: x["count"], reverse=True)
    return result


def main() -> None:
    records = load_unresolved()
    if not records:
        print("✅ 暂无未命中问题（eval/unresolved_questions.jsonl 为空）")
        return

    print(f"未命中问题总数: {len(records)}\n")
    print("按意图分布:")
    by_intent = Counter(r.get("intent", "unknown") for r in records)
    for intent, cnt in by_intent.most_common():
        print(f"  {intent:<18} {cnt}")

    print("\n高频问题聚类（知识更新候选）:")
    for g in group_questions(records):
        print(f"  [{g['count']}条] {g['label']}")
        for s in g["samples"]:
            print(f"      - {s[:40]}")

    print("\n【建议动作】")
    print("  1. 连续 2 期出现在 TOP 聚类的问题 -> 写进 data/faq/ 对应主题文档")
    print("  2. 用 scripts/import_faq.py 增量灌库（幂等：同 item_name 先删后插）")
    print("  3. 重跑 scripts/eval_cs_smoke.py + scripts/report_kpi.py 验证转人工率下降")
    print(f"  FAQ 目录: {FAQ_DIR}")


if __name__ == "__main__":
    main()
