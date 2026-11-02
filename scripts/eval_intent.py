"""
意图路由评估脚本（P2 验证）

用 50 条客服评估集（app/eval/cs_eval_set.json，5 类意图各 10 条）评估 Triage 分类准确率。
直接复用 node_triage.classify_intent，与线上逻辑完全一致。

用法：python scripts/eval_intent.py
输出：总体准确率 + 每类准确率 + 混淆矩阵（简要）
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.query_process.agent.nodes.node_triage import classify_intent

EVAL_SET = Path(__file__).resolve().parent.parent / "app" / "eval" / "cs_eval_set.json"
INTENTS = ("order_status", "logistics", "policy", "complex_aftersale", "chitchat")


def main() -> None:
    data = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    print(f"评估集共 {len(data)} 条\n")

    total = 0
    correct = 0
    per_class = defaultdict(lambda: {"total": 0, "correct": 0})
    confusion = defaultdict(Counter)  # gold -> {pred: count}

    for item in data:
        gold = item["intent"]
        question = item["question"]
        pred = classify_intent(question)["intent"]

        total += 1
        per_class[gold]["total"] += 1
        confusion[gold][pred] += 1
        if pred == gold:
            correct += 1
            per_class[gold]["correct"] += 1
        else:
            print(f"  ✗ [{gold} → {pred}] {question[:50]}")

    print(f"\n总体准确率: {correct}/{total} = {correct / total:.1%}\n")
    print("每类准确率:")
    for intent in INTENTS:
        t = per_class[intent]["total"]
        c = per_class[intent]["correct"]
        acc = c / t if t else 0
        print(f"  {intent:<18} {c}/{t} = {acc:.1%}")

    print("\n混淆矩阵 (gold → pred):")
    for gold in INTENTS:
        row = dict(confusion[gold])
        print(f"  {gold:<18} {row}")


if __name__ == "__main__":
    main()
