# Agent 评估与可观测性（Evaluation & Observability）

## 是什么
评估（Eval）是「自动打分系统好不好」，可观测性（Observability）是「运行时看清每一步发生了什么」。两者是 Agent 能「上线而非玩具」的关键。

## 评估层面
- **检索层**：context_precision / recall（见 A8 RAGAS）；
- **生成层**：faithfulness / answer_relevancy；
- **端到端**：用「黄金集（标准问答对）」「对抗集（刁难问题）」跑回归；
- **延迟**：TTFT（首字延迟）、总耗时。

## 可观测性工具
- **Tracing**：LangSmith、Langfuse、Phoenix 记录每次调用的输入/输出/耗时/链路；
- **日志**：结构化日志 + 关键埋点（如本项目 `[TTFT]` 日志）；
- **指标面板**：把 TTFT、成功率、各节点耗时可视化。

## 与本项目的关系
本项目已落地两块：
- **评估**：`app/eval/ragas_eval.py` 跑 RAGAS（faithfulness 0.833 等）；
- **可观测**：SSE 把每个节点进度实时推前端；`node_answer_output.py` 埋 `[TTFT]` 首字延迟到 `ttft_metrics.jsonl`，可统计均值（实测 ~507ms）。

这套「评估 + 埋点」正好对应 Agent 系统里常说的「全链路评估可观测」：单点指标看不出退化，只有把每一步的输入输出都留下来，才能定位是哪一环掉了。

## 建议演进
接 Langfuse 做 tracing，把每次问答的全链路（检索了哪些块、重排分多少）可视化，debug 和汇报都方便。
