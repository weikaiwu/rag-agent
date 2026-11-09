# RAG 评估：RAGAS 框架

## 是什么
RAGAS 是业界主流的 RAG 自动化评估框架，不需要人工标注，用 LLM 当裁判给四个核心指标打分：
- **faithfulness（忠实度）**：答案是否只基于检索到的上下文，不胡编；
- **answer_relevancy（答案相关性）**：答案是否切题；
- **context_precision（上下文精度）**：检索到的片段里，相关的排前面没；
- **context_recall（上下文召回）**：该有的相关片段是否都检索到了。

## 为什么评估很重要
不评估就不知道 RAG 哪环弱：召回差（precision/recall 低）还是生成飘（faithfulness 低）。评估驱动优化，是项目「工程化」的体现。

## 与本项目的关系
`app/eval/ragas_eval.py` 复用项目自己的 LLM 和 BGE-M3 客户端，跑 `sample_dataset.json` 里的问题/上下文/答案/标准答案。实测样本：faithfulness 0.833、answer_relevancy 0.856、context_precision 1.0、context_recall 1.0。

- context 双 1.0 说明检索召回质量高；
- faithfulness 0.833 由单条样本被 RAGAS 判 0.5 拉低 → 可讲「评估发现波动→定位噪声片段→加动态 TopK 截断优化」的闭环故事。

## 注意
- 样本量要够（建议 10+），3 条样本均值置信度低；
- ragas 版本要和 langchain 兼容，新版本可能 import 报错（本项目钉 0.2.15）。
