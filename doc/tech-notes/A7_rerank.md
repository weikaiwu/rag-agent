# 重排序 Rerank（Cross-Encoder）

## 是什么
检索阶段（bi-encoder，查询和文档分别编码）为了快，精度有限。Rerank 阶段用 **Cross-Encoder** 把「查询+文档」拼一起过一次模型，输出一个相关性分数，对 TopN 候选精排。精度和延迟的权衡点：检索取 20–50 个，重排只排这几十个，所以不慢。

## 常见模型
- **BGE-reranker**（如 `BGE-reranker-v2-m3`）：多语言、效果好，本项目采用；
- Cohere Rerank：API 服务，免部署；
- Jina reranker：长文本友好。

## 与本项目的关系
`node_rerank` 节点：对 RRF 融合后的 TopN 文档，用 BGE-reranker 逐一打分给 (query, doc) 对，按分数降序，再用「动态 TopK + 悬崖检测（cliff detection）」截断——如果第 K 和第 K+1 名分差突然拉大，就在 K 处截断，避免把弱相关片段塞进上下文。

## 注意
- Rerank 是同步推理，会加一点延迟，但只对几十个候选跑，TTFT 影响可控；
- 重排模型要和系统语言匹配（中文场景选多语言版）。
