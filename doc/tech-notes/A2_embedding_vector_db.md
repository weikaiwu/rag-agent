# Embedding 与向量数据库

## 是什么
Embedding（向量化）是把文本映射成一串浮点数向量（如 1024 维）的过程，语义相近的文本向量距离也近。向量数据库（如 Milvus、Qdrant、Chroma）负责存储这些向量并提供「按相似度（通常是余弦距离）检索 TopK」的能力。

## 核心概念
- **稠密向量（Dense）**：模型整体语义编码，能捕捉语义相似（"手机没电了" 和 "电池耗尽" 近）。
- **稀疏向量（Sparse）**：类似 BM25 的关键词权重（如 SPLADE、BGE-M3 的稀疏输出），擅长精确关键词匹配。
- **相似度度量**：余弦相似度（COSINE）、内积（IP）、欧氏距离（L2）。
- **索引类型**：Milvus 里常用 HNSW（图索引，快但占内存）、IVF（聚类，省内存）。

## 与本项目的关系
本项目用 **BGE-M3** 作为 embedding 模型，一次推理同时产出「稠密向量 + 稀疏向量」：
- 稠密向量存为 `FLOAT_VECTOR` 字段，用 HNSW + COSINE 检索；
- 稀疏向量存为 `SPARSE_FLOAT_VECTOR` 字段，用 `SPARSE_INVERTED_INDEX` + IP 检索；
- 两路结果在 `node_rrf` 里做倒数排名融合。

## 选型对比
- Chroma：轻量、单机、适合原型；
- Qdrant：Rust 实现、性能好；
- Milvus：分布式、生产首选、运维重；
- pgvector：已有 Postgres 时零额外组件。
