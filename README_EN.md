# Technical-Document QA System (RAG + Agent)

> A LangGraph-orchestrated RAG application for technical documentation (AI / LLM / Agent / Python, etc.): hybrid retrieval + reranking + streaming answers + multi-turn dialogue.

A complete Retrieval-Augmented Generation (RAG) system covering two LangGraph pipelines — **document ingestion** and **query reasoning** — with dense + sparse hybrid vector retrieval, HyDE hypothetical-document expansion, weighted RRF fusion, BGE cross-encoder reranking, Web Search tool-use (MCP), SSE streaming, and MongoDB-based multi-turn memory.

---

## ✨ Key Features

- **Dual-graph orchestration (LangGraph)**: ingestion graph (8 nodes) + query graph (7 nodes), with state flowing between nodes for clear, observable logic.
- **Hybrid retrieval**: BGE-M3 dual vectors (dense `HNSW + COSINE` / sparse `SPARSE_INVERTED_INDEX + IP`) retrieved in parallel.
- **HyDE expansion**: rewrite the query into a "hypothetical answer document" before vector search to improve semantic recall.
- **Weighted RRF fusion**: unified ranking across multiple recall paths (vector / HyDE / web / knowledge graph).
- **BGE reranking**: cross-encoder rerank + dynamic TopK + cliff-detection to filter noisy chunks.
- **Tool-use (MCP)**: Web Search via `openai-agents` StreamableHttp (DashScope), supplementing real-time / out-of-corpus knowledge.
- **Streaming UX**: SSE pushes retrieval progress (node-level visualization) + token-by-token answers, TTFT ~507ms.
- **Multi-turn memory**: MongoDB stores conversation history for contextual follow-ups.
- **Multimodal**: MinIO stores document images; answers can cite figures.
- **Evaluable**: built-in RAGAS evaluation (faithfulness / answer_relevancy / context_precision·recall).

---

## 🏗️ Architecture

Two FastAPI services:

- **Import service** (`:8000`) — PDF → Markdown (MinerU) → split → BGE-M3 dual-vector embed → Milvus.
- **Query service** (`:8001`) — entity/query rewrite → parallel recall (vector / HyDE / web / KG) → RRF fusion → BGE rerank → SSE streaming answer with multi-turn history.

External stores: Milvus (vectors), MongoDB (history), MinIO (images), Neo4j (optional KG), MCP server (web search).

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|------------|
| Orchestration | LangGraph 1.2.x |
| LLM | OpenAI-compatible API (configurable model) |
| Embedding / Rerank | BGE-M3 (dense + sparse), BGE cross-encoder reranker |
| Vector DB | Milvus (`pymilvus` 3.x) |
| Document parsing | MinerU (`magic-pdf`) |
| Object storage | MinIO |
| Conversation history | MongoDB |
| Tool-use | Web Search MCP (`openai-agents` StreamableHttp) |
| Evaluation | RAGAS 0.2.15 |
| API / Streaming | FastAPI + SSE |
| Runtime | Python ≥ 3.12 |

---

## 🚀 Quick Start

### Prerequisites
- Python ≥ 3.12
- Deployed Milvus, MongoDB, MinIO (deployment not included in this repo)
- Available LLM API (OpenAI-compatible), MinerU API, DashScope MCP credentials

### 1. Install dependencies
```bash
uv sync
# or: pip install -e .
```

### 2. Configure environment
```bash
cp .env.example .env   # then fill in real API keys and config
```

### 3. Start services (two terminals)
```bash
# Terminal 1: import service (port 8000)
python -m app.import_process.api.import_server

# Terminal 2: query service (port 8001)
python -m app.query_process.api.query_server
```

### 4. Open the chat UI
Browser: `http://127.0.0.1:8001/chat.html`

---

## 📊 Evaluation Metrics (real runs)

Run `python -m app.eval.ragas_eval` (samples in `app/eval/sample_dataset.json`):

| Metric | Value |
|--------|-------|
| faithfulness | `0.898` |
| answer_relevancy | `0.761` |
| context_precision | `0.900` |
| context_recall | `1.000` |
| avg. TTFT | `~507 ms` (5 Q&A runs, range 368–659 ms) |

---

## 🐳 Container Deployment

One command to bring up all dependencies (Milvus / etcd / MinIO / MongoDB) and app services:

```bash
cp .env.example .env
docker compose up -d
```

- Import: http://localhost:8000/import
- Query: http://localhost:8001/chat.html
- On first run, create MinIO bucket `knowledge-base-files` in the MinIO console.

---

## 🗺️ Roadmap

- Containerization & one-command deployment (`Dockerfile` + `docker-compose.yml`).
- Stronger evaluation loop: cross-document / multi-hop RAGAS samples, regression baseline.
- Broader Agent capability: planning loop (ReAct reflection) and multi-agent collaboration.
- Performance & observability: caching layer, async task queue, end-to-end metrics (success rate / latency / token usage).
