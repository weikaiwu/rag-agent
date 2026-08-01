# RAGAS 质量评估 · 运行说明

本目录下的 `ragas_eval.py` 用于对「技术文档智能问答系统」做 **RAGAS 自动评估**，
复用线上同一套 LLM（`get_llm_client()`）与 BGE-M3 嵌入（`generate_embeddings()`），
保证评估口径与生产链路一致。

## 前置条件
1. 项目根目录 `.env` 已配置好 LLM 与 Milvus 连接（生产环境本来就在用）。
2. Milvus 服务处于运行状态（评估会从真实向量库检索，VM 上需先启动）。
3. Python 依赖已安装（见下方）。

## 安装依赖
依赖已钉版写入 `pyproject.toml`（`ragas==0.2.15` + `datasets`），直接同步即可：

```bash
uv sync
```

> ⚠️ 不要手动 `uv add ragas`，否则可能装到 0.4.x 最新版——在你当前的
> `langchain>=1.3.11` 环境下会 import 失败（ragas 依赖已被新版 langchain 移除的
> `vertexai` 模块）。评估脚本已内置兼容补丁，无需降级 langchain。

## 运行
```bash
python -m app.eval.ragas_eval
```

## 输出
- 终端打印 4 个指标：faithfulness / answer_relevancy / context_precision / context_recall
- 同时生成 `app/eval/ragas_report.json` 落盘，便于存档与前后对比。

## 自定义数据集
编辑 `app/eval/sample_dataset.json`，每条需含以下字段：
- `question`：用户问题
- `contexts`：检索到的上下文片段（字符串列表）
- `answer`：系统生成的答案
- `ground_truth`：标准答案（用于 context_recall 等需要参考的指标）

## 已知坑
- **`node_search_embedding_hyde.py:4` 残留导入**：
  `from langchain_classic.chains.question_answering.map_reduce_prompt import messages`
  这一行是无效残留（`langchain_classic` 模块不存在，且 `messages` 从未被使用）。
  若运行评估或导入该模块时报 `ModuleNotFoundError: langchain_classic`，
  **直接删除第 4 行即可**，不影响任何功能。
- 评估会真实调用 LLM 与 Milvus，注意 API 额度消耗与网络连通性。
