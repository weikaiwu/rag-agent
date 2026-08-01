"""
RAGAS 全链路评估模块
===================
对本项目的 RAG 问答质量做自动化评估，覆盖四个核心指标：
  - faithfulness       忠实度    ：答案是否严格基于检索到的上下文，不编造事实
  - answer_relevancy   答案相关性：回答是否切题、有用
  - context_precision  上下文精确度：检索召回的内容是否真的相关（去噪能力）
  - context_recall     上下文召回率：标准答案所需信息是否被检索覆盖

设计原则：
  1. 完全复用项目已有的大模型客户端(get_llm_client)与 BGE-M3 向量客户端(generate_embeddings)，
     不引入任何额外模型配置，保证评估口径与线上一致。
  2. 评估数据集来自 sample_dataset.json（技术文档领域样例），可替换为自己的标注集。
  3. 结果同时打印并落盘为 ragas_report.json，便于写进简历 / README 作为质量证据。

运行：
  cd 项目根目录
  uv add ragas datasets        # 或 pip install ragas datasets
  python -m app.eval.ragas_eval

依赖：ragas>=0.2.0, datasets
"""
import json
import sys
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

def _ensure_ragas_importable():
    """
    兼容补丁：ragas 部分版本在导入时会引用 `langchain_community.chat_models.vertexai`，
    而本项目钉在 langchain 1.x（该模块已被新版移除），直接 import 会报 ModuleNotFoundError。
    这里不改动任何依赖版本，仅用 shim 补齐该子模块使 ragas 能正常导入。
    （运行期我们只用 ChatOpenAI，永远不会实例化 vertexai，故 shim 安全无害。）
    """
    import importlib
    import sys
    import types

    name = "langchain_community.chat_models.vertexai"
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
    except Exception:
        mod = types.ModuleType(name)
        mod.ChatVertexAI = object  # ragas 仅需在导入时解析到此名字
        sys.modules[name] = mod


_ensure_ragas_importable()
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from datasets import Dataset
except ImportError as e:  # 友好提示，避免裸 Traceback
    print("缺少依赖 ragas / datasets，请先执行：uv sync")
    print(f"原始错误：{e}")
    sys.exit(1)

from langchain_core.embeddings import Embeddings

from app.lm.lm_utils import get_llm_client
from app.lm.embedding_utils import generate_embeddings
from app.core.logger import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ProjectBGEEmbeddings(Embeddings):
    """把项目自带的 BGE-M3 稠密向量包装成 LangChain Embeddings，供 RAGAS 上下文类指标使用。"""

    def embed_documents(self, texts):
        return generate_embeddings(texts)["dense"]

    def embed_query(self, text):
        return generate_embeddings([text])["dense"][0]


def load_sample_dataset(path: Path | None = None) -> Dataset:
    """加载评估数据集（JSON 列表，字段：question/contexts/answer/ground_truth）。"""
    path = path or (PROJECT_ROOT / "app" / "eval" / "sample_dataset.json")
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return Dataset.from_list(rows)


def main():
    # 1. 用项目已有客户端构建 RAGAS 所需的 LLM 与 Embeddings 包装
    llm = LangchainLLMWrapper(get_llm_client())
    embeddings = LangchainEmbeddingsWrapper(ProjectBGEEmbeddings())

    # 2. 载入数据集
    dataset = load_sample_dataset()
    logger.info(f"载入评估样本 {len(dataset)} 条")

    # 3. 执行评估
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    # 4. 输出与落盘
    print("\n===== RAGAS 评估结果 =====")
    print(result)
    try:
        df = result.to_pandas()
        report_path = PROJECT_ROOT / "app" / "eval" / "ragas_report.json"
        df.to_json(report_path, orient="records", force_ascii=False, indent=2)
        print(f"\n报告已写入：{report_path}")
    except Exception as e:  # 落盘失败不影响评估结论
        logger.warning(f"结果保存失败（不影响评估）：{e}")


if __name__ == "__main__":
    main()
