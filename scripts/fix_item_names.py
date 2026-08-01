"""
修复已入库文档的 item_name 字段（数据层面修复）

背景：
    之前 item_name_recognition.prompt 是旧口径，导致 32 份技术文档
    被误识别为具体的产品型号名。代码已修复为「主题识别」，
    但已入库的旧数据 item_name 仍是错的。本脚本原地修复 kb_chunks 与 item_name_collection。

运行环境：
    必须在能连接 Milvus + 调用 LLM 的机器上执行（即运行 import / query 服务的 VM）。
    在项目根目录执行：python scripts/fix_item_names.py
"""
import os
import sys

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.clients.milvus_utils import get_milvus_client
from app.conf.milvus_config import milvus_config
from app.lm.embedding_utils import generate_embeddings
from app.core.load_prompt import load_prompt
from app.lm.lm_utils import get_llm_client
from langchain_core.messages import HumanMessage, SystemMessage


def _norm_sparse(sv):
    """把 query 读回的稀疏向量统一成 {index: value} 字典，便于重新写入。
    pymilvus query 返回的稀疏向量可能是 CSR 格式 {"indices":[...],"values":[...]}，
    也可能是 SparseVector 对象或直接的 {index:value} 字典。"""
    if isinstance(sv, dict):
        if "indices" in sv and "values" in sv:
            return {int(i): float(v) for i, v in zip(sv["indices"], sv["values"])}
        return {int(k): float(val) for k, val in sv.items()}
    if hasattr(sv, "indices") and hasattr(sv, "values"):  # SparseVector 对象
        return {int(i): float(v) for i, v in zip(sv.indices, sv.values)}
    return sv


def recognize_topic(file_title: str, content_sample: str) -> str:
    """用域化后的提示词，让 LLM 识别文档主题"""
    context = content_sample[:2500]
    human = load_prompt('item_name_recognition', file_title=file_title, context=context)
    system = load_prompt('product_recognition_system')
    llm = get_llm_client(json_mode=False)
    resp = llm.invoke([HumanMessage(content=human), SystemMessage(content=system)])
    name = (resp.content or "").strip()
    # 兜底：空则回退 file_title
    return name or file_title


def main():
    client = get_milvus_client()
    chunks_coll = milvus_config.chunks_collection
    name_coll = milvus_config.item_name_collection

    # 1. 读取全部 chunks（含主键与向量）
    client.load_collection(chunks_coll)
    all_chunks = client.query(
        collection_name=chunks_coll,
        output_fields=["chunk_id", "file_title", "item_name", "content",
                       "title", "parent_title", "part", "dense_vector", "sparse_vector"],
        limit=10000,
    )
    if not all_chunks:
        print("⚠️ kb_chunks 为空，无需修复。")
        return

    # 2. 按 file_title 分组，识别正确主题
    groups = {}
    for c in all_chunks:
        groups.setdefault(c["file_title"], []).append(c)

    new_topic_by_file = {}
    for file_title, chunks in groups.items():
        sample = "\n\n".join(c.get("content", "") for c in chunks[:5])
        topic = recognize_topic(file_title, sample)
        new_topic_by_file[file_title] = topic
        print(f"  {file_title}  ->  {topic}")

    # 3. 原地更新 kb_chunks 的 item_name（整行 upsert：auto_id 主键需保留用于匹配，
    #    且 upsert 要求提供所有非空字段，所以复制整行只改 item_name）
    update_data = []
    for c in all_chunks:
        row = dict(c)  # 复制整行（含 chunk_id / 向量等所有字段）
        row["item_name"] = new_topic_by_file[c["file_title"]]
        row["sparse_vector"] = _norm_sparse(c.get("sparse_vector"))
        update_data.append(row)
    client.upsert(collection_name=chunks_coll, data=update_data)
    print(f"✅ 已更新 kb_chunks 中 {len(update_data)} 条记录的 item_name")

    # 4. 重建 item_name_collection（清空旧主题名，插入新主题向量）
    client.load_collection(name_coll)
    old = client.query(collection_name=name_coll, output_fields=["pk"], limit=10000)
    if old:
        old_ids = [r["pk"] for r in old]
        client.delete(collection_name=name_coll, ids=old_ids)
        print(f"🗑️  已删除 item_name_collection 中 {len(old_ids)} 条旧记录")
    # 去重新主题
    seen = set()
    items_to_insert = []
    for file_title, topic in new_topic_by_file.items():
        if topic in seen:
            continue
        seen.add(topic)
        emb = generate_embeddings([topic])
        items_to_insert.append({
            "file_title": file_title,
            "item_name": topic,
            "dense_vector": emb["dense"][0],
            "sparse_vector": emb["sparse"][0],
        })
    if items_to_insert:
        client.insert(collection_name=name_coll, data=items_to_insert)
        client.load_collection(name_coll)
    print(f"✅ 已重建 item_name_collection，写入 {len(items_to_insert)} 个主题")


if __name__ == "__main__":
    main()
