"""
一次性脚本：在重新入库前清空 Milvus 的 kb_chunks collection。
仅删除该 collection（下次入库会自动重建），不影响其他数据。

用法（在项目虚拟环境中运行）：
    cd <项目根目录>
    python scripts/clear_milvus.py

说明：
    本项目入库逻辑是「按 item_name 增量删除」，不会自动清掉上一批
    不同 item_name 的旧文档。重新导入知识库前，
    必须先手动清空，否则新旧文档混库、污染检索与评估结果。
"""
import os
from pymilvus import connections, utility

# 默认值取自项目实际配置（VM 上的 Milvus）
MILVUS_HOST = os.getenv("MILVUS_HOST", "192.168.196.100")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION = os.getenv("CHUNKS_COLLECTION", "kb_chunks")


def main():
    print(f"连接 Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    try:
        if utility.has_collection(COLLECTION):
            utility.drop_collection(COLLECTION)
            print(f"[OK] 已删除 collection: {COLLECTION}（下次入库会自动重建）")
        else:
            print(f"[INFO] collection 不存在，无需删除: {COLLECTION}")
    finally:
        connections.disconnect("default")


if __name__ == "__main__":
    main()
