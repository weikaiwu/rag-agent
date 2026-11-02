"""
重建 kb_chunks：清空后重导 doc/tech-notes/ 下的知识库语料。

为什么需要它：
    本项目入库逻辑是「按 item_name 增量删除」，不会自动清掉上一批不同 item_name
    的旧切片。改了语料（例如重写文档内容）后如果不整库重导，新旧切片会混库，
    检索结果里会同时召回旧版本片段。

用法（项目根目录 + 项目虚拟环境）：
    python scripts/reimport_tech_notes.py --clear        # 清空后全量重导（推荐）
    python scripts/reimport_tech_notes.py                # 只增量重导（不推荐，可能混库）
    python scripts/reimport_tech_notes.py --only A1      # 只导文件名含 A1 的，用于先试一篇

退出码：0 全部成功 / 1 参数或目录问题 / 2 有失败项
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SRC_DIR = PROJECT_ROOT / "doc" / "tech-notes"


def _load_graph():
    """延迟导入：避免 --help 也去加载模型/连接数据库。"""
    from app.import_process.agent.main_graph import kb_import_app
    from app.import_process.agent.state import get_default_state

    return kb_import_app, get_default_state


def clear_collection() -> None:
    """删除 kb_chunks（下次入库自动重建）。"""
    from pymilvus import connections, utility

    host = os.getenv("MILVUS_HOST", "192.168.196.100")
    port = os.getenv("MILVUS_PORT", "19530")
    collection = os.getenv("CHUNKS_COLLECTION", "kb_chunks")

    print(f"[清空] 连接 Milvus {host}:{port}")
    connections.connect(alias="default", host=host, port=port)
    try:
        if utility.has_collection(collection):
            utility.drop_collection(collection)
            print(f"[清空] 已删除 collection: {collection}")
        else:
            print(f"[清空] {collection} 不存在，跳过")
    finally:
        connections.disconnect("default")


def import_one(md_path: Path, out_root: Path, graph, get_state) -> bool:
    """跑一遍导入图处理单个 md，返回是否成功。"""
    task_id = str(uuid.uuid4())
    work_dir = out_root / task_id
    work_dir.mkdir(parents=True, exist_ok=True)

    local_file = work_dir / md_path.name
    shutil.copy2(md_path, local_file)

    state = get_state()
    state["task_id"] = task_id
    state["local_file_path"] = str(local_file)
    state["local_dir"] = str(work_dir)

    try:
        for event in graph.stream(state):
            for node_name in event:
                print(f"    · {node_name}")
        return True
    except Exception as e:  # 单篇失败不阻断整批
        print(f"    ✗ 失败: {type(e).__name__}: {e}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="重建 kb_chunks 知识库语料")
    parser.add_argument("--clear", action="store_true", help="先清空 kb_chunks 再导入")
    parser.add_argument("--only", default="", help="只处理文件名包含该子串的文档")
    args = parser.parse_args()

    if not SRC_DIR.is_dir():
        print(f"[错误] 语料目录不存在: {SRC_DIR}")
        return 1

    files = sorted(p for p in SRC_DIR.glob("*.md") if p.is_file())
    if args.only:
        files = [p for p in files if args.only in p.name]
    if not files:
        print(f"[错误] 没有匹配的文档（--only={args.only!r}）")
        return 1

    print(f"待导入 {len(files)} 篇，源目录: {SRC_DIR}")
    if args.clear:
        try:
            clear_collection()
        except Exception as e:
            print(f"[错误] 清空失败: {type(e).__name__}: {e}")
            print("       请确认 Milvus 可连（虚拟机是否已启动）")
            return 1

    out_root = PROJECT_ROOT / "output" / datetime.now().strftime("%Y%m%d") / "reimport"
    graph, get_state = _load_graph()
    ok: list[str] = []
    failed: list[str] = []

    for idx, md in enumerate(files, 1):
        print(f"\n[{idx}/{len(files)}] {md.name}")
        (ok if import_one(md, out_root, graph, get_state) else failed).append(md.name)

    print(f"\n{'=' * 50}")
    print(f"完成：成功 {len(ok)} / 失败 {len(failed)}")
    if failed:
        print("失败清单：")
        for name in failed:
            print(f"  - {name}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
