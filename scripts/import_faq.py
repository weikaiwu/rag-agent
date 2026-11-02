"""
FAQ 知识库灌库脚本（电商售后客服系统）

把 data/faq/*.md 通过现有导入图（kb_import_app）灌入 Milvus。
复用线上导入链路：md → 切片 → BGE-M3 向量化 → kb_chunks 入库，零新代码。

用法：python scripts/import_faq.py
"""
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.state import get_default_state

FAQS_DIR = Path(__file__).resolve().parent.parent / "data" / "faq"
OUT_ROOT = Path(__file__).resolve().parent.parent / "output" / datetime.now().strftime("%Y%m%d")


def main() -> None:
    md_files = sorted(FAQS_DIR.glob("*.md"))
    if not md_files:
        print("❌ data/faq/ 下没有 md 文件")
        return

    print(f"共 {len(md_files)} 个 FAQ 文档待导入: {[f.name for f in md_files]}")
    for md in md_files:
        task_id = str(uuid.uuid4())
        dir_path = OUT_ROOT / task_id
        dir_path.mkdir(parents=True, exist_ok=True)
        local_file_path = dir_path / md.name
        shutil.copyfile(md, local_file_path)

        state = get_default_state()
        state["task_id"] = task_id
        state["local_file_path"] = str(local_file_path)
        state["local_dir"] = str(dir_path)

        print(f"\n>>> 导入: {md.name} (task={task_id[:8]})")
        try:
            for event in kb_import_app.stream(state):
                for node_name in event:
                    print(f"    ✅ 节点完成: {node_name}")
            print(f"<<< {md.name} 导入成功")
        except Exception as e:
            print(f"❌ {md.name} 导入失败: {e}")
            continue


if __name__ == "__main__":
    main()
