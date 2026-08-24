"""让 pytest 能从项目根目录导入 app 包（CI / 本地通用）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
