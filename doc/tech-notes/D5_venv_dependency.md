# 虚拟环境与依赖管理（uv / pip）

## 是什么
Python 项目必须隔离依赖，否则不同项目的包会互相污染。虚拟环境（venv / conda）给每个项目独立一套包；依赖文件（requirements.txt / pyproject.toml）锁定版本，保证别人复现。

## 工具对比
- **pip + requirements.txt**：经典，慢；
- **venv**：标准库自带，建隔离环境；
- **uv**：Rust 写的新型工具，装包速度比 pip 快 10–100 倍，用 `pyproject.toml` + `uv.lock` 锁版本，2025 起成主流。

## 与本项目的关系
本项目用 `pyproject.toml`（uv 风格）管理依赖：langgraph、milvus、flagembedding(BGE-M3)、openai-agents、ragas==0.2.15 等。部分包钉了版本（如 ragas）以避免和新 langchain 不兼容。

## 最佳实践
- 永远在虚拟环境里跑，别装全局；
- 锁版本（lock file），别写 `>=` 敞口导致他人装到不兼容新版；
- 定期更新但小步快跑，避免一次性大升级爆雷。
