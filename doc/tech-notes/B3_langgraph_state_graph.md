# LangGraph：状态图编排

## 是什么
LangGraph 是 LangChain 团队出的「用图编排多步 LLM 流程」框架。把流程拆成 **节点（Node，一个函数）** 和 **边（Edge，流转条件）**，节点间通过共享的 **状态（State）** 传递数据。适合做有循环、有分支、可回溯的 Agent 流程——比线性 chain 灵活得多。

## 核心概念
- **State**：用 TypedDict / Pydantic 定义的共享状态，节点读写它（常带 reducer 做累加，如把多路结果 append）；
- **Node**：普通 Python 函数，输入 State、返回要更新的字段；
- **Edge / Conditional Edge**：普通边顺序走，条件边根据 State 决定下一步（实现 ReAct 循环）；
- **checkpointer**：把状态落盘，支持断点续跑、人工介入（human-in-the-loop）。

## 与本项目的关系
本项目两张图都是 LangGraph：
- **导入图（8 节点）**：PDF→MD → 切分 → 主题识别 → 建 KG → BGE 向量化 → 入库；
- **查询图（7 节点）**：确认主题 → 多路检索 → RRF 融合 → 重排 → 生成 →（SSE 流式推送）。
进度面板上那 8 个节点打钩，就是 LangGraph 节点执行顺序的可视化。

## 对比
- LangChain LCEL：线性链，不适合循环/复杂分支；
- 原生 asyncio：灵活但样板代码多；
- LangGraph：图 + 状态 + 检查点，Agent 场景最省心。
