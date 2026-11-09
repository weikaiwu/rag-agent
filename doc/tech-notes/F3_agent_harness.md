# Agent Harness（智能体运行时 / 编排框架）

## 是什么
Harness（马具/框架）指把「LLM + 工具 + 记忆 + 循环 + 错误处理」打包起来的那层运行时。开发者不直接写裸循环，而是用 harness 声明「有哪些工具、什么记忆、怎么决策」，框架负责把循环跑起来、管状态、处理失败重试。

## 主流 harness
- **LangGraph**：图编排，状态显式、可控、可持久化（本项目采用）；
- **AutoGen / AgentFlow**：多 Agent 对话编排；
- **CrewAI**：角色化多 Agent；
- **openai-agents / Agno / LlamaIndex**：轻量 Agent 运行时；
- **原生手写 asyncio 循环**：最灵活但样板多。

## 与本项目的关系
本项目的 LangGraph 就是 harness：它管着查询图 7 个节点的状态流转、SSE 进度推送、错误兜底。我不用手写 while 循环调 LLM，而是声明节点和边，框架跑循环。

## 怎么选
- 要精细控制流程、可观测、可中断 → LangGraph；
- 快速做个单 Agent 原型 → openai-agents / Agno；
- 多角色协作 → CrewAI / AutoGen。
