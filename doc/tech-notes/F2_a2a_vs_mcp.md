# A2A vs MCP（Agent 间通信 vs 工具协议）

## 是什么
2025 年 Agent 生态出现两个常被一起提的协议：
- **MCP（Model Context Protocol）**：解决「单个 Agent ↔ 工具/数据源」怎么连（见 B4）。Agent 调数据库、调搜索，用 MCP。
- **A2A（Agent2Agent，Google 2025 提出）**：解决「Agent ↔ Agent」怎么协作。一个 Agent 怎么发现别的 Agent、把任务委派出去、拿回结果。

## 一句话区分
- MCP = Agent 用手（调工具）；
- A2A = Agent 用嘴（找同伴协作）。

## 为什么需要 A2A
复杂任务（"调研竞品并出报告"）单 Agent 干不过来，要拆给「搜索 Agent」「分析 Agent」「写作 Agent」。它们可能用不同框架、不同厂商，A2A 提供统一「Agent 卡片（Agent Card）」描述能力，让彼此可发现、可委派。

## 与本项目的关系
本项目目前是「单 Agent + MCP 工具」形态（Web Search 通过 MCP）。若演进成多 Agent（见 F5），A2A 是 Agent 间协作的标准候选。我在 README/知识库里把它作为「行业新方向」记录，体现跟风口。

## 趋势
2026 年「MCP 管工具、A2A 管协作」正成为多 Agent 系统的标准分层。
