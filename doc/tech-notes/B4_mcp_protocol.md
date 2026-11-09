# MCP：模型上下文协议

## 是什么
MCP（Model Context Protocol，模型上下文协议）是 2024 年底 Anthropic 提出的开放标准，用来**标准化「LLM 应用 ↔ 外部工具/数据源」的连接方式**。类比：USB-C 统一了硬件接口，MCP 统一了 AI 工具接口。在此之前每个工具都要单独适配，MCP 让「写一次工具，任意客户端可用」。

## 三种原语
- **Tools**：可被模型调用的函数（如 web search、查数据库）；
- **Resources**：只读上下文（如文件、配置），喂给模型；
- **Prompts**：预置的提示词模板。

## 传输方式
- **stdio**：本地进程间通信；
- **Streamable HTTP / SSE**：远程服务（本项目用的就是 HTTP 形式接阿里云百炼的 Web Search MCP）。

## 与本项目的关系
查询图的「网络搜索」节点通过 openai-agents 的 MCP 客户端连到百炼的 StreamableHttp MCP Server，模型按需调用搜索工具补充召回。这是标准 MCP 用法，不绑定特定厂商。

## MCP vs Function Calling
- Function Calling 是「模型调函数的能力」，各家 API 格式不同；
- MCP 是「客户端和工具服务之间的传输协议」，解决「工具怎么暴露、怎么被发现」；
- 二者互补：MCP 负责接，Function Calling 负责模型决策。
