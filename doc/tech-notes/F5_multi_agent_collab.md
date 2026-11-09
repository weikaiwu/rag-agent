# 多智能体协作（Multi-Agent）

## 是什么
把「一个大 Agent 包打天下」拆成多个专职 Agent 协作：比如「规划 Agent」拆任务、「检索 Agent」查资料、「分析 Agent」推理、「写作 Agent」成稿、「校验 Agent」审查。各 Agent 可用不同模型/工具，通过消息或协议（如 A2A）通信。

## 常见协作模式
- ** supervisor（主管）**：一个中枢 Agent 派活给多个 worker；
- **peer-to-peer（平级）**：Agent 间直接对话；
- ** debate（辩论）**：多个 Agent 互相质疑提升答案质量。

## 好处与代价
- 好处：职责清晰、可并行、单 Agent 提示更聚焦、容易扩展；
- 代价：通信开销大、调试难、token 贵、容易「踢皮球」。

## 与本项目的关系
本项目当前是单 Agent（查询图）编排多工具。若升级，可拆成「主题确认 Agent + 检索 Agent + 重排 Agent + 生成 Agent」，或加「校验 Agent」做 Reflection。我在知识库记录多 Agent 模式，作为演进方向。

## 什么时候该用
- 任务明显可分阶段；
- 单 prompt 塞不下所有角色；
- 需要并行提速。
