# 配置管理：.env 与 python-dotenv

## 是什么
项目的敏感信息（API Key、数据库地址、密码）和可变配置，不该写死在代码里，而是放 `.env` 文件，运行时用 `python-dotenv` 加载成环境变量。好处：配置和代码分离、不同环境（开发/生产）用不同 .env、密钥不进版本库。

## 关键约定
- `.env` **绝不提交 Git**（必须在 .gitignore）；
- 提供 `.env.example` 列变量名、值留空，方便别人 clone 后填；
- 代码里只 `os.getenv("XXX")`，不硬编码。

## 与本项目的关系
项目根 `.env` 含：模型 API Key、Milvus 地址/端口、MongoDB 地址、MinIO 地址、Neo4j 配置（可选，留空则跳过）、日志级别开关。`node_import_milvus` 等节点自动从项目根找 `.env`。

## 安全红线
- 推 GitHub 前必须确认 `.env` 在 `.gitignore`；
- 泄露的 Key 立即吊销重置；
- 生产环境用密钥管理服务（Vault / 云秘钥），不止 .env。
