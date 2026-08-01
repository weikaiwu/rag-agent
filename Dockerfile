# 技术文档智能问答系统 - 应用镜像
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# 系统级依赖：magic-pdf / torch / opencv 等所需的基础库
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    ffmpeg poppler-utils git curl \
    && rm -rf /var/lib/apt/lists/*

# 使用 uv 做依赖解析与安装（快、可复现）
RUN pip install --no-cache-dir uv

# 先复制依赖清单，利用镜像层缓存
COPY pyproject.toml uv.lock* ./
RUN uv export --no-dev --no-emit-project -o /tmp/requirements.txt 2>/dev/null \
    || uv pip compile pyproject.toml -o /tmp/requirements.txt \
    && uv pip install --system -r /tmp/requirements.txt

# 再复制源码
COPY . .

EXPOSE 8000 8001

# 默认启动查询服务；导入服务使用同一镜像另起容器（见 docker-compose.yml）
CMD ["python", "-m", "app.query_process.api.query_server"]
