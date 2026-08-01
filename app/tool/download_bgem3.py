import os
from modelscope.hub.snapshot_download import snapshot_download

# 下载模型到本地缓存目录（可用环境变量 BGE_M3_CACHE_DIR 覆盖，默认 ./models/bge-m3）
cache_dir = os.getenv("BGE_M3_CACHE_DIR", "./models/bge-m3")
model_dir = snapshot_download('BAAI/bge-m3', cache_dir=cache_dir)
print(f"模型已下载到: {model_dir}")