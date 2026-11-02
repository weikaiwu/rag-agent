# 订单库（MySQL）配置 —— 客服业务化 P3
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class MysqlConfig:
    host: str
    port: int
    user: str
    password: str
    db: str


mysql_config = MysqlConfig(
    host=os.getenv("MYSQL_HOST", "127.0.0.1"),
    port=int(os.getenv("MYSQL_PORT", "3306")),
    user=os.getenv("MYSQL_USER", "root"),
    password=os.getenv("MYSQL_PASSWORD", ""),
    db=os.getenv("MYSQL_DB", "order_service"),
)
