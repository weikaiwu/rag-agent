"""
模拟订单造数脚本（电商售后客服系统 · 演示数据）

用途：生成与 data/mock_orders.sql 同结构的订单数据，供本地开发/测试使用。
设计要点（面试可讲）：
1. 状态机覆盖完整：pending_payment/paid/shipped/delivering/completed/refunding/refunded/cancelled，
   保证客服系统所有查询分支都有数据可测（如"追问订单号""查不到""退款中"）。
2. 手机号脱敏：统一 138****xxxx 掩码格式——演示数据也遵守公司数据安全基线。
3. 订单号规则：DD + yyyyMMdd + 4位序号，与 SQL 种子数据一致，避免"order1"式假数据。

用法：python scripts/gen_mock_orders.py [条数] [起始日期]
"""
from __future__ import annotations

import argparse
import datetime as dt
import random
import string
import sys
from pathlib import Path

# 确保可从项目根运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---- 商品池（与 FAQ 知识库商品保持一致，方便对答）----
PRODUCTS = [
    ("无线蓝牙耳机 Pro", "SKU-AUD-001", 299.00),
    ("智能手表 S2", "SKU-WCH-002", 899.00),
    ("机械键盘 87 键", "SKU-KBD-003", 459.00),
    ("便携充电宝 20000mAh", "SKU-PWR-004", 129.00),
    ("USB-C 扩展坞 8合1", "SKU-DCK-005", 199.00),
    ("人体工学椅", "SKU-CHR-006", 1299.00),
    ("无线鼠标 M3", "SKU-MOU-007", 89.00),
    ("27 寸 4K 显示器", "SKU-MON-008", 1899.00),
    ("桌面麦克风", "SKU-MIC-009", 349.00),
    ("蓝牙音箱 Go", "SKU-SPK-010", 259.00),
]

STATUSES = [
    "pending_payment", "paid", "shipped", "delivering",
    "completed", "refunding", "refunded", "cancelled",
]
# 状态 -> (物流公司, 物流节点) 的合理组合，保证状态与物流语义一致
LOGISTICS = {
    "shipped":   [("中通快递", "已揽收，待发出"), ("圆通速递", "已揽收")],
    "delivering": [("顺丰速运", "运输中：已到达深圳转运中心"),
                   ("德邦快递", "运输中：已到达广州分拨中心"),
                   ("京东物流", "运输中")],
    "completed": [("顺丰速运", "已签收"), ("中通快递", "已签收"), ("京东物流", "已签收")],
    "refunding": [("圆通速递", "运输中"), ("顺丰速运", "已签收")],
    "refunded":  [("顺丰速运", "已签收"), ("中通快递", "已签收")],
}

# 各状态对应的退款状态（保证语义一致）
REFUND_BY_STATUS = {
    "pending_payment": "none", "paid": "none", "shipped": "none",
    "delivering": "none", "completed": "none", "cancelled": "none",
    "refunding": "applied", "refunded": "refunded",
}


def mask_phone() -> str:
    """生成脱敏手机号：真实前缀 + 掩码 + 后4位（138****xxxx）"""
    head = random.choice(["13", "15", "18"]) + str(random.randint(0, 9))
    tail = f"{random.randint(0, 9999):04d}"
    return f"{head}****{tail}"


def gen_order_no(date: dt.date, seq: int) -> str:
    return f"DD{date.strftime('%Y%m%d')}{seq:04d}"


def gen_orders(count: int = 40, start_date: dt.date | None = None) -> list[dict]:
    start_date = start_date or dt.date.today() - dt.timedelta(days=14)
    orders: list[dict] = []
    seq_counter: dict[str, int] = {}

    for i in range(count):
        date = start_date + dt.timedelta(days=i % 14)
        key = date.strftime("%Y%m%d")
        seq_counter[key] = seq_counter.get(key, 0) + 1
        order_no = gen_order_no(date, seq_counter[key])

        product_name, sku, amount = random.choice(PRODUCTS)
        status = random.choice(STATUSES)
        created = dt.datetime.combine(date, dt.time(random.randint(8, 22), random.randint(0, 59)))

        order: dict = {
            "order_no": order_no,
            "user_phone": mask_phone(),
            "product_name": product_name,
            "sku": sku,
            "amount": amount,
            "status": status,
            "refund_status": REFUND_BY_STATUS[status],
            "created_at": created.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": (created + dt.timedelta(hours=random.randint(6, 72))).strftime("%Y-%m-%d %H:%M:%S"),
        }
        if status in LOGISTICS:
            company, node = random.choice(LOGISTICS[status])
            order["logistics_company"] = company
            order["tracking_no"] = f"{company[0]}{random.randint(10**9, 10**10 - 1)}"
            order["logistics_status"] = node
        else:
            order["logistics_company"] = None
            order["tracking_no"] = None
            order["logistics_status"] = None
        orders.append(order)

    # 按时间排序，保证插入顺序自然
    orders.sort(key=lambda o: o["created_at"])
    return orders


def to_sql_insert(orders: list[dict]) -> str:
    """生成可直接执行的 INSERT 语句（与 data/mock_orders.sql 表结构一致）"""
    esc = lambda s: "NULL" if s is None else f"'{str(s).replace(chr(39), chr(39)*2)}'"
    rows = []
    for o in orders:
        rows.append(
            f"({esc(o['order_no'])}, {esc(o['user_phone'])}, {esc(o['product_name'])}, "
            f"{esc(o['sku'])}, {o['amount']}, {esc(o['status'])}, "
            f"{esc(o['logistics_company'])}, {esc(o['tracking_no'])}, {esc(o['logistics_status'])}, "
            f"{esc(o['refund_status'])}, {esc(o['created_at'])}, {esc(o['updated_at'])})"
        )
    return (
        "INSERT INTO orders (order_no, user_phone, product_name, sku, amount, status, "
        "logistics_company, tracking_no, logistics_status, refund_status, created_at, updated_at) VALUES\n"
        + ",\n".join(rows) + ";\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="生成模拟订单 SQL")
    parser.add_argument("count", nargs="?", type=int, default=40, help="订单条数（默认40）")
    parser.add_argument("--out", default="data/mock_orders_generated.sql", help="输出 SQL 路径")
    args = parser.parse_args()

    orders = gen_orders(args.count)
    sql = to_sql_insert(orders)
    out = Path(__file__).resolve().parent.parent / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(sql, encoding="utf-8")
    print(f"✅ 已生成 {len(orders)} 条模拟订单 -> {out}")
    # 状态分布统计，方便验证覆盖
    from collections import Counter
    print("状态分布:", dict(Counter(o["status"] for o in orders)))


if __name__ == "__main__":
    main()
