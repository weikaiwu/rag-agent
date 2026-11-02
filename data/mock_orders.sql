-- ============================================================
-- 电商售后智能客服系统 - 模拟订单库（演示数据）
-- 说明：本表为演示用途，数据为脱敏模拟数据，替换真实业务库即可。
-- 数据库：order_service（MySQL 8+）
-- ============================================================

CREATE DATABASE IF NOT EXISTS order_service DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE order_service;

CREATE TABLE IF NOT EXISTS orders (
    id               BIGINT AUTO_INCREMENT PRIMARY KEY,
    order_no         VARCHAR(32)  NOT NULL UNIQUE COMMENT '订单号，格式 DD+日期+4位序号',
    user_phone       VARCHAR(20)  NOT NULL COMMENT '下单手机号（演示脱敏，掩码格式）',
    product_name     VARCHAR(128) NOT NULL COMMENT '商品名称',
    sku              VARCHAR(64)  NOT NULL COMMENT 'SKU 编码',
    amount           DECIMAL(10,2) NOT NULL COMMENT '订单金额（元）',
    status           ENUM('pending_payment','paid','shipped','delivering','completed','refunding','refunded','cancelled') NOT NULL COMMENT '订单状态',
    logistics_company VARCHAR(32) DEFAULT NULL COMMENT '物流公司',
    tracking_no      VARCHAR(64)  DEFAULT NULL COMMENT '运单号',
    logistics_status VARCHAR(128) DEFAULT NULL COMMENT '物流节点描述（最新）',
    refund_status    ENUM('none','applied','approved','refunded','rejected') NOT NULL DEFAULT 'none' COMMENT '退款状态',
    created_at       DATETIME     NOT NULL COMMENT '下单时间',
    updated_at       DATETIME     NOT NULL COMMENT '更新时间',
    KEY idx_status (status),
    KEY idx_phone (user_phone),
    KEY idx_created (created_at)
) ENGINE=InnoDB COMMENT='订单表（电商售后演示数据）';

-- ------------------------------------------------------------
-- 示例数据：订单号规则 DD + yyyyMMdd + 4 位序号
-- 手机号统一掩码格式 138****xxxx（演示脱敏，公司数据安全基线）
-- ------------------------------------------------------------
INSERT INTO orders (order_no, user_phone, product_name, sku, amount, status, logistics_company, tracking_no, logistics_status, refund_status, created_at, updated_at) VALUES
('DD202608150001', '138****0248', '无线蓝牙耳机 Pro', 'SKU-AUD-001', 299.00, 'delivering',  '顺丰速运', 'SF1380000001', '运输中：已到达深圳转运中心', 'none', '2026-08-15 10:23:00', '2026-08-17 08:30:00'),
('DD202608150002', '139****5612', '智能手表 S2',     'SKU-WCH-002', 899.00, 'shipped',     '中通快递', 'ZT2234567890', '已揽收，待发出', 'none', '2026-08-15 14:02:00', '2026-08-17 09:15:00'),
('DD202608160003', '137****8873', '机械键盘 87 键', 'SKU-KBD-003', 459.00, 'completed',   '京东物流', 'JD1100222333', '已签收', 'none', '2026-08-16 09:47:00', '2026-08-18 11:20:00'),
('DD202608160004', '138****3356', '便携充电宝 20000mAh', 'SKU-PWR-004', 129.00, 'refunding',   '圆通速递', 'YT5566778899', '运输中', 'applied', '2026-08-16 16:38:00', '2026-08-19 10:05:00'),
('DD202608170005', '135****9921', 'USB-C 扩展坞 8合1', 'SKU-DCK-005', 199.00, 'pending_payment', NULL, NULL, NULL, 'none', '2026-08-17 20:11:00', '2026-08-17 20:11:00'),
('DD202608170006', '136****4478', '人体工学椅',     'SKU-CHR-006', 1299.00, 'delivering',  '德邦快递', 'DB9988776655', '运输中：已到达广州分拨中心', 'none', '2026-08-17 11:52:00', '2026-08-19 07:42:00'),
('DD202608180007', '139****2103', '无线鼠标 M3',     'SKU-MOU-007', 89.00,  'paid',         NULL, NULL, NULL, 'none', '2026-08-18 08:19:00', '2026-08-18 08:19:00'),
('DD202608180008', '138****7745', '27 寸 4K 显示器', 'SKU-MON-008', 1899.00, 'refunded',    '顺丰速运', 'SF1380000002', '已签收', 'refunded', '2026-08-18 15:33:00', '2026-08-21 14:26:00'),
('DD202608190009', '137****6620', '桌面麦克风',      'SKU-MIC-009', 349.00, 'completed',   '中通快递', 'ZT5566123490', '已签收', 'none', '2026-08-19 09:26:00', '2026-08-21 16:48:00'),
('DD202608190010', '136****1189', '蓝牙音箱 Go',     'SKU-SPK-010', 259.00, 'cancelled',   NULL, NULL, NULL, 'none', '2026-08-19 21:07:00', '2026-08-20 09:12:00');
