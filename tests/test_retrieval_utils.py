"""检索链路纯函数单元测试。

覆盖 app/utils/retrieval_utils.py 的核心行为：
- 加权 RRF 融合（排序 / 权重 / top_k / chunk_id 兼容）
- 动态 TopK + 断崖检测（绝对断崖 / 相对断崖 / 上下限保护）
- Milvus 主题过滤表达式构造
"""

from app.utils.retrieval_utils import (
    build_item_name_expr,
    dynamic_topk_with_gap,
    reciprocal_rank_fusion,
)

# ----------------------------- reciprocal_rank_fusion -----------------------------

def _chunk(cid):
    return {"id": cid, "content": f"chunk-{cid}"}


class TestReciprocalRankFusion:
    def test_single_source_keeps_rank_order(self):
        """单路输入：返回顺序按 RRF 分数降序，原第 1 名仍在首位。"""
        source = [_chunk("a"), _chunk("b"), _chunk("c")]
        result = reciprocal_rank_fusion([(source, 1.0)], top_k=5)
        assert [c["id"] for c in result] == ["a", "b", "c"]

    def test_multi_source_accumulates_score(self):
        """双路召回命中同一 chunk：分数累加，排到最前。"""
        source_a = [_chunk("dup"), _chunk("x")]
        source_b = [_chunk("dup"), _chunk("y")]
        result = reciprocal_rank_fusion([(source_a, 1.0), (source_b, 1.0)], top_k=5)
        assert result[0]["id"] == "dup"

    def test_weight_affects_ranking(self):
        """权重生效：weight=2 的路第 2 名 > weight=1 的路第 1 名（1/62*2 > 1/61*1）。"""
        source_light = [_chunk("first_light")]
        source_heavy = [_chunk("first_heavy"), _chunk("second_heavy")]
        result = reciprocal_rank_fusion(
            [(source_light, 1.0), (source_heavy, 2.0)], top_k=5
        )
        assert result[0]["id"] == "first_heavy"

    def test_topk_truncation(self):
        """top_k 截断：融合后只保留前 top_k 条。"""
        source = [_chunk(str(i)) for i in range(10)]
        result = reciprocal_rank_fusion([(source, 1.0)], top_k=3)
        assert len(result) == 3

    def test_chunk_id_via_entity(self):
        """chunk 无 "id" 字段时兼容 "entity.chunk_id" 结构。"""
        chunks = [{"entity": {"chunk_id": "e1"}}, {"entity": {"chunk_id": "e2"}}]
        result = reciprocal_rank_fusion([(chunks, 1.0)], top_k=5)
        assert [c["entity"]["chunk_id"] for c in result] == ["e1", "e2"]

    def test_empty_sources(self):
        """空输入：返回空列表，不抛异常。"""
        assert reciprocal_rank_fusion([], top_k=5) == []


# ----------------------------- dynamic_topk_with_gap -----------------------------

def _scored(scores):
    return [{"score": s} for s in scores]


class TestDynamicTopkWithGap:
    def test_no_gap_returns_all(self):
        """分数均匀下降无断崖：返回全部候选。"""
        result = dynamic_topk_with_gap(_scored([0.9, 0.85, 0.8, 0.75]))
        assert len(result) == 4

    def test_absolute_gap_truncates(self):
        """绝对断崖：0.9 -> 0.35（落差 0.55 >= 0.5），截断为 1 条。"""
        result = dynamic_topk_with_gap(_scored([0.9, 0.35, 0.3]))
        assert len(result) == 1

    def test_relative_gap_truncates(self):
        """相对断崖：0.9 -> 0.6（落差 0.3 < 0.5，但相对 33% >= 25%），截断为 1 条。"""
        result = dynamic_topk_with_gap(_scored([0.9, 0.6, 0.5]))
        assert len(result) == 1

    def test_empty_list(self):
        """空输入：返回空列表。"""
        assert dynamic_topk_with_gap([]) == []

    def test_max_topk_cap(self):
        """超过 10 条候选：最多保留 10 条（硬上限）。"""
        result = dynamic_topk_with_gap(_scored([0.9 - i * 0.01 for i in range(15)]))
        assert len(result) == 10

    def test_min_topk_keeps_at_least_one(self):
        """即使首对即断崖，也至少保留 1 条（最小保护）。"""
        result = dynamic_topk_with_gap(_scored([0.9, 0.1, 0.05]))
        assert len(result) == 1


# ----------------------------- build_item_name_expr -----------------------------

class TestBuildItemNameExpr:
    def test_with_items(self):
        """多主题：生成逗号连接的双引号列表表达式。"""
        assert build_item_name_expr(["LangGraph", "AutoGen"]) == 'item_name in ["LangGraph","AutoGen"]'

    def test_empty_list_returns_none(self):
        """空列表：返回 None（全库检索）。"""
        assert build_item_name_expr([]) is None

    def test_none_returns_none(self):
        """None：返回 None（全库检索）。"""
        assert build_item_name_expr(None) is None
