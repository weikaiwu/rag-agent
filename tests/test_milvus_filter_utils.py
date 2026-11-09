"""导入链路删除键单元测试（回归两次实测事故）。

背景：重导同一篇文档时，必须先删掉它自己的旧切片，否则新旧切片会混库。
删除键的选择踩过两次坑，这里把结论钉住：

  事故 1（按 item_name 单键删）：B1_agent_reasoning_modes 与 F3_agent_harness
      都被 LLM 抽成「LangGraph 多智能体编排」，导入 F3 时把 B1 的切片一起删了，
      27 篇文档导进去只剩 26 篇，而导入日志每篇都报成功。
  事故 2（item_name + file_title 联合删）：item_name 会漂移，F10 重导抽出的名字
      和上次不同，联合条件一条也匹配不上，旧切片留成孤儿——F10 变成 12 条
      （6 旧 + 6 新），全库 146 → 152，且旧文本（含"招聘"字样）仍会被检索命中。

结论：删除键只用 file_title（来自文件名，稳定且能区分文档），item_name 只做兜底。
"""

from app.utils.milvus_filter_utils import build_doc_delete_expr


class TestBuildDocDeleteExpr:
    def test_uses_file_title_only(self):
        """正常情况：只看 file_title，item_name 不参与，避免撞名/漂移。"""
        expr = build_doc_delete_expr("F10_ai_coding_tools", "AI编程工具提效、Cursor")
        assert expr == 'file_title == "F10_ai_coding_tools"'
        assert "item_name" not in expr

    def test_same_title_different_item_name_same_expr(self):
        """item_name 漂移不影响删除范围：两次抽取结果不同，表达式必须一致。"""
        first = build_doc_delete_expr("B1_agent_reasoning_modes", "LangGraph 多智能体编排")
        second = build_doc_delete_expr("B1_agent_reasoning_modes", "Agent 推理模式、ReAct、Plan-Execute")
        assert first == second

    def test_escapes_quotes_and_backslashes(self):
        """文件名含引号/反斜杠时必须转义，否则 filter 表达式语法会断。"""
        expr = build_doc_delete_expr('a"b\\c.md')
        assert expr == 'file_title == "a\\"b\\\\c.md"'

    def test_falls_back_to_item_name_without_title(self):
        """file_title 缺失才退回 item_name（有误删风险，调用方需打警告）。"""
        expr = build_doc_delete_expr("", "RAG 检索增强生成")
        assert expr == 'item_name == "RAG 检索增强生成"'

    def test_empty_both_returns_empty(self):
        """两个都为空时返回空串，调用方必须跳过删除，否则等于清空全库。"""
        assert build_doc_delete_expr("", "") == ""
        assert build_doc_delete_expr(None, None) == ""

    def test_newlines_are_flattened(self):
        """换行/制表符会被压成空格，保证表达式单行有效。"""
        expr = build_doc_delete_expr("a\nb\tc.md")
        assert expr == 'file_title == "a b c.md"'
