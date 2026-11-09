"""Milvus 过滤表达式构造（纯函数，便于单测，不依赖 pymilvus / LLM）。

抽出来的原因：删除键的选法踩过两次事故，需要能脱离 Milvus 做回归测试。
"""

from app.utils.escape_milvus_string_utils import escape_milvus_string


def build_doc_delete_expr(file_title: str, item_name: str = "") -> str:
    """构造「删除同一篇文档旧切片」的过滤表达式。

    只用 file_title（来自文件名，同一篇文档永远不变，且能区分文档）。
    item_name 不能进删除键，两次实测踩出来的：

      1. 它由 LLM 抽取，不同文档会抽到同一个名字，只用它会误删别篇
         （B1_agent_reasoning_modes 与 F3_agent_harness 都抽成
          「LangGraph 多智能体编排」，导致先导入那篇整篇消失）；
      2. 它还会漂移，同一篇文档重导时抽取结果和上次不同，于是
         「item_name == 新值 and file_title == 本文」一条也匹配不上，
         旧切片留成孤儿（F10_ai_coding_tools 重导后残留 6 条，总数 146 → 152）。

    只有在 file_title 缺失时才退回 item_name（有误删风险，调用方应打警告）。

    Returns:
        过滤表达式；两者都为空时返回空串，调用方必须跳过删除，避免清空全库。
    """
    if file_title:
        return f'file_title == "{escape_milvus_string(file_title)}"'
    if item_name:
        return f'item_name == "{escape_milvus_string(item_name)}"'
    return ""
