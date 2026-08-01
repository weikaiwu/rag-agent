import os
from typing import Any, List, Dict

from langchain_core.utils import batch_iterate
from pymongo import results

from app.import_process.agent.state import ImportGraphState
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
from app.utils.task_utils import add_running_task,add_done_task
from app.core.logger import logger

def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 向量化 (node_bge_embedding)
    为什么叫这个名字: 使用 BGE-M3 模型将文本转换为向量 (Embedding)。
    未来要实现:
    1. 加载 BGE-M3 模型。
    2. 对每个 Chunk 的文本进行 Dense (稠密) 和 Sparse (稀疏) 向量化。
    3. 准备好写入 Milvus 的数据格式。
    """
    # 获取当前节点名称，用于日志和任务状态记录
    current_node = "node_bge_embedding"
    logger.info(f">>> 开始执行LangGraph节点：{current_node}")

    # 标记任务运行状态，用于任务监控/前端进度展示
    add_running_task(state.get("task_id", ""), current_node)
    logger.info("--- BGE-M3 文本向量化处理启动 ---")

    try:
        # 获取要生成向量的chunks
        chunks=state.get("chunks")
        if not chunks or not isinstance(chunks, list):
            logger.error(f'chunks数据无效，请检查')
            raise ValueError(f'chunks数据无效，请检查')
        # 给每个chunks生成向量
        final_chunks = []
        batch_size = 5
        for i in range(0, len(chunks), batch_size):
            batch_items = chunks[i:i+batch_size]
            current_texts=[]
            for item in batch_items:
                item_name=item.get("item_name")
                item_content=item.get("item_content")
                item_text=f'文档主题：{item_name},内容：{item_content}'
                current_texts.append(item_text)
            #     当前批次生成的向量
            result=generate_embeddings(current_texts)
            # 完善chunks的属性生成稠密稀疏向量
            for i,chunk in enumerate(batch_items):
                chunk_item=chunk.copy()
                chunk_item['dense_vector']=result['dense'][i]
                chunk_item['sparse_vector']=result['sparse'][i]
                final_chunks.append(chunk_item)
        state['chunks'] = final_chunks
        logger.info(f"--- BGE-M3 向量化处理完成，处理 {final_chunks} 条文本切片 ---")
        add_done_task(state.get("task_id", ""), current_node)
    except Exception as e:
        # 捕获节点所有异常，记录错误堆栈，不中断整体流程
        logger.error(f"BGE-M3向量化节点执行失败：{str(e)}", exc_info=True)

    # 返回更新后的状态对象，传递至下游节点
    return state