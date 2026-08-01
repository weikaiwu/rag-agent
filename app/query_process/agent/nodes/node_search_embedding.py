
from app.conf.milvus_config import milvus_config
from app.utils.task_utils import add_running_task,add_done_task
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import create_hybrid_search_requests,hybrid_search,get_milvus_client
from app.core.logger import logger
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())
def node_search_embedding(state):
    """
    节点功能：进行向量内容检索
    带着问题去查询chunks切片
    需要重写的问题和item_name
    """
    add_running_task(state["session_id"], "node_search_embedding", state.get("is_stream"))

    # 搜索假设性答案
    # 1、先从state获取数据
    rewritten_query=state.get("rewritten_query")
    item_names=state.get("item_names")
    # 2、将重写问题转换成向量
    embeddings=generate_embeddings([rewritten_query])
    # 3、进行向量数据库的混合查询
    # 3.1创建混合查询请求对象（向量检索）
    # 若已确认 item_names 则按主题过滤；否则（空列表）进行全库全局语义检索
    if item_names:
        item_name_str=','.join(f'"{item}"' for item in item_names)
        expr=f'item_name in [{item_name_str}]'
    else:
        expr=None
    hybrid_search_requests=create_hybrid_search_requests(
        dense_vector=embeddings['dense'][0],
        sparse_vector=embeddings['sparse'][0],
        expr=expr
    )
    # 3.2进行混合查询出发
    milvus_client=get_milvus_client()
    resp=hybrid_search(
        client=milvus_client,
        collection_name=milvus_config.chunks_collection,
        reqs=hybrid_search_requests,
        ranker_weights=(0.9,0.1),
        norm_score=True,
        limit=5,
        output_fields=["chunk_id","content","file_title","title","parent_title","item_name"]
    )
    # 4、处理查询结果赋值chunks的属性
    embedding_chunks=resp[0] if resp else []
    # ...
    add_done_task(state["session_id"], "node_search_embedding", state.get("is_stream"))

    return {"embedding_chunks":embedding_chunks }

