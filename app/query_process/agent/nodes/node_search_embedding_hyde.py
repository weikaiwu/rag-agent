# HyDE节点

from langchain_core.messages import HumanMessage


from app.utils.task_utils import add_running_task, add_done_task
from app.lm.lm_utils import *
from app.lm.embedding_utils import *
from app.clients.milvus_utils import *
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


def step_1_create_hyde_doc(rewritten_query):
    llm=get_llm_client()
    hyde_prompt=load_prompt("hyde_prompt",rewritten_query=rewritten_query)
    messages=[
        HumanMessage(content=hyde_prompt)
    ]
    response=llm.invoke(messages)
    hyde_doc=response.content
    logger.info(f"使用模型生成的假设性答案，问题是{rewritten_query}，答案是{hyde_doc}")
    return hyde_doc


def step_2_search_embedding_hyde(rewritten_query, hyde_doc, item_names):
    # 根据问题和答案查询向量数据库
    # 1、拼接问题和答案
    query_str=rewritten_query+hyde_doc
    # 2、拼接查询对应的向量
    embeddings=generate_embeddings([query_str])
    # 3、生成AnnSearchRequest
    # 若已确认 item_names 则按主题过滤；否则（空列表）进行全库全局语义检索
    if item_names:
        item_name_str = ','.join(f'"{item}"' for item in item_names)
        expr = f'item_name in [{item_name_str}]'
    else:
        expr = None
    reqs=create_hybrid_search_requests(
        dense_vector=embeddings['dense'][0],
        sparse_vector=embeddings['sparse'][0],
        expr=expr
    )
    # 4、进行混合查询
    milvus_client=get_milvus_client()
    resp=hybrid_search(
        client=milvus_client,
        collection_name=milvus_config.chunks_collection,
        reqs=reqs,
        ranker_weights=(0.9, 0.1),
        output_fields=["item_name", "content", "title", "parent_title", "chunk_id"]
    )
    # 5、处理返回结果
    result=resp[0] if resp else []
    logger.info(f"假设性检索结果为{result}")
    return result



def node_search_embedding_hyde(state):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """
    add_running_task(state["session_id"], "node_search_embedding_hyde", state.get("is_stream"))

    # 1、提取参数（item_name和重写的问题）
    rewritten_query = state.get("rewritten_query")
    item_names = state.get("item_names")
    # 2、调用llm大模型生成一个假设性答案
    hyde_doc=step_1_create_hyde_doc(rewritten_query)
    # 3、问题加答案进行向量检索
    resp=step_2_search_embedding_hyde(rewritten_query,hyde_doc,item_names)
    # 4、赋值返回结果
    # ...
    add_done_task(state["session_id"], "node_search_embedding_hyde", state.get("is_stream"))

    return {"hyde_embedding_chunks":resp}

