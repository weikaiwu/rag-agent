# 导入基础库：系统、路径、类型注解（类型注解提升代码可读性和可维护性）
import os
from typing import List, Dict, Any, Tuple

from langchain_core import messages
from openai.types.responses import response_custom_tool_call
# 导入Milvus客户端（向量数据库核心操作）、数据类型枚举（定义集合Schema）
from pymilvus import MilvusClient, DataType
# 导入LangChain消息类（标准化大模型对话消息格式）
from langchain_core.messages import SystemMessage, HumanMessage

from app.conf import milvus_config
# 导入自定义模块：
# 1. 流程状态载体：ImportGraphState为LangGraph流程的统一状态管理对象
from app.import_process.agent.state import ImportGraphState
# 2. Milvus工具：获取单例Milvus客户端，实现连接复用
from app.clients.milvus_utils import get_milvus_client
# 3. 大模型工具：获取大模型客户端，统一模型调用入口
from app.lm.lm_utils import get_llm_client
# 4. 向量工具：BGE-M3模型实例、向量生成方法（稠密+稀疏向量）
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
# 5. 稀疏向量工具：归一化处理，保证向量长度为1，提升检索准确性
from app.utils.normalize_sparse_vector import normalize_sparse_vector
# 6. 任务工具：更新任务运行状态，用于任务监控和管理
from app.utils.task_utils import add_running_task, add_done_task
# 7. 日志工具：项目统一日志入口，分级输出（info/warning/error）
from app.core.logger import logger
# 8. 提示词工具：加载本地prompt模板，实现提示词与代码解耦
from app.core.load_prompt import load_prompt

from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.conf.milvus_config import milvus_config

# --- 配置参数 (Configuration) ---
# 大模型识别文档主题的上下文切片数：取前5个切片，避免上下文过长导致大模型输入超限
DEFAULT_ITEM_NAME_CHUNK_K = 5
# 单个切片内容截断长度：防止单切片内容过长，占满大模型上下文
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
# 大模型上下文总字符数上限：适配主流大模型输入限制，默认2500
CONTEXT_TOTAL_MAX_CHARS = 2500


def step_1_get_chunks(state):
    # 获取chunks和file_name
    file_title=state.get('file_title')
    chunks = state.get('chunks')
    if not chunks:
        raise ValueError('chunks里没有值，直接报错')
    if not file_title:
        file_title=os.path.basename(state.get('md_path'))
        state['file_title']=file_title
    return file_title,chunks



def step_2_build_context(chunks):
    # 根据chunks拼接context
    # 最多获取top个，字符不得超过CONTEXT_TOTAL_MAX_CHARS
    # 内容处理为切片：{1}，标题：{title}，内容{context} \n\n
    # 前置准备
    parts=[]
    total_chars=0
    # 循环处理context
    for index,chunk in enumerate(chunks[:DEFAULT_ITEM_NAME_CHUNK_K],start=1):
        chunk_content=chunk['content']
        chunk_title=chunk['title']
        data=f'切片：{index},标题:{chunk_title},内容：{chunk_content}'
        parts.append(data)
        total_chars+=len(data)
        if total_chars>=SINGLE_CHUNK_CONTENT_MAX_LEN:
            logger.info(f'已超过最大字符数{total_chars}')
            break
    context='\n\n'.join(parts)
    final_context=context[:SINGLE_CHUNK_CONTENT_MAX_LEN]
    return final_context


def step_3_call_llm(context, file_title):
    # 大模型调用获取item_name
    # 1、构建提示词
    human_prompt=load_prompt('item_name_recognition',file_title=file_title,context=context)
    system_prompt=load_prompt('product_recognition_system')
    # 2.获取模型对象
    llm=get_llm_client(json_mode=False)
    # 3\执行调用
    messages=[
        HumanMessage(content=human_prompt),
        SystemMessage(content=system_prompt),
    ]
    response=llm.invoke(messages)
    # 4、进行判断和兜底
    item_name=response.content
    if not item_name:
        item_name=file_title
    # 5、返回
    return item_name


def step_4_update_chunks_and_state(state, item_name, chunks):
    # 修改state中的item_name和chunks中的item_name
    state['item_name']=item_name
    for chunk in chunks:
        chunk['item_name']=item_name
    state['chunks']=chunks
    logger.info(f'完成state和chunks中item_name的修改')


def step_5_generate_embeddings(item_name):
    # 根据item_name生成向量
    # dense_vector[稠密]  sparse_vector[稀疏]
    # generate_embeddings是自己封装的嵌入式模型生成向量的函数
    # embeddings list对应的向量=model.encode_documents(text)传入的list
    # 生成向量的字符串['1','2','3']
    # result={
    #         'dense':[1的稠密，2的稠密，3的稠密]
    #         'sparse':[1的稀疏,2的稀疏,3的稀疏]
#               }
    result=generate_embeddings([item_name])
    dense_vector,sparse_vector=result['dense'][0],result['sparse'][0]
    return dense_vector,sparse_vector

def step_6_save_to_vector_db(file_title, item_name, dense_vector, sparse_vector):
    # 将向量和对应字符串保存到向量库中
    # 1、获取milvus的客户端
    milvus_client = get_milvus_client()
    # 2、判断是否存在集合（表），不存在就自己创建
    if not milvus_client.has_collection(collection_name=milvus_config.item_name_collection):
    # 3、创建集合
    # 3.1创建集合对应的列信息
        schema=milvus_client.create_schema(
            auto_id=True,#主键自增长
            enable_dynamic_field=True,#动态字段
        )
    # 3.2添加数据进去
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True,auto_id=True)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR,)
    # 3.3配置索引
        index_params = milvus_client.prepare_index_params()

    # 3.4. Add indexes
        index_params.add_index(
            field_name="dense_vector",#给哪个列创建索引(稠密）
            index_name="dense_vector_index",#索引的名字
            index_type="HNSW",#配置查找索引的算法（负责找到向亮的）
            metric_type="COSINE",#配置向量匹配和对比的（负责对比相似度的）
            params={
                "M": 16,
                "efConstruction": 200
            }
        )
    #     # 10000M=16  efConstruction=200
    #     # 50000M=32 efConstruction=300
    #     # 100000M=64 efConstruction=400
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            index_name="sparse_vector_index",
            metric_type="IP",
            params={"inverted_index_algo": "DAAT_MAXSCORE"},
        )
        milvus_client.create_collection(
            collection_name=milvus_config.item_name_collection,
            schema=schema,
            index_params=index_params
        )
        #3、先删除之前的item_name
        #加载和选中集合
    milvus_client.load_collection(collection_name=milvus_config.item_name_collection)
    milvus_client.delete(collection_name=milvus_config.item_name_collection,
                         filter=f"item_name=='{item_name}'")
    #4、向集合中插入最新的item_name数据和对应的向量
    item={
        "file_title":file_title,
        "item_name":item_name,
        "dense_vector":dense_vector,
        "sparse_vector":sparse_vector
    }
    milvus_client.insert(collection_name=milvus_config.item_name_collection,
                         data=[item])
    milvus_client.load_collection(collection_name=milvus_config.item_name_collection)
    logger.info(f'对应的{item_name}的向量数据插入完成')

def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 主体识别 (node_item_name_recognition)
    为什么叫这个名字: 识别文档核心描述的物品/文档主题 (Item Name)。
    未来要实现:
    1. 取文档前几段内容。
    2. 调用 LLM 识别这篇文档讲的是什么主题 (如: "RAG 混合检索技术文档")。
    3. 存入 state["item_name"] 用于后续数据幂等性清理。
    """
    function_name = "node_item_name_recognition"
    logger.info(f">>> [{function_name}] 开始执行，当前状态为: {state}")
    add_running_task(state['task_id'], function_name)

    try:
        #1、校验和取值（file_title,chunks)万一没有item_name用file_title查询
        file_title,chunks=step_1_get_chunks(state)
        #2、构建上下文环境（chunks）拿五个前面的chunks拼接成一段context
        context=step_2_build_context(chunks)
        #3、调用大模型，拼接提示词，识别chunks对应的item_name
        item_name=step_3_call_llm(context,file_title)
        #4、修改state chunks chunks里的item_name
        step_4_update_chunks_and_state(state,item_name,chunks)
        #5、item_name生成向量（稠密，稀疏）
        dense_vector,sparse_vector=step_5_generate_embeddings(item_name)
        #6、将向量保存到向量数据库(id/file_title/item_name/稠密和稀疏）
        step_6_save_to_vector_db(file_title,item_name,dense_vector,sparse_vector)
    except Exception as e:
        logger.error(f">>>[{function_name}]主体识别发生了错误，具体异常为{e}")
        raise
    finally:
        # 6、结束的日志与任务的配置
        logger.info(f">>> [{function_name}] 开始结束，当前状态为: {state}")
        add_done_task(state['task_id'], function_name)
    return state

