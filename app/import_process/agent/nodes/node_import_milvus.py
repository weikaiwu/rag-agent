import os
from typing import List, Dict, Any
# 导入Milvus相关依赖
from pymilvus import DataType
# 导入自定义模块
from app.import_process.agent.state import ImportGraphState
from app.clients.milvus_utils import get_milvus_client
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger
from app.conf.milvus_config import milvus_config
from app.utils.escape_milvus_string_utils import escape_milvus_string
# 从配置文件读取切片集合名称，与配置解耦，便于环境切换
CHUNKS_COLLECTION_NAME = milvus_config.chunks_collection


def step_2_prepare_collections(state):
    milvus_client = get_milvus_client()
    # 2、判断是否存在集合（表），不存在就自己创建
    if not milvus_client.has_collection(collection_name=milvus_config.chunks_collection):
        # 3、创建集合
        # 3.1创建集合对应的列信息
        schema = milvus_client.create_schema(
            auto_id=True,  # 主键自增长
            enable_dynamic_field=True,  # 动态字段
        )
        # 3.2添加数据进去
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="part", datatype=DataType.INT8)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR )
        # 3.3配置索引
        index_params = milvus_client.prepare_index_params()

        # 3.4. Add indexes
        index_params.add_index(
            field_name="dense_vector",  # 给哪个列创建索引(稠密）
            index_name="dense_vector_index",  # 索引的名字
            index_type="HNSW",  # 配置查找索引的算法（负责找到向亮的）
            metric_type="COSINE",  # 配置向量匹配和对比的（负责对比相似度的）
            params={
                "M": 32,
                "efConstruction": 300
            }
        )
        # 10000M=16  efConstruction=200
        # 50000M=32 efConstruction=300
        # 100000M=64 efConstruction=400
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            index_name="sparse_vector_index",
            metric_type="IP",
            params={"inverted_index_algo": "DAAT_MAXSCORE"},
        )
        milvus_client.create_collection(
            collection_name=milvus_config.chunks_collection,
            schema=schema,
            index_params=index_params
        )
        milvus_client.load_collection(milvus_config.chunks_collection)
    return milvus_client


def step_3_delete_old_data(milvus_client, item_name):
    # 根据item_name删除
    if not milvus_client.has_collection(collection_name=CHUNKS_COLLECTION_NAME):
        logger.warning(f"集合 {CHUNKS_COLLECTION_NAME} 不存在，跳过删除。")
        return
    milvus_client.load_collection(collection_name=CHUNKS_COLLECTION_NAME)
    milvus_client.delete(collection_name=CHUNKS_COLLECTION_NAME,
                                    filter=f'item_name == "{item_name}"')
    # milvus_client.load_collection(collection_name=CHUNKS_COLLECTION_NAME)


def step_4_insert_collections(milvus_client,chunks):
    # 插入集合的数值
    insert_result=milvus_client.insert(collection_name=CHUNKS_COLLECTION_NAME,data=chunks)
    # 成功插入了几条
    insert_count=insert_result.get('insert_count',0)
    logger.info(f'成功插入了{insert_count}条数据')
    ids=insert_result.get('ids',[])

    if ids and len(ids)==len(chunks):
        for index,chunk in enumerate(chunks):
            chunk['chunk_id']=ids[index]
    return chunks



def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 导入向量库 (node_import_milvus)
    为什么叫这个名字: 将处理好的向量数据写入 Milvus 数据库。
    未来要实现:
    1. 连接 Milvus。
    2. 根据 item_name 删除旧数据 (幂等性)。
    3. 批量插入新的向量数据。
    """
    function_name = "node_import_milvus"
    logger.info(f">>> [{function_name}] 开始执行，当前状态为: {state}")
    add_running_task(state['task_id'], function_name)

    try:
    # 1、校验获取数据
        chunks=state.get('chunks')
        if not chunks:
            logger.error(f'[{function_name}]没有chunks的向量数据')
            raise ValueError('没有chunks数据')
    # 2、获取集合，如果没有就建立一个
        milvus_client = step_2_prepare_collections(state)
    # 3、删除旧数据
        step_3_delete_old_data(milvus_client,chunks[0]['item_name'])
    # 4、插入chunks的数据
        with_id_chunks=step_4_insert_collections(milvus_client,chunks)
        state['chunks'] = with_id_chunks
    except Exception as e:
        logger.error(f">>>[{function_name}]导入chunks的向量数据库异常，具体异常为{e}")
        raise
    finally:
        # 6、结束的日志与任务的配置
        logger.info(f">>> [{function_name}] 开始结束，当前状态为: {state}")
        add_done_task(state['task_id'], function_name)
    return state

