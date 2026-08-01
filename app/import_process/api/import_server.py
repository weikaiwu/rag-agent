import os
import shutil
import uuid
from typing import List, Dict, Any
from datetime import datetime
import uvicorn
# 第三方库
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from h11 import Request

# 项目内部工具/配置/客户端
from app.clients.minio_utils import get_minio_client
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import (
    add_running_task,
    add_done_task,
    get_done_task_list,
    get_running_task_list,
    update_task_status,
    get_task_status,
)
from app.import_process.agent.state import get_default_state
from app.import_process.agent.main_graph import kb_import_app  # LangGraph全流程编译实例
from app.core.logger import logger  # 项目统一日志工具


# 初始化FastAPI应用实例
# 标题和描述会在Swagger文档(http://ip:port/docs)中展示
app = FastAPI(
    title="File Import Service",
    description="Web service for uploading files to Knowledge Base (PDF/MD → 解析 → 切分 → 向量化 → Milvus入库)"
)

# 跨域中间件配置：解决前端调用后端接口的跨域限制
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有前端域名访问（生产环境建议指定具体域名）
    allow_credentials=True,  # 允许携带Cookie等认证信息
    allow_methods=["*"],  # 允许所有HTTP方法（GET/POST/PUT/DELETE等）
    allow_headers=["*"],  # 允许所有请求头
)


# 8080/import ->  import.html
@app.get("/import",response_class=FileResponse)
async def get_import_page():
    import_html_path=PROJECT_ROOT/"app"/"import_process"/"page"/"import.html"
    if not import_html_path.exists():
        raise HTTPException(status_code=404,detail="Page not found")
    return FileResponse(import_html_path,media_type="text/html")



 # 定义调用import_graph的函数
def run_import_graph(task_id:str,local_file_path:str,local_dir:str):
    # 需要传入三个数据才能执行：文件的地址：local_file_path(str) 任务的标识：task_id 输出文件夹的地址：local_dir(str)
    # 每个任务里面每个节点的状态
    #     add_done_task(task_id, "upload_file")
    #     add_running_task(task_id,"upload_file")
        try:
        #本次任务的总状态
            update_task_status(task_id,"processing")

            init_state=get_default_state()
            init_state["task_id"]=task_id
            init_state["local_file_path"]=local_file_path
            init_state["local_dir"]=local_dir

            # 执行图节点
            for event in kb_import_app.stream(init_state):
            # event里面有{节点名，状态)
                for node_name,result in event.items():
                    logger.info(f'节点：{node_name}已执行，当前结果为{result}')
            update_task_status(task_id,"completed")
            logger.info(f'{task_id}执行完毕')
        except Exception as e:
            logger.exception("执行失败，发生异常")
            update_task_status(task_id,"failed")
# 8080/upload post 文件上传加开始导入
@app.post("/upload")
async def upload_file(background_tasks: BackgroundTasks,
                      files:List[UploadFile] = File(...)):
    # 1、整理输出的位置output/日期文件夹
    today_str=datetime.now().strftime("%Y%m%d")
    base_out_path=PROJECT_ROOT/"output"/today_str
    # 2、记录每个文件上传的任务id
    task_ids=[]
    # 3、循环处理每个上传的文件+进行异步图调用
    for file in files:
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
    # 记录每个文件上传了
        add_running_task(task_id,"upload_file")
    # 文件的dir_path
        dir_path=base_out_path/task_id
    #     没有文件夹
        dir_path.mkdir(parents=True,exist_ok=True)
    # 文件的local_file_path
        local_file_path=dir_path/file.filename
    # 将上传的文件写入local_file_path
        with open(local_file_path,"wb") as buffer:
            shutil.copyfileobj(file.file,buffer)
    # 异步执行
    #     参数1：run_import_graph执行的参数
    #     参数2；参数列表 task_id,local_file_path,dir_path数据放入run_import_graph中
        background_tasks.add_task(run_import_graph,task_id,str(local_file_path),str(dir_path))
        logger.info(f"{task_id}完成上传，开启了对应的异步任务")
        add_done_task(task_id,"upload_file")
        # 4、返回最终结果
    return {
        "code":200,
        "message":f"完成上传，开启了对应的异步任务!文件数量为{len(files)}",
        "task_ids":task_ids
    }


# --------------------------
# 核心接口：任务状态查询接口
# 前端轮询此接口获取单个任务的处理进度和状态
# 访问地址：http://localhost:8000/status/{task_id} （GET请求）
# --------------------------
@app.get("/status/{task_id}", summary="任务状态查询", description="根据TaskID查询单个文件的处理进度和全局状态")
async def get_task_progress(task_id: str):
    """
    任务状态查询接口
    前端轮询此接口（如每秒1次），获取任务的实时处理进度
    返回数据均来自内存中的任务管理字典（task_utils.py），高性能无IO

    :param task_id: 全局唯一任务ID（由/upload接口返回）
    :return: 包含任务全局状态、已完成节点、运行中节点的JSON响应
    """
    # 构造任务状态返回体
    task_status_info: Dict[str, Any] = {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),  # 任务全局状态：pending/processing/completed/failed
        "done_list": get_done_task_list(task_id),  # 已完成的节点/阶段列表
        "running_list": get_running_task_list(task_id)  # 正在运行的节点/阶段列表
    }
    # 记录状态查询日志，方便追踪前端轮询情况
    logger.info(
        f"[{task_id}] 任务状态查询，当前状态：{task_status_info['status']}，已完成节点：{task_status_info['done_list']}")
    return task_status_info
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)