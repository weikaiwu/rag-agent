
from pathlib import Path

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task


def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry)
    为什么叫这个名字: 作为图的 Entry Point，负责接收外部输入并决定流程走向。
    未来要实现:
    1. 接收文件路径。
    2. 判断文件类型 (PDF/MD)。
    3. 设置 state 中的路由标记 (is_pdf_read_enabled / is_md_read_enabled)。
    """
    # 1、进入节点的日志输出（节点+参数）会推送到前端
    function_name="node_entry"
    logger.info(f">>> [{function_name}] 开始执行，当前状态为: {state}")
    add_running_task(state['task_id'],function_name)
    # 2、进行必要的非空判定
    local_file_path = state['local_file_path']
    if not local_file_path:
        logger.error(f"[{function_name}]检查发现没有输入文件，无法继续")
        return state
    # 3、判定是否为md/pdf格式文件
    if local_file_path.endswith('.md'):
            # 处理md
            state['is_md_read_enabled']=True
            state['md_path'] = local_file_path
            # 处理pdf
    elif local_file_path.endswith('.pdf'):
        state['is_pdf_read_enabled']=True
        state['pdf_path'] = local_file_path
    else:
        logger.error(f"[{function_name}]文件不是md/pd格式，无法继续")
    # 拿到file_title
    file_title=Path(local_file_path).stem
    state['file_title'] = file_title
    # 4、输出节点的日志输出（节点+参数）会推送给前端
    logger.info(f">>> [{function_name}] 开始结束，当前状态为: {state}")
    add_done_task(state['task_id'],function_name)
    return state