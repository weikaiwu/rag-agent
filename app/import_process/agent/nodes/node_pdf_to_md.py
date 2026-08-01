import os
import shutil
import time
import zipfile
from pathlib import Path
from urllib import error

import requests
from app.core.logger import logger, PROJECT_ROOT
from app.import_process.agent.state import ImportGraphState, create_default_state
from app.utils.task_utils import add_running_task, add_done_task
from app.conf.mineru_config import mineru_config, MineruConfig


def step_1_validate_paths(state):
    logger.debug(f">>> 在pdf转md下开始文件校验")
    pdf_path=state['pdf_path']
    local_dir=state['local_dir']
    if not pdf_path:
        logger.error(f"step_1_validate_paths检查发现没有输入文件，无法继续")
        raise ValueError('step_1_validate_paths检查发现没有输入文件，无法继续')
    if not local_dir:
        # 如果没有，给一个默认值
        local_dir=PROJECT_ROOT/'output'
        logger.info(f"step_1_validate_paths检查发现没有赋值，于是给默认值: {local_dir}")
    pdf_path_obj=Path(pdf_path)
    local_dir_obj=Path(local_dir)

    if not pdf_path_obj.exists():
        logger.error(f"step_1_validate_paths检查发现没有输入文件，无法继续")
        raise FileNotFoundError('step_1_validate_paths检查发现没有输入文件，无法继续')
    if not local_dir_obj.exists():
        logger.error(f"step_1_validate_paths检查发现没有文件夹，主动创建一个文件夹")
        local_dir_obj.mkdir(parents=True,exist_ok=True)
    return pdf_path_obj,local_dir_obj


def step_2_upload_and_poll(pdf_path_obj):
    # 通过mineru将pdf解析成md文件，并且获得他解析后的zip包的下载地址url
    # 1、申请上传的解析地址
    # 需要提供token 解析pdf文件的url 一个写死的头文件
    token = mineru_config.api_key
    url = f"{mineru_config.base_url}/file-urls/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "files": [
            {"name": f"{pdf_path_obj.name}"}
        ],
        "model_version": "vlm"
    }
    file_path = ["demo.pdf"]
    response = requests.post(url,headers=header,json=data)
    # 检查请求状态码是不是200 检查返回状态码是不是0
    if response.status_code != 200 or response.json()["code"] != 0:
        logger.error(f"[step_2_upload_and_poll]请求mineru解析失败，检查文件路径输入是否正确")
        raise   RuntimeError(f"[step_2_upload_and_poll]请求mineru解析失败，检查文件路径输入是否正确")
    # 获取上传地址url
    upload_url = response.json()["data"]["file_urls"][0]
    # 获取等待id
    batch_id=response.json()["data"]["batch_id"]
#     2、将文件上传到解析地址
#     使用put请求将pdf_path_obj传到upload_url里面
#      先禁止走代理
    http_session = requests.Session()
    http_session.trust_env=False
    try:
        with open(pdf_path_obj,'rb') as f:
            file_data = f.read()
        upload_response = http_session.put(upload_url,data=file_data)
        if upload_response.status_code != 200:
            logger.error(f"[step_2_upload_and_poll]请求mineru解析失败，检查文件路径输入是否正确")
            raise RuntimeError(f"[step_2_upload_and_poll]请求mineru解析失败，检查文件路径输入是否正确")
    except Exception as e:
        logger.error(f"[step_2_upload_and_poll]请求mineru解析失败，检查文件路径输入是否正确")
        raise RuntimeError(f"[step_2_upload_and_poll]请求mineru解析失败，检查文件路径输入是否正确")
    finally:
        http_session.close()
    #3、轮询解析结果
    #循环获取结果，设计一个循环3秒一次 等待时间最多为600秒
    url = f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
    timeout_seconds = 600
    poll_interval = 3
    start_time = time.time()
    while True:
        #3.1超时判断，不能假定他是第一次循环
        if time.time() - start_time > timeout_seconds:
            logger.error(f"[step_2_upload_and_poll]请求mineru解析失败，检查文件路径输入是否正确")
            raise TimeoutError(f"[step_2_upload_and_poll]请求mineru解析失败，检查文件路径输入是否正确")
        #3.2向指定的url地址获取解析结果
        res = requests.get(url, headers=header)
        #3.3判断有没有获得解析结果，并获得zip下载的url地址
        #先判断res状态返回数是否正常，如果不为200则有错误如果是5xx则休眠3秒再继续循环别的直接报错，如果等于200成功就开始获取
        #本次结果json，判断有没有获取成功状态码是否为0，接着获取解析结果，还要判断解析状态是否为done最后返回full_zip_url
        if res.status_code != 200:
            if 500 <= res.status_code < 600:
                time.sleep(poll_interval)
                continue
            raise RuntimeError(f"[step_2_upload_and_poll]请求mineru解析失败，检查文件路径输入是否正确")
        json_data = res.json()
        if json_data["code"] != 0:
            raise RuntimeError(f"[step_2_upload_and_poll]请求mineru解析失败，检查文件路径输入是否正确")
        extract_result = json_data["data"]["extract_result"][0]
        if extract_result["state"] == 'done':
            full_zip_url = extract_result["full_zip_url"]
            logger.info(f"解析完成，获得解析zip地址")
            return full_zip_url
        else:
            time.sleep(poll_interval)
def step_3_download_and_extract(zip_url, local_dir_obj, stem) -> str:
    # 先把zip下载下来，再解压到相应文件夹，返回md文件的地址
    # zip_url是要下载的地址，local_dir_obj是下载进去的文件夹，stem是下载下来的文件名
    # 1、先下载zip文件
    extract_target_dir = local_dir_obj / stem
    if extract_target_dir.exists():
        shutil.rmtree(extract_target_dir)
    response = requests.get(zip_url)
    if response.status_code != 200:
        logger.error(f"[step_3_upload_and_poll]下载zip文件失败，检查文件路径输入是否正确")
        raise RuntimeError(f"[step_3_upload_and_poll]下载zip文件失败，检查文件路径输入是否正确")
    # 2、将相应的下载的文件保存到本地
    zip_save_path = local_dir_obj /f"{stem}" / f"{stem}_result.zip"
    zip_save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(zip_save_path,'wb') as f:
        f.write(response.content)
        logger.info(f"[step_3_upload_and_poll]下载zip成功，保存地址为{zip_save_path}")

    # 3、清空下就目录

    extract_target_dir.mkdir(parents=True, exist_ok=True) #新建一个文件夹
    # 4、解压zip文件
    with zipfile.ZipFile(zip_save_path, 'r') as zip_file_object:#创建一个zip对象
        zip_file_object.extractall(extract_target_dir)##解压zip文件
    # 5、将解压出来的结果丢到md地址
    md_file_list=list(extract_target_dir.rglob('*.md'))
    if not md_file_list:
        logger.error(f"[step_3_upload_and_poll]下载zip文件失败，检查文件路径输入是否正确")
        raise RuntimeError(f"[step_3_upload_and_poll]下载zip文件失败，检查文件路径输入是否正确")
    target_md_file=None
    for md_file in md_file_list:
        if md_file.name==stem+".md":
            target_md_file = md_file
            break
    if not target_md_file:
        for md_file in md_file_list:
            if md_file.name.lower()=='full.md':
                target_md_file = md_file
                break
    if not target_md_file:
        target_md_file = md_file_list[0]
    if target_md_file.stem!=stem:
        target_md_file = target_md_file.rename(target_md_file.with_name(f"{stem}.md"))
    final_md_str_path=str(target_md_file.resolve())
    logger.info(f"[step_3_upload_and_poll]已完成md解压最终保存路径为f{final_md_str_path}")
    return final_md_str_path


def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    为什么叫这个名字: 核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    未来要实现:
    1. 调用 MinerU (magic-pdf) 工具。
    2. 将 PDF 转换成 Markdown 格式。
    3. 将结果保存到 state["md_content"]。
    """
    # 1、进入任务的日志和配置
    function_name = "node_pdf_to_md"
    logger.info(f">>> [{function_name}] 开始执行，当前状态为: {state}")
    add_running_task(state['task_id'], function_name)

    try:
        # 2、进行参数校验
        # 参数有state_local_path,local_dir,返回校验后的path文件和目录
        pdf_path_obj,local_dir_obj= step_1_validate_paths(state)
        # 3、调用mineru解析pdf文件，返回一个下载的url地址
        # 参数有要解析的PDF文件的路径，返回一个下载zip包的url
        zip_url=step_2_upload_and_poll(pdf_path_obj)
        #4、下载zip包，并解析与提取
        # 参数有下载zip包的url和解压的文件夹local_dir和解压的zip的文件名
        # 返回一个md文件的路径
        md_path=step_3_download_and_extract(zip_url,local_dir_obj,pdf_path_obj.stem)
        #5、把md_path进行赋值，读取md文件内容丢到md_context中
        # 更新下数据
        state['md_path'] = md_path
        state['local_dir'] = str(local_dir_obj)
        # md内容的读取并赋值给md_context
        with open(md_path,'r',encoding='utf_8') as f:
            state['md_content'] = f.read()
    except Exception as e:
        logger.error(f">>>[{function_name}]使用minerU解析发生了错误，具体异常为{e}")
        raise
    finally:

    # 6、结束的日志与任务的配置
        logger.info(f">>> [{function_name}] 开始结束，当前状态为: {state}")
        add_done_task(state['task_id'], function_name)
    return state

