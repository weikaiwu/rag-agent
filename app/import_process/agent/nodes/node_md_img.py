import os
import re
import base64
from encodings import utf_8
from pathlib import Path
from typing import Dict, List, Tuple
from collections import deque

from langchain_classic.chains.question_answering.map_reduce_prompt import messages
from langgraph.prebuilt.tool_node import msg_content_output
# MinIO相关依赖
from minio import Minio, deleteobjects
from minio.deleteobjects import DeleteObject
from openai.types.beta.threads import image_file
from torch.utils.tensorboard import summary
from transformers.generation.continuous_batching import requests

# 【核心改造1：移除原生OpenAI，导入LangChain工具类和多模态消息模块】
from app.clients.minio_utils import get_minio_client, bucket_name
from app.import_process.agent.nodes.node_pdf_to_md import step_1_validate_paths
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task
# LLM客户端工具类（核心复用，替换原生OpenAI调用）
from app.lm.lm_utils import get_llm_client
# LangChain多模态依赖（消息构造+异常捕获）
from langchain.messages import HumanMessage
from langchain_core.exceptions import LangChainException
# 项目配置
from app.conf.minio_config import minio_config
from app.conf.lm_config import lm_config
# 项目日志工具（统一使用）
from app.core.logger import logger
# api访问限速工具
from app.utils.rate_limit_utils import apply_api_rate_limit
# 提示词加载工具
from app.core.load_prompt import load_prompt

# MinIO支持的图片格式集合（小写后缀，统一匹配标准）
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

def is_supported_image(filename:str)->bool:
#     判断文件是否为图片
    return os.path.splitext(filename)[1] in IMAGE_EXTENSIONS

def step_1_get_content(state:ImportGraphState) -> Tuple[str,Path,Path]:
    # 1、获取md的地址
    md_file_path=state["md_path"]
    if not md_file_path:
        raise ValueError("md地址不能为空")
    md_path_obj = Path(md_file_path)
    if not md_path_obj.exists():
        raise FileNotFoundError(f'md_path:{md_file_path}不存在')
    # 2、读取md_content
    if not state['md_content']:
        with md_path_obj.open("r",encoding='utf_8') as f:
            state['md_content']= f.read()
    # 3、提取图片文件夹obj
    images_dir_obj=md_path_obj.parent/'images'
    return state['md_content'],md_path_obj,images_dir_obj


def find_image_in_md_content(md_content, image_file,context_length:int=100):
    # 定义正则表达式
    pattern = re.compile(r"!\[.*?"+image_file+r".*?\)")

    content = None#存储图片的结果
    items=list(pattern.finditer(md_content))
    if not items:
        return None
    # 查询符合位置
    if item:=items[0]:
        start,end=item.span()#获取图片的起始和终止位置
    #    截取上下文
    # 截取上文
        pre_text=md_content[max(start-context_length,0):start]# 考虑前面有没有content_length没有就从0开始
    # 截取下文
        post_text=md_content[end:min(end+context_length,len(md_content))]# 考虑后面有没有content_length没有就取最大长度
    # 截取位置前后的内容
        content=(pre_text,post_text)
    if content:
        logger.info(f"图片名字为{image_file},在{md_content[:100]}，该图片的上下文为{content}")
        return content


def step_2_scan_images(md_content, images_dir_obj:Path)-> List[Tuple[str,str,Tuple[str,str]]]:
    # 1、先创建一个目标集合
    targets=[]
    # 2、循环读取images中图图片，检查md中有没有使用图片，有的话就读取上下文
    for image_file in os.listdir(images_dir_obj):
    #检查图片是否可用
     if not is_supported_image(image_file):
        logger.warning("当前文件不是图片")
        continue
        # 是图片就查看是否在md中，是就提取上下文
     content_data=find_image_in_md_content(md_content,image_file)
     if not content_data:
        logger.warning("该图片没有使用，不存在上下文")
        continue
     targets.append((image_file,str(images_dir_obj/image_file),content_data))
    return targets


def step_3_generate_img_summaries(targets, stem):
    summaries={}
    request_times=deque()
    for image_file,image_path,context in targets:
    # 解构 图片名 图片地址 上下文
    # 1、访问限速问题
        apply_api_rate_limit(request_times,max_requests=9)
    # 2、向视觉模型发送图片
    # 2.1模型对象
        vm_model=get_llm_client(model=lm_config.lv_model)
    # 2.2提示词
        prompt=load_prompt('image_summary',root_folder=stem,image_content=context)
        with open(image_path,'rb') as f:
            image_base64=base64.b64encode(f.read()).decode('utf-8')
        messages = [
         {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    },
                },
                {"type": "text", "text": f"{prompt}"},
            ],
        },
    ]
    # 2.3调用大模型
        response = vm_model.invoke(messages)
        summary=response.content.strip().replace('\n','')
        summaries[image_file]=summary
        logger.info(f'图片{image_file},总体结果:{summary}')
    logger.info(f"图片结果为{summaries}")
    return summaries


def step_4_upload_images_and_replace_md(summaries, targets, md_content, stem):
    """
     将图片传到minio
     替换md中的描述和图片
     summaries:图片名 ：描述
     targets:(图片命，地址，（上下文））
     md_content:md原内容
     stem:文件名
     return 新md

    """
    # minio的储存结果为 桶/upload_images/文件夹名字/图片对象.jpg
    minio_client = get_minio_client()
    # 1、删除minio中对应文件的图片
    # 1.1获取你要删除的图片
    object_list=minio_client.list_objects(minio_config.bucket_name,
                                          prefix=f"{minio_config.minio_img_dir[1:]}/{stem}",
                                          recursive=True)
    delete_object_list=[DeleteObject(obj.object_name) for obj in object_list]
    # 1.2调用方法删除

    errors=minio_client.remove_objects(minio_config.bucket_name,delete_object_list)
    for errors in errors:
        logger.error(f'删除对象失效{errors}')
    logger.info(f"已完成{stem}的清空")
    # 2、上传到minio服务器
    # 先声明一个字典收集上传图片结果
    images_url={}
    # 从targets里获得图片名和图片地址，需要桶名，对象名，文件地址，文件类型
    for image_file,image_path,_ in targets:
        try:
            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                object_name=f"{minio_config.minio_img_dir}/{stem}/{image_file}",
                file_path=image_path,
                content_type="image/jpeg"
            )
            # 图片地址=协议＋端点+桶名+对象名
            images_url[image_file]=f"http://{minio_config.endpoint}/{minio_config.bucket_name}{minio_config.minio_img_dir}/{stem}/{image_file}"
    # 上传完毕以后记录
            logger.info(f"完成图片{image_file}的上传，地址为{images_url[image_file]}")
        except Exception as e:
            logger.error(f"图片上传失败:{image_file}，失败原因为{e}")
            # 3、md中图片的替换
            #汇总：{图片名：（描述，url地址）}
    image_infos={}
    for image_file,summary in summaries.items():
        if url:=images_url.get(image_file):
            image_infos[image_file]=(summary,url)
    logger.info(f"图片的汇总结果为:{image_infos}")

    if image_infos:
        for image_file,(summary,url) in image_infos.items():
            rep=re.compile(r"!\[.*?\]\(.*?"+image_file+r".*?\)")
            md_content=rep.sub(f"![{summary}]({url})",md_content)
        logger.info(f'已完成内容的替换，新内容为{md_content}')
    return md_content


def step_5_replace_md_and_save(new_md_content, md_path_obj):
    # 完成新md的磁盘，并返回老地址
    # 新地址:new_md_content
    # 老地址md_path_obj
    new_md_path_str=os.path.splitext(md_path_obj)[0]+'_new.md'
    with open(new_md_path_str,'w',encoding='utf-8') as f:
        f.write(new_md_content)
    logger.info(f"已经完成新内容写入{new_md_path_str}")
    return new_md_path_str

def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)。
    未来要实现:
    1. 扫描 Markdown 中的图片链接。
    2. 将图片上传到 MinIO 对象存储。
    3. (可选) 调用多模态模型生成图片描述。
    4. 替换 Markdown 中的图片链接为 MinIO URL。
    """
    function_name = "node_md_img"
    logger.info(f">>> [{function_name}] 开始执行，当前状态为: {state}")
    add_running_task(state['task_id'], function_name)
    # 1、检验并获取本次处理所需数据
    md_content,md_path_obj,images_dir_obj=step_1_get_content(state)
    if not images_dir_obj.exists():
        logger.info(f"[md文件中没有图片]")
        return state
    # 2、识别images中图片信息，进行图片总结
    targets=step_2_scan_images(md_content,images_dir_obj)
    # 3、进行图片内容的总结和处理（视觉模型）
    summaries=step_3_generate_img_summaries(targets,md_path_obj.stem)
    # 4、上传图片到minio并替换md中的图片（描述＋url）
    new_md_content=step_4_upload_images_and_replace_md(summaries,targets,md_content,md_path_obj.stem)
    # 5、新的md内容的替换和保存
    new_md_file_path=step_5_replace_md_and_save(new_md_content,md_path_obj)
    state['md_path']=new_md_file_path
    state['md_content']=new_md_content
    logger.info(f">>> [{function_name}] 开始结束，当前状态为: {state}")
    add_done_task(state['task_id'], function_name)
    return state

