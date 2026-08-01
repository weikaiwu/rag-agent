import re
import json
import os
# 统一类型注解，避免混用any/Any
from typing import List, Dict, Any, Tuple
# LangChain文本分割器（标注核心用途，便于理解）
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from modelscope.models.science.unifold.config import chunk_size

# 项目内部工具/状态/日志导入（保持原有路径）
from app.utils.task_utils import add_running_task, add_done_task
from app.import_process.agent.state import ImportGraphState
from app.core.logger import logger  # 项目统一日志工具，核心替换print

# --- 配置参数 (Configuration) ---
# 单个Chunk最大字符长度：超过则触发二次切分（适配大模型上下文窗口）
DEFAULT_MAX_CONTENT_LENGTH = 2000
# 短Chunk合并阈值：同父标题的短Chunk会被合并，减少碎片化
MIN_CONTENT_LENGTH = 500


def step_1_get_content(state):
    # 读取要切片的内容
    md_content = state['md_content']
    if not md_content:
        logger.error(f"[step_1_get_content]中md_content文件不存在")
        raise Exception('请检查输入路径是否正确')
    # 处理md中的换行符号
    md_content = md_content.replace('\r\n', '\n').replace('\r', '\n')
    file_title = state.get('file_title','default_title')
    return md_content, file_title


def step_2_split_by_title(md_content, file_title):
    # 1、准备前置工作
    # 1.1正则
    title_pattern=r'^\s*#{1,6}\s+.+'
    # 1.2md——content切割
    lines=md_content.split('\n')
    # 1.3定义临时储存变量
    current_title = ''
    current_lines=[]
    title_count=0
    is_code_block = False
    # 1.4最终存储的列表
    sections = []
    # 2、循环每行列表
    for line in lines:
        strip_line = line.strip()#把每一行前后的空白都去掉放到strip_line
    # 2.1判断代码快状态
        if strip_line.startswith('```') or strip_line.startswith('~~~'):
            is_code_block = not is_code_block
            current_lines.append(line)
            continue

    # 2.2判断是不是标题
        is_title=(not is_code_block)and re.match(title_pattern,strip_line)
        if is_title:
        #检查是不是第一次 不是就先存储
            if current_title:
                sections.append({
                    'title': current_title,
                    'content': '\n'.join(current_lines),
                    'file_title': file_title,
                })
    # 2.3是标题怎么处理
            current_title=strip_line
            current_lines=[current_title]
            title_count+=1
        else:
    # 2.4不是标题怎么处理
            current_lines.append(line)
    if current_title:
        sections.append({
            'title': current_title,
            'content': '\n'.join(current_lines),
            'file_title': file_title,
        })
    # 3、返回结果
    logger.info(f'已经完成chunks语义的粗切,chunks数量为{title_count},切片内容为{sections}')
    return sections,title_count,len(lines)


def split_long_section(section, max_length):
    # 将当前的chunk内容过长二次切割
    # 返回切割后的[{},{}]
    # 1、content获取
    content = section.get('content')
    # 2、判断chunk是否过长，没过长就直接返回
    if len(content) <= max_length:
        logger.info(f'[split_long_section]:{content},当前的chunk未超过{max_length},所以不切割')
        return [section]
    # 3、超长了进行二次切割
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_length,#chunk的长度不可能超过你的max_length
        chunk_overlap=100,#chunk的重叠长度
        separators=['\n\n','\n','。','，','!','：',' ']#在什么地方切割
    )
    # title=标题名  _1,_2,_3  part_1,_2,_3    parent_title=section.title
    sub_sections=[]
    for index,chunk in enumerate(splitter.split_text(content),start=1):
        text=chunk.strip()
        title=f'{section.get("title")}_{index}'
        parent_title=section.get('title')
        part=index
        file_title=section.get('file_title')
        sub_sections.append({
            'title': title,
            'content':text,
            'file_title': file_title,
            'parent_title': parent_title,
            'part': part,
        })
    # 4、返回
    return sub_sections


def merge_short_sections(final_sections, min_length):
    merged_sections = []
    pre_section =None
    for section in final_sections:
        if pre_section is None:
            pre_section=section
            continue
    #     判断pre的长度是否小于最小值且两个切块的parent名字是否相同
        is_pre_short=len(pre_section.get('content'))<min_length
        is_same_parent_title=pre_section.get('parent_title') and (pre_section.get('parent_title')==section.get('parent_title'))
        if is_pre_short and is_same_parent_title:
            pre_section['content']+='\n\n'+section.get('content')
            pre_section['part']=section.get('part')
        else:
            merged_sections.append(pre_section)
            pre_section=section
    if pre_section is not None:
        merged_sections.append(pre_section)
    return merged_sections


def step_3_refine_chunks(sections, max_length,min_length):
    # 1、超过了MAX_CONTENT_LENGTH的块要做切割
    # 2、超过了MIN_CONTENT_LENGTH的块要做合并
    final_sections = []#储存处理以后的块
    # 超过的先切碎
    for section in sections:
        # [{title content file_title,parent_title,part},{},{}]
        sub_section = split_long_section(section,max_length)
        final_sections.extend(sub_section)
    # 小于的再合并
    final_sections=merge_short_sections(final_sections,min_length)
    # 补全属性和参数
    for section in final_sections:
        section['part']=section.get('part')or 1
        section['parent_title']=section.get('parent_title')or section.get('title')
    # 返回
    return final_sections


def step_4_backup_chunks(state, sections):
    local_dir=state.get('local_dir')
    backup_file_path=os.path.join(local_dir,"chunks.json")
    with open(backup_file_path,'w',encoding='utf-8') as f:
        json.dump(sections,f,ensure_ascii=False,indent=4)
    logger.info(f'已经将内容存到{backup_file_path}')


def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 文档切分 (node_document_split)
    为什么叫这个名字: 将长文档切分成小的 Chunks (切片) 以便检索。
    未来要实现:
    1. 基于 Markdown 标题层级进行递归切分。
    2. 对过长的段落进行二次切分。
    3. 生成包含 Metadata (标题路径) 的 Chunk 列表。
    """
    function_name = "node_document_split"
    logger.info(f">>> [{function_name}] 开始执行，当前状态为: {state}")
    add_running_task(state['task_id'], function_name)

    try:
        # 1、参数校验
        md_content,file_title = step_1_get_content(state)
        # 2、粗粒度切割md文件，在保证语义完善的情况下，根据标题切割
        # [{content:标题内容}{title：标题}{file_title:文件名}{}{}]
        sections,title_count,lines_count=step_2_split_by_title(md_content,file_title)
        #3、特殊场景一个文档没有标题，给他一个默认标题
        if title_count==0:
            sections=[{'title':'没有主题','md_content':md_content,'file_title':file_title}]
        #4、细粒度切割md 大小和重叠合适
        sections=step_3_refine_chunks(sections,DEFAULT_MAX_CONTENT_LENGTH,MIN_CONTENT_LENGTH)
        #5、数据的备份和chunks属性的修改
        state['chunks']=sections
        step_4_backup_chunks(state,sections)


    except Exception as e:
        logger.error(f">>>[{function_name}]使用minerU解析发生了错误，具体异常为{e}")
        raise
    finally:
        # 6、结束的日志与任务的配置
        logger.info(f">>> [{function_name}] 开始结束，当前状态为: {state}")
        add_done_task(state['task_id'], function_name)
    return state

