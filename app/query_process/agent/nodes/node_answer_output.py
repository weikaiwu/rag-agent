from app.utils.task_utils import add_running_task, add_done_task, set_task_result
from app.utils.sse_utils import push_to_session, SSEEvent
from app.query_process.agent.state import QueryGraphState
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from app.lm.lm_utils import get_llm_client
from app.clients.mongo_history_utils import save_chat_message
import re
import time
import json
from pathlib import Path
from datetime import datetime

_IMAGE_BLOCK_MARKER = "【图片】"
MAX_CONTEXT_CHARS = 12000  # 限制 prompt 的最大上下文长度，防止超出模型窗口

# TTFT 指标落盘（不受 .env 日志级别限制，便于直接提取首字延迟）
# node_answer_output.py 位于 app/query_process/agent/nodes/，上溯 4 层到 app/，再进 eval/
_TTFT_LOG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "eval" / "ttft_metrics.jsonl"


def _log_ttft(session_id: str, ttft_ms: float) -> None:
    """将首字延迟追加写入结构化指标文件，供外部直接读取/统计。"""
    try:
        _TTFT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "session_id": session_id,
            "ttft_ms": round(ttft_ms, 1),
        }
        with open(_TTFT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"[TTFT] 指标落盘失败：{e}")


def step_1_check_answer(state):
    """若上游已产出确定答案（如 item_name 命中），直接回写，跳过模型润色。"""
    answer = state.get("answer")
    is_stream = state.get("is_stream", False)
    if answer:
        if is_stream:
            push_to_session(state["session_id"], SSEEvent.DELTA, {"delta": answer})
        else:
            set_task_result(state["session_id"], "answer", answer)
        return True
    return False


def step_2_load_prompt(state):
    """拼接上下文（重排片段 + 历史对话 + 项目名）生成答案润色 prompt。"""
    rewritten_query = state.get("rewritten_query") or state.get("original_query")
    reranked_docs = state.get("reranked_docs", [])
    item_names = state.get("item_names", [])
    history = state.get("history", [])

    docs = []
    used_length = 0
    for i, doc in enumerate(reranked_docs, start=1):
        text = doc.get("text")
        source = doc.get("source")
        title = doc.get("title")
        score = doc.get("score")
        content = f"[{i}][source={source}][title={title}][score={score}]\n\n{text}"
        if used_length + len(content) > MAX_CONTEXT_CHARS:
            logger.info(f"本次内容停止追加了！已经大于限制长度！")
            break
        docs.append(content)
        used_length += len(content)

    history_str = ""
    if history and len(history) > 0:
        for message in history:
            role = message.get("role")
            text = message.get("text")
            if role == "user" and text:
                history_str += f"【用户】: {text}\n"
            elif role == "assistant" and text:
                history_str += f"【助手】: {text}\n"
            used_length += len(history_str)
            if used_length > MAX_CONTEXT_CHARS:
                logger.info(f"本次内容停止追加了！已经大于限制长度！")
                break
    else:
        history_str = "没有历史对话记录！"

    item_names_str = ",".join(item_names)
    answer_out_prompt = load_prompt("answer_out",
                                    context="\n\n".join(docs),
                                    history=history_str,
                                    item_names=item_names_str,
                                    question=rewritten_query)
    logger.info(f"已经完成了提示词生成：{answer_out_prompt}")
    return answer_out_prompt


def step_3_create_answer(state, prompt):
    """调用大模型生成最终答案；流式场景测量并落盘 TTFT（首字延迟）。"""
    model = get_llm_client()
    is_stream = state.get("is_stream", False)
    answer = ''
    if is_stream:
        t0 = time.time()  # TTFT 测量起点：prompt 就绪
        first_token_logged = False
        for chunk in model.stream(prompt):
            delta = chunk.content
            if delta and not first_token_logged:
                first_token_logged = True
                ttft_ms = (time.time() - t0) * 1000
                logger.info(f"[TTFT] 首字延迟 = {ttft_ms:.1f} ms")
                _log_ttft(state["session_id"], ttft_ms)
            answer += delta
            push_to_session(state["session_id"], SSEEvent.DELTA, {"delta": delta})
    else:
        response = model.invoke(prompt)
        answer = response.content
        set_task_result(state["session_id"], "answer", answer)

    state['answer'] = answer
    logger.info(f"lm模型最终返回的结果：{answer}")
    return answer


def step_4_extract_images_url(state):
    """从候选片段与联网结果中提取去重后的图片 URL，单独回传供前端渲染。"""
    images = []
    seen_images = set()

    image_reg = re.compile(r"!\[.*?\]\((.*?)\)")
    reranked_docs = state.get("reranked_docs", [])
    for doc in reranked_docs:
        url = doc.get("url")
        if url and url.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")) and url not in seen_images:
            images.append(url)
            seen_images.add(url)

        text = doc.get("text")
        if text:
            for image_url in image_reg.findall(text):
                if image_url not in seen_images:
                    images.append(image_url)
                    seen_images.add(image_url)

    logger.info(f"已经完成图片提取。数量:{len(images)},提取内容：{images}")
    state['image_urls'] = images
    return images


def step_5_write_history(state):
    """将本轮助手回答写入 MongoDB 会话历史。"""
    session_id = state.get("session_id")
    answer = state.get("answer")
    rewritten_query = state.get("rewritten_query") or state.get("original_query")
    item_names = state.get("item_names", [])

    if answer:
        save_chat_message(
            session_id=session_id,
            role="assistant",
            text=answer,
            item_names=item_names,
            rewritten_query=rewritten_query
        )
    logger.info(f"完成了本次对话的记录存储！")


def node_answer_output(state):
    """
    宏观：将最终 topk -> 大模型 -> 润色 -> 结果 ->
        【流式】sse -> 前端（push_to_session）
        【非流式】set_task_result
    """
    add_running_task(state["session_id"], "node_answer_output", state.get("is_stream"))

    answer_exists = step_1_check_answer(state)
    if not answer_exists:
        prompt = step_2_load_prompt(state)
        answer = step_3_create_answer(state, prompt)
        images_url = step_4_extract_images_url(state)
        if images_url:
            push_to_session(state["session_id"],
                            SSEEvent.FINAL,
                            {"answer": answer,
                             "status": "completed",
                             "image_urls": images_url})
    # 无论是否命中直答，都记录会话历史
    step_5_write_history(state)
    add_done_task(state['session_id'], "node_answer_output", state.get("is_stream"))
    return state
