import json
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from langgraph.errors import GraphRecursionError

from ..agent.graph import build_agent, RECURSION_LIMIT
from ..config import settings

# 步数超限时给用户的可执行提示——不是"服务出错请稍后重试"（那暗示是瞬时故障），
# 而是诚实告知检索未在限定步数内收敛，并指引缩小范围。
_RECURSION_HINT = "这个问题涉及的检索步骤较多，未能在限定步数内得到答案。请把问题缩小一点（例如指定具体文档或工序名）后再试。"
from .chatlog import format_args, setup_logging
from .sse import events_from_astream, extract_cite, text_of

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_HTML = PROJECT_ROOT / "frontend" / "index.html"

# 请求体上限：单条消息字符数、保留的历史轮数、单条历史字符数
MAX_MESSAGE_CHARS = 8000
MAX_HISTORY_ITEMS = 40
MAX_HISTORY_ITEM_CHARS = 8000
# 整个请求体硬上限。前端每轮回传全量 history（约 40×8000 字，UTF-8 下 ~1MB），2MB 留足余量；
# 超限在读 body 之前就 413 拒掉——pydantic 的约束发生在 JSON 已解析进内存之后，挡不住超大 body 的 OOM。
MAX_BODY_BYTES = 2 * 1024 * 1024

log = setup_logging(settings.data_dir)

app = FastAPI(title="KB Agent")


@app.middleware("http")
async def _limit_body_size(request, call_next):
    cl = request.headers.get("content-length")
    if cl is not None:
        try:
            if int(cl) > MAX_BODY_BYTES:
                return JSONResponse({"error": "请求体过大"}, status_code=413)
        except ValueError:
            return JSONResponse({"error": "非法的 Content-Length"}, status_code=400)
    return await call_next(request)

_agent = None


def get_agent():
    # 无 checkpointer：Web 端多轮上下文来自前端 history（见 build_agent 文档）
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=MAX_MESSAGE_CHARS)
    thread_id: str | None = None
    history: list | None = None


def _build_messages(req: "ChatRequest") -> list:
    """把前端 history（[{role,content,...}]）+ 本轮 message 拼成对话，支持网页多轮上下文。
    对历史做有界裁剪：只保留最近 MAX_HISTORY_ITEMS 条，每条截断到 MAX_HISTORY_ITEM_CHARS。"""
    msgs = []
    history = (req.history or [])[-MAX_HISTORY_ITEMS:]
    for h in history:
        role = h.get("role") if isinstance(h, dict) else None
        content = h.get("content", "") if isinstance(h, dict) else ""
        if role in ("user", "assistant") and content:
            msgs.append({"role": role, "content": str(content)[:MAX_HISTORY_ITEM_CHARS]})
    msgs.append({"role": "user", "content": req.message})
    return msgs


@app.get("/health")
def health():
    return {"status": "ok"}


def _collect_sources(messages: list) -> list:
    """从完整消息序列里的 ToolMessage 抽引用，去重保序——与 /chat/stream 的口径一致。"""
    sources, seen = [], set()
    for m in messages or []:
        if m.__class__.__name__ != "ToolMessage":
            continue
        for c in extract_cite(getattr(m, "content", "")):
            if c not in seen:
                seen.add(c)
                sources.append(c)
    return sources


@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        result = await get_agent().ainvoke(
            {"messages": _build_messages(req)},
            config={"recursion_limit": RECURSION_LIMIT},
        )
        messages = result["messages"]
        answer = text_of(messages[-1].content)
        return JSONResponse({"answer": answer, "sources": _collect_sources(messages)})
    except GraphRecursionError:
        log.warning("chat hit recursion limit")
        return JSONResponse({"answer": _RECURSION_HINT, "sources": []})
    except Exception:
        log.exception("chat failed")
        return JSONResponse({"error": "服务出错，请稍后重试"}, status_code=500)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    turn = uuid.uuid4().hex[:8]

    async def gen():
        log.info("Query[%s turns=%d]: %s", turn, len(req.history or []), req.message)
        step = 0
        try:
            astream = get_agent().astream(
                {"messages": _build_messages(req)},
                stream_mode=["updates", "messages"],
                config={"recursion_limit": RECURSION_LIMIT},
            )
            async for ev in events_from_astream(astream):
                etype = ev.get("type")
                if etype == "tool":
                    step += 1
                    log.info("  tool[%s #%d]: %s(%s)",
                             turn, step, ev.get("name", ""), format_args(ev.get("args", {})))
                elif etype == "answer":
                    log.info("Answer[%s]: %s", turn, (ev.get("text", "") or "")[:2000])
                    log.info("  sources[%s]: %s", turn, ev.get("sources", []))
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except GraphRecursionError:
            # 步数超限：发独立 error 事件给出可执行提示，而非伪装成完成的回答
            log.warning("chat_stream hit recursion limit [%s]", turn)
            err = {"type": "error", "text": _RECURSION_HINT}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
        except Exception:
            log.exception("chat_stream failed [%s]", turn)
            # 用独立的 error 事件，前端按错误处理；不可发 answer 假装回答完成
            err = {"type": "error", "text": "服务出错，请稍后重试"}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
def index():
    return FileResponse(INDEX_HTML)
