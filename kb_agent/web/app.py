import json
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..agent.graph import build_agent
from ..config import settings
from .chatlog import format_args, setup_logging
from .sse import events_from_astream

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_HTML = PROJECT_ROOT / "frontend" / "index.html"

# 请求体上限：单条消息字符数、保留的历史轮数、单条历史字符数
MAX_MESSAGE_CHARS = 8000
MAX_HISTORY_ITEMS = 40
MAX_HISTORY_ITEM_CHARS = 8000

log = setup_logging(settings.data_dir)

app = FastAPI(title="KB Agent")

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


@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        result = await get_agent().ainvoke({"messages": _build_messages(req)})
        answer = result["messages"][-1].content
        return JSONResponse({"answer": answer, "sources": []})
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
