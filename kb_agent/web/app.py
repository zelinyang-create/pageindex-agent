import json
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from ..agent.graph import build_agent
from ..config import settings
from .chatlog import format_args, setup_logging
from .sse import events_from_stream

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEX_HTML = PROJECT_ROOT / "frontend" / "index.html"

log = setup_logging(settings.data_dir)

app = FastAPI(title="KB Agent")

_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None
    history: list | None = None


def _build_messages(req: "ChatRequest") -> list:
    """把前端 history（[{role,content,...}]）+ 本轮 message 拼成对话，支持网页多轮上下文。"""
    msgs = []
    for h in (req.history or []):
        role = h.get("role") if isinstance(h, dict) else None
        content = h.get("content", "") if isinstance(h, dict) else ""
        if role in ("user", "assistant") and content:
            msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": req.message})
    return msgs


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    cfg = {"configurable": {"thread_id": req.thread_id or uuid.uuid4().hex}}
    result = get_agent().invoke({"messages": _build_messages(req)}, cfg)
    answer = result["messages"][-1].content
    return JSONResponse({"answer": answer, "sources": []})


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    thread_id = req.thread_id or uuid.uuid4().hex
    cfg = {"configurable": {"thread_id": thread_id}}
    turn = uuid.uuid4().hex[:8]

    def gen():
        log.info("Query[%s thread=%s turns=%d]: %s",
                 turn, thread_id, len(req.history or []), req.message)
        step = 0
        try:
            stream = get_agent().stream(
                {"messages": _build_messages(req)},
                cfg, stream_mode=["updates", "messages"],
            )
            for ev in events_from_stream(stream):
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
            err = {"type": "answer", "text": "（服务出错，请稍后重试）", "sources": []}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/")
def index():
    return FileResponse(INDEX_HTML)
