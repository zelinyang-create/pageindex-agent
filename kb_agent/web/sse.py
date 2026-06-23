import json


def text_of(content) -> str:
    """把 LangChain 消息的 content 归一成字符串。
    某些 provider（经 OpenRouter 的部分模型）返回 list 形式的 content
    （[{"type":"text","text":...}, ...]），直接当字符串用会在拼接/裁剪处抛 TypeError，
    把正常回答变成 500。这里统一抽取其中的文本片段。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for it in content:
            if isinstance(it, str):
                parts.append(it)
            elif isinstance(it, dict):
                # {"type":"text","text":...} 或退化成其它带 text 的结构
                t = it.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    return str(content)


def _cite_str(cite: dict) -> str:
    return f"{cite.get('doc','')} · {cite.get('section','')} · 行{cite.get('lines','')}"


def _cite_one(item) -> str:
    """优先用工具预格式化好的 cite_text（与 agent 正文粘贴的是同一字符串，保证一致）；
    旧结果或缺字段时回退到从 cite dict 拼接。"""
    if not isinstance(item, dict):
        return ""
    t = item.get("cite_text")
    if isinstance(t, str) and t.strip():
        return t
    if isinstance(item.get("cite"), dict):
        return _cite_str(item["cite"])
    return ""


def extract_cite(content) -> list:
    """从工具返回内容抽引用字符串。content 可能是 dict、list 或 JSON 字符串。"""
    data = content
    if isinstance(content, str):
        try:
            data = json.loads(content)
        except Exception:
            return []
    out = []
    if isinstance(data, dict):
        c = _cite_one(data)
        if c:
            out.append(c)
    elif isinstance(data, list):
        for item in data:
            c = _cite_one(item)
            if c:
                out.append(c)
    return out


def _new_state():
    return {"final": [], "sources": [], "seen": set()}


def _map_event(mode, data, state):
    """把一个 (mode,data) 事件映射成 0..N 个前端事件，并累积最终答案/引用到 state。"""
    events = []
    if mode == "messages":
        chunk, meta = data
        text = text_of(getattr(chunk, "content", ""))
        node = (meta or {}).get("langgraph_node")
        # 只把 agent 节点产出的、有内容的 token 当答案流
        if text and node in (None, "agent"):
            state["final"].append(text)
            events.append({"type": "chunk", "text": text})
    elif mode == "updates":
        for _node, upd in (data or {}).items():
            for m in (upd or {}).get("messages", []) if isinstance(upd, dict) else []:
                for tc in (getattr(m, "tool_calls", None) or []):
                    events.append({"type": "tool", "name": tc.get("name", ""), "args": tc.get("args", {})})
                if m.__class__.__name__ == "ToolMessage":
                    for c in extract_cite(getattr(m, "content", "")):
                        if c not in state["seen"]:
                            state["seen"].add(c)
                            state["sources"].append(c)
    return events


def _answer_event(state):
    return {"type": "answer", "text": "".join(state["final"]), "sources": state["sources"]}


def events_from_stream(stream):
    """把同步 agent.stream(stream_mode=['updates','messages']) 映射成前端事件 tool/chunk/answer。"""
    state = _new_state()
    for mode, data in stream:
        for ev in _map_event(mode, data, state):
            yield ev
    yield _answer_event(state)


async def events_from_astream(astream):
    """异步版：消费 agent.astream(...)，避免在 LLM 流式期间阻塞事件循环。"""
    state = _new_state()
    async for mode, data in astream:
        for ev in _map_event(mode, data, state):
            yield ev
    yield _answer_event(state)
