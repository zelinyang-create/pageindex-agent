import json


def _cite_str(cite: dict) -> str:
    return f"{cite.get('doc','')} · {cite.get('section','')} · 行{cite.get('lines','')}"


def extract_cite(content) -> list:
    """从工具返回内容抽引用字符串。content 可能是 dict、list 或 JSON 字符串。"""
    data = content
    if isinstance(content, str):
        try:
            data = json.loads(content)
        except Exception:
            return []
    out = []
    if isinstance(data, dict) and isinstance(data.get("cite"), dict):
        out.append(_cite_str(data["cite"]))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("cite"), dict):
                out.append(_cite_str(item["cite"]))
    return out


def events_from_stream(stream):
    """把 agent.stream(stream_mode=['updates','messages']) 的 (mode,data) 流，
    映射成前端事件：tool / chunk / answer。"""
    final_parts = []
    sources = []
    seen_src = set()

    for mode, data in stream:
        if mode == "messages":
            chunk, meta = data
            text = getattr(chunk, "content", "") or ""
            node = (meta or {}).get("langgraph_node")
            # 只把 agent 节点产出的、有内容的 token 当答案流
            if text and node in (None, "agent"):
                final_parts.append(text)
                yield {"type": "chunk", "text": text}
        elif mode == "updates":
            for _node, upd in (data or {}).items():
                for m in (upd or {}).get("messages", []) if isinstance(upd, dict) else []:
                    for tc in (getattr(m, "tool_calls", None) or []):
                        yield {"type": "tool", "name": tc.get("name", ""), "args": tc.get("args", {})}
                    if m.__class__.__name__ == "ToolMessage":
                        for c in extract_cite(getattr(m, "content", "")):
                            if c not in seen_src:
                                seen_src.add(c)
                                sources.append(c)

    yield {"type": "answer", "text": "".join(final_parts), "sources": sources}
