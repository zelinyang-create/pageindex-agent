import json
from kb_agent.web import app as app_mod
from fastapi.testclient import TestClient


class FakeAgent:
    def invoke(self, inp, cfg):
        class M:  # 末条消息
            content = "答案。来源：D · S · 行1-2"
        return {"messages": [M()]}

    def stream(self, inp, cfg, stream_mode=None):
        tool_msg = type("TM", (), {"content": json.dumps({"cite": {"doc": "D", "section": "S", "lines": "1-2"}}, ensure_ascii=False)})()
        tool_msg.__class__.__name__ = "ToolMessage"
        ai = type("AI", (), {"tool_calls": [{"name": "read_node", "args": {"node_id": "d:0"}}], "content": ""})()
        chunk = type("CK", (), {"content": "答案。", "tool_calls": []})()
        yield ("updates", {"agent": {"messages": [ai]}})
        yield ("updates", {"tools": {"messages": [tool_msg]}})
        yield ("messages", (chunk, {"langgraph_node": "agent"}))


def _client(monkeypatch):
    monkeypatch.setattr(app_mod, "get_agent", lambda: FakeAgent())
    return TestClient(app_mod.app)


def test_health(monkeypatch):
    assert _client(monkeypatch).get("/health").json() == {"status": "ok"}


def test_chat_returns_answer(monkeypatch):
    r = _client(monkeypatch).post("/chat", json={"message": "问题"})
    assert r.status_code == 200
    assert "答案" in r.json()["answer"]


def test_chat_stream_emits_events(monkeypatch):
    r = _client(monkeypatch).post("/chat/stream", json={"message": "问题"})
    assert r.status_code == 200
    body = r.text
    assert '"type": "tool"' in body
    assert '"type": "answer"' in body
    assert "D · S · 行1-2" in body


def test_build_messages_includes_history():
    from kb_agent.web.app import _build_messages, ChatRequest
    req = ChatRequest(message="那它呢", history=[
        {"role": "user", "content": "HJ900001 结论?"},
        {"role": "assistant", "content": "符合要求", "sources": ["x"]},
    ])
    msgs = _build_messages(req)
    assert msgs[0] == {"role": "user", "content": "HJ900001 结论?"}
    assert msgs[1] == {"role": "assistant", "content": "符合要求"}  # sources 被剥离
    assert msgs[-1] == {"role": "user", "content": "那它呢"}
