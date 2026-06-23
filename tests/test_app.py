import json
from kb_agent.web import app as app_mod
from fastapi.testclient import TestClient


def _tool_msg(cite):
    tm = type("TM", (), {"content": json.dumps({"cite": cite}, ensure_ascii=False)})()
    tm.__class__.__name__ = "ToolMessage"
    return tm


class FakeAgent:
    async def ainvoke(self, inp, config=None):
        class M:  # 末条消息
            content = "答案。来源：D · S · 行1-2"
        # 末条之前夹一个 ToolMessage：守护 /chat 的 _collect_sources 抽取路径
        return {"messages": [_tool_msg({"doc": "D", "section": "S", "lines": "1-2"}), M()]}

    async def astream(self, inp, stream_mode=None, config=None):
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


def test_chat_returns_sources(monkeypatch):
    # /chat（非流式）应与 /chat/stream 口径一致：从 ToolMessage 抽出引用
    r = _client(monkeypatch).post("/chat", json={"message": "问题"})
    assert r.status_code == 200
    assert r.json()["sources"] == ["D · S · 行1-2"]


class ListContentAgent:
    async def ainvoke(self, inp, config=None):
        # 某些 provider 的 content 是 list（[{"type":"text","text":...}]）——不能让它变成 500
        m = type("M", (), {"content": [{"type": "text", "text": "列表"}, {"type": "text", "text": "答案"}]})()
        return {"messages": [m]}


def test_chat_handles_list_content(monkeypatch):
    monkeypatch.setattr(app_mod, "get_agent", lambda: ListContentAgent())
    r = TestClient(app_mod.app).post("/chat", json={"message": "问题"})
    assert r.status_code == 200
    assert r.json()["answer"] == "列表答案"


def test_oversized_body_rejected(monkeypatch):
    # Content-Length 超限应在读 body / 进 pydantic 之前被 413 拒掉
    monkeypatch.setattr(app_mod, "MAX_BODY_BYTES", 5)  # 任何正常 body 都会超
    r = _client(monkeypatch).post("/chat", json={"message": "问题"})
    assert r.status_code == 413


def test_chat_passes_recursion_limit(monkeypatch):
    # 调用 agent 时必须显式传 recursion_limit（否则落到 LangGraph 默认 25）
    seen = {}

    class CapturingAgent:
        async def ainvoke(self, inp, config=None):
            seen["config"] = config
            return {"messages": [type("M", (), {"content": "ok"})()]}

    monkeypatch.setattr(app_mod, "get_agent", lambda: CapturingAgent())
    TestClient(app_mod.app).post("/chat", json={"message": "问题"})
    assert seen["config"]["recursion_limit"] == app_mod.RECURSION_LIMIT


class RecursionAgent:
    async def ainvoke(self, inp, config=None):
        from langgraph.errors import GraphRecursionError
        raise GraphRecursionError("limit")

    async def astream(self, inp, stream_mode=None, config=None):
        from langgraph.errors import GraphRecursionError
        raise GraphRecursionError("limit")
        yield  # async generator


def test_chat_recursion_limit_graceful(monkeypatch):
    # 步数超限：返回 200 + 可执行提示，而非 500 "服务出错"
    monkeypatch.setattr(app_mod, "get_agent", lambda: RecursionAgent())
    r = TestClient(app_mod.app).post("/chat", json={"message": "问题"})
    assert r.status_code == 200
    assert r.json()["answer"] == app_mod._RECURSION_HINT


def test_chat_stream_recursion_limit_emits_error(monkeypatch):
    # 流式步数超限：发 error 事件给出提示，绝不伪装成完成的 answer
    monkeypatch.setattr(app_mod, "get_agent", lambda: RecursionAgent())
    body = TestClient(app_mod.app).post("/chat/stream", json={"message": "问题"}).text
    assert '"type": "error"' in body
    assert app_mod._RECURSION_HINT in body
    assert '"type": "answer"' not in body


def test_chat_stream_emits_events(monkeypatch):
    r = _client(monkeypatch).post("/chat/stream", json={"message": "问题"})
    assert r.status_code == 200
    body = r.text
    assert '"type": "tool"' in body
    assert '"type": "answer"' in body
    assert "D · S · 行1-2" in body


class BoomAgent:
    async def astream(self, inp, stream_mode=None, config=None):
        raise RuntimeError("boom")
        yield  # 使其成为 async generator


def test_chat_stream_error_emits_error_not_answer(monkeypatch):
    monkeypatch.setattr(app_mod, "get_agent", lambda: BoomAgent())
    r = TestClient(app_mod.app).post("/chat/stream", json={"message": "问题"})
    assert r.status_code == 200
    body = r.text
    assert '"type": "error"' in body          # 出错时发 error
    assert '"type": "answer"' not in body      # 绝不伪装成完成的回答


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
