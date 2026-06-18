import json
from kb_agent.web.sse import events_from_stream, extract_cite


class FakeAI:
    """模拟带 tool_calls 的 AIMessage（updates 模式里出现）。"""
    def __init__(self, tool_calls=None, content=""):
        self.tool_calls = tool_calls or []
        self.content = content


class FakeChunk:
    """模拟 messages 模式的 AIMessageChunk。"""
    def __init__(self, content):
        self.content = content
        self.tool_calls = []


class FakeTool:
    """模拟 ToolMessage。"""
    def __init__(self, content):
        self.content = content
    @property
    def __class__name__(self): return "ToolMessage"


def test_extract_cite_from_read_node_json():
    payload = json.dumps({"cite": {"doc": "HJ900001鉴定报告", "section": "检验报告", "lines": "43-79"}}, ensure_ascii=False)
    assert extract_cite(payload) == ["HJ900001鉴定报告 · 检验报告 · 行43-79"]


def test_extract_cite_from_search_hits_json():
    payload = json.dumps([{"cite": {"doc": "D", "section": "S1", "lines": "1-2"}},
                          {"cite": {"doc": "D", "section": "S2", "lines": "3-4"}}], ensure_ascii=False)
    assert extract_cite(payload) == ["D · S1 · 行1-2", "D · S2 · 行3-4"]


def test_events_sequence_tool_chunk_answer():
    # 模拟：agent 节点先发一个 tool_call(updates) → tool 结果(updates) → 答案 token(messages)
    tool_ai = FakeAI(tool_calls=[{"name": "read_node", "args": {"node_id": "doc_a:0003"}}])
    tool_msg = type("TM", (), {"content": json.dumps({"cite": {"doc": "工艺文件", "section": "5.3 回流焊", "lines": "8-14"}}, ensure_ascii=False)})()
    tool_msg.__class__.__name__ = "ToolMessage"
    stream = [
        ("updates", {"agent": {"messages": [tool_ai]}}),
        ("updates", {"tools": {"messages": [tool_msg]}}),
        ("messages", (FakeChunk("峰值"), {"langgraph_node": "agent"})),
        ("messages", (FakeChunk("245℃"), {"langgraph_node": "agent"})),
    ]
    evs = list(events_from_stream(stream))
    types = [e["type"] for e in evs]
    assert types == ["tool", "chunk", "chunk", "answer"]
    assert evs[0] == {"type": "tool", "name": "read_node", "args": {"node_id": "doc_a:0003"}}
    assert evs[1]["text"] == "峰值" and evs[2]["text"] == "245℃"
    assert evs[3]["type"] == "answer"
    assert evs[3]["text"] == "峰值245℃"
    assert "工艺文件 · 5.3 回流焊 · 行8-14" in evs[3]["sources"]
