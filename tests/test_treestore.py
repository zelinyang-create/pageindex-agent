import json
from pathlib import Path
from kb_agent.treestore import TreeStore


def _make_data(tmp_path):
    ws = tmp_path / "workspace"; ws.mkdir(parents=True)
    (tmp_path / "catalog").mkdir(parents=True)
    doc = {"id": "doc_a", "type": "md", "path": "x.md",
           "doc_name": "工艺文件", "doc_description": "工艺说明",
           "line_count": 20,
           "structure": [
               {"title": "5 SMT", "node_id": "0001", "line_num": 1, "summary": "贴片回流",
                "text": "本章工艺", "nodes": [
                   {"title": "5.1 印刷", "node_id": "0002", "line_num": 3, "summary": "锡膏",
                    "text": "印刷正文", "nodes": []},
                   {"title": "5.3 回流焊", "node_id": "0003", "line_num": 8, "summary": "回流曲线",
                    "text": "峰值 245℃", "nodes": []}]},
               {"title": "6 检验", "node_id": "0004", "line_num": 15, "summary": "检验项目",
                "text": "外观检验", "nodes": []}]}
    (ws / "doc_a.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    (ws / "_meta.json").write_text(json.dumps({"doc_a": {"doc_name": "工艺文件"}}, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "catalog" / "document_catalog.json").write_text(
        json.dumps([{"doc_id": "doc_a", "doc_name": "工艺文件", "doc_description": "工艺说明"}], ensure_ascii=False),
        encoding="utf-8")
    return tmp_path


def test_list_catalog(tmp_path):
    ts = TreeStore(_make_data(tmp_path))
    cat = ts.list_catalog()
    assert cat[0]["doc_id"] == "doc_a" and cat[0]["doc_name"] == "工艺文件"


def test_get_outline_top_level_only(tmp_path):
    ts = TreeStore(_make_data(tmp_path))
    out = ts.get_outline("doc_a")
    ids = [n["id"] for n in out["nodes"]]
    assert ids == ["doc_a:0001", "doc_a:0004"]      # 只有顶层
    assert out["nodes"][0]["has_children"] is True


def test_open_node_children(tmp_path):
    ts = TreeStore(_make_data(tmp_path))
    ch = ts.open_node("doc_a:0001")
    assert [c["id"] for c in ch["children"]] == ["doc_a:0002", "doc_a:0003"]


def test_read_node_breadcrumb_and_cite(tmp_path):
    ts = TreeStore(_make_data(tmp_path))
    r = ts.read_node("doc_a:0003")
    assert r["text"] == "峰值 245℃"
    assert r["parent_id"] == "doc_a:0001"
    assert r["prev_id"] == "doc_a:0002"
    assert r["path"] == "工艺文件 > 5 SMT > 5.3 回流焊"
    assert r["cite"]["doc"] == "工艺文件"
    assert r["cite"]["section"] == "5.3 回流焊"
    # 行范围：0003 起于 8，下一个文档序节点 0004 起于 15 → 8-14
    assert r["cite"]["lines"] == "8-14"
