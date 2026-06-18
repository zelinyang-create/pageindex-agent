import json
from pathlib import Path

from .index.bm25_index import BM25Index
from .snippet import make_snippet


class TreeStore:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self._catalog = []
        self._docs = {}          # doc_id -> doc dict
        self._nodes = {}         # handle -> record
        self._top = {}           # doc_id -> [handle,...] 顶层
        self._bm25 = None
        self._load()

    # ---- 加载 ----
    def _load(self):
        cat_path = self.data_dir / "catalog" / "document_catalog.json"
        if cat_path.exists():
            self._catalog = json.loads(cat_path.read_text(encoding="utf-8"))
        ws = self.data_dir / "workspace"
        for f in sorted(ws.glob("doc_*.json")):
            doc = json.loads(f.read_text(encoding="utf-8"))
            self._docs[doc["id"]] = doc
            self._index_doc(doc)
        idx_dir = self.data_dir / "indexes"
        if (idx_dir / "meta.json").exists():
            self._bm25 = BM25Index.load(idx_dir)

    def _index_doc(self, doc):
        doc_id = doc["id"]
        doc_name = doc.get("doc_name", "")
        line_count = doc.get("line_count", 0)
        order = []   # 文档序的 (handle, line_num)，用于算行范围

        def walk(nodes, parent_handle, path_titles):
            handles = []
            for i, n in enumerate(nodes):
                h = f"{doc_id}:{n['node_id']}"
                handles.append(h)
                my_path = path_titles + [n.get("title", "")]
                self._nodes[h] = {
                    "handle": h, "doc_id": doc_id, "doc_name": doc_name,
                    "title": n.get("title", ""), "summary": n.get("summary", "") or "",
                    "text": n.get("text", "") or "", "line_num": n.get("line_num", 0),
                    "parent": parent_handle, "path_titles": my_path,
                    "child_handles": [], "prev": None, "next": None,
                }
                order.append((h, n.get("line_num", 0)))
                child_handles = walk(n.get("nodes", []), h, my_path)
                self._nodes[h]["child_handles"] = child_handles
                # 兄弟 prev/next
                if i > 0:
                    self._nodes[h]["prev"] = handles[i - 1]
                    self._nodes[handles[i - 1]]["next"] = h
            return handles

        self._top[doc_id] = walk(doc.get("structure", []), None, [doc_name])

        # 行范围：按文档序，end = 下一节点 line_num - 1；末节点 = line_count
        order.sort(key=lambda x: x[1])
        for idx, (h, ln) in enumerate(order):
            end = (order[idx + 1][1] - 1) if idx + 1 < len(order) else line_count
            if end < ln:
                end = ln
            self._nodes[h]["lines"] = f"{ln}-{end}"

    # ---- 工具方法 ----
    def list_catalog(self):
        return self._catalog

    def _brief(self, h):
        n = self._nodes[h]
        return {"id": h, "title": n["title"], "summary": n["summary"],
                "lines": n.get("lines", ""), "has_children": bool(n["child_handles"])}

    def get_outline(self, doc_id):
        doc = self._docs.get(doc_id)
        if not doc:
            return {"error": f"unknown doc: {doc_id}"}
        return {"doc": doc_id, "name": doc.get("doc_name", ""),
                "nodes": [self._brief(h) for h in self._top.get(doc_id, [])]}

    def open_node(self, node_id):
        n = self._nodes.get(node_id)
        if not n:
            return {"error": f"unknown node: {node_id}"}
        return {"node": node_id, "title": n["title"],
                "children": [self._brief(c) for c in n["child_handles"]]}

    def read_node(self, node_id):
        n = self._nodes.get(node_id)
        if not n:
            return {"error": f"unknown node: {node_id}"}
        out = {
            "id": node_id, "title": n["title"], "text": n["text"],
            "cite": {"doc": n["doc_name"], "section": n["title"], "lines": n.get("lines", "")},
            "path": " > ".join(n["path_titles"]),
            "parent_id": n["parent"], "prev_id": n["prev"], "next_id": n["next"],
            "has_children": bool(n["child_handles"]),
        }
        sec = self._section_info(node_id)
        if sec:
            out["section"] = sec
        return out

    def _node(self, node_id):
        return self._nodes.get(node_id)

    def _section_span(self, node_id):
        """一个长工序/章节常被切成多个【连续同名兄弟节点】（窗口）。
        返回与本节点标题相同、且在兄弟链上首尾相连的那一串 handle（含自身，按文档序）。"""
        n = self._nodes.get(node_id)
        if not n:
            return [node_id]
        title = (n["title"] or "").strip()
        if not title:
            return [node_id]
        seq = [node_id]
        p = n["prev"]
        while p and (self._nodes.get(p, {}).get("title") or "").strip() == title:
            seq.insert(0, p)
            p = self._nodes[p]["prev"]
        nx = n["next"]
        while nx and (self._nodes.get(nx, {}).get("title") or "").strip() == title:
            seq.append(nx)
            nx = self._nodes[nx]["next"]
        return seq

    def _section_info(self, node_id):
        """若本节点属于一个跨窗口段落（同名兄弟 > 1），返回 {part,total,span}，否则 None。"""
        span = self._section_span(node_id)
        if len(span) <= 1:
            return None
        return {"part": span.index(node_id) + 1, "total": len(span), "span": span}

    def search_nodes(self, query: str, top_k: int = 5):
        if self._bm25 is None:
            return []
        out = []
        for hit in self._bm25.search(query, top_k=top_k):
            h = hit["node_id_full"]
            n = self._nodes.get(h)
            if not n:
                continue
            item = {
                "id": h, "title": n["title"], "score": hit["score"],
                "snippet": make_snippet(n["text"], query),
                "cite": {"doc": n["doc_name"], "section": n["title"], "lines": n.get("lines", "")},
                "path": " > ".join(n["path_titles"]),
                "parent_id": n["parent"], "prev_id": n["prev"], "next_id": n["next"],
            }
            sec = self._section_info(h)
            if sec:
                item["section"] = sec
            out.append(item)
        return out
