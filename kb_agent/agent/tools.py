from langchain_core.tools import tool

from ..config import settings
from ..treestore import TreeStore

_store = None


def get_store() -> TreeStore:
    global _store
    if _store is None:
        _store = TreeStore(settings.data_dir)
    return _store


def set_store(ts: TreeStore) -> None:
    global _store
    _store = ts


@tool
def list_catalog() -> list:
    """列出知识库里有哪些文档（doc_id、名称、描述）。不知道有哪些文档或该看哪本时调用。"""
    return get_store().list_catalog()


@tool
def get_outline(doc_id: str) -> dict:
    """看某文档的顶层章节大纲（标题+摘要+节点 id）。doc_id 来自 list_catalog。"""
    return get_store().get_outline(doc_id)


@tool
def open_node(node_id: str) -> dict:
    """展开某节点的直接子章节（标题+摘要+子节点 id）。node_id 是工具结果里给你的句柄，原样传回。"""
    return get_store().open_node(node_id)


@tool
def read_node(node_id: str) -> dict:
    """读某章节的正文，返回正文 + 引用(cite) + 面包屑(path/parent_id/prev_id/next_id)。node_id 原样传回。"""
    return get_store().read_node(node_id)


@tool
def search_nodes(query: str, top_k: int = 5) -> list:
    """按内容检索最相关的章节节点（BM25），返回命中节点的片段+引用+面包屑。细节/不知道在哪/跨文档时用。"""
    return get_store().search_nodes(query, top_k=top_k)


KB_TOOLS = [list_catalog, get_outline, open_node, read_node, search_nodes]
