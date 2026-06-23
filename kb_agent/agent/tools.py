from langchain_core.tools import tool

from ..config import settings
from ..treestore import TreeStore, DEFAULT_READ_CHARS

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
    """列出知识库里有哪些文档。返回 [{doc_id, doc_name, doc_description}]。
    不知道库里有哪些文档、或该看哪本时先调它。"""
    return get_store().list_catalog()


@tool
def get_outline(doc_id: str) -> dict:
    """看某文档的顶层章节大纲。doc_id 来自 list_catalog。
    返回 {doc, name, nodes:[{id, title, summary, lines, has_children}]}；
    has_children=true 的节点可用 open_node 继续下钻。返回含 error 字段则 doc_id 无效。"""
    return get_store().get_outline(doc_id)


@tool
def open_node(node_id: str) -> dict:
    """展开某节点的直接子章节（只看结构、不读正文）。返回 {node, title, children:[{id,title,summary,lines,has_children}]}。
    node_id 必须原样传回工具结果里给你的句柄，不要自己拼造。返回含 error 字段则 id 无效，请改用 search_nodes/list_catalog 重新定位。"""
    return get_store().open_node(node_id)


@tool
def read_node(node_id: str, offset: int = 0, max_chars: int = DEFAULT_READ_CHARS) -> dict:
    """读某章节正文。返回 {id, title, text, total_chars, cite:{doc,section,lines}, cite_text, path,
    parent_id, prev_id, next_id, has_children, section?, truncated?, next_offset?}。
    正文较长会被截断：若返回 truncated=true，只给了前 max_chars 个字符，要读后续就带 offset=next_offset 再调本工具翻页。
    写引用直接用 cite_text 字段（已格式化好）原样附到回答末尾，不要自己拼接 doc/section/行号；
    parent_id/prev_id/next_id 用于看上级/相邻章节；
    若带 section:{part,total,span}，说明本节点只是"第 part/共 total 段"，完整内容（含附表等子节点）在 span 列出的所有句柄里。
    node_id 原样传回；返回含 error 字段则 id 无效，改用 search_nodes 重新定位。"""
    return get_store().read_node(node_id, offset=offset, max_chars=max_chars)


@tool
def search_nodes(query: str, top_k: int = 8) -> list:
    """按内容检索最相关的章节节点（BM25），细节/针尖问题、不知道在哪本哪节、跨文档查找时用。
    返回 [{id, title, score, snippet, cite, cite_text, path, parent_id, prev_id, next_id, section?}]；命中后视情况用 read_node 读正文。
    写引用用 cite_text 字段（已格式化好）原样粘贴，不要自己拼接。
    若某命中带 section.span，整段已被折叠成这一条代表，完整内容在 span 列出的句柄里。
    注意：返回的是 dict 且含 error 字段时，表示检索索引不可用（不是"没搜到"），不要据此回答"库里没有"。"""
    return get_store().search_nodes(query, top_k=top_k)


KB_TOOLS = [list_catalog, get_outline, open_node, read_node, search_nodes]
