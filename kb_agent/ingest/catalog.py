import hashlib

from .identifiers import extract_identifiers
from ..tokenize import write_domain_terms


def make_doc_id(doc_name: str, uniq: str = "") -> str:
    """doc_id = doc_name(+ 可选 uniq，通常是源文件路径) 的 sha1。
    传 uniq 可避免不同文件同名（doc_name 相同）时 doc_id 碰撞、互相覆盖工作区文件、
    并在 BM25 索引里留下指向已不存在文档的悬挂 handle。"""
    basis = f"{uniq}\n{doc_name}" if uniq else doc_name
    h = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:8]
    return f"doc_{h}"


def build_card(doc_id: str, tree: dict, full_text: str, domain_dict_path) -> dict:
    """地图卡片只留 doc_id/doc_name/doc_description。
    标识符抽取仍执行，但只把型号/编号写进 domain_dict（供 BM25 认整词），不进卡片，避免地图被噪声污染。
    """
    ids = extract_identifiers(full_text)
    write_domain_terms(
        ids["project_ids"] + ids["product_models"] + ids["report_ids"] + ids["drawing_ids"],
        domain_dict_path,
    )
    return {
        "doc_id": doc_id,
        "doc_name": tree.get("doc_name", ""),
        "doc_description": tree.get("doc_description", ""),
    }
