import json
from pathlib import Path

from .tree import build_tree
from .catalog import make_doc_id, build_card
from ..index.nodes import iter_nodes
from ..index.bm25_index import build_index


def ingest_dir(md_dir, out, model: str) -> None:
    md_dir = Path(md_dir)
    out = Path(out)
    ws = out / "workspace"; ws.mkdir(parents=True, exist_ok=True)
    (out / "catalog").mkdir(parents=True, exist_ok=True)
    domain_dict = out / "domain_dict_auto.txt"

    cards = []
    meta = {}
    all_records = []

    for md_path in sorted(md_dir.glob("*.md")):
        full_text = md_path.read_text(encoding="utf-8")
        tree = build_tree(str(md_path), model=model)
        # uniq=源文件路径：两个不同 .md 即使 doc_name 相同也不会拿到同一个 doc_id
        doc_id = make_doc_id(tree["doc_name"], uniq=str(md_path.resolve()))
        if doc_id in meta:
            raise ValueError(f"doc_id 碰撞：{doc_id}（{md_path}）——同路径同名重复入库？")

        # 落树
        doc = {"id": doc_id, "type": "md", "path": str(md_path.resolve()),
               "doc_name": tree["doc_name"], "doc_description": tree.get("doc_description", ""),
               "line_count": tree.get("line_count", 0), "structure": tree["structure"]}
        (ws / f"{doc_id}.json").write_text(
            json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        meta[doc_id] = {"type": "md", "doc_name": doc["doc_name"],
                        "doc_description": doc["doc_description"],
                        "path": doc["path"], "line_count": doc["line_count"]}

        # 卡片（瘦身 + 写词典）；把 doc_name 并入搜索文本，确保文件名中的项目号/型号也被抽出
        # _spacing() inside extract_identifiers handles CJK/ASCII boundary — no patch needed here
        search_text = tree.get("doc_name", "") + "\n" + full_text
        cards.append(build_card(doc_id, tree, search_text, domain_dict))
        # 收集节点
        all_records.extend(iter_nodes(doc_id, tree))

    (ws / "_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "catalog" / "document_catalog.json").write_text(
        json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8")

    if all_records:
        build_index(all_records).save(out / "indexes")

    print(f"ingested {len(cards)} docs, {len(all_records)} nodes → {out}")
