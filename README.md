# pageindex-agent

一个基于 **LangGraph** 的只读知识库检索 agent：把文档解析成层次「文档树」，让模型在树上自主漫游、按内容检索，并给出带引用（文档 → 章节 → 行号）的回答——**没有写死的业务流程**，工具调用的顺序与次数由模型自行决定。

## 架构（三层堆叠，运行时只有一个循环）

- **引擎层**：LangGraph `create_react_agent`（ChatOpenAI 走 OpenRouter + MemorySaver 会话记忆 + `.stream` 流式）。
- **工具层**：5 个 `@tool` —— `list_catalog` / `get_outline` / `open_node` / `read_node` / `search_nodes`。
- **数据底座**：`TreeStore` 仓储层（PageIndex 文档树 + `bm25s` 节点级索引 + 文档目录）。

## 模块

```
kb_agent/
  config.py            # 配置（env）
  tokenize.py          # jieba + 词典分词
  ingest/              # 入库：建树 / 标识符抽取 / 瘦身目录
  index/               # BM25 节点索引（bm25s）
  treestore.py         # 仓储层：加载树/目录/索引，导航 + 检索
  agent/               # tools / prompt / graph（create_react_agent 组装）
  web/                 # FastAPI + SSE 适配 + 服务前端
pageindex/             # vendored：PageIndex 建树
frontend/index.html    # 聊天页（流式 + 工具过程 + 引用）
tests/                 # pytest
```

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env        # 填入 OPENAI_*（建树）与 OPENROUTER_*（对话）的 key

# 1) 入库：把 .md 放进 data/md/，建树 + 目录 + BM25 索引
python -c "from kb_agent.config import settings; from kb_agent.ingest.run import ingest_dir; ingest_dir(settings.md_dir, settings.data_dir, settings.index_model)"

# 2) 启动服务（前后端一体）
uvicorn kb_agent.web.app:app --port 8000
# 浏览器打开 http://127.0.0.1:8000/
```

> 说明：入库/建树用一套 OpenAI 兼容端点（`OPENAI_*` + `PAGEINDEX_INDEX_MODEL`），agent 对话用另一套（`OPENROUTER_*` + `CHAT_MODEL`）。入库新文档后需重启服务。

## 测试

```bash
python -m pytest tests/ -v
```

## License

MIT
