from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from ..config import settings
from .tools import KB_TOOLS
from .prompt import SYSTEM_PROMPT


def build_chat_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=0,
        timeout=60,        # 上游卡住时不要无限挂住一个 worker
        max_retries=2,
    )


def build_agent(checkpointer=None):
    """checkpointer=None → 无服务端记忆（Web 端靠前端传 history 维持多轮）；
    需要服务端会话记忆（如 CLI REPL）时显式传入 MemorySaver 并配合稳定 thread_id。
    切忌"既传全量 history 又用 checkpointer"——会双重叠加上下文。"""
    model = build_chat_model()
    return create_react_agent(
        model=model,
        tools=KB_TOOLS,
        prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
