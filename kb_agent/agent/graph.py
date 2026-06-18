from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from ..config import settings
from .tools import KB_TOOLS
from .prompt import SYSTEM_PROMPT


def build_chat_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=0,
    )


def build_agent(checkpointer=None):
    model = build_chat_model()
    return create_react_agent(
        model=model,
        tools=KB_TOOLS,
        prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer or MemorySaver(),
    )
