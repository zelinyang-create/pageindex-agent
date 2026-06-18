import uuid

from langgraph.checkpoint.memory import MemorySaver

from .agent.graph import build_agent


def main():
    # CLI 不回传 history，靠 checkpointer + 固定 thread_id 维持 REPL 多轮记忆
    agent = build_agent(MemorySaver())
    thread_id = uuid.uuid4().hex
    cfg = {"configurable": {"thread_id": thread_id}}
    print("KB agent CLI（输入 q 退出）")
    while True:
        try:
            q = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("q", "exit", ""):
            break
        result = agent.invoke({"messages": [{"role": "user", "content": q}]}, cfg)
        print("\nagent>", result["messages"][-1].content)


if __name__ == "__main__":
    main()
