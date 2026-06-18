import uuid

from .agent.graph import build_agent


def main():
    agent = build_agent()
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
