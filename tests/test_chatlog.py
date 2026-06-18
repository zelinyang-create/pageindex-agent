from kb_agent.web.chatlog import format_args, setup_logging


def test_format_args_compact():
    assert format_args({"query": "涂布"}) == '{"query": "涂布"}'


def test_format_args_truncates_long():
    out = format_args({"q": "x" * 500}, limit=50)
    assert len(out) <= 51 and out.endswith("…")


def test_format_args_non_serializable_falls_back():
    out = format_args({"a", "b"})  # set 不是 JSON 可序列化 → 退回 str()
    assert isinstance(out, str) and out


def test_setup_logging_writes_file(tmp_path):
    logger = setup_logging(tmp_path)
    logger.info("hello-test-line")
    for h in logger.handlers:
        h.flush()
    log_file = tmp_path / "logs" / "chat.log"
    assert log_file.exists()
    assert "hello-test-line" in log_file.read_text(encoding="utf-8")
