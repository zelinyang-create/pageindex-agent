"""对话日志：记录每轮 query / 工具调用 / 最终回答+引用，便于事后复盘。

日志同时写到控制台（uvicorn stdout）和 data_dir/logs/chat.log。
"""
import json
import logging
from pathlib import Path

_FMT = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")


def setup_logging(data_dir) -> logging.Logger:
    """配置并返回 kb_agent 日志器。幂等：重复调用不会重复挂同一个 handler，
    但对一个新的 data_dir 仍会补挂对应的文件 handler（方便测试与切库）。"""
    logger = logging.getLogger("kb_agent")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 控制台 handler 只挂一次
    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_console:
        sh = logging.StreamHandler()
        sh.setFormatter(_FMT)
        logger.addHandler(sh)

    # 文件 handler 按目标路径去重
    try:
        log_dir = Path(data_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        target = (log_dir / "chat.log").resolve()
        already = any(
            isinstance(h, logging.FileHandler)
            and Path(getattr(h, "baseFilename", "")).resolve() == target
            for h in logger.handlers
        )
        if not already:
            fh = logging.FileHandler(target, encoding="utf-8")
            fh.setFormatter(_FMT)
            logger.addHandler(fh)
    except Exception:
        logger.warning("无法创建文件日志，仅输出到控制台", exc_info=True)

    return logger


def format_args(args, limit: int = 300) -> str:
    """把工具参数压成单行字符串，过长截断。"""
    try:
        s = json.dumps(args, ensure_ascii=False)
    except Exception:
        s = str(args)
    return s if len(s) <= limit else s[:limit] + "…"
