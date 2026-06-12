"""JSON 结构化日志格式化器

将标准 logging 输出转换为 JSON 格式，便于后续解析和检索。
"""
import json
import logging
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    """输出 JSON 格式的结构化日志。"""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        # 附加自定义字段（通过 record 的 extra 传递）
        for key in ("task_id", "node", "duration_ms", "model", "input_tokens",
                     "output_tokens", "collection", "result_count", "error_type"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        # 异常信息
        if record.exc_info and record.exc_info[1]:
            log_entry["error_type"] = type(record.exc_info[1]).__name__
            log_entry["error_message"] = str(record.exc_info[1])

        return json.dumps(log_entry, ensure_ascii=False, default=str)


def setup_json_logging(level: int = logging.INFO):
    """配置 JSON 格式的结构化日志。

    Args:
        level: 日志级别，默认 INFO。
    """
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()  # 清除已有 handler
    root_logger.setLevel(level)
    root_logger.addHandler(handler)
