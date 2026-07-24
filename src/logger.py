"""
pyCinemetrics 日志配置
统一日志格式和输出
"""

import logging
import sys
from pathlib import Path


def setup_logging(
    level: int = logging.INFO,
    log_file: str | None = None,
    log_format: str | None = None,
) -> None:
    """
    配置全局日志

    Args:
        level: 日志级别
        log_file: 日志文件路径（可选）
        log_format: 日志格式（可选）
    """
    if log_format is None:
        log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout)
    ]

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers,
        force=True,
    )

    # 设置第三方库日志级别
    logging.getLogger('tensorflow').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    获取模块日志器

    Args:
        name: 模块名称

    Returns:
        logging.Logger 实例
    """
    return logging.getLogger(name)
