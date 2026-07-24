"""
pyCinemetrics 辅助工具
资源路径管理和启动画面
"""

import sys
import os
import platform
import importlib
from pathlib import Path
from typing import Optional

from src.logger import get_logger

logger = get_logger(__name__)


def resource_path(relative_path: str) -> Optional[Path]:
    """
    获取资源文件的绝对路径
    支持 PyInstaller 打包和普通运行

    Args:
        relative_path: 相对路径

    Returns:
        资源文件的绝对路径，失败返回 None
    """
    try:
        if platform.system() == 'Windows':
            base_path = getattr(sys, '_MEIPASS', Path.cwd())
        else:
            base_path = getattr(sys, '_MEIPASS', Path(__file__).parent)

        return Path(base_path) / relative_path
    except Exception as e:
        logger.error("Failed to resolve resource path '%s': %s", relative_path, e)
        return None


class Splash:
    """PyInstaller 启动画面管理"""

    def __init__(self):
        self.splash = None
        if '_PYIBoot_SPLASH' in os.environ:
            try:
                if importlib.util.find_spec('pyi_splash'):
                    import pyi_splash  # type: ignore
                    self.splash = pyi_splash
            except Exception as e:
                logger.debug("Splash screen not available: %s", e)

    def close(self) -> None:
        """关闭启动画面"""
        if self.splash:
            try:
                self.splash.close()
            except Exception as e:
                logger.debug("Failed to close splash: %s", e)

    def update(self, text: str) -> None:
        """更新启动画面文本

        Args:
            text: 显示文本
        """
        if self.splash:
            try:
                self.splash.update_text(text)
            except Exception as e:
                logger.debug("Failed to update splash: %s", e)
