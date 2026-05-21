import logging
import sys
from dataclasses import dataclass, field

from app.config import Settings


@dataclass
class LoggingConfig:
    """日志配置结构化模型（与Settings解耦，便于扩展）"""

    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    disabled_loggers: list[str] = field(
        default_factory=lambda: [
            "uvicorn",
            "uvicorn.error",
            "uvicorn.access",
            "uvicorn.asgi",
            "watchfiles.main",
        ]
    )
    # 可选：文件日志配置（拓展能力）
    file_path: str | None = None
    file_max_bytes: int = 10 * 1024 * 1024  # 10MB
    file_backup_count: int = 5


class LogConfigurator:
    """日志配置器（封装配置逻辑，便于复用/测试）"""

    _configured: bool = False  # 避免重复配置

    @classmethod
    def configure(cls, settings: Settings) -> None:
        """
        初始化日志配置
        :param settings: 应用配置（从Settings中解析日志相关配置）
        """
        if cls._configured:
            return  # 防止重复配置

        # 1. 从Settings构建日志配置（支持通过环境变量覆盖默认值）
        _defaults = LoggingConfig()
        log_config = LoggingConfig(
            level="DEBUG" if settings.debug else "INFO",
            format=getattr(settings, "log_format", _defaults.format),
            date_format=getattr(settings, "log_date_format", _defaults.date_format),
            disabled_loggers=getattr(settings, "disabled_loggers", _defaults.disabled_loggers),
            file_path=getattr(settings, "log_file_path", _defaults.file_path),
        )

        # 2. 配置根日志器
        cls._configure_root_logger(log_config)

        # 3. 禁用指定日志器
        cls._disable_loggers(log_config.disabled_loggers)

        # 标记配置完成
        cls._configured = True

    @classmethod
    def _configure_root_logger(cls, config: LoggingConfig) -> None:
        """配置根日志器（控制台+可选文件处理器）"""
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, config.level.upper()))

        # 清除默认处理器（避免重复输出）
        root_logger.handlers.clear()

        # 1. 添加控制台处理器
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(getattr(logging, config.level.upper()))
        console_formatter = logging.Formatter(fmt=config.format, datefmt=config.date_format)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)

        # 2. 可选：添加文件处理器（支持日志轮转）
        if config.file_path:
            from logging.handlers import RotatingFileHandler

            file_handler = RotatingFileHandler(
                filename=config.file_path,
                maxBytes=config.file_max_bytes,
                backupCount=config.file_backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(getattr(logging, config.level.upper()))
            file_handler.setFormatter(console_formatter)
            root_logger.addHandler(file_handler)

    @classmethod
    def _disable_loggers(cls, logger_names: list[str]) -> None:
        """禁用指定日志器（更优雅的方式，避免propagate/handler问题）"""
        for logger_name in logger_names:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.CRITICAL + 1)  # 高于所有级别，等同于禁用
            logger.propagate = False  # 阻止向上传播
            logger.handlers.clear()  # 清除现有处理器

    @classmethod
    def reset(cls) -> None:
        """重置日志配置（测试场景使用）"""
        cls._configured = False
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(logging.WARNING)


def setup_logging(settings: Settings) -> None:
    """应用启动时配置根日志（与 ``main.create_app`` 对接）。"""
    LogConfigurator.configure(settings)


def get_logger(name: str, level: str | None = None) -> logging.Logger:
    """
    获取日志器（支持自定义级别，规范命名）
    :param name: 日志器名称（建议使用__name__）
    :param level: 自定义日志级别（如"DEBUG"，优先级高于根配置）
    :return: 配置好的日志器
    """
    logger = logging.getLogger(name)

    # 可选：设置自定义级别
    if level:
        logger.setLevel(getattr(logging, level.upper()))

    return logger
