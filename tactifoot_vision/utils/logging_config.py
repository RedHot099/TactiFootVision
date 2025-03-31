# tactifoot_vision/utils/logging_config.py
import logging
import sys

from loguru import logger as loguru_logger


def setup_logging(level="INFO"):
    """
    Configures logging using Loguru.

    Removes default handlers and adds a new one that formats messages
    and sinks to stderr. Intercepts standard logging messages.

    Args:
        level (str): The minimum logging level (e.g., "DEBUG", "INFO").
    """
    loguru_logger.remove()  # Remove default handler
    loguru_logger.add(
        sys.stderr,
        level=level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # Intercept standard logging messages
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # Get corresponding Loguru level if it exists
            try:
                level = loguru_logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find caller from where originated the logged message
            frame, depth = logging.currentframe(), 2
            while frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            loguru_logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )

    logging.basicConfig(handlers=[InterceptHandler()], level=0)
    loguru_logger.info(f"Loguru logging configured with level: {level.upper()}")
