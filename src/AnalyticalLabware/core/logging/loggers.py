import logging
from typing import Optional


def get_logger(logger_name: Optional[str] = None) -> logging.Logger:
    if logger_name is None:
        return logging.getLogger("AnalyticalLabware")

    return logging.getLogger(f"AnalyticalLabware.{logger_name}")
