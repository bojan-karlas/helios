"""Logging setup (run.log + console)."""
from __future__ import annotations

import logging
from pathlib import Path


def get_logger(name: str = "helios") -> logging.Logger:
    return logging.getLogger(name)


def configure_logging(log_dir: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("helios")
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "run.log")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
