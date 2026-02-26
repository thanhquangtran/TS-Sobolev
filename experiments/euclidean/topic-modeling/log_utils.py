# log_utils.py
# Safe helpers for file logging (stdlib) and optional W&B.

from pathlib import Path
import time
import logging

def create_file_logger(log_dir: Path, name: str = "train", level: str = "INFO") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # file
    ts = time.strftime("%Y%m%d-%H%M%S")
    fh = logging.FileHandler(log_dir / f"{name}_{ts}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
