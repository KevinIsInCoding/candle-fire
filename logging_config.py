from __future__ import annotations

import json
import logging
import logging.handlers
import os
from datetime import datetime, timezone

_LOG_LEVEL = os.getenv("CANDLE_LOG_LEVEL", "WARNING").upper()
_LOG_DIR = os.getenv("CANDLE_LOG_DIR", "logs")

_configured = False


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "data"):
            entry["data"] = record.data
        return json.dumps(entry, default=str)


def _setup() -> None:
    global _configured
    if _configured:
        return
    _configured = True

    level = getattr(logging, _LOG_LEVEL, logging.WARNING)

    root = logging.getLogger("candle_fire")
    root.setLevel(logging.DEBUG)
    root.propagate = False

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    root.addHandler(console)

    os.makedirs(_LOG_DIR, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        os.path.join(_LOG_DIR, "candle_fire.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_JSONFormatter())
    root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    """Return a ``candle_fire.<name>`` logger, configuring handlers on first call."""
    _setup()
    return logging.getLogger(f"candle_fire.{name}")
