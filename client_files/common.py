from __future__ import annotations
import logging
import sys
from pathlib import Path

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QColor

logger = logging.getLogger("gui")

COLOR_MAP = {
    "danger": QColor(255, 50, 50),
    "workspace": QColor(0, 191, 255),
    "blank": QColor(50, 205, 50),
}
DEFAULT_COLOR = QColor(200, 200, 200)

class QLogSignal(QObject):
    log_msg = pyqtSignal(str)

class QLogHandler(logging.Handler):
    """Кастомный логгер, который отправляет сообщения в GUI (терминал)."""
    def __init__(self, emitter):
        super().__init__()
        self.emitter = emitter
        self.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S"))

    def emit(self, record):
        msg = self.format(record)
        self.emitter.log_msg.emit(msg)

def setup_logging(log_file_path: Path) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    try:
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logging.info(f"Logging initialized. Log file: {log_file_path}")
    except Exception as e:
        print(f"Failed to create file log handler: {e}", file=sys.stderr)


