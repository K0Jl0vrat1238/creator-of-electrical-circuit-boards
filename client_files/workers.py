from __future__ import annotations
import sys
import logging
import time
import json
import base64
import urllib.request
import urllib.error
from pathlib import Path

from PyQt5.QtCore import pyqtSignal, QThread, QObject
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


class ApiWorker(QThread):
    """Универсальный воркер для общения с API станка без зависания интерфейса."""
    success = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, url: str, method: str = "POST", data: dict | None = None):
        super().__init__()
        self.url = url
        self.method = method
        self.data = data

    def run(self):
        try:
            logger.info(f"API Request [{self.method}] {self.url}")
            req = urllib.request.Request(self.url, method=self.method)
            if self.data is not None:
                req.data = json.dumps(self.data).encode('utf-8')
                req.add_header('Content-Type', 'application/json')
                
            with urllib.request.urlopen(req, timeout=500) as response:
                raw = response.read().decode('utf-8')
                res_data = json.loads(raw) if raw else {}
                logger.info(f"API Success: {self.url}")
                self.success.emit(res_data)
                
        except urllib.error.HTTPError as e:
            err = e.read().decode('utf-8')
            try:
                err = json.loads(err).get("detail", err)
            except Exception:
                pass
            logger.error(f"API HTTP Error {e.code}: {err}")
            self.failed.emit(f"Ошибка {e.code}: {err}")
        except Exception as e:
            logger.error(f"API Network Error: {str(e)}")
            self.failed.emit(f"Сбой сети: {str(e)}")


class FetchWorker(QThread):
    success = pyqtSignal(dict)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, ip_addr: str, model_type: str):
        super().__init__()
        self.ip_addr = ip_addr
        self.model_type = model_type

    def run(self):
        try:
            logger.info("Начинаем процесс получения фото и разметки...")
            self.progress.emit("Прогрев камеры...")
            warmup_url = f"http://{self.ip_addr}:8000/api/v0/get_photo?camera=up&cut=true"
            try:
                req = urllib.request.Request(warmup_url, headers={'User-Agent': 'PyQt5'})
                with urllib.request.urlopen(req, timeout=10) as response:
                    response.read()
            except Exception:
                pass 

            time.sleep(2.0)

            result_data = {}
            if self.model_type == "none":
                self.progress.emit("Получение фото без модели...")
                logger.info("Запрашиваем сырое фото...")
                url = f"http://{self.ip_addr}:8000/api/v0/get_photo?camera=up&cut=true"
                req = urllib.request.Request(url, headers={'User-Agent': 'PyQt5'})
                with urllib.request.urlopen(req, timeout=30) as response:
                    img_bytes = response.read()
                    result_data["image"] = base64.b64encode(img_bytes).decode("utf-8")
                    result_data["detections"] = []
                    result_data["segments"] = []
            else:
                self.progress.emit(f"Получение разметки ({self.model_type})...")
                logger.info(f"Запрашиваем yoled фото с моделью {self.model_type}...")
                url = f"http://{self.ip_addr}:8000/api/v0/get_yoled_photo?camera=up&confidence_threshold=0.3&model_type={self.model_type}&cut=true"
                req = urllib.request.Request(url, headers={'User-Agent': 'PyQt5'})
                with urllib.request.urlopen(req, timeout=80) as response:
                    raw_data = response.read().decode('utf-8')
                    result_data = json.loads(raw_data)

            self.progress.emit("Запрос лимитов станка...")
            limits_url = f"http://{self.ip_addr}:8000/api/v0/get_limits"
            try:
                req_lim = urllib.request.Request(limits_url, headers={'User-Agent': 'PyQt5'})
                with urllib.request.urlopen(req_lim, timeout=10) as response:
                    result_data["limits"] = json.loads(response.read().decode('utf-8')).get("limits", {})
            except Exception as e:
                logger.warning(f"Ошибка лимитов: {e}, ставим дефолтные.")
                self.progress.emit(f"Ошибка лимитов: {e}")
                result_data["limits"] = {"x": {"min": 0, "max": 260}, "y": {"min": -160, "max": 20}}

            if result_data.get("segments") and self.model_type != "none":
                self.progress.emit("Конвертация координат (px -> mm)...")
                logger.info("Конвертируем сегменты в миллиметры...")
                conv_url = f"http://{self.ip_addr}:8000/api/v0/pixels_to_mm_bulk"
                payload = json.dumps({"segments": result_data["segments"]}).encode('utf-8')
                req_conv = urllib.request.Request(conv_url, data=payload, headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req_conv, timeout=60) as response:
                    conv_resp = json.loads(response.read().decode('utf-8'))
                    result_data["segments_mm"] = conv_resp.get("segments_mm", [])
            else:
                result_data["segments_mm"] = []

            logger.info("Получение данных успешно завершено!")
            self.success.emit(result_data)

        except Exception as e:
            logger.error(f"Сбой FetchWorker: {e}")
            self.failed.emit(str(e))