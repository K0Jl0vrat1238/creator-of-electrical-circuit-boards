# workers.py
from __future__ import annotations
import logging
import time
import json
import math
import base64
import threading
import concurrent.futures
import urllib.request
import urllib.error

from PyQt5.QtCore import pyqtSignal, QThread, QObject, QPointF
from PyQt5.QtGui import QPainterPath

logger = logging.getLogger("gui")

def _format_json(data: dict | None) -> str:
    """Форматирует словарь в компактную JSON-строку с обрезкой до 250 символов."""
    if not data: return "{}"
    try:
        dumped = json.dumps(data, ensure_ascii=False)
        return dumped[:250] + ("..." if len(dumped) > 250 else "")
    except Exception:
        text = str(data)
        return text[:250] + ("..." if len(text) > 250 else "")

class QLogSignal(QObject):
    log_msg = pyqtSignal(str)

class QLogHandler(logging.Handler):
    def __init__(self, emitter):
        super().__init__()
        self.emitter = emitter
        self.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%H:%M:%S"))

    def emit(self, record):
        msg = self.format(record)
        self.emitter.log_msg.emit(msg)

class ApiWorker(QThread):
    success = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, url: str, method: str = "POST", data: dict | None = None, timeout: float = None, silent: bool = False):
        super().__init__()
        self.url = url
        self.method = method
        self.data = data
        self.timeout = timeout
        self.silent = silent

    def run(self):
        try:
            if not self.silent:
                endpoint = self.url.split('/')[-1]
                req_body = _format_json(self.data) if self.data else ""
                log_req = f"📡 [API Запрос] {self.method} /{endpoint} | {req_body}" if req_body else f"📡 [API Запрос] {self.method} /{endpoint}"
                logger.info(log_req)
                
            req = urllib.request.Request(self.url, method=self.method)
            if self.data is not None:
                req.data = json.dumps(self.data).encode('utf-8')
                req.add_header('Content-Type', 'application/json')
                
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                raw = response.read().decode('utf-8')
                res_data = json.loads(raw) if raw else {}
                
                if not self.silent:
                    res_body = _format_json(res_data)
                    logger.info(f"📥 [API Ответ] /{endpoint} | {res_body}")
                    
                self.success.emit(res_data)
                
        except urllib.error.HTTPError as e:
            err = e.read().decode('utf-8')
            try: err = json.loads(err).get("detail", err)
            except Exception: pass
            
            if not self.silent:
                logger.error(f"❌ API Ошибка HTTP {e.code} ({self.url.split('/')[-1]}): {err}")
            self.failed.emit(f"Ошибка {e.code}: {err}")
            
        except Exception as e:
            if not self.silent:
                logger.error(f"❌ API Сбой сети ({self.url.split('/')[-1]}): {str(e)}")
            self.failed.emit(f"Сбой сети: {str(e)}")

class FetchWorker(QThread):
    success = pyqtSignal(dict)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, ip_addr: str, model_type: str):
        super().__init__()
        self.ip_addr = ip_addr
        self.model_type = model_type

    def _sync_get(self, endpoint: str, timeout: float = 30) -> dict | bytes:
        url = f"http://{self.ip_addr}:8000/api/v0/{endpoint}"
        logger.info(f"📡 [API Запрос] GET /{endpoint}")
        req = urllib.request.Request(url, headers={'User-Agent': 'PyQt5'})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = response.read()
            try:
                res_json = json.loads(data.decode('utf-8'))
                logger.info(f"📥 [API Ответ] /{endpoint} | {_format_json(res_json)}")
                return res_json
            except Exception:
                logger.info(f"📥 [API Ответ] /{endpoint} | (Бинарные данные картинки)")
                return data

    def _sync_post(self, endpoint: str, payload: dict, timeout: float = None) -> dict:
        url = f"http://{self.ip_addr}:8000/api/v0/{endpoint}"
        logger.info(f"📡 [API Запрос] POST /{endpoint} | {_format_json(payload)}")
        req = urllib.request.Request(url, method="POST", data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            res_json = json.loads(response.read().decode('utf-8'))
            logger.info(f"📥 [API Ответ] /{endpoint} | {_format_json(res_json)}")
            return res_json

    def run(self):
        try:
            self.progress.emit("Прогрев камеры...")
            try: self._sync_get("get_photo?camera=up&cut=true", timeout=10)
            except Exception: pass 
            time.sleep(2.0)

            result_data = {}
            if self.model_type == "none":
                self.progress.emit("Получение фото без модели...")
                img_bytes = self._sync_get("get_photo?camera=up&cut=true", timeout=30)
                result_data["image"] = base64.b64encode(img_bytes).decode("utf-8")
                result_data["detections"] = []
                result_data["segments"] = []
            else:
                self.progress.emit(f"Получение разметки ({self.model_type})...")
                result_data = self._sync_get(f"get_yoled_photo?camera=up&confidence_threshold=0.3&model_type={self.model_type}&cut=true", timeout=80)

            self.progress.emit("Запрос лимитов станка...")
            try:
                lim_data = self._sync_get("get_limits", timeout=10)
                result_data["limits"] = lim_data.get("limits", {})
            except Exception as e:
                result_data["limits"] = {"x": {"min": 0, "max": 260}, "y": {"min": -160, "max": 20}}

            if result_data.get("segments") and self.model_type != "none":
                self.progress.emit("Конвертация координат (px -> mm)...")
                conv_resp = self._sync_post("pixels_to_mm_bulk", {"segments": result_data["segments"]}, timeout=60)
                result_data["segments_mm"] = conv_resp.get("segments_mm", [])
            else:
                result_data["segments_mm"] = []

            self.success.emit(result_data)

        except Exception as e:
            self.failed.emit(str(e))

class AutoPlacementWorker(QThread):
    finished_success = pyqtSignal(object, float)
    finished_failed = pyqtSignal(str)

    def __init__(self, candidates, angles, precalc_pcb, limits_dict, inverted_blank, danger_path, global_blank_bbox, global_danger_bbox):
        super().__init__()
        self.candidates = candidates
        self.angles = angles
        self.precalc_pcb = precalc_pcb
        
        self.l_min_x = limits_dict.get("x", {}).get("min", 0.0)
        self.l_max_x = limits_dict.get("x", {}).get("max", 260.0)
        self.l_min_y = limits_dict.get("y", {}).get("min", -160.0)
        self.l_max_y = limits_dict.get("y", {}).get("max", 20.0)
        
        self.inverted_blank = inverted_blank
        self.danger_path = danger_path
        self.global_blank_bbox = global_blank_bbox
        self.global_danger_bbox = global_danger_bbox
        
        self._cancel_event = threading.Event()

    def cancel(self):
        self._cancel_event.set()

    def run(self):
        def check_chunk(chunk_data):
            loc_inverted_blank = QPainterPath(self.inverted_blank)
            loc_danger_path = QPainterPath(self.danger_path)
            
            for idx, pt in chunk_data:
                if self._cancel_event.is_set(): return None
                
                for angle in self.angles:
                    pb, pb_bbox, pd, pd_bbox = self.precalc_pcb[angle]
                    t_pb_bbox = pb_bbox.translated(pt.x(), pt.y())
                    
                    if t_pb_bbox.left() < self.l_min_x or t_pb_bbox.right() > self.l_max_x or t_pb_bbox.top() < self.l_min_y or t_pb_bbox.bottom() > self.l_max_y:
                        continue
                    if not self.global_blank_bbox.contains(t_pb_bbox):
                        continue
                        
                    t_pb = pb.translated(pt.x(), pt.y())
                    if loc_inverted_blank.intersects(t_pb): 
                        continue
                        
                    if self.global_danger_bbox is not None:
                        t_pd_bbox = pd_bbox.translated(pt.x(), pt.y())
                        if t_pd_bbox.intersects(self.global_danger_bbox):
                            t_pd = pd.translated(pt.x(), pt.y())
                            if loc_danger_path.intersects(t_pd):
                                continue
                                
                    return (idx, pt, angle)
            return None

        indexed_candidates = list(enumerate(self.candidates))
        if not indexed_candidates:
            self.finished_failed.emit("Нет кандидатов для размещения")
            return
            
        chunk_size = math.ceil(len(indexed_candidates) / 4)
        chunks = [indexed_candidates[i:i + chunk_size] for i in range(0, len(indexed_candidates), chunk_size)]
        
        valid_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(check_chunk, c) for c in chunks]
            for future in concurrent.futures.as_completed(futures):
                if self._cancel_event.is_set(): break
                res = future.result()
                if res is not None:
                    valid_results.append(res)
                    
        if self._cancel_event.is_set():
            self.finished_failed.emit("Отменено пользователем")
            return
            
        if valid_results:
            valid_results.sort(key=lambda x: x[0])
            best_match = valid_results[0]
            self.finished_success.emit(best_match[1], best_match[2])
        else:
            self.finished_failed.emit("Не найдено безопасного места")