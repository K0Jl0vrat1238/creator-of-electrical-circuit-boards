import os
import time
import base64
import threading
import torch
import torch.nn as nn
import numpy as np
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import cv2 as cv
from fastapi import FastAPI, HTTPException, Query, Response
import uvicorn
from ultralytics import YOLO
from pydantic import BaseModel
from contextlib import asynccontextmanager

from cnc_control import CNCMachine
from cnc_config import *

class MoveRequest(BaseModel):
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    diagonal: bool = False


class ZProbePoint(BaseModel):
    x: float
    y: float
    z: Optional[float] = None


class ZProbeRequest(BaseModel):
    points: List[ZProbePoint]
    V_MAX: Optional[float] = None
    A_MAX: Optional[float] = None

    

# ================= Настройки геометрии стола =================
# Угол поворота картинки (в градусах). Положительный — против часовой, отрицательный — по часовой.
ROTATION_ANGLE = 0.5

# Координаты ббокса для обрезки стола (для разрешения 1920x1080)
# Формат: [ymin, xmin, ymax, xmax]
TABLE_CROP_BOX = [95, 105, 1035, 1665]
# =============================================================

cnc = CNCMachine()
cnc.setup()
cnc.shutdown()
# --- НЕЙРОСЕТЬ ДЛЯ КАЛИБРОВКИ ---
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(),
            nn.Linear(dim, dim), nn.SiLU()
        )
    def forward(self, x): return x + self.block(x)

class LensCalibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(2, 128), nn.SiLU())
        self.core = nn.Sequential(ResBlock(128), ResBlock(128), ResBlock(128))
        self.head = nn.Linear(128, 2)
    def forward(self, x): return self.head(self.core(self.embed(x)))

# Загрузка модели
calibrator_model = None
calibrator_mean = None
calibrator_std = None

try:
    checkpoint = torch.load('calibrator.pth', map_location='cpu', weights_only=False)
    calibrator_model = LensCalibrator()
    calibrator_model.load_state_dict(checkpoint['model_state'])
    calibrator_model.eval()
    calibrator_mean = checkpoint['x_mean']
    calibrator_std = checkpoint['x_std']
    print("[INFO] Нейросеть калибровки линзы загружена!")
except Exception as e:
    print(f"[WARN] Ошибка загрузки calibrator.pth: {e}")

class MovePixelRequest(BaseModel):
    px_x: float
    px_y: float
    z: Optional[float] = None
    diagonal: bool = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Здесь код, который выполняется при старте (если ничего не надо, просто пропускаем)
    yield
    # А здесь — то, что сработает строго при выключении сервера
    cnc.shutdown()

app = FastAPI(title="CNC_Server", lifespan=lifespan)

# Настройки путей к камерам
UP_CAMERA_ID = os.getenv("UP_CAMERA_ID", "/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._A4tech_FHD_1080P_PC_Camera_SN0001-video-index0")
HAND_CAMERA_ID = os.getenv("HAND_CAMERA_ID", "/dev/v4l/by-id/usb-Generic_USB2.0_PC_CAMERA-video-index0")
CAMERA_TIMEOUT_SECONDS = 30.0
SERIALIZE_USB_CAMERAS = os.getenv("SERIALIZE_USB_CAMERAS", "0").lower() in {"1", "true", "yes", "on"}

# Пути к моделям YOLO
YOLO_MODEL_PATH_FAST = os.getenv("YOLO_MODEL_PATH_FAST", "nano_last_ncnn_model")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "best-medium.pt")

# Блокировки ресурсов
camera_lock = threading.Lock()
target_cache_lock = threading.Lock()
target_cache: Dict[str, "CameraTarget"] = {}

# Глобальная инициализация моделей YOLO
yolo_model_fast: Optional[YOLO] = None
yolo_model_standard: Optional[YOLO] = None

try:
    path_fast = Path(YOLO_MODEL_PATH_FAST)
    if path_fast.exists():
        yolo_model_fast = YOLO(str(path_fast), task='segment')
        print(f"[INFO] Быстрая YOLO модель успешно загружена: {path_fast}")
except Exception as e:
    print(f"[WARN] Ошибка при загрузке быстрой модели YOLO: {e}")

try:
    path_std = Path(YOLO_MODEL_PATH)
    if path_std.exists():
        yolo_model_standard = YOLO(str(path_std), task='segment')
        print(f"[INFO] Стандартная YOLO модель успешно загружена: {path_std}")
except Exception as e:
    print(f"[WARN] Ошибка при загрузке стандартной модели YOLO: {e}")

@dataclass
class CameraTarget:
    source: Union[int, str]
    label: str

def _open_capture(target: CameraTarget) -> cv.VideoCapture:
    return cv.VideoCapture(target.source, cv.CAP_ANY)

def resolve_camera(camera_id: Union[int, str]) -> CameraTarget:
    wanted = str(camera_id).strip()
    if not wanted:
        raise HTTPException(status_code=400, detail="Camera id is empty")

    with target_cache_lock:
        cached = target_cache.get(wanted)
        if cached is not None:
            return cached

    if wanted.lower().startswith("index:"):
        wanted = wanted.split(":", 1)[1].strip()

    if wanted.isdigit():
        target = CameraTarget(int(wanted), f"index:{wanted}")
        with target_cache_lock:
            target_cache[str(camera_id).strip()] = target
        return target

    if wanted.startswith("/dev/") or Path(wanted).exists():
        target = CameraTarget(wanted, wanted)
        with target_cache_lock:
            target_cache[str(camera_id).strip()] = target
        return target

    for base in (Path("/dev/v4l/by-id"), Path("/dev/v4l/by-path")):
        if base.exists():
            for link in base.iterdir():
                if wanted.lower() in link.name.lower():
                    resolved_path = str(link.resolve())
                    target = CameraTarget(resolved_path, link.name)
                    with target_cache_lock:
                        target_cache[str(camera_id).strip()] = target
                    return target

    raise HTTPException(status_code=404, detail=f"Camera '{wanted}' was not found")

def _io_lock():
    return camera_lock if SERIALIZE_USB_CAMERAS else nullcontext()


class AutoTimeoutCamera:
    """
    Класс управления камерой со встроенным автоотключением и фиксированными параметрами.
    Использует разделение grab/retrieve для предотвращения отставания кадров в буфере.
    """
    def __init__(self, camera_id: Union[int, str], width: int, height: int, target_fps: float, exposure: Optional[float] = None):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.exposure = exposure
        self.timeout = CAMERA_TIMEOUT_SECONDS

        self.lock = threading.Lock()
        self.cap: Optional[cv.VideoCapture] = None
        self.thread: Optional[threading.Thread] = None
        
        self.frame: Optional[Any] = None
        self.ret: bool = False
        self.running: bool = False
        self.last_access: float = 0.0

    def _reader(self, target: CameraTarget):
        self.cap = _open_capture(target)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            with self.lock:
                self.running = False
            return

        # Настройка формата MJPEG для предотвращения перегрузки USB шины
        self.cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Минимизируем буфер на уровне драйвера
        self.cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

        # Принудительная ручная экспозиция, если задана (1.0 = ручной режим в V4L2)
        if self.exposure is not None:
            self.cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 1.0)
            self.cap.set(cv.CAP_PROP_EXPOSURE, float(self.exposure))

        # Быстрый прогрев сенсора
        for _ in range(5):
            self.cap.grab()

        delay = 1.0 / self.target_fps
        last_saved = 0.0

        try:
            while True:
                now = time.monotonic()
                with self.lock:
                    if now - self.last_access > self.timeout:
                        self.running = False
                        break

                # Шаг 1: Вызываем grab() непрерывно на максимальной скорости камеры.
                # Это очищает буфер ОС и гарантирует отсутствие задержки.
                # grab() заблокирует поток ровно до прихода нового кадра с сенсора.
                with _io_lock():
                    grabbed = self.cap.grab()

                if not grabbed:
                    time.sleep(0.01)  # Защита от зависания при сбое камеры
                    continue

                # Шаг 2: Декодируем (retrieve) кадр только если пришло время по нашей частоте (FPS)
                if now - last_saved >= delay:
                    with _io_lock():
                        ret, frame = self.cap.retrieve()
                    if ret and frame is not None:
                        with self.lock:
                            self.frame = frame
                            self.ret = True
                        last_saved = now
        finally:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            with self.lock:
                self.frame = None
                self.ret = False
                self.running = False

    def get_frame(self) -> Optional[Any]:
        now = time.monotonic()
        target = resolve_camera(self.camera_id)

        with self.lock:
            self.last_access = now
            if not self.running:
                self.running = True
                self.frame = None
                self.ret = False
                self.thread = threading.Thread(target=self._reader, args=(target,), daemon=True)
                self.thread.start()

        # Ожидание первого кадра при старте
        start_wait = time.monotonic()
        while time.monotonic() - start_wait < 5.0:
            with self.lock:
                if self.ret and self.frame is not None:
                    return self.frame
                if not self.running:
                    return None
            time.sleep(0.05)
        return None

# Инстанцирование камер с фиксированными параметрами
up_camera_instance = AutoTimeoutCamera(
    UP_CAMERA_ID, 
    width=1920, 
    height=1080, 
    target_fps=1.0, 
    exposure=15.0  # Принудительная экспозиция для верхней камеры
)

hand_camera_instance = AutoTimeoutCamera(
    HAND_CAMERA_ID, 
    width=640, 
    height=480, 
    target_fps=5.0, 
    exposure=None  # Экспозиция нижней камеры не настраивается принудительно
)

def _get_camera_instance(camera: str) -> AutoTimeoutCamera:
    resolved = camera.strip().lower()
    if resolved == "up":
        return up_camera_instance
    elif resolved == "hand":
        return hand_camera_instance
    raise HTTPException(status_code=400, detail="Поддерживаются только камеры 'up' и 'hand'")

def _get_active_yolo(model_type: str) -> YOLO:
    if model_type == "fast":
        model = yolo_model_fast
        name = "Быстрая модель"
    else:
        model = yolo_model_standard
        name = "Стандартная модель"

    if model is None:
        raise HTTPException(
            status_code=500,
            detail=f"{name} не была инициализирована или не найдена на сервере."
        )
    return model

@app.get("/api/v0/get_photo")
def get_photo(
    camera: str = Query("up", description="'up' или 'hand'"),
    cut: bool = Query(False, description="Обрезать изображение по границам стола (только для камеры 'up')"),
) -> Response:
    cam = _get_camera_instance(camera)
    frame = cam.get_frame()
    if frame is None:
        raise HTTPException(status_code=500, detail=f"Не удалось получить кадр с камеры {camera}")
    
    # Геометрические преобразования делаем только для верхней камеры
    if camera.strip().lower() == "up":
        # И поворот, и кроп выполняются ТОЛЬКО если cut=True
        if cut:
            h, w = frame.shape[:2]
            
            # Шаг 1: Поворот картинки вокруг центра, если задан угол
            if ROTATION_ANGLE != 0.0:
                center = (w // 2, h // 2)
                matrix = cv.getRotationMatrix2D(center, ROTATION_ANGLE, 1.0)
                frame = cv.warpAffine(frame, matrix, (w, h), flags=cv.INTER_LINEAR)
            
            # Шаг 2: Кропаем по ббоксу
            ymin, xmin, ymax, xmax = TABLE_CROP_BOX
            ymin_clamped = max(0, min(ymin, h))
            ymax_clamped = max(0, min(ymax, h))
            xmin_clamped = max(0, min(xmin, w))
            xmax_clamped = max(0, min(xmax, w))
            
            if ymax_clamped > ymin_clamped and xmax_clamped > xmin_clamped:
                frame = frame[ymin_clamped:ymax_clamped, xmin_clamped:xmax_clamped]
            
            # Шаг 3: Дополнительный поворот на 180 градусов по часовой
            frame = cv.rotate(frame, cv.ROTATE_180)
    
    # Кодирование в JPEG с фиксированным качеством 85%
    success, encoded_img = cv.imencode(
        ".jpg",
        frame,
        [int(cv.IMWRITE_JPEG_QUALITY), 85],
    )
    if not success:
        raise HTTPException(status_code=500, detail="Ошибка кодирования кадра")
        
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")

@app.get("/api/v0/get_yoled_photo")
def get_yoled_photo(
    camera: str = Query("up", description="'up' или 'hand'"),
    model_type: str = Query("fast", description="Тип модели: 'fast' или 'standard'"),
    confidence_threshold: float = Query(0.25, ge=0.0, le=1.0, description="Порог уверенности детекции"),
    cut: bool = Query(False, description="Обрезать изображение по границам стола до подачи в YOLO (только для камеры 'up')"),
) -> Dict[str, Any]:
    cam = _get_camera_instance(camera)
    frame = cam.get_frame()
    if frame is None:
        raise HTTPException(status_code=500, detail=f"Не удалось получить кадр с камеры {camera}")

    # Геометрическая предобработка перед нейронкой только для верхней камеры
    if camera.strip().lower() == "up" and cut:
        h, w = frame.shape[:2]
        
        # Шаг 1: Поворот картинки вокруг центра, если задан угол
        if ROTATION_ANGLE != 0.0:
            center = (w // 2, h // 2)
            matrix = cv.getRotationMatrix2D(center, ROTATION_ANGLE, 1.0)
            frame = cv.warpAffine(frame, matrix, (w, h), flags=cv.INTER_LINEAR)
        
        # Шаг 2: Кропаем по ббоксу стола
        ymin, xmin, ymax, xmax = TABLE_CROP_BOX
        ymin_clamped = max(0, min(ymin, h))
        ymax_clamped = max(0, min(ymax, h))
        xmin_clamped = max(0, min(xmin, w))
        xmax_clamped = max(0, min(xmax, w))
        
        if ymax_clamped > ymin_clamped and xmax_clamped > xmin_clamped:
            frame = frame[ymin_clamped:ymax_clamped, xmin_clamped:xmax_clamped]
            
        # Шаг 3: Дополнительный поворот на 180 градусов по часовой
        frame = cv.rotate(frame, cv.ROTATE_180)

    model = _get_active_yolo(model_type)
    results = model(frame, conf=confidence_threshold, verbose=False)
    result = results[0]

    detections = []
    if hasattr(result, "boxes") and result.boxes is not None:
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            name = model.names[class_id] if class_id in model.names else str(class_id)
            detections.append(
                {
                    "box": [float(val) for val in xyxy],
                    "confidence": confidence,
                    "class_id": class_id,
                    "name": name,
                }
            )

    segments = []
    if hasattr(result, "masks") and result.masks is not None:
        for segment in result.masks.xy:
            segments.append([[float(pt[0]), float(pt[1])] for pt in segment])

    # Кодирование обработанного (или оригинального) изображения в JPEG
    success, encoded_img = cv.imencode(
        ".jpg",
        frame,
        [int(cv.IMWRITE_JPEG_QUALITY), 85],
    )
    if not success:
        raise HTTPException(status_code=500, detail="Ошибка кодирования результирующего кадра")

    image_base64 = base64.b64encode(encoded_img.tobytes()).decode("utf-8")

    return {
        "camera_id": str(camera),
        "image": image_base64,
        "orig_shape": list(result.orig_shape) if hasattr(result, "orig_shape") else None,
        "detections": detections,
        "segments": segments,
    }
    
@app.post("/api/v0/parking")
def api_parking():
    if cnc.is_busy:
        raise HTTPException(status_code=400, detail="Станок сейчас занят другой операцией")
    cnc.is_busy = True
    try:
        # Вся магия умной парковки теперь запускается одной строчкой!
        cnc.home_all()
        
        return {"status": "success", "message": "Парковка успешна. Координаты обнулены.", "current_pos": cnc.get_coordinates_mm()}
    except Exception as e:
        cnc.is_homed = False
        raise HTTPException(status_code=500, detail=f"Ошибка парковки: {str(e)}")
    finally:
        cnc.is_busy = False
    
@app.post("/api/v0/move_absolute")
def api_move_absolute(req: MoveRequest):
    if cnc.is_busy:
        raise HTTPException(status_code=400, detail="Станок сейчас занят другой операцией")
    if not cnc.is_homed:
        raise HTTPException(status_code=400, detail="Сперва запустите паркинг")
    cnc.is_busy = True
    try:
        cnc.move_absolute(x=req.x, y=req.y, z=req.z, diagonal=req.diagonal)
        return {"status": "success", "current_pos": cnc.get_coordinates_mm()}
    except RuntimeError as err:
        raise HTTPException(status_code=400, detail=str(err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cnc.is_busy = False

@app.post("/api/v0/move_relative")
def api_move_relative(req: MoveRequest):
    if cnc.is_busy:
        raise HTTPException(status_code=400, detail="Станок сейчас занят другой операцией")
    if not cnc.is_homed:
        raise HTTPException(status_code=400, detail="Сперва запустите паркинг")
    cnc.is_busy = True
    try:
        cnc.move_relative(x=req.x, y=req.y, z=req.z, diagonal=req.diagonal)
        return {"status": "success", "current_pos": cnc.get_coordinates_mm()}
    except RuntimeError as err:
        raise HTTPException(status_code=400, detail=str(err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cnc.is_busy = False

@app.post("/api/v0/z_probe")
def api_z_probe(req: ZProbeRequest):
    if cnc.is_busy:
        raise HTTPException(status_code=400, detail="Станок занят")
    if not cnc.is_homed:
        raise HTTPException(status_code=400, detail="Сперва выполните паркинг")

    cnc.is_busy = True
    try:
        points = [point.dict() for point in req.points]
        result_points = cnc.z_probe_points(points, v_max=req.V_MAX, a_max=req.A_MAX)
        return {
            "status": "success",
            "points": result_points,
            "current_pos": cnc.get_coordinates_mm()
        }
    except RuntimeError as err:
        raise HTTPException(status_code=400, detail=str(err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cnc.is_busy = False

@app.get("/api/v0/get_coords")
def api_get_coords():
    if not cnc.is_homed:
        return {"coordinates": None}
    coords = cnc.get_coordinates_mm()
    return {
        "coordinates": {
            "x": coords['X'],
            "y": coords['Y'],
            "z": coords['Z']
        }
    }

@app.get("/api/v0/get_limits")
def api_get_limits():
    return {
        "limits": {
            "x": {"min": MIN_X, "max": MAX_X},
            "y": {"min": MIN_Y, "max": MAX_Y},
            "z": {"min": MIN_Z, "max": MAX_Z}
        }
    }

@app.post("/api/v0/stop")
def api_stop():
    cnc.pause = True
    cnc.is_homed = False
    cnc.z_probe_active = False
    cnc.active_operation = None
    cnc.interrupted_operation = None
    cnc.resume_target = None
    
    # Полностью сбрасываем кэш пробинга при экстренном стопе
    if hasattr(cnc, 'remaining_z_probe_points'):
        cnc.remaining_z_probe_points = []
    if hasattr(cnc, 'z_probe_results'):
        cnc.z_probe_results = []
    if hasattr(cnc, 'first_touch_z'):
        cnc.first_touch_z = None
    
    cnc.setup()
    cnc.shutdown()
    
    return {
        "status": "stopped",
        "message": "СТОП! Все обесточено, пробинг сброшен, выполните паркинг.",
        "current_pos": cnc.get_coordinates_mm() if cnc.is_homed else None
    }

@app.post("/api/v0/pause")
def api_pause():
    if not cnc.is_homed:
        return {
            "status": "not_homed",
            "message": "Станок не запаркован, выполните паркинг.",
            "current_pos": cnc.get_coordinates_mm() if cnc.is_homed else None
        }
    if not cnc.is_busy:
        return {
            "status": "ready",
            "message": "Станок не занят, нет необходимости в паузе.",
            "current_pos": cnc.get_coordinates_mm() if cnc.is_homed else None
        }
    # Активируем стоп-флаг, чтобы циклы шагов прервались
    cnc.pause = True
    if getattr(cnc, 'active_operation', None) == 'z_probe':
        cnc.resume_target = None
                
    return {
        "status": "interrupted",
        "message": "Пауза! Движение заблокировано.",
        "current_pos": cnc.get_coordinates_mm() if cnc.is_homed else None
    }

@app.post("/api/v0/resume")
def api_resume():
    if cnc.is_busy:
        raise HTTPException(status_code=400, detail="Станок сейчас занят другой операцией")
    if not cnc.is_homed:
        raise HTTPException(status_code=400, detail="Сперва запустите паркинг")
        
    cnc.is_busy = True
    try:
        # Снимаем паузу
        cnc.pause = False
        
        # Проверяем, какая операция была прервана
        if getattr(cnc, 'interrupted_operation', None) == 'z_probe':
            v_max = getattr(cnc, 'z_probe_v_max', None)
            a_max = getattr(cnc, 'z_probe_a_max', None)
            # Запускаем пробинг в режиме возобновления
            result_points = cnc.z_probe_points([], v_max=v_max, a_max=a_max, is_resume=True)
            return {
                "status": "success",
                "message": "Z-probing успешно возобновлен и завершен!",
                "points": result_points,
                "current_pos": cnc.get_coordinates_mm()
            }
        else:
            # Обычное возобновление перемещения
            if not getattr(cnc, 'resume_target', None):
                raise HTTPException(status_code=400, detail="Нет сохраненной точки для возобновления работы")
            target = cnc.resume_target
            cnc.move_absolute(
                x=target['x'], 
                y=target['y'], 
                z=target['z'], 
                diagonal=target.get('diagonal', False)
            )
            return {
                "status": "success", 
                "message": "Работа успешно возобновлена с места остановки!", 
                "current_pos": cnc.get_coordinates_mm()
            }
    except RuntimeError as err:
        raise HTTPException(status_code=400, detail=str(err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cnc.is_busy = False
        
@app.post("/api/v0/move_pixel")
def api_move_pixel(req: MovePixelRequest):
    if calibrator_model is None:
        raise HTTPException(status_code=500, detail="Модель калибровки не загружена на сервере")
    if cnc.is_busy:
        raise HTTPException(status_code=400, detail="Станок сейчас занят другой операцией")
    if not cnc.is_homed:
        raise HTTPException(status_code=400, detail="Сперва запустите паркинг")

    # Конвертация пикселей в мм
    pt = np.array([req.px_x, req.px_y])
    norm_pt = (pt - calibrator_mean) / calibrator_std
    inputs = torch.FloatTensor([norm_pt])
    with torch.no_grad():
        preds = calibrator_model(inputs)
    
    mm_x, mm_y = preds[0][0].item(), preds[0][1].item()
    
    cnc.is_busy = True
    try:
        # Едем в вычисленные координаты
        cnc.move_absolute(x=mm_x, y=mm_y, z=req.z, diagonal=req.diagonal)
        return {"status": "success", "calculated_mm": {"x": mm_x, "y": mm_y}, "current_pos": cnc.get_coordinates_mm()}
    except RuntimeError as err:
        raise HTTPException(status_code=400, detail=str(err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cnc.is_busy = False
        
class BulkPixelRequest(BaseModel):
    segments: List[List[List[float]]]

@app.post("/api/v0/pixels_to_mm_bulk")
def api_pixels_to_mm_bulk(req: BulkPixelRequest):
    if calibrator_model is None:
        raise HTTPException(status_code=500, detail="Модель калибровки не загружена на сервере")
    
    results = []
    for segment in req.segments:
        if not segment:
            results.append([])
            continue
        
        # Перегоняем сразу весь сегмент (список точек) через модель
        pts = np.array(segment) 
        norm_pts = (pts - calibrator_mean) / calibrator_std
        inputs = torch.FloatTensor(norm_pts)
        with torch.no_grad():
            preds = calibrator_model(inputs)
        
        mm_pts = preds.numpy().tolist()
        results.append(mm_pts)
        
    return {"status": "success", "segments_mm": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
