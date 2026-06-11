# main.py
from __future__ import annotations
import shutil
import sys
import math
import logging
import argparse
import time
import json
import base64
import urllib.request
import urllib.error
from pathlib import Path
import cv2
import numpy as np
import yaml

from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal, QMimeData, QThread
from PyQt5.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QBrush, QPixmap, QPolygonF, QFont
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QGraphicsPixmapItem,
    QGraphicsScene, QGraphicsView, QMainWindow, QMessageBox, QPushButton,
    QRadioButton, QStyleFactory, QTabWidget, QVBoxLayout, QWidget, QCheckBox, QFrame, QSlider
)
from pipeline import PipelineConfig, PipelineError, PipelineResult, run_pipeline, compose_layers_to_image
from svg_export import export_svg_bundle
from gerber_importer import render_gerber_and_excellon

from ip_by_mac import get_ip_by_mac

logger = logging.getLogger("gui")

COLOR_MAP = {
    "danger": QColor(255, 50, 50),
    "workspace": QColor(0, 191, 255),
    "blank": QColor(50, 205, 50),
}
DEFAULT_COLOR = QColor(200, 200, 200)

# --- ГЕОМЕТРИЧЕСКИЕ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
def polygon_area(poly: QPolygonF) -> float:
    """Вычисление площади полигона методом Гаусса (Shoelace formula)."""
    n = poly.count()
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += poly[i].x() * poly[j].y()
        area -= poly[j].x() * poly[i].y()
    return abs(area) / 2.0

def point_to_segment_distance(p: QPointF, s1: QPointF, s2: QPointF) -> float:
    """Минимальное расстояние от точки до отрезка."""
    dx = s2.x() - s1.x()
    dy = s2.y() - s1.y()
    if dx == 0 and dy == 0:
        return math.hypot(p.x() - s1.x(), p.y() - s1.y())
    t = ((p.x() - s1.x()) * dx + (p.y() - s1.y()) * dy) / (dx*dx + dy*dy)
    t = max(0.0, min(1.0, t))
    proj_x = s1.x() + t * dx
    proj_y = s1.y() + t * dy
    return math.hypot(p.x() - proj_x, p.y() - proj_y)

def polygon_to_polygon_distance(poly1: QPolygonF, poly2: QPolygonF) -> float:
    """Минимальное расстояние между границами двух полигонов."""
    min_dist = float('inf')
    n1 = poly1.count()
    n2 = poly2.count()
    if n1 < 2 or n2 < 2:
        return 0.0
    
    for i in range(n1):
        p = poly1[i]
        for j in range(n2):
            s1 = poly2[j]
            s2 = poly2[(j + 1) % n2]
            dist = point_to_segment_distance(p, s1, s2)
            if dist < min_dist:
                min_dist = dist
                
    for i in range(n2):
        p = poly2[i]
        for j in range(n1):
            s1 = poly1[j]
            s2 = poly1[(j + 1) % n1]
            dist = point_to_segment_distance(p, s1, s2)
            if dist < min_dist:
                min_dist = dist
                
    return min_dist


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
            req = urllib.request.Request(self.url, method=self.method)
            if self.data is not None:
                req.data = json.dumps(self.data).encode('utf-8')
                req.add_header('Content-Type', 'application/json')
                
            with urllib.request.urlopen(req, timeout=500) as response:
                raw = response.read().decode('utf-8')
                self.success.emit(json.loads(raw) if raw else {})
                
        except urllib.error.HTTPError as e:
            err = e.read().decode('utf-8')
            try:
                err = json.loads(err).get("detail", err)
            except Exception:
                pass
            self.failed.emit(f"Ошибка {e.code}: {err}")
        except Exception as e:
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
                url = f"http://{self.ip_addr}:8000/api/v0/get_photo?camera=up&cut=true"
                req = urllib.request.Request(url, headers={'User-Agent': 'PyQt5'})
                with urllib.request.urlopen(req, timeout=30) as response:
                    img_bytes = response.read()
                    result_data["image"] = base64.b64encode(img_bytes).decode("utf-8")
                    result_data["detections"] = []
                    result_data["segments"] = []
            else:
                self.progress.emit(f"Получение разметки ({self.model_type})...")
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
                self.progress.emit(f"Ошибка лимитов: {e}")
                result_data["limits"] = {"x": {"min": 0, "max": 260}, "y": {"min": -160, "max": 20}}

            if result_data.get("segments") and self.model_type != "none":
                self.progress.emit("Конвертация координат (px -> mm)...")
                conv_url = f"http://{self.ip_addr}:8000/api/v0/pixels_to_mm_bulk"
                payload = json.dumps({"segments": result_data["segments"]}).encode('utf-8')
                req_conv = urllib.request.Request(conv_url, data=payload, headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req_conv, timeout=60) as response:
                    conv_resp = json.loads(response.read().decode('utf-8'))
                    result_data["segments_mm"] = conv_resp.get("segments_mm", [])
            else:
                result_data["segments_mm"] = []

            self.success.emit(result_data)

        except Exception as e:
            self.failed.emit(str(e))


class LayerComposerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Сборка из отдельных слоев")
        self.setMinimumWidth(550)
        layout = QVBoxLayout(self)
        
        grid = QGridLayout()
        layers_info = [
            ('holes', 'Отверстия (Blue):'),
            ('green', 'Зеленый / Шелкография (Green):'),
            ('paths', 'Дорожки (Black):'),
            ('cut', 'Контур (Red):')
        ]
        
        self.edits = {}
        for row, (key, label_text) in enumerate(layers_info):
            grid.addWidget(QLabel(label_text), row, 0)
            edit = QLineEdit()
            edit.setReadOnly(True)
            self.edits[key] = edit
            grid.addWidget(edit, row, 1)
            
            btn = QPushButton("Выбрать...")
            btn.clicked.connect(lambda _, k=key: self._browse(k))
            grid.addWidget(btn, row, 2)
            
        layout.addLayout(grid)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
    def _browse(self, key: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, f"Выберите слой: {key}", "", "Растровые форматы (*.jpg *.jpeg *.png *.bmp)"
        )
        if path:
            self.edits[key].setText(path)
            
    def get_paths(self) -> dict[str, str]:
        return {k: edit.text() for k, edit in self.edits.items() if edit.text()}


class GerberComposerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Импорт Gerber / Excellon")
        self.setMinimumWidth(550)
        layout = QVBoxLayout(self)
        
        grid = QGridLayout()
        layers_info = [
            ('holes', 'Карта сверловки (.drl / .txt):'),
            ('green', 'Шелкография (.gto / .gbo / .gbr):'),
            ('paths', 'Проводящие дорожки (.gtl / .gbl / .gbr):'),
            ('cut', 'Контур платы (.gko / .gbr):')
        ]
        
        self.edits = {}
        for row, (key, label_text) in enumerate(layers_info):
            grid.addWidget(QLabel(label_text), row, 0)
            edit = QLineEdit()
            edit.setReadOnly(True)
            self.edits[key] = edit
            grid.addWidget(edit, row, 1)
            
            btn = QPushButton("Выбрать...")
            btn.clicked.connect(lambda _, k=key: self._browse(k))
            grid.addWidget(btn, row, 2)
            
        layout.addLayout(grid)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
    def _browse(self, key: str) -> None:
        file_filter = "Координаты сверловки (*.drl *.txt *.xln)" if key == 'holes' else "Файлы Gerber (*.gbr *.gtl *.gbl *.gko *.gto *.gbo)"
        path, _ = QFileDialog.getOpenFileName(self, f"Выберите файл для: {key}", "", file_filter)
        if path:
            self.edits[key].setText(path)
            
    def get_paths(self) -> dict[str, str]:
        return {k: edit.text() for k, edit in self.edits.items() if edit.text()}


class PreviewView(QGraphicsView):
    pixelClicked = pyqtSignal(float, float)
    mouseMovedInScene = pyqtSignal(float, float)
    rightClickedInScene = pyqtSignal(float, float)
    keyPressed = pyqtSignal(int)

    def __init__(self) -> None:
        super().__init__(QGraphicsScene())
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(QColor("#efefef"))
        self.setMinimumSize(400, 300)
        self.setStyleSheet("border: 1px solid #808080;")
        self._zoom = 1.0
        self._panning = False
        self._pan_start = None
        self._pan_moved = False
        self._selection_mode = False
        self.setCursor(Qt.OpenHandCursor)

    def set_selection_mode(self, enabled: bool) -> None:
        self._selection_mode = enabled
        if enabled:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.OpenHandCursor)

    def fit_to_scene(self) -> None:
        self.resetTransform()
        self._zoom = 1.0
        if self.scene().items():
            self.scene().setSceneRect(self.scene().itemsBoundingRect())
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)
            self._zoom = self.transform().m11()

    def wheelEvent(self, event) -> None:
        if not self.scene().items() or event.angleDelta().y() == 0:
            return
        factor = 1.2 if event.angleDelta().y() > 0 else 1.0 / 1.2
        next_zoom = self._zoom * factor
        if next_zoom < 0.05 or next_zoom > 80:
            return
        self.scale(factor, factor)
        self._zoom = next_zoom

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._panning = True
            self._pan_moved = False
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        elif event.button() == Qt.RightButton:
            pos = self.mapToScene(event.pos())
            self.rightClickedInScene.emit(pos.x(), pos.y())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        pos = self.mapToScene(event.pos())
        self.mouseMovedInScene.emit(pos.x(), pos.y())

        if self._panning and self._pan_start is not None:
            delta = event.pos() - self._pan_start
            if delta.manhattanLength() > 3:
                self._pan_moved = True
                self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
                self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y()) 
            self._pan_start = event.pos()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._panning = False
            self._pan_start = None
            if self._selection_mode:
                self.setCursor(Qt.CrossCursor)
            else:
                self.setCursor(Qt.OpenHandCursor)
            
            if self._selection_mode and not self._pan_moved:
                pos = self.mapToScene(event.pos())
                self.pixelClicked.emit(pos.x(), pos.y())
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        self.keyPressed.emit(event.key())
        super().keyPressEvent(event)


class LayerListWidget(QListWidget):
    orderChanged = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropOverwriteMode(False) 
        self.setDefaultDropAction(Qt.MoveAction) 
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setAlternatingRowColors(True)
        self._dragged_row: int | None = None

    def startDrag(self, supportedActions: Qt.DropActions) -> None:
        self._dragged_row = self.currentRow()
        super().startDrag(supportedActions)

    def dropMimeData(self, index: int, data: QMimeData, action: Qt.DropAction) -> bool:
        if action == Qt.MoveAction and self._dragged_row is not None:
            item_to_move = self.takeItem(self._dragged_row)
            if item_to_move is None:
                self._dragged_row = None
                return False

            target_row = index
            if self._dragged_row < target_row:
                target_row -= 1
            
            self.insertItem(target_row, item_to_move)
            self.setCurrentItem(item_to_move) 
            self._dragged_row = None
            self.orderChanged.emit()
            return True
        return super().dropMimeData(index, data, action)


class MainWindow(QMainWindow):
    def __init__(self, skip_validation: bool = False) -> None:
        super().__init__()
        self.setWindowTitle("Фрезеровщик печатных плат")
        self.setMinimumSize(1140, 760)
        self._base_dir = Path(__file__).resolve().parent
        self._image_path: Path | None = None
        self._source_qimage: QImage | None = None
        self._last_result: PipelineResult | None = None
        self._initial_skip_validation = skip_validation

        # Состояния станка
        self.current_x = None
        self.current_y = None
        self.current_z = None
        self.is_homed = False
        self.api_worker = None

        # Данные для вкладки Размещение
        self.placement_img: QImage | None = None
        self.placement_detections = []
        self.placement_segments = []
        self.placement_segments_mm = []
        self.placement_limits = {}

        # Интерактивное позиционирование платы на бланке
        self.placement_active = False
        self.placement_preview_item = None
        self.placed_pcb_pos = QPointF(0, 0)
        self.placed_pcb_angle = 0.0
        self.placed_pcbs = []  # Содержит dict {'pos': QPointF, 'angle': float, 'polygon_mm': QPolygonF}

        self._build_ui()
        self._init_layer_list()
        
        QTimer.singleShot(1000, self._api_get_coords)

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        self.tabs = QTabWidget()
        outer.addWidget(self.tabs)

        self.controls_tab = QWidget()
        self.tabs.addTab(self.controls_tab, "Настройки")
        self._build_controls_tab(self.controls_tab)

        self.preview_tab = QWidget()
        self.tabs.addTab(self.preview_tab, "Предпросмотр")
        self._build_preview_tab(self.preview_tab)

        self.placement_tab = QWidget()
        self.tabs.addTab(self.placement_tab, "Размещение")
        self._build_placement_tab(self.placement_tab)

    def _build_controls_tab(self, parent: QWidget) -> None:
        outer = QVBoxLayout(parent)
        outer.setSpacing(10)

        top_panel = QHBoxLayout()
        self.coord_label = QLabel("Координаты: X: --- | Y: --- | Z: ---")
        self.coord_label.setFont(QFont("Arial", 12, QFont.Bold))
        top_panel.addWidget(self.coord_label)

        btn_refresh = QPushButton("🔄")
        btn_refresh.setToolTip("Обновить координаты")
        btn_refresh.setFixedSize(35, 35)
        btn_refresh.clicked.connect(self._api_get_coords)
        top_panel.addWidget(btn_refresh)
        
        top_panel.addSpacing(20)

        btn_home = QPushButton("🏠")
        btn_home.setToolTip("Выполнить паркинг")
        btn_home.setFixedSize(35, 35)
        btn_home.clicked.connect(self.run_homing)
        top_panel.addWidget(btn_home)

        btn_pause = QPushButton("⏸️")
        btn_pause.setToolTip("Приостановить текущую команду")
        btn_pause.setFixedSize(35, 35)
        btn_pause.clicked.connect(self.pause_command)
        top_panel.addWidget(btn_pause)

        btn_play = QPushButton("▶️")
        btn_play.setToolTip("Возобновить выполнение команды")
        btn_play.setFixedSize(35, 35)
        btn_play.clicked.connect(self.resume_command)
        top_panel.addWidget(btn_play)

        btn_stop = QPushButton("🛑 СТОП")
        btn_stop.setToolTip("Экстренная остановка")
        btn_stop.setFont(QFont("Arial", 14, QFont.Bold))
        btn_stop.setFixedSize(140, 40)
        btn_stop.setStyleSheet("background-color: #FF3232; color: white; border-radius: 6px;")
        btn_stop.clicked.connect(self.emergency_stop)
        top_panel.addWidget(btn_stop)

        top_panel.addStretch()
        outer.addLayout(top_panel)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        outer.addWidget(line)

        manual_group = QGroupBox("Ручное позиционирование")
        manual_layout = QHBoxLayout(manual_group)
        manual_layout.setContentsMargins(8, 6, 8, 6)
        manual_layout.setSpacing(10)

        manual_layout.addWidget(QLabel("X:"))
        self.target_x_spin = QDoubleSpinBox()
        self.target_x_spin.setRange(-1000.0, 1000.0)
        self.target_x_spin.setValue(0.0)
        self.target_x_spin.setSuffix(" мм")
        manual_layout.addWidget(self.target_x_spin)

        manual_layout.addWidget(QLabel("Y:"))
        self.target_y_spin = QDoubleSpinBox()
        self.target_y_spin.setRange(-1000.0, 1000.0)
        self.target_y_spin.setValue(0.0)
        self.target_y_spin.setSuffix(" мм")
        manual_layout.addWidget(self.target_y_spin)

        manual_layout.addWidget(QLabel("Z:"))
        self.target_z_spin = QDoubleSpinBox()
        self.target_z_spin.setRange(-1000.0, 1000.0)
        self.target_z_spin.setValue(0.0)
        self.target_z_spin.setSuffix(" мм")
        manual_layout.addWidget(self.target_z_spin)

        self.manual_diag_cb = QCheckBox("Ехать диагонально")
        manual_layout.addWidget(self.manual_diag_cb)

        btn_go = QPushButton("Ехать")
        btn_go.clicked.connect(self._on_manual_move_clicked)
        manual_layout.addWidget(btn_go)

        manual_layout.addStretch()
        outer.addWidget(manual_group)

        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setFrameShadow(QFrame.Sunken)
        outer.addWidget(line2)

        cols = QHBoxLayout()

        col1 = QVBoxLayout()
        server_group = QGroupBox("Подключение к станку (Сервер)")
        s_layout = QGridLayout(server_group)
        self.radio_mac = QRadioButton("MAC Адрес")
        self.radio_ip = QRadioButton("IP Адрес")
        self.radio_mac.setChecked(True)
        s_layout.addWidget(self.radio_mac, 0, 0)
        s_layout.addWidget(self.radio_ip, 1, 0)
        self.server_address_edit = QLineEdit("e4-5f-01-1e-15-f8")
        s_layout.addWidget(self.server_address_edit, 0, 1, 2, 1)
        col1.addWidget(server_group)

        col1.addWidget(self._build_source_group())
        col1.addWidget(self._build_scale_group())
        col1.addStretch()

        col2 = QVBoxLayout()
        col2.addWidget(self._build_tool_group())
        col2.addWidget(self._build_future_gcode_group())
        col2.addStretch()

        cols.addLayout(col1, stretch=1)
        cols.addLayout(col2, stretch=1)
        outer.addLayout(cols)

        actions = QHBoxLayout()
        self.status_label = QLabel("Готов.")
        self.status_label.setObjectName("statusLabel")
        actions.addWidget(self.status_label)
        actions.addStretch(1)
        self.generate_button = QPushButton("Создать SVG")
        self.generate_button.clicked.connect(self._generate)
        actions.addWidget(self.generate_button)
        outer.addLayout(actions)

    def _build_preview_tab(self, parent: QWidget) -> None:
        layout = QHBoxLayout(parent)
        layout.setSpacing(10)

        left = QVBoxLayout()
        left.addWidget(QLabel("Слои (Сверху -> Вниз):"))

        self.layer_list = LayerListWidget() 
        self.layer_list.itemChanged.connect(lambda _: self._update_preview())
        self.layer_list.orderChanged.connect(self._update_preview)
        left.addWidget(self.layer_list)

        self.reset_layers_btn = QPushButton("Сбросить слои")
        self.reset_layers_btn.clicked.connect(self._init_layer_list)
        left.addWidget(self.reset_layers_btn)

        left.addStretch()

        self.btn_next_tab = QPushButton("Далее ->")
        self.btn_next_tab.clicked.connect(lambda: self.tabs.setCurrentIndex(2))
        left.addWidget(self.btn_next_tab)

        layout.addLayout(left, 1)

        self.preview_label = PreviewView() 
        layout.addWidget(self.preview_label, 4)

    def _build_placement_tab(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)

        top_controls = QHBoxLayout()
        top_controls.addWidget(QLabel("Модель:"))
        self.combo_model = QComboBox()
        self.combo_model.addItems(["fast", "standart", "Без модели"])
        self.combo_model.setCurrentText("standart")
        top_controls.addWidget(self.combo_model)

        self.btn_fetch_photo = QPushButton("Получить фото")
        self.btn_fetch_photo.clicked.connect(self._fetch_placement_photo)
        top_controls.addWidget(self.btn_fetch_photo)

        self.btn_select_pixel = QPushButton("🎯 Выбор пикселя")
        self.btn_select_pixel.setCheckable(True)
        self.btn_select_pixel.toggled.connect(self._on_select_pixel_toggled)
        top_controls.addWidget(self.btn_select_pixel)

        # Инструменты размещения
        self.btn_start_placement = QPushButton("🧩 Разместить плату")
        self.btn_start_placement.setEnabled(False)
        self.btn_start_placement.clicked.connect(self._start_placement_mode)
        top_controls.addWidget(self.btn_start_placement)

        self.btn_auto_place = QPushButton("🤖 Авто-размещение")
        self.btn_auto_place.setEnabled(False)
        self.btn_auto_place.clicked.connect(self._on_auto_placement_clicked)
        top_controls.addWidget(self.btn_auto_place)

        self.placement_status = QLabel("Готов.")
        top_controls.addWidget(self.placement_status)
        top_controls.addStretch()

        layout.addLayout(top_controls)

        self.placement_tabs = QTabWidget()
        
        self.photo_tab = QWidget()
        photo_layout = QVBoxLayout(self.photo_tab)
        
        chk_layout = QHBoxLayout()
        self.chk_workspace = QCheckBox("Workspace")
        self.chk_workspace.setChecked(True)
        self.chk_workspace.stateChanged.connect(self._update_placement_views)
        self.chk_blank = QCheckBox("Blank")
        self.chk_blank.setChecked(True)
        self.chk_blank.stateChanged.connect(self._update_placement_views)
        self.chk_danger = QCheckBox("Danger")
        self.chk_danger.setChecked(True)
        self.chk_danger.stateChanged.connect(self._update_placement_views)
        
        chk_layout.addWidget(self.chk_workspace)
        chk_layout.addWidget(self.chk_blank)
        chk_layout.addWidget(self.chk_danger)
        chk_layout.addStretch()
        photo_layout.addLayout(chk_layout)
        
        self.view_yolo = PreviewView()
        self.view_yolo.pixelClicked.connect(self._on_pixel_clicked)
        photo_layout.addWidget(self.view_yolo)
        
        self.proj_tab = QWidget()
        proj_layout = QVBoxLayout(self.proj_tab)
        self.view_map = PreviewView()
        self.view_map.setBackgroundBrush(QBrush(QColor("#ffffff")))
        proj_layout.addWidget(self.view_map)
        
        self.placement_tabs.addTab(self.photo_tab, "Фото")
        self.placement_tabs.addTab(self.proj_tab, "Миллиметровая проекция")
        
        layout.addWidget(self.placement_tabs)

    def show_error(self, title, message):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setStyleSheet("QLabel{ min-width: 300px; font-size: 11px;} QPushButton{ width: 80px; height: 25px; }")
        msg.exec_()

    def resolve_server_ip(self) -> str | None:
        val = self.server_address_edit.text().strip()
        if not val:
            self.show_error("Ошибка", "Введите IP или MAC адрес сервера!")
            return None
        if self.radio_ip.isChecked():
            return val
        return get_ip_by_mac(val)

    def _api_get_coords(self):
        ip = self.resolve_server_ip()
        if not ip: return
        self.api_worker = ApiWorker(f"http://{ip}:8000/api/v0/get_coords", "GET")
        self.api_worker.success.connect(self._on_coords_update)
        self.api_worker.start()

    def _on_coords_update(self, data):
        coords = data.get("coordinates")
        if coords is None:
            self.is_homed = False
            self.current_x = self.current_y = self.current_z = None
            self.coord_label.setText("Координаты: --- | --- | ---")
            self.coord_label.setStyleSheet("color: #aa0000;")
        else:
            self.is_homed = True
            self.current_x = coords.get("x")
            self.current_y = coords.get("y")
            self.current_z = coords.get("z")
            self.coord_label.setText(f"Координаты: X: {self.current_x:.2f} | Y: {self.current_y:.2f} | Z: {self.current_z:.2f}")
            self.coord_label.setStyleSheet("color: #00aa00;")
            
            if hasattr(self, 'target_x_spin') and not self.target_x_spin.hasFocus():
                self.target_x_spin.setValue(self.current_x)
            if hasattr(self, 'target_y_spin') and not self.target_y_spin.hasFocus():
                self.target_y_spin.setValue(self.current_y)
            if hasattr(self, 'target_z_spin') and not self.target_z_spin.hasFocus():
                self.target_z_spin.setValue(self.current_z)

    def run_homing(self):
        ip = self.resolve_server_ip()
        if not ip: return
        self.coord_label.setText("Парковка...")
        self.api_worker = ApiWorker(f"http://{ip}:8000/api/v0/parking", "POST")
        self.api_worker.success.connect(self._on_command_success)
        self.api_worker.failed.connect(self._on_command_failed)
        self.api_worker.start()

    def pause_command(self):
        ip = self.resolve_server_ip()
        if not ip: return
        self.api_worker = ApiWorker(f"http://{ip}:8000/api/v0/pause", "POST")
        self.api_worker.success.connect(self._on_command_success)
        self.api_worker.failed.connect(self._on_command_failed)
        self.api_worker.start()

    def resume_command(self):
        ip = self.resolve_server_ip()
        if not ip: return
        self.api_worker = ApiWorker(f"http://{ip}:8000/api/v0/resume", "POST")
        self.api_worker.success.connect(self._on_command_success)
        self.api_worker.failed.connect(self._on_command_failed)
        self.api_worker.start()

    def emergency_stop(self):
        ip = self.resolve_server_ip()
        if not ip: return
        self.api_worker = ApiWorker(f"http://{ip}:8000/api/v0/stop", "POST")
        self.api_worker.success.connect(self._on_command_success)
        self.api_worker.failed.connect(self._on_command_failed)
        self.api_worker.start()

    def _on_command_success(self, data):
        msg = data.get("message", "Успешно выполнено!")
        QMessageBox.information(self, "Станок", msg)
        self._api_get_coords()

    def _on_command_failed(self, err):
        self.show_error("Ошибка станка", err)
        self._api_get_coords()

    def _on_manual_move_clicked(self) -> None:
        tx = self.target_x_spin.value()
        ty = self.target_y_spin.value()
        tz = self.target_z_spin.value()

        if not self.is_homed:
            self.show_error(
                "Требуется паркинг", 
                "Координаты станка неопределены. Сначала выполните процедуру паркинга (🏠)."
            )
            return

        min_x = self.placement_limits.get("x", {}).get("min", 0.0)
        max_x = self.placement_limits.get("x", {}).get("max", 260.0)
        min_y = self.placement_limits.get("y", {}).get("min", -160.0)
        max_y = self.placement_limits.get("y", {}).get("max", 20.0)
        min_z = self.placement_limits.get("z", {}).get("min", 0.0)
        max_z = self.placement_limits.get("z", {}).get("max", 30.0)

        if not (min_x <= tx <= max_x):
            self.show_error(
                "Ошибка лимитов",
                f"Координата X ({tx:.2f} мм) выходит за рабочие пределы станка [{min_x:.2f}, {max_x:.2f}]."
            )
            return

        if not (min_y <= ty <= max_y):
            self.show_error(
                "Ошибка лимитов",
                f"Координата Y ({ty:.2f} мм) выходит за рабочие пределы станка [{min_y:.2f}, {max_y:.2f}]."
            )
            return

        if not (min_z <= tz <= max_z):
            self.show_error(
                "Ошибка лимитов",
                f"Координата Z ({tz:.2f} мм) выходит за рабочие пределы станка [{min_z:.2f}, {max_z:.2f}]."
            )
            return

        ip = self.resolve_server_ip()
        if not ip: return

        payload = {
            "x": float(tx),
            "y": float(ty),
            "z": float(tz),
            "diagonal": self.manual_diag_cb.isChecked()
        }
        self.status_label.setText("Отправка команды перемещения...")
        self.api_worker = ApiWorker(f"http://{ip}:8000/api/v0/move_absolute", "POST", payload)
        self.api_worker.success.connect(self._on_command_success)
        self.api_worker.failed.connect(self._on_command_failed)
        self.api_worker.start()

    def _on_select_pixel_toggled(self, checked: bool) -> None:
        self.view_yolo.set_selection_mode(checked)
        if checked:
            self.placement_status.setText("Режим выбора пикселя активен. Кликните по фото.")
        else:
            self.placement_status.setText("Режим перемещения активен (ручной панораминг).")

    def _fetch_placement_photo(self):
        coords_invalid = False
        if not self.is_homed or self.current_x is None or self.current_y is None or self.current_z is None:
            coords_invalid = True
        else:
            if abs(self.current_x) > 0.01 or abs(self.current_y) > 0.01 or abs(self.current_z) > 0.01:
                coords_invalid = True

        if coords_invalid:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Предупреждение о положении станка")
            msg_box.setText(
                "Возможно, станок находится в неправильном положении для фото. "
                "Если сделать фото сейчас, перемещение к пикселю и размещение будет работать неверно.\n\n"
                "Продолжить?"
            )
            msg_box.setStyleSheet("QLabel{ min-width: 350px; font-size: 11px; }")
            
            yes_button = msg_box.addButton("Да", QMessageBox.YesRole)
            home_button = msg_box.addButton("Выполнить паркинг", QMessageBox.ActionRole)
            cancel_button = msg_box.addButton("Отмена", QMessageBox.RejectRole)
            
            msg_box.exec_()
            
            clicked = msg_box.clickedButton()
            if clicked == cancel_button:
                return
            elif clicked == home_button:
                self.run_homing()
                return

        ip = self.resolve_server_ip()
        if not ip: return

        mdl_text = self.combo_model.currentText()
        model_type = "fast" if mdl_text == "fast" else "standard" if mdl_text == "standart" else "none"

        self.btn_fetch_photo.setEnabled(False)
        self.placement_status.setText("Запуск воркера...")

        self.worker = FetchWorker(ip, model_type)
        self.worker.progress.connect(self.placement_status.setText)
        self.worker.success.connect(self._on_fetch_success)
        self.worker.failed.connect(self._on_fetch_failed)
        self.worker.start()

    def _on_fetch_success(self, data):
        self.placement_status.setText("Данные загружены, отрисовка...")
        try:
            img_b64 = data.get("image", "")
            img_bytes = base64.b64decode(img_b64)
            image = QImage.fromData(img_bytes)
            
            self.placement_img = image
            self.placement_detections = data.get("detections", [])
            self.placement_segments = data.get("segments", [])
            self.placement_segments_mm = data.get("segments_mm", [])
            self.placement_limits = data.get("limits", {"x": {"min": 0, "max": 260}, "y": {"min": -160, "max": 20}})
            self.placed_pcbs = []

            self._update_placement_views()
            self.view_yolo.fit_to_scene()
            self.view_map.fit_to_scene()

            self.placement_status.setText("Успешно.")
            self._update_placement_buttons_state()
        except Exception as e:
            self.placement_status.setText(f"Ошибка парсинга: {e}")
        finally:
            self.btn_fetch_photo.setEnabled(True)

    def _on_fetch_failed(self, err):
        self.placement_status.setText(f"Ошибка сети: {err}")
        self.btn_fetch_photo.setEnabled(True)
        self.show_error("Ошибка", f"Не удалось получить данные:\n{err}")

    def _update_placement_views(self):
        if not self.placement_img:
            return

        # 1. ОТРИСОВКА ВЕРХА (ФОТО)
        scene_yolo = self.view_yolo.scene()
        scene_yolo.clear()

        pixmap = QPixmap.fromImage(self.placement_img)
        scene_yolo.addPixmap(pixmap)

        show_flags = {
            "workspace": self.chk_workspace.isChecked(),
            "blank": self.chk_blank.isChecked(),
            "danger": self.chk_danger.isChecked()
        }

        for i, det in enumerate(self.placement_detections):
            c_name = str(det.get("name", "")).lower()
            if not show_flags.get(c_name, True):
                continue
                
            color = COLOR_MAP.get(c_name, DEFAULT_COLOR)
            
            if i < len(self.placement_segments):
                seg = self.placement_segments[i]
                polygon = QPolygonF([QPointF(pt[0], pt[1]) for pt in seg])
                pen = QPen(QColor(color.red(), color.green(), color.blue(), 180), 2)
                brush = QBrush(QColor(color.red(), color.green(), color.blue(), 60))
                scene_yolo.addPolygon(polygon, pen, brush)

        # 2. ОТРИСОВКА НИЗА (МИЛЛИМЕТРЫ)
        scene_map = self.view_map.scene()
        scene_map.clear()

        min_x = self.placement_limits.get("x", {}).get("min", 0.0)
        max_x = self.placement_limits.get("x", {}).get("max", 260.0)
        min_y = self.placement_limits.get("y", {}).get("min", -160.0)
        max_y = self.placement_limits.get("y", {}).get("max", 20.0)

        w = max_x - min_x
        h = max_y - min_y
        rect = QRectF(min_x, -max_y, w, h)
        scene_map.addRect(rect, QPen(Qt.black, 1), QBrush(QColor("#fcfcfc")))

        draw_order = []
        for i, det in enumerate(self.placement_detections):
            c_name = str(det.get("name", "")).lower()
            if c_name == "workspace":
                continue 
            
            if i < len(self.placement_segments_mm) and self.placement_segments_mm[i]:
                priority = 0 if c_name == "blank" else 1 
                draw_order.append((priority, c_name, self.placement_segments_mm[i]))

        draw_order.sort(key=lambda x: x[0])

        for _, c_name, seg_mm in draw_order:
            color = COLOR_MAP.get(c_name, DEFAULT_COLOR)
            polygon = QPolygonF([QPointF(pt[0], -pt[1]) for pt in seg_mm])
            pen = QPen(color, 1)
            brush = QBrush(color) 
            scene_map.addPolygon(polygon, pen, brush)

        # Отрисовка размещенных плат
        for placement in self.placed_pcbs:
            pos = placement['pos']
            angle = placement['angle']
            
            paths_mm = self._get_pcb_centered_draw_paths()
            rad = math.radians(angle)
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)
            scx = pos.x()
            scy = -pos.y()
            
            qpath = QPainterPath()
            for polyline in paths_mm:
                if len(polyline) < 2: continue
                rx = polyline[0][0] * cos_a - polyline[0][1] * sin_a
                ry = polyline[0][0] * sin_a + polyline[0][1] * cos_a
                qpath.moveTo(scx + rx, scy - ry)
                for px, py in polyline[1:]:
                    rx = px * cos_a - py * sin_a
                    ry = px * sin_a + py * cos_a
                    qpath.lineTo(scx + rx, scy - ry)
                    
            perm_item = scene_map.addPath(qpath)
            
            scale = 1.0
            if self._source_qimage:
                img_w = self._source_qimage.width()
                img_h = self._source_qimage.height()
                config = self._build_config()
                if config.width_mm is not None:
                    scale = config.width_mm / img_w
                elif config.height_mm is not None:
                    scale = config.height_mm / img_h
                    
            cutter_diameter_mm = 1.0
            if self._last_result and hasattr(self._last_result, 'cut_tool_radius_px'):
                cutter_diameter_mm = self._last_result.cut_tool_radius_px * 2 * scale
                
            pen = QPen(QColor(0, 0, 255, 255))
            pen.setWidthF(max(0.1, cutter_diameter_mm))
            perm_item.setPen(pen)
            perm_item.setBrush(QBrush(QColor(0, 0, 255, 40)))

    def _on_pixel_clicked(self, x, y):
        self.btn_select_pixel.setChecked(False)

        if not self.placement_img: return
        w, h = self.placement_img.width(), self.placement_img.height()
        if not (0 <= x < w and 0 <= y < h):
            return 
            
        if not self.is_homed or self.current_z is None:
            reply = QMessageBox.question(
                self, "Требуется паркинг", 
                "Координаты станка не определены.\nХотите выполнить паркинг прямо сейчас?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.run_homing()
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Подтверждение перемещения")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(f"Хотите ли вы приехать в пиксель ({int(x)}, {int(y)})?"))
        
        z_layout = QHBoxLayout()
        z_layout.addWidget(QLabel("Координата Z:"))
        z_spin = QDoubleSpinBox()
        z_spin.setRange(-50.0, 50.0)
        z_spin.setValue(self.current_z)
        z_layout.addWidget(z_spin)
        layout.addLayout(z_layout)
        
        diag_cb = QCheckBox("Ехать диагонально")
        diag_cb.setChecked(False)
        layout.addWidget(diag_cb)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Yes | QDialogButtonBox.No)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        if dialog.exec_() == QDialog.Accepted:
            ip = self.resolve_server_ip()
            if not ip: return
            
            payload = {
                "px_x": float(x), 
                "px_y": float(y), 
                "z": z_spin.value(), 
                "diagonal": diag_cb.isChecked()
            }
            self.api_worker = ApiWorker(f"http://{ip}:8000/api/v0/move_pixel", "POST", payload)
            self.api_worker.success.connect(self._on_command_success)
            self.api_worker.failed.connect(self._on_command_failed)
            self.api_worker.start()

    # --- ГЕОМЕТРИЧЕСКОЕ И АВТОМАТИЧЕСКОЕ РАЗМЕЩЕНИЕ ПЛАТЫ ---
    def _update_placement_buttons_state(self):
        """Управление доступностью кнопок позиционирования."""
        has_pcb = (self._last_result is not None)
        has_blank = False
        if self.placement_segments_mm and self.placement_detections:
            for det in self.placement_detections:
                if str(det.get("name", "")).lower() == "blank":
                    has_blank = True
                    break
        can_place = has_pcb and has_blank
        self.btn_start_placement.setEnabled(can_place)
        self.btn_auto_place.setEnabled(can_place)

    def _get_pcb_pixel_bbox(self) -> tuple[float, float, float, float]:
        if not self._last_result:
            return 0, 0, 0, 0
        xmin = ymin = float('inf')
        xmax = ymax = float('-inf')
        
        for polyline in self._last_result.cut_paths:
            for x, y in polyline:
                xmin = min(xmin, x)
                xmax = max(xmax, x)
                ymin = min(ymin, y)
                ymax = max(ymax, y)
                
        for polyline in self._last_result.black_paths:
            for x, y in polyline:
                xmin = min(xmin, x)
                xmax = max(xmax, x)
                ymin = min(ymin, y)
                ymax = max(ymax, y)
                
        for polyline in self._last_result.green_paths:
            for x, y in polyline:
                xmin = min(xmin, x)
                xmax = max(xmax, x)
                ymin = min(ymin, y)
                ymax = max(ymax, y)
                
        for hole in self._last_result.holes:
            r = hole.radius_px
            xmin = min(xmin, hole.x_px - r)
            xmax = max(xmax, hole.x_px + r)
            ymin = min(ymin, hole.y_px - r)
            ymax = max(ymax, hole.y_px + r)
            
        if xmin == float('inf'):
            return 0, 0, 0, 0
        return xmin, ymin, xmax, ymax

    def _get_pcb_centered_draw_paths(self) -> list[list[tuple[float, float]]]:
        if not self._last_result:
            return []
        xmin, ymin, xmax, ymax = self._get_pcb_pixel_bbox()
        cx_px = (xmin + xmax) / 2.0
        cy_px = (ymin + ymax) / 2.0
        
        config = self._build_config()
        img_w = self._source_qimage.width()
        img_h = self._source_qimage.height()
        scale = 1.0
        if config.width_mm is not None:
            scale = config.width_mm / img_w
        elif config.height_mm is not None:
            scale = config.height_mm / img_h
            
        if self._last_result.cut_paths:
            paths_mm = []
            for polyline in self._last_result.cut_paths:
                # Зеркалируем относительную координату Y для соответствия декартовой системе координат станка
                paths_mm.append([((x - cx_px) * scale, -(y - cy_px) * scale) for (x, y) in polyline])
            return paths_mm
        else:
            w = ((xmax - xmin) * scale) / 2.0
            h = ((ymax - ymin) * scale) / 2.0
            return [
                [(-w, -h), (w, -h), (w, h), (-w, h), (-w, -h)]
            ]

    def _get_rotated_pcb_local_poly(self, angle: float) -> QPolygonF:
        xmin, ymin, xmax, ymax = self._get_pcb_pixel_bbox()
        
        config = self._build_config()
        img_w = self._source_qimage.width()
        img_h = self._source_qimage.height()
        scale = 1.0
        if config.width_mm is not None:
            scale = config.width_mm / img_w
        elif config.height_mm is not None:
            scale = config.height_mm / img_h
            
        pcb_w_mm = (xmax - xmin) * scale
        pcb_h_mm = (ymax - ymin) * scale
        
        w = pcb_w_mm / 2.0
        h = pcb_h_mm / 2.0
        local_vertices = [
            QPointF(-w, -h),
            QPointF(w, -h),
            QPointF(w, h),
            QPointF(-w, h)
        ]
        
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        rotated_poly = QPolygonF()
        for pt in local_vertices:
            rx = pt.x() * cos_a - pt.y() * sin_a
            ry = pt.x() * sin_a + pt.y() * cos_a
            rotated_poly.append(QPointF(rx, ry))
        return rotated_poly

    def _validate_placement_geometry(self, pcb_poly_mach: QPolygonF, blanks: list[QPolygonF] = None) -> str | None:
        min_x = self.placement_limits.get("x", {}).get("min", 0.0)
        max_x = self.placement_limits.get("x", {}).get("max", 260.0)
        min_y = self.placement_limits.get("y", {}).get("min", -160.0)
        max_y = self.placement_limits.get("y", {}).get("max", 20.0)
        
        for i in range(pcb_poly_mach.count()):
            pt = pcb_poly_mach[i]
            if not (min_x <= pt.x() <= max_x) or not (min_y <= pt.y() <= max_y):
                return f"Плата выходит за пределы рабочей области станка! Лимиты X: [{min_x}, {max_x}], Y: [{min_y}, {max_y}]"
        
        if blanks is None:
            blanks = []
            for i, det in enumerate(self.placement_detections):
                if str(det.get("name", "")).lower() == "blank":
                    if i < len(self.placement_segments_mm):
                        seg = self.placement_segments_mm[i]
                        blanks.append(QPolygonF([QPointF(pt[0], pt[1]) for pt in seg]))
        
        if not blanks:
            return "Не найден заготовочный материал (класс 'blank')!"
            
        inside_any_blank = False
        best_blank = None
        for b in blanks:
            intersection = pcb_poly_mach.intersected(b)
            area_pcb = polygon_area(pcb_poly_mach)
            area_intersection = polygon_area(intersection)
            if area_pcb > 0 and (area_intersection / area_pcb) > 0.99:
                inside_any_blank = True
                best_blank = b
                break
                
        if not inside_any_blank:
            return "Плата должна быть полностью размещена внутри заготовки (класса 'blank')!"
            
        dist_to_blank = polygon_to_polygon_distance(pcb_poly_mach, best_blank)
        if dist_to_blank < 1.0:
            return f"Контур слишком близко к краю заготовки ({dist_to_blank:.2f} мм < 1.0 мм)!"
            
        dangers = []
        for i, det in enumerate(self.placement_detections):
            if str(det.get("name", "")).lower() == "danger":
                if i < len(self.placement_segments_mm):
                    seg = self.placement_segments_mm[i]
                    dangers.append(QPolygonF([QPointF(pt[0], pt[1]) for pt in seg]))
                    
        for d in dangers:
            intersection = pcb_poly_mach.intersected(d)
            if polygon_area(intersection) > 0.01:
                return "Плата пересекает запретную зону (класс 'danger')!"
                
            dist_to_danger = polygon_to_polygon_distance(pcb_poly_mach, d)
            if dist_to_danger < 3.0:
                return f"Контур слишком близко к запретной зоне ({dist_to_danger:.2f} мм < 3.0 мм)!"
                
        return None

    def _start_placement_mode(self) -> None:
        if not self._last_result:
            self.show_error("Ошибка", "Сперва создайте плату (нажмите 'Создать SVG' на первой вкладке)!")
            return
        
        # Сбрасываем старые зафиксированные размещения при новом запуске
        self.placed_pcbs = []
        self._update_placement_views()
        
        self.placement_active = True
        self.placed_pcb_angle = 0.0
        self.placed_pcb_pos = QPointF(0, 0)
        
        self.placement_tabs.setCurrentWidget(self.proj_tab)
        
        self.view_map.set_selection_mode(True)
        self.view_map.setMouseTracking(True)
        self.view_map.viewport().setMouseTracking(True)
        self.view_map.setFocus()
        
        self.view_map.mouseMovedInScene.connect(self._on_placement_mouse_move)
        self.view_map.rightClickedInScene.connect(self._on_placement_right_click)
        self.view_map.keyPressed.connect(self._on_placement_key_press)
        
        self.placement_status.setText("Размещение активно. Двигайте мышь. [- / =] - Вращение, ПКМ - Зафиксировать.")
        self._update_placement_preview_item()
        
    def _stop_placement_mode(self) -> None:
        self.placement_active = False
        self.view_map.set_selection_mode(False)
        
        try:
            self.view_map.mouseMovedInScene.disconnect(self._on_placement_mouse_move)
            self.view_map.rightClickedInScene.disconnect(self._on_placement_right_click)
            self.view_map.keyPressed.disconnect(self._on_placement_key_press)
        except Exception:
            pass
            
        if self.placement_preview_item:
            try:
                self.view_map.scene().removeItem(self.placement_preview_item)
            except RuntimeError:
                pass
            self.placement_preview_item = None
            
        self.placement_status.setText("Размещение завершено.")

    def _on_placement_mouse_move(self, scene_x, scene_y) -> None:
        if not self.placement_active:
            return
        self.placed_pcb_pos = QPointF(scene_x, -scene_y)
        self._update_placement_preview_item()

    def _on_placement_key_press(self, key: int) -> None:
        if not self.placement_active:
            return
        if key in (Qt.Key_Minus, 1045, 45):
            self.placed_pcb_angle = (self.placed_pcb_angle - 5.0) % 360
            self._update_placement_preview_item()
        elif key in (Qt.Key_Equal, Qt.Key_Plus, 61, 43):
            self.placed_pcb_angle = (self.placed_pcb_angle + 5.0) % 360
            self._update_placement_preview_item()

    def _update_placement_preview_item(self) -> None:
        paths_mm = self._get_pcb_centered_draw_paths()
        rad = math.radians(self.placed_pcb_angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        
        scx = self.placed_pcb_pos.x()
        scy = -self.placed_pcb_pos.y()
        
        qpath = QPainterPath()
        for polyline in paths_mm:
            if len(polyline) < 2: continue
            rx = polyline[0][0] * cos_a - polyline[0][1] * sin_a
            ry = polyline[0][0] * sin_a + polyline[0][1] * cos_a
            qpath.moveTo(scx + rx, scy - ry)
            for px, py in polyline[1:]:
                rx = px * cos_a - py * sin_a
                ry = px * sin_a + py * cos_a
                qpath.lineTo(scx + rx, scy - ry)
                
        need_creation = True
        if self.placement_preview_item:
            try:
                self.placement_preview_item.setPath(qpath)
                need_creation = False
            except RuntimeError:
                pass

        if need_creation:
            self.placement_preview_item = self.view_map.scene().addPath(qpath)
            
        scale = 1.0
        if self._source_qimage:
            img_w = self._source_qimage.width()
            img_h = self._source_qimage.height()
            config = self._build_config()
            if config.width_mm is not None:
                scale = config.width_mm / img_w
            elif config.height_mm is not None:
                scale = config.height_mm / img_h
                
        cutter_diameter_mm = 1.0
        if self._last_result and hasattr(self._last_result, 'cut_tool_radius_px'):
            cutter_diameter_mm = self._last_result.cut_tool_radius_px * 2 * scale
            
        pen = QPen(QColor(255, 0, 0, 180))
        pen.setWidthF(max(0.1, cutter_diameter_mm))
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        self.placement_preview_item.setPen(pen)

    def _on_placement_right_click(self, scene_x, scene_y) -> None:
        if not self.placement_active:
            return
        
        pcb_poly_mach = self._get_rotated_pcb_local_poly(self.placed_pcb_angle)
        translated_poly = QPolygonF()
        for i in range(pcb_poly_mach.count()):
            translated_poly.append(pcb_poly_mach[i] + self.placed_pcb_pos)
            
        err = self._validate_placement_geometry(translated_poly)
        if err is not None:
            self.show_error("Ошибка размещения", err)
            return
            
        placement_data = {
            'pos': self.placed_pcb_pos,
            'angle': self.placed_pcb_angle,
            'polygon_mm': translated_poly
        }
        self.placed_pcbs.append(placement_data)
        
        self._stop_placement_mode()
        self._update_placement_views()
        
        QMessageBox.information(
            self, "Успех", 
            f"Плата успешно зафиксирована:\nX: {self.placed_pcb_pos.x():.2f} мм, Y: {self.placed_pcb_pos.y():.2f} мм\nУгол: {self.placed_pcb_angle:.1f}°"
        )

    def find_auto_placement(self) -> tuple[QPointF, float] | None:
        blanks = []
        for i, det in enumerate(self.placement_detections):
            if str(det.get("name", "")).lower() == "blank":
                if i < len(self.placement_segments_mm):
                    seg = self.placement_segments_mm[i]
                    blanks.append(QPolygonF([QPointF(pt[0], pt[1]) for pt in seg]))
        if not blanks:
            return None
        
        all_pts = [pt for b in blanks for pt in b]
        b_xmin = min(pt.x() for pt in all_pts)
        b_xmax = max(pt.x() for pt in all_pts)
        b_ymin = min(pt.y() for pt in all_pts)
        b_ymax = max(pt.y() for pt in all_pts)
        
        step_x = 4.0
        step_y = 4.0
        angles = [0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0]
        
        for angle in angles:
            local_poly = self._get_rotated_pcb_local_poly(angle)
            curr_x = b_xmin
            while curr_x <= b_xmax:
                curr_y = b_ymin
                while curr_y <= b_ymax:
                    pt_center = QPointF(curr_x, curr_y)
                    center_inside = False
                    for b in blanks:
                        if b.containsPoint(pt_center, Qt.OddEvenFill):
                            center_inside = True
                            break
                            
                    if center_inside:
                        candidate_poly = QPolygonF()
                        for i in range(local_poly.count()):
                            candidate_poly.append(local_poly[i] + pt_center)
                            
                        err = self._validate_placement_geometry(candidate_poly, blanks)
                        if err is None:
                            return pt_center, angle
                            
                    curr_y += step_y
                curr_x += step_x
        return None

    def _on_auto_placement_clicked(self) -> None:
        # Сбрасываем старые размещения перед автоматическим расчетом
        self.placed_pcbs = []
        self._update_placement_views()
        
        res = self.find_auto_placement()
        if res is None:
            self.show_error("Ошибка авто-размещения", "Невозможно разместить автоматически: нет свободного места.")
            return
            
        pos, angle = res
        self.placed_pcb_pos = pos
        self.placed_pcb_angle = angle
        
        pcb_poly_mach = self._get_rotated_pcb_local_poly(angle)
        translated_poly = QPolygonF()
        for i in range(pcb_poly_mach.count()):
            translated_poly.append(pcb_poly_mach[i] + pos)
            
        placement_data = {
            'pos': pos,
            'angle': angle,
            'polygon_mm': translated_poly
        }
        self.placed_pcbs.append(placement_data)
        
        self.placement_tabs.setCurrentWidget(self.proj_tab)
        self._update_placement_views()
        
        QMessageBox.information(
            self, "Авто-размещение выполнено", 
            f"Оптимальные координаты размещения найдены:\nX: {pos.x():.2f} мм, Y: {pos.y():.2f} мм\nУгол: {angle:.1f}°"
        )

    # --- ОСТАЛЬНЫЕ МЕТОДЫ MAINWINDOW ---
    def _init_layer_list(self) -> None:
        self.layer_list.clear()
        layers = [
            ("source", "Оригинал", True),
            ("holes", "Отверстия", True),
            ("green", "Текст", True),
            ("paths", "Дорожки", True),
            ("cut", "Контур", True),
        ]
        
        self.layer_list.blockSignals(True)
        for key, title, checked in layers:
            item = QListWidgetItem(title)
            item.setData(Qt.UserRole, key) 
            item.setFlags(
                Qt.ItemIsUserCheckable
                | Qt.ItemIsDragEnabled
                | Qt.ItemIsEnabled
                | Qt.ItemIsSelectable
            )
            item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            self.layer_list.addItem(item)
        self.layer_list.blockSignals(False)
        self._update_preview()
        self._update_placement_buttons_state()

    def _build_source_group(self) -> QGroupBox:
        group = QGroupBox("Входы")
        layout = QGridLayout(group)
        layout.setColumnStretch(1, 1)

        choose_button = QPushButton("Выбрать картинку")
        choose_button.clicked.connect(self._select_image)
        layout.addWidget(choose_button, 0, 0)

        self.image_path_edit = QLineEdit()
        self.image_path_edit.setReadOnly(True)
        layout.addWidget(self.image_path_edit, 0, 1)

        compose_button = QPushButton("Выбрать слои (собрать PNG)")
        compose_button.clicked.connect(self._show_layer_composer)
        layout.addWidget(compose_button, 1, 0)

        gerber_button = QPushButton("Выбрать Gerber")
        gerber_button.clicked.connect(self._show_gerber_stub)
        layout.addWidget(gerber_button, 2, 0)

        hint = QLabel("Поддерживаемые форматы: JPG, PNG, BMP")
        layout.addWidget(hint, 1, 1, 2, 1) 
        return group

    def _build_scale_group(self) -> QGroupBox:
        group = QGroupBox("Физический размер")
        layout = QGridLayout(group)

        self.width_radio = QRadioButton("Ширина (мм)")
        self.width_radio.setChecked(True)
        self.width_radio.toggled.connect(self._sync_scale_mode)
        layout.addWidget(self.width_radio, 0, 0)

        self.width_mm = QDoubleSpinBox()
        self.width_mm.setRange(0.001, 1_000_000.0)
        self.width_mm.setDecimals(4)
        self.width_mm.setValue(100.0)
        self.width_mm.setSuffix(" мм")
        layout.addWidget(self.width_mm, 0, 1)

        self.height_radio = QRadioButton("Высота (мм)")
        self.height_radio.toggled.connect(self._sync_scale_mode)
        layout.addWidget(self.height_radio, 1, 0)

        self.height_mm = QDoubleSpinBox()
        self.height_mm.setRange(0.001, 1_000_000.0)
        self.height_mm.setDecimals(4)
        self.height_mm.setValue(100.0)
        self.height_mm.setSuffix(" мм")
        self.height_mm.setEnabled(False)
        layout.addWidget(self.height_mm, 1, 1)
        return group

    def _build_tool_group(self) -> QGroupBox:
        group = QGroupBox("Инструменты и векторные пути")
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.tool_angle_deg = self._spin(0.1, 179.0, 30.0, suffix=" °")
        form.addRow("Угол фрезы (град):", self.tool_angle_deg)

        self.drill_diameter_mm = self._spin(0.01, 1000.0, 0.8, suffix=" мм")
        form.addRow("Диаметр сверла:", self.drill_diameter_mm)

        self.stepover_percent = self._spin(1.0, 100.0, 50.0, suffix=" %")
        form.addRow("Разряженность:", self.stepover_percent)

        self.simplify_tolerance_mm = self._spin(0.0, 100.0, 0.0, suffix=" мм", decimals=4)
        form.addRow("Грубость симплификации путей:", self.simplify_tolerance_mm)

        self.green_mode = QComboBox()
        self.green_mode.addItem("По центру линий (Centerline)", userData="centerline")
        self.green_mode.addItem("Змейка (scanline)", userData="scanline")
        self.green_mode.addItem("Контуры (contour-offset)", userData="contour-offset")
        form.addRow("Режим гравировки (Green):", self.green_mode)
        
        self.skip_validation_checkbox = QCheckBox("Игнорировать ошибки трассировки")
        self.skip_validation_checkbox.setChecked(self._initial_skip_validation)
        self.skip_validation_checkbox.setHidden(True)
        form.addRow(self.skip_validation_checkbox)

        return group

    def _build_future_gcode_group(self) -> QGroupBox:
        group = QGroupBox("Параметры для создания G-кода")
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.stock_thickness_mm = self._spin(0.001, 1000.0, 1.5, suffix=" мм")
        form.addRow("Толщина заготовки:", self.stock_thickness_mm)

        self.cut_speed_steps_s = self._spin(0.001, 100000.0, 15000.0, suffix=" шагов/с")
        form.addRow("Скорость реза:", self.cut_speed_steps_s)

        self.max_accel_steps_s2 = self._spin(0.001, 1_000_000.0, 10000.0, suffix=" шагов/с²")
        form.addRow("Макс. ускорение:", self.max_accel_steps_s2)

        self.trace_depth_mm = self._spin(0.001, 1000.0, 0.2, suffix=" мм")
        form.addRow("Глубина прореза дорожки:", self.trace_depth_mm)

        self.green_depth_mm = self._spin(0.001, 1000.0, 0.1, suffix=" мм")
        form.addRow("Глубина гравировки:", self.green_depth_mm)
        
        spindle_layout = QHBoxLayout()
        self.spindle_slider = QSlider(Qt.Horizontal)
        self.spindle_slider.setRange(0, 100)
        self.spindle_slider.setValue(0)
        self.spindle_label = QLabel("0%")
        self.spindle_slider.valueChanged.connect(lambda val: self.spindle_label.setText(f"{val}%"))
        
        spindle_layout.addWidget(self.spindle_slider)
        spindle_layout.addWidget(self.spindle_label)
        form.addRow("Скорость Шпинделя:", spindle_layout)
        
        return group

    @staticmethod
    def _spin(minimum: float, maximum: float, value: float, suffix: str = "", decimals: int = 3) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setValue(value)
        if suffix:
            spin.setSuffix(suffix)
        return spin

    def _sync_scale_mode(self) -> None:
        by_width = self.width_radio.isChecked()
        self.width_mm.setEnabled(by_width)
        self.height_mm.setEnabled(not by_width)

    def _select_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите исходную картинку",
            str(self._base_dir),
            "Растровые форматы (*.jpg *.jpeg *.png *.bmp)",
        )
        if not path: return
        self._load_image_state(Path(path))

    def _show_layer_composer(self) -> None:
        dialog = LayerComposerDialog(self)
        if dialog.exec_():
            paths = dialog.get_paths()
            if paths:
                self.status_label.setText("Склеиваем слои...")
                QApplication.processEvents()
                try:
                    self._compose_and_load_layers(paths)
                except Exception as e:
                    self.show_error("Ошибка сборки", f"Не удалось склеить слои:\n{e}")
                    self.status_label.setText("Ошибка сборки слоев.")

    def _compose_and_load_layers(self, paths_dict: dict[str, str]) -> None:
        resolved_paths = {k: Path(v) for k, v in paths_dict.items() if v}
        out_path = self._base_dir / "composed_source.png"
        compose_layers_to_image(resolved_paths, out_path)
        self._load_image_state(out_path)
        self.status_label.setText("Слои успешно собраны.")

    def _show_gerber_stub(self) -> None:
        dialog = GerberComposerDialog(self)
        if dialog.exec_():
            paths = dialog.get_paths()
            if not paths: return
            self.status_label.setText("Импортируем Gerber-файлы...")
            QApplication.processEvents()
            try:
                layers_paths = {k: Path(v) for k, v in paths.items()}
                composed, width_mm, height_mm = render_gerber_and_excellon(layers_paths)
                
                out_path = self._base_dir / "composed_source.png"
                is_success, buffer = cv2.imencode(".png", cv2.cvtColor(composed, cv2.COLOR_RGBA2BGRA))
                if is_success: buffer.tofile(str(out_path))
                
                self._load_image_state(out_path)
                self.width_radio.setChecked(True)
                self.width_mm.setValue(width_mm)
                self._sync_scale_mode()
                self.status_label.setText(f"Gerber успешно импортирован.")
            except Exception as e:
                self.show_error("Ошибка импорта Gerber", f"Не удалось прочитать векторные файлы:\n{e}")
                self.status_label.setText("Ошибка импорта Gerber.")
    
    def _load_image_state(self, path: Path) -> None:
        self._image_path = path
        self.image_path_edit.setText(str(path))
        self.status_label.setText("Картинка загружена.")
        
        raw = np.fromfile(str(path), dtype=np.uint8)
        rgba = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        if rgba.ndim == 2: rgba = cv2.cvtColor(rgba, cv2.COLOR_GRAY2RGBA)
        elif rgba.ndim == 3 and rgba.shape[2] == 3: rgba = cv2.cvtColor(rgba, cv2.COLOR_BGR2RGBA)
        elif rgba.ndim == 3 and rgba.shape[2] == 4: rgba = cv2.cvtColor(rgba, cv2.COLOR_BGRA2RGBA)
            
        arr = np.ascontiguousarray(rgba)
        h, w = arr.shape[:2]
        self._source_qimage = QImage(arr.data, w, h, w * 4, QImage.Format_RGBA8888).copy()
        
        self._last_result = None  
        self._update_preview()
        self.preview_label.fit_to_scene() 

    def load_config_from_yaml(self, path: Path) -> None:
        try:
            config = PipelineConfig.from_yaml(path, skip_validation=self._initial_skip_validation)
            if config.image_path.exists(): self._load_image_state(config.image_path)
            
            if config.width_mm is not None:
                self.width_radio.setChecked(True)
                self.width_mm.setValue(config.width_mm)
            elif config.height_mm is not None:
                self.height_radio.setChecked(True)
                self.height_mm.setValue(config.height_mm)
            self._sync_scale_mode()

            self.tool_angle_deg.setValue(config.tool_angle_deg)
            self.drill_diameter_mm.setValue(config.drill_diameter_mm)
            self.stepover_percent.setValue(config.stepover_percent)
            self.simplify_tolerance_mm.setValue(config.simplify_tolerance_mm)
            
            idx = self.green_mode.findData(config.green_mode)
            if idx != -1: self.green_mode.setCurrentIndex(idx)

            self.stock_thickness_mm.setValue(config.stock_thickness_mm)
            self.cut_speed_steps_s.setValue(config.cut_speed_mm_s) 
            self.max_accel_steps_s2.setValue(config.max_accel_mm_s2)
            self.trace_depth_mm.setValue(config.trace_depth_mm)
            self.green_depth_mm.setValue(config.green_depth_mm)
            self.skip_validation_checkbox.setChecked(config.skip_validation)

            self.status_label.setText(f"Конфиг загружен.")
        except Exception:
            self.status_label.setText("Ошибка разбора YAML.")

    def _build_config(self) -> PipelineConfig:
        if self._image_path is None: raise PipelineError("Сперва выберите исходную картинку.")
        w = float(self.width_mm.value()) if self.width_radio.isChecked() else None
        h = float(self.height_mm.value()) if not self.width_radio.isChecked() else None

        return PipelineConfig(
            image_path=self._image_path,
            width_mm=w, height_mm=h,
            tool_angle_deg=float(self.tool_angle_deg.value()),
            drill_diameter_mm=float(self.drill_diameter_mm.value()),
            stepover_percent=float(self.stepover_percent.value()),
            simplify_tolerance_mm=float(self.simplify_tolerance_mm.value()),
            green_mode=self.green_mode.currentData(),
            stock_thickness_mm=float(self.stock_thickness_mm.value()),
            cut_speed_mm_s=float(self.cut_speed_steps_s.value()),
            max_accel_mm_s2=float(self.max_accel_steps_s2.value()),
            trace_depth_mm=float(self.trace_depth_mm.value()),
            green_depth_mm=float(self.green_depth_mm.value()),
            skip_validation=self.skip_validation_checkbox.isChecked()
        )

    def _generate(self) -> None:
        try:
            self.status_label.setText("Обработка...")
            QApplication.processEvents()

            config = self._build_config()
            out_dir = self._base_dir / "outputs"
            
            if out_dir.exists(): shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            result = run_pipeline(config)
            files = export_svg_bundle(result, config, out_dir)
            
            self._last_result = result
            self._update_preview()

            self.status_label.setText("SVG сгенерировано.")
            details = "Созданные файлы:\n" + "\n".join(f"- {path.name}" for path in files.values())
            if result.warnings:
                details += "\n\n⚠️ Предупреждения:\n" + "\n".join(f"• {w}" for w in result.warnings)

            QMessageBox.information(self, "Успешно завершено", details)
            self.tabs.setCurrentIndex(1)
            self._update_placement_buttons_state()
            
        except PipelineError as exc:
            self.status_label.setText("Ошибка генерации.")
            self.show_error("Ошибка", str(exc))
        except Exception as exc:
            self.status_label.setText("Ошибка.")
            self.show_error("Ошибка", f"Непредвиденная ошибка: {exc}")

    def _update_preview(self) -> None:
        scene = self.preview_label.scene()
        scene.clear()  

        if self._source_qimage is None and self._last_result is None: return

        for layer_key, enabled in reversed(self._layer_state()):
            if not enabled: continue
            if layer_key == "source" and self._source_qimage is not None:
                pixmap = QPixmap.fromImage(self._source_qimage)
                item = scene.addPixmap(pixmap)
                item.setTransformationMode(Qt.SmoothTransformation)
                item.setOpacity(0.3)
                continue
                
            if self._last_result is None: continue
            
            if layer_key == "holes":
                brush, pen = QColor(0, 80, 255, 220), QPen(Qt.NoPen)
                for hole in self._last_result.holes:
                    r = hole.radius_px
                    scene.addEllipse(hole.x_px - r, hole.y_px - r, r * 2, r * 2, pen, brush)
            elif layer_key == "green":
                self._add_paths_to_scene(scene, self._last_result.green_paths, QColor(0, 200, 0, 255), self._last_result.green_tool_radius_px * 2)
            elif layer_key == "paths":
                self._add_paths_to_scene(scene, self._last_result.black_paths, QColor(20, 20, 20, 255), self._last_result.black_tool_radius_px * 2)
            elif layer_key == "cut":
                self._add_paths_to_scene(scene, self._last_result.cut_paths, QColor(255, 0, 0, 255), self._last_result.cut_tool_radius_px * 2)

        if scene.items(): scene.setSceneRect(scene.itemsBoundingRect())

    def _add_paths_to_scene(self, scene: QGraphicsScene, paths: list[list[tuple[float, float]]], color: QColor, width: float) -> None:
        if not paths: return
        qpath = QPainterPath()
        for polyline in paths:
            if len(polyline) < 2: continue
            qpath.moveTo(QPointF(polyline[0][0], polyline[0][1]))
            for x, y in polyline[1:]: qpath.lineTo(QPointF(x, y))

        pen_thick = QPen(QColor(color.red(), color.green(), color.blue(), 90))
        pen_thick.setWidthF(max(1.0, float(width)))
        pen_thick.setCosmetic(False)
        pen_thick.setCapStyle(Qt.RoundCap)
        pen_thick.setJoinStyle(Qt.RoundJoin)
        scene.addPath(qpath, pen_thick)

        pen_thin = QPen(color)
        pen_thin.setWidthF(0)
        pen_thin.setCosmetic(True)
        scene.addPath(qpath, pen_thin)

    def _layer_state(self) -> list[tuple[str, bool]]:
        return [(str(self.layer_list.item(idx).data(Qt.UserRole)), self.layer_list.item(idx).checkState() == Qt.Checked) for idx in range(self.layer_list.count())]

def build_stylesheet() -> str:
    return """
QWidget { background-color: #c0c0c0; color: #000000; font-family: "MS Sans Serif", sans-serif; font-size: 11px; }
QTabWidget::pane { border: 1px solid #7f7f7f; top: -1px; }
QTabBar::tab { background: #c0c0c0; border: 1px solid #7f7f7f; padding: 4px 10px; }
QTabBar::tab:selected { background: #d6d6d6; }
QGroupBox { border: 1px solid #7f7f7f; margin-top: 10px; padding-top: 10px; font-weight: bold; }
QGroupBox::title { left: 8px; top: -2px; padding: 0 3px; }
QLineEdit, QDoubleSpinBox, QComboBox, QListWidget, QCheckBox { background-color: #ffffff; border: 1px solid #808080; padding: 2px 4px; }
QPushButton { background-color: #c0c0c0; border-top: 2px solid #ffffff; border-left: 2px solid #ffffff; border-right: 2px solid #404040; border-bottom: 2px solid #404040; min-height: 20px; padding: 2px 10px; }
QPushButton:pressed { border-top: 2px solid #404040; border-left: 2px solid #404040; border-right: 2px solid #ffffff; border-bottom: 2px solid #ffffff; }
"""

def run_headless_workflow(config_path: Path, output_dir: Path | None, skip_validation: bool) -> int:
    """Выполняет обработку в консольном (безголовом) режиме без инициализации GUI."""
    try:
        logger.info("Initializing headless workflow execution...")
        config = PipelineConfig.from_yaml(config_path, skip_validation=skip_validation)
        
        # Определяем директорию для сохранения результатов
        if output_dir is None:
            resolved_output_dir = config_path.parent / "outputs"
        else:
            resolved_output_dir = Path(output_dir)
            
        logger.info(f"Preparing target output directory: {resolved_output_dir}")
        if resolved_output_dir.exists():
            shutil.rmtree(resolved_output_dir)
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Запуск конвейера обработки изображений
        logger.info("Executing processing pipeline...")
        result = run_pipeline(config)
        
        # Генерация итоговых векторных SVG слоев
        logger.info("Exporting processed vector tracks to SVG...")
        files = export_svg_bundle(result, config, resolved_output_dir)
        
        logger.info(f"Processing successfully finalized. Created files ({len(files)}):")
        for key, path in files.items():
            logger.info(f"  [{key}] -> {path.resolve()}")
            
        # =====================================================================
        # ТОЧКА ИНТЕГРАЦИИ ДЛЯ ПОСЛЕДУЮЩИХ СТАДИЙ
        # В будущем здесь можно разместить вызовы генерации G-кода, например:
        #
        # logger.info("Generating G-code files...")
        # gcode_files = generate_gcode_bundle(result, config, resolved_output_dir)
        # =====================================================================
        
        return 0
    except Exception as e:
        logger.error(f"Execution failed with critical error: {e}", exc_info=True)
        return 1

def main() -> int:
    parser = argparse.ArgumentParser(description="PCB Toolpath Generator")
    parser.add_argument(
        "-s", "--skip-validation",
        action="store_true",
        help="Игнорировать некритичные ошибки трассировки и коллизии (по умолчанию: выключено)"
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        help="Путь к YAML-файлу конфигурации для предзагрузки настроек"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Запуск программы в консольном режиме без графического интерфейса"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Альтернативный путь для сохранения результатов (только для режима headless)"
    )
    
    parsed_args, qt_args = parser.parse_known_args()
    
    app = QApplication([sys.argv[0]] + qt_args)
    
    base_dir = Path(__file__).resolve().parent
    setup_logging(base_dir / "app.log")
    
    logger.info(f"Command line parsed. skip_validation={parsed_args.skip_validation}, config={parsed_args.config}")
    
    # 1. Сценарий выполнения в безголовом режиме (headless)
    if parsed_args.headless:
        if not parsed_args.config:
            logger.error("Error: --headless requires a config file. Please specify --config <path_to_yaml>")
            return 1
        return run_headless_workflow(
            config_path=Path(parsed_args.config),
            output_dir=Path(parsed_args.output_dir) if parsed_args.output_dir else None,
            skip_validation=parsed_args.skip_validation
        )

    # 2. Обычный запуск с графическим интерфейсом
    if "windows" in {name.lower() for name in QStyleFactory.keys()}: 
        app.setStyle("windows")
        
    app.setStyleSheet(build_stylesheet())
    
    window = MainWindow(skip_validation=parsed_args.skip_validation)
    
    # Если передан аргумент --config, загружаем настройки
    if parsed_args.config:
        config_path = Path(parsed_args.config)
        if config_path.exists():
            window.load_config_from_yaml(config_path)
        else:
            logger.error(f"Config file not found: {parsed_args.config}")
            
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())