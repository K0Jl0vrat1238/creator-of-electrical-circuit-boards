# main_window.py
from __future__ import annotations
import shutil
import math
import logging
import base64
from pathlib import Path
import cv2
import numpy as np

from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt5.QtGui import QColor, QImage, QPainterPath, QPen, QBrush, QPixmap, QPolygonF, QFont, QPainterPathStroker, QKeySequence
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QShortcut, QSlider,
    QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidgetItem, QGraphicsScene, QMainWindow, QMessageBox, QPushButton, QRadioButton, QTabWidget,
    QVBoxLayout, QWidget, QCheckBox, QFrame, QGraphicsItemGroup, QGraphicsPathItem
)
from pipeline import PipelineConfig, PipelineError, PipelineResult, run_pipeline, compose_layers_to_image
from svg_export import export_svg_bundle
from gerber_importer import render_gerber_and_excellon

from common import COLOR_MAP, DEFAULT_COLOR, QLogHandler, QLogSignal, logger
from workers import ApiWorker, FetchWorker, AutoPlacementWorker
from widgets import GerberComposerDialog, LayerComposerDialog, LayerListWidget, PreviewView, Spline3DWidget
from ip_by_mac import get_ip_by_mac

from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator

# --- Утилита для генерации G-кода ---
class GCodeBuilder:
    def __init__(self, feedrate=600):
        self.lines = []
        self.feedrate = feedrate
    def raw(self, line): self.lines.append(line)
    def motor_on(self): self.lines.append("M3")
    def motor_off(self): self.lines.append("M5")
    def set_feed(self, f):
        self.feedrate = f
        self.lines.append(f"F{f}")
    def rapid(self, x=None, y=None, z=None):
        parts = ["G0"]
        if x is not None: parts.append(f"X{x:.3f}")
        if y is not None: parts.append(f"Y{y:.3f}")
        if z is not None: parts.append(f"Z{z:.3f}")
        if len(parts) > 1: self.lines.append(" ".join(parts))
    def feed(self, x=None, y=None, z=None, f=None):
        parts = ["G1"]
        if x is not None: parts.append(f"X{x:.3f}")
        if y is not None: parts.append(f"Y{y:.3f}")
        if z is not None: parts.append(f"Z{z:.3f}")
        if f is not None: parts.append(f"F{f}")
        if len(parts) > 1: self.lines.append(" ".join(parts))

class MainWindow(QMainWindow):
    def __init__(self, skip_validation: bool = False) -> None:
        super().__init__()
        self.setWindowTitle("Фрезеровщик печатных плат")
        self.setMinimumSize(1200, 860)
        self._base_dir = Path(__file__).resolve().parent
        self._image_path: Path | None = None
        self._source_qimage: QImage | None = None
        self._last_result: PipelineResult | None = None
        self._initial_skip_validation = skip_validation

        self._active_workers = []

        self.current_x = None
        self.current_y = None
        self.current_z = None
        self.is_homed = False

        self.placement_img: QImage | None = None
        self.placement_detections = []
        self.placement_segments = []
        self.placement_segments_mm = []
        self.machine_limits = {"x": {"min": 0.0, "max": 260.0}, "y": {"min": -160.0, "max": 20.0}, "z": {"min": -50.0, "max": 5.0}}
        self.limits_fetched = False

        self.placement_active = False
        self.placement_preview_item = None
        self.placed_pcb_pos = QPointF(0, 0)
        self.placed_pcb_angle = 0.0
        self.placed_pcbs = []  
        self._z_interpolator = None

        self._build_ui()
        self._setup_shortcuts()

        self.log_emitter = QLogSignal()
        self.log_emitter.log_msg.connect(self.terminal.append)
        gui_logger = QLogHandler(self.log_emitter)
        logging.getLogger().addHandler(gui_logger)

        self._init_layer_list()
        
        self.ping_timer = QTimer()
        self.ping_timer.setSingleShot(True)
        self.ping_timer.setInterval(3000)
        self.ping_timer.timeout.connect(lambda: self._startup_ping)
        
        self.server_address_edit.textChanged.connect(self.ping_timer.start)
        self.radio_mac.toggled.connect(lambda: self._startup_ping)
        self.radio_ip.toggled.connect(lambda: self._startup_ping)

        QTimer.singleShot(1000, lambda: self._startup_ping)

    def _track_worker(self, worker):
        self._active_workers.append(worker)
        worker.finished.connect(lambda: self._cleanup_worker(worker))
        worker.start()

    def _cleanup_worker(self, worker):
        if worker in self._active_workers:
            self._active_workers.remove(worker)

    def _setup_shortcuts(self):
        QShortcut(QKeySequence("Esc"), self).activated.connect(self.emergency_stop)
        QShortcut(QKeySequence("Ctrl+Space"), self).activated.connect(self.pause_command)
        QShortcut(QKeySequence("Ctrl+P"), self).activated.connect(self.resume_command)
        QShortcut(QKeySequence("Ctrl+R"), self).activated.connect(self.run_homing)

    def log_human(self, ok: bool, msg: str):
        prefix = "✅ " if ok else "❌ "
        if ok: logger.info(prefix + msg)
        else: logger.error(prefix + msg)

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

        self.btn_global_stop = QPushButton("🛑 ЭКСТРЕННЫЙ СТОП (Esc)")
        self.btn_global_stop.setStyleSheet("background-color: #FF3232; color: white; font-weight: bold; font-size: 15px; border-radius: 4px; padding: 10px;")
        self.btn_global_stop.clicked.connect(self.emergency_stop)
        outer.addWidget(self.btn_global_stop)

        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, stretch=4)

        self.controls_tab = QWidget()
        self.tabs.addTab(self.controls_tab, "Настройки")
        self._build_controls_tab(self.controls_tab)

        self.preview_tab = QWidget()
        self.tabs.addTab(self.preview_tab, "Предпросмотр")
        self._build_preview_tab(self.preview_tab)

        self.placement_tab = QWidget()
        self.tabs.addTab(self.placement_tab, "Размещение")
        self._build_placement_tab(self.placement_tab)

        self.spline_tab = QWidget()
        self.tabs.addTab(self.spline_tab, "3Д-сплайн")
        self._build_spline_tab(self.spline_tab)

        from PyQt5.QtWidgets import QTextEdit
        term_group = QGroupBox("Логи:")
        term_layout = QVBoxLayout(term_group)
        term_layout.setContentsMargins(4, 4, 4, 4)
        self.terminal = QTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: Consolas, monospace; font-size: 12px;")
        term_layout.addWidget(self.terminal)
        outer.addWidget(term_group, stretch=1)

    def _build_controls_tab(self, parent: QWidget) -> None:
        outer = QVBoxLayout(parent)
        outer.setSpacing(10)

        top_panel = QHBoxLayout()
        self.coord_label = QLabel("Координаты: Нет подключения")
        self.coord_label.setStyleSheet("color: #aaaaaa;")
        self.coord_label.setFont(QFont("Arial", 12, QFont.Bold))
        top_panel.addWidget(self.coord_label)

        btn_refresh = QPushButton("🔄")
        btn_refresh.setToolTip("Обновить координаты")
        btn_refresh.setFixedSize(35, 35)
        btn_refresh.clicked.connect(lambda: self._api_get_coords(silent=False))
        top_panel.addWidget(btn_refresh)
        
        top_panel.addSpacing(20)

        btn_home = QPushButton("🏠")
        btn_home.setToolTip("Выполнить паркинг (R)")
        btn_home.setFixedSize(35, 35)
        btn_home.clicked.connect(self.run_homing)
        top_panel.addWidget(btn_home)

        btn_pause = QPushButton("⏸️")
        btn_pause.setToolTip("Приостановить текущую команду (Пробел)")
        btn_pause.setFixedSize(35, 35)
        btn_pause.clicked.connect(self.pause_command)
        top_panel.addWidget(btn_pause)

        btn_play = QPushButton("▶️")
        btn_play.setToolTip("Возобновить выполнение команды (P)")
        btn_play.setFixedSize(35, 35)
        btn_play.clicked.connect(self.resume_command)
        top_panel.addWidget(btn_play)

        # --- НАЧАЛО НОВОГО БЛОКА ПОДКЛЮЧЕНИЯ ---
        top_panel.addSpacing(15)
        
        v_line = QFrame()
        v_line.setFrameShape(QFrame.VLine)
        v_line.setFrameShadow(QFrame.Sunken)
        top_panel.addWidget(v_line)
        
        top_panel.addSpacing(10)
        
        top_panel.addWidget(QLabel("Сервер:"))
        self.radio_mac = QRadioButton("MAC")
        self.radio_ip = QRadioButton("IP")
        self.radio_mac.setChecked(True)
        top_panel.addWidget(self.radio_mac)
        top_panel.addWidget(self.radio_ip)
        
        self.server_address_edit = QLineEdit("e4-5f-01-1e-15-f8")
        self.server_address_edit.setFixedWidth(160)
        top_panel.addWidget(self.server_address_edit)
        # --- КОНЕЦ НОВОГО БЛОКА ПОДКЛЮЧЕНИЯ ---

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
        col1.addWidget(self._build_source_group())
        col1.addWidget(self._build_scale_group())
        col1.addWidget(self._build_drills_group())  # <-- Теперь размеры сверл здесь
        col1.addStretch()

        col2 = QVBoxLayout()
        col2.addWidget(self._build_tool_group())
        col2.addWidget(self._build_future_gcode_group())
        col2.addStretch()

        cols.addLayout(col1, stretch=1)
        cols.addLayout(col2, stretch=1)
        outer.addLayout(cols)

        actions = QHBoxLayout()
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
        self.combo_model.setCurrentText("fast")
        top_controls.addWidget(self.combo_model)

        self.btn_fetch_photo = QPushButton("Получить фото")
        self.btn_fetch_photo.clicked.connect(self._fetch_placement_photo)
        top_controls.addWidget(self.btn_fetch_photo)

        self.btn_select_pixel = QPushButton("🎯 Ехать в пиксель")
        self.btn_select_pixel.setCheckable(True)
        self.btn_select_pixel.setEnabled(False)
        self.btn_select_pixel.toggled.connect(self._on_select_pixel_toggled)
        top_controls.addWidget(self.btn_select_pixel)

        self.btn_start_placement = QPushButton("🧩 Разместить плату")
        self.btn_start_placement.setEnabled(False)
        self.btn_start_placement.setCheckable(True)
        self.btn_start_placement.toggled.connect(self._on_placement_toggled)
        top_controls.addWidget(self.btn_start_placement)

        self.btn_auto_place = QPushButton("🤖 Авто-размещение")
        self.btn_auto_place.setEnabled(False)
        self.btn_auto_place.clicked.connect(self._on_auto_placement_clicked)
        top_controls.addWidget(self.btn_auto_place)

        self.btn_fabricate = QPushButton("⚡ НАЧАТЬ ИЗГОТОВЛЕНИЕ")
        self.btn_fabricate.setEnabled(False)
        self.btn_fabricate.setStyleSheet("background-color: #0078D7; color: white; font-weight: bold;")
        self.btn_fabricate.clicked.connect(self._start_fabrication_workflow)
        top_controls.addWidget(self.btn_fabricate)
        top_controls.addStretch()

        layout.addLayout(top_controls)

        self.placement_tabs = QTabWidget()
        
        self.photo_tab = QWidget()
        photo_layout = QVBoxLayout(self.photo_tab)
        
        chk_layout = QHBoxLayout()
        self.chk_workspace = QCheckBox("Workspace"); self.chk_workspace.setChecked(True)
        self.chk_blank = QCheckBox("Blank"); self.chk_blank.setChecked(True)
        self.chk_danger = QCheckBox("Danger"); self.chk_danger.setChecked(True)
        self.chk_workspace.stateChanged.connect(self._update_placement_views)
        self.chk_blank.stateChanged.connect(self._update_placement_views)
        self.chk_danger.stateChanged.connect(self._update_placement_views)
        chk_layout.addWidget(self.chk_workspace); chk_layout.addWidget(self.chk_blank); chk_layout.addWidget(self.chk_danger); chk_layout.addStretch()
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

    def _build_spline_tab(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)
        self.spline_widget = Spline3DWidget()
        layout.addWidget(self.spline_widget)

    def resolve_server_ip(self, silent=False) -> str | None:
        val = self.server_address_edit.text().strip()
        if not val:
            if not silent: logger.error("❌ Введите IP или MAC адрес сервера!")
            return None
        if self.radio_ip.isChecked(): return val
        return get_ip_by_mac(val)

    def _on_command_success(self, data, action_name="Операция"):
        status = data.get("status", "success")
        msg = data.get("message", "Выполнено")
        
        if status == "success":
            self.log_human(True, f"{action_name}: {msg}")
            if action_name == "Парковка":
                self._api_get_limits(silent=True)
        else:
            self.log_human(False, f"{action_name} отклонено станком: {msg}")
            
        self._api_get_coords()

    def _on_command_failed(self, err, action_name="Операция"):
        self.log_human(False, f"Сбой сети при выполнении '{action_name}': {err}")
        self._api_get_coords()
    
    def _api_get_coords(self, silent=False):
        ip = self.resolve_server_ip(silent=silent)
        if not ip:
            self._on_coords_failed("")
            return
        worker = ApiWorker(f"http://{ip}:8000/api/v0/get_coords", "GET", timeout=3.0, silent=silent)
        worker.success.connect(self._on_coords_update)
        worker.failed.connect(self._on_coords_failed)
        self._track_worker(worker)

    def _startup_ping(self):
        self._api_get_coords(silent=True)
        self._api_get_limits(silent=True)

    def _api_get_limits(self, silent=True):
        ip = self.resolve_server_ip(silent=silent)
        if not ip: return
        worker = ApiWorker(f"http://{ip}:8000/api/v0/get_limits", "GET", timeout=3.0, silent=silent)
        worker.success.connect(self._on_limits_update)
        self._track_worker(worker)

    def _on_limits_update(self, data):
        limits = data.get("limits")
        if limits:
            self.machine_limits = limits
            self.limits_fetched = True
            if not self.placement_active:
                self._update_placement_views()

    def _on_coords_failed(self, err):
        self.is_homed = False
        self.coord_label.setText("Координаты: Нет подключения")
        self.coord_label.setStyleSheet("color: #aaaaaa;")

    def _on_coords_update(self, data):
        coords = data.get("coordinates")
        if coords is None:
            self.is_homed = False
            self.current_x = self.current_y = self.current_z = None
            self.coord_label.setText("Не отпаркован")
            self.coord_label.setStyleSheet("color: #FFA500;")
        else:
            self.is_homed = True
            self.current_x = coords.get("x")
            self.current_y = coords.get("y")
            self.current_z = coords.get("z")
            self.coord_label.setText(f"X: {self.current_x:.2f} | Y: {self.current_y:.2f} | Z: {self.current_z:.2f}")
            self.coord_label.setStyleSheet("color: #00AA00;")
            
            if hasattr(self, 'target_x_spin') and not self.target_x_spin.hasFocus(): self.target_x_spin.setValue(self.current_x)
            if hasattr(self, 'target_y_spin') and not self.target_y_spin.hasFocus(): self.target_y_spin.setValue(self.current_y)
            if hasattr(self, 'target_z_spin') and not self.target_z_spin.hasFocus(): self.target_z_spin.setValue(self.current_z)

    def run_homing(self, _=False, on_success=None):
        ip = self.resolve_server_ip()
        if not ip: return
        worker = ApiWorker(f"http://{ip}:8000/api/v0/parking", "POST", timeout=None)
        
        worker.success.connect(lambda d: self._on_command_success(d, "Парковка"))
        worker.failed.connect(lambda e: self._on_command_failed(e, "Парковка"))
        self._track_worker(worker)

    def _check_homing_and_execute(self, action_callback):
        if self.is_homed:
            action_callback()
            return

        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setWindowTitle("Требуется паркинг")
        msg_box.setText("Для перемещения необходим паркинг, сделать его сейчас?")
        btn_home_and_move = msg_box.addButton("Сделать паркинг и выполнить перемещение", QMessageBox.AcceptRole)
        btn_home_only = msg_box.addButton("Сделать паркинг и отменить перемещение", QMessageBox.ActionRole)
        btn_cancel = msg_box.addButton("Отменить операцию", QMessageBox.RejectRole)
        msg_box.exec_()
        clicked = msg_box.clickedButton()
        
        if clicked == btn_cancel:
            self.log_human(False, "Операция перемещения отменена.")
        elif clicked == btn_home_only:
            self.log_human(True, "Запущен паркинг. Перемещение отменено.")
            self.run_homing()
        elif clicked == btn_home_and_move:
            self.log_human(True, "Запущен паркинг. После завершения будет выполнено перемещение.")
            self.run_homing(on_success=action_callback)

    def pause_command(self):
        ip = self.resolve_server_ip()
        if not ip: return
        worker = ApiWorker(f"http://{ip}:8000/api/v0/pause", "POST")
        worker.success.connect(lambda d: self._on_command_success(d, "Пауза"))
        worker.failed.connect(lambda e: self._on_command_failed(e, "Пауза"))
        self._track_worker(worker)

    def resume_command(self):
        ip = self.resolve_server_ip()
        if not ip: return
        worker = ApiWorker(f"http://{ip}:8000/api/v0/resume", "POST")
        worker.success.connect(lambda d: self._on_command_success(d, "Возобновление"))
        worker.failed.connect(lambda e: self._on_command_failed(e, "Возобновление"))
        self._track_worker(worker)

    def emergency_stop(self):
        ip = self.resolve_server_ip()
        if not ip: return
        worker = ApiWorker(f"http://{ip}:8000/api/v0/stop", "POST")
        worker.success.connect(lambda d: self._on_command_success(d, "Экстренный стоп"))
        worker.failed.connect(lambda e: self._on_command_failed(e, "Экстренный стоп"))
        self._track_worker(worker)

    def _on_manual_move_clicked(self) -> None:
        tx, ty, tz = self.target_x_spin.value(), self.target_y_spin.value(), self.target_z_spin.value()
        def do_move():
            ip = self.resolve_server_ip()
            if not ip: return
            payload = {"x": float(tx), "y": float(ty), "z": float(tz), "diagonal": self.manual_diag_cb.isChecked()}
            worker = ApiWorker(f"http://{ip}:8000/api/v0/move_absolute", "POST", payload, timeout=None)
            worker.success.connect(lambda d: self._on_command_success(d, "Ручное перемещение"))
            worker.failed.connect(lambda e: self._on_command_failed(e, "Ручное перемещение"))
            worker.success.connect(lambda: self._api_get_coords(silent=True))
            self._track_worker(worker)
        self._check_homing_and_execute(do_move)

    def _on_select_pixel_toggled(self, checked: bool) -> None:
        self.view_yolo.set_selection_mode(checked)

    def _fetch_placement_photo(self):
        coords_invalid = not self.is_homed or self.current_x is None
        if not coords_invalid and (abs(self.current_x) > 0.01 or abs(self.current_y) > 0.01 or abs(self.current_z) > 0.01):
            coords_invalid = True

        if coords_invalid:
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("Предупреждение")
            msg_box.setText("Станок находится не в нулевой точке. Выполнить парковку перед снимком?")
            btn_park = msg_box.addButton("🏠 Да, выполнить", QMessageBox.AcceptRole)
            msg_box.addButton("📸 Нет, сделать как есть", QMessageBox.DestructiveRole)
            btn_cancel = msg_box.addButton("❌ Отменить", QMessageBox.RejectRole)
            msg_box.exec_()
            clicked = msg_box.clickedButton()
            if clicked == btn_cancel: return
            elif clicked == btn_park:
                self.run_homing()
                return

        ip = self.resolve_server_ip()
        if not ip: return

        mdl_text = self.combo_model.currentText()
        model_type = "fast" if mdl_text == "fast" else "standard" if mdl_text == "standart" else "none"

        self.btn_fetch_photo.setEnabled(False)
        self.btn_select_pixel.setEnabled(False)
        worker = FetchWorker(ip, model_type)
        worker.success.connect(self._on_fetch_success)
        worker.failed.connect(self._on_fetch_failed)
        self._track_worker(worker)

    def _on_fetch_success(self, data):
        try:
            img_b64 = data.get("image", "")
            img_bytes = base64.b64decode(img_b64)
            self.placement_img = QImage.fromData(img_bytes)
            self.placement_detections = data.get("detections", [])
            self.placement_segments = data.get("segments", [])
            self.placement_segments_mm = data.get("segments_mm", [])
            self.placed_pcbs = []

            self._update_placement_views()
            self.view_yolo.fit_to_scene()
            self.view_map.fit_to_scene()
            if self.placement_segments_mm: self.btn_select_pixel.setEnabled(True)
            self.log_human(True, "Фото и миллиметровая проекция успешно загружены.")
            self._update_placement_buttons_state()
        except Exception as e:
            logger.error(f"Fetch Parse Error: {e}")
        finally:
            self.btn_fetch_photo.setEnabled(True)

    def _on_fetch_failed(self, err):
        self.btn_fetch_photo.setEnabled(True)
        self.log_human(False, f"Не удалось получить данные: {err}")

    def _get_scale(self) -> float:
        if not self._source_qimage: return 1.0
        config = self._build_config()
        if config.width_mm is not None: return config.width_mm / self._source_qimage.width()
        elif config.height_mm is not None: return config.height_mm / self._source_qimage.height()
        return 1.0

    def _get_center_px(self) -> tuple[float, float]:
        if not self._last_result: return 0.0, 0.0
        xmin = ymin = float('inf')
        xmax = ymax = float('-inf')
        for polys in [self._last_result.cut_paths, self._last_result.black_paths, self._last_result.green_paths]:
            for poly in polys:
                for x, y in poly:
                    xmin, xmax = min(xmin, x), max(xmax, x)
                    ymin, ymax = min(ymin, y), max(ymax, y)
        for hole in self._last_result.holes:
            r = hole.radius_px
            xmin, xmax = min(xmin, hole.x_px - r), max(xmax, hole.x_px + r)
            ymin, ymax = min(ymin, hole.y_px - r), max(ymax, hole.y_px + r)
        if xmin == float('inf'): return 0.0, 0.0
        return (xmin + xmax) / 2.0, (ymin + ymax) / 2.0

    def _get_pcb_combined_path(self, pos: QPointF, angle: float, margin: float = 0.0) -> QPainterPath:
        res = QPainterPath()
        if not self._last_result: return res
        scale = self._get_scale()
        cx, cy = self._get_center_px()
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        
        def map_pt(x, y):
            rel_x = (x - cx) * scale
            rel_y = -(y - cy) * scale 
            rx = rel_x * cos_a - rel_y * sin_a
            ry = rel_x * sin_a + rel_y * cos_a
            return QPointF(pos.x() + rx, pos.y() + ry)
            
        layers = [(self._last_result.cut_paths, self._last_result.cut_tool_radius_px), (self._last_result.black_paths, self._last_result.black_tool_radius_px), (self._last_result.green_paths, self._last_result.green_tool_radius_px)]
        for polys, r_px in layers:
            if not polys: continue
            dia_mm = (r_px * 2 * scale) + (margin * 2)
            stroker = QPainterPathStroker()
            stroker.setWidth(max(0.01, dia_mm))
            stroker.setCapStyle(Qt.RoundCap); stroker.setJoinStyle(Qt.RoundJoin)
            layer_path = QPainterPath()
            for poly in polys:
                if len(poly) < 2: continue
                layer_path.moveTo(map_pt(*poly[0]))
                for pt in poly[1:]: layer_path.lineTo(map_pt(*pt))
            res.addPath(stroker.createStroke(layer_path))
            
        for hole in self._last_result.holes:
            center = map_pt(hole.x_px, hole.y_px)
            r_mm = (hole.radius_px * scale) + margin
            res.addEllipse(center, r_mm, r_mm)
        return res

    def _get_pcb_convex_hull_physical(self, pos: QPointF, angle: float) -> QPainterPath:
        if not self._last_result: return QPainterPath()
        scale = self._get_scale()
        cx, cy = self._get_center_px()
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        def map_pt_phys(x, y):
            rel_x = (x - cx) * scale
            rel_y = -(y - cy) * scale
            rx = rel_x * cos_a - rel_y * sin_a
            ry = rel_x * sin_a + rel_y * cos_a
            return QPointF(pos.x() + rx, pos.y() + ry)

        all_pts = []
        for polys in [self._last_result.cut_paths, self._last_result.black_paths, self._last_result.green_paths]:
            for poly in polys:
                for pt in poly: all_pts.append(map_pt_phys(*pt))
        for hole in self._last_result.holes:
            p_center = map_pt_phys(hole.x_px, hole.y_px)
            r_mm = hole.radius_px * scale
            all_pts.extend([QPointF(p_center.x() + r_mm, p_center.y()), QPointF(p_center.x() - r_mm, p_center.y()), QPointF(p_center.x(), p_center.y() + r_mm), QPointF(p_center.x(), p_center.y() - r_mm)])

        if not all_pts: return QPainterPath()
        from scipy.spatial import ConvexHull
        pts_array = np.array([[p.x(), p.y()] for p in all_pts])
        hull_path = QPainterPath()
        try:
            hull = ConvexHull(pts_array)
            first = True
            for vertex_idx in hull.vertices:
                pt = pts_array[vertex_idx]
                if first: hull_path.moveTo(pt[0], pt[1]); first = False
                else: hull_path.lineTo(pt[0], pt[1])
            hull_path.closeSubpath()
        except: pass
        return hull_path

    def _get_danger_path_physical(self) -> QPainterPath:
        danger_path = QPainterPath()
        for i, det in enumerate(self.placement_detections):
            if str(det.get("name", "")).lower() == "danger" and i < len(self.placement_segments_mm):
                danger_path.addPolygon(QPolygonF([QPointF(pt[0], pt[1]) for pt in self.placement_segments_mm[i]]))
        stroker = QPainterPathStroker()
        stroker.setWidth(6.0) 
        res = QPainterPath()
        res.addPath(danger_path)
        res.addPath(stroker.createStroke(danger_path))
        return res

    def _create_pcb_graphics_group(self, pos: QPointF, angle: float) -> QGraphicsItemGroup:
        group = QGraphicsItemGroup()
        if not self._last_result: return group
        scale = self._get_scale()
        cx, cy = self._get_center_px()
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        
        def map_pt_scene(x, y):
            rel_x = (x - cx) * scale
            rel_y = -(y - cy) * scale 
            rx = rel_x * cos_a - rel_y * sin_a
            ry = rel_x * sin_a + rel_y * cos_a
            return QPointF(pos.x() + rx, -(pos.y() + ry))
            
        def add_layer(polys, r_px, color_rgba):
            if not polys: return
            dia_mm = max(0.05, r_px * 2 * scale)
            path = QPainterPath()
            for poly in polys:
                if len(poly) < 2: continue
                path.moveTo(map_pt_scene(*poly[0]))
                for pt in poly[1:]: path.lineTo(map_pt_scene(*pt))
            item = QGraphicsPathItem(path)
            pen = QPen(QColor(*color_rgba))
            pen.setWidthF(dia_mm); pen.setCapStyle(Qt.RoundCap); pen.setJoinStyle(Qt.RoundJoin)
            item.setPen(pen)
            group.addToGroup(item)
            
        add_layer(self._last_result.cut_paths, self._last_result.cut_tool_radius_px, (255, 0, 0, 140))
        add_layer(self._last_result.black_paths, self._last_result.black_tool_radius_px, (20, 20, 20, 140))
        add_layer(self._last_result.green_paths, self._last_result.green_tool_radius_px, (0, 200, 0, 140))
        
        for hole in self._last_result.holes:
            center = map_pt_scene(hole.x_px, hole.y_px)
            r_mm = hole.radius_px * scale
            path = QPainterPath()
            path.addEllipse(center, r_mm, r_mm)
            item = QGraphicsPathItem(path)
            item.setPen(QPen(Qt.NoPen))
            c = QColor(*hole.color, 220)
            item.setBrush(QBrush(c))
            group.addToGroup(item)
        return group

    def _update_placement_views(self):
        if not self.placement_img: return
        scene_yolo = self.view_yolo.scene()
        scene_yolo.clear()
        scene_yolo.addPixmap(QPixmap.fromImage(self.placement_img))
        show_flags = {"workspace": self.chk_workspace.isChecked(), "blank": self.chk_blank.isChecked(), "danger": self.chk_danger.isChecked()}

        for i, det in enumerate(self.placement_detections):
            c_name = str(det.get("name", "")).lower()
            if not show_flags.get(c_name, True): continue
            color = COLOR_MAP.get(c_name, DEFAULT_COLOR)
            if i < len(self.placement_segments):
                scene_yolo.addPolygon(QPolygonF([QPointF(pt[0], pt[1]) for pt in self.placement_segments[i]]), QPen(QColor(color.red(), color.green(), color.blue(), 180), 2), QBrush(QColor(color.red(), color.green(), color.blue(), 60)))

        scene_map = self.view_map.scene()
        scene_map.clear()

        min_x = self.machine_limits.get("x", {}).get("min", 0.0)
        max_x = self.machine_limits.get("x", {}).get("max", 260.0)
        min_y = self.machine_limits.get("y", {}).get("min", -160.0)
        max_y = self.machine_limits.get("y", {}).get("max", 20.0)
        scene_map.addRect(QRectF(min_x, -max_y, max_x - min_x, max_y - min_y), QPen(Qt.black, 1), QBrush(QColor("#fcfcfc")))

        draw_order = []
        for i, det in enumerate(self.placement_detections):
            c_name = str(det.get("name", "")).lower()
            if c_name == "workspace": continue 
            if i < len(self.placement_segments_mm) and self.placement_segments_mm[i]:
                draw_order.append((0 if c_name == "blank" else 1, c_name, self.placement_segments_mm[i]))

        draw_order.sort(key=lambda x: x[0])
        for _, c_name, seg_mm in draw_order:
            color = COLOR_MAP.get(c_name, DEFAULT_COLOR)
            scene_map.addPolygon(QPolygonF([QPointF(pt[0], -pt[1]) for pt in seg_mm]), QPen(color, 1), QBrush(color))

        for placement in self.placed_pcbs:
            scene_map.addItem(self._create_pcb_graphics_group(placement['pos'], placement['angle']))

    def _on_pixel_clicked(self, x, y):
        self.btn_select_pixel.setChecked(False)
        if not self.placement_img: return
        w, h = self.placement_img.width(), self.placement_img.height()
        if not (0 <= x < w and 0 <= y < h): return 
            
        dialog = QDialog(self)
        dialog.setWindowTitle("Параметры перемещения")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(f"Ехать в пиксель ({int(x)}, {int(y)})"))
        z_layout = QHBoxLayout()
        z_layout.addWidget(QLabel("Координата Z:"))
        z_spin = QDoubleSpinBox()
        z_spin.setRange(-50.0, 50.0)
        z_spin.setValue(self.current_z if self.current_z is not None else 0.0)
        z_layout.addWidget(z_spin)
        layout.addLayout(z_layout)
        diag_cb = QCheckBox("Ехать диагонально"); diag_cb.setChecked(True)
        layout.addWidget(diag_cb)
        buttons = QDialogButtonBox(QDialogButtonBox.Yes | QDialogButtonBox.No)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        if dialog.exec_() == QDialog.Accepted:
            target_z, is_diag = z_spin.value(), diag_cb.isChecked()
            def do_move():
                ip = self.resolve_server_ip()
                if not ip: return
                worker = ApiWorker(f"http://{ip}:8000/api/v0/move_pixel", "POST", {"px_x": float(x), "px_y": float(y), "z": target_z, "diagonal": is_diag}, timeout=None)
                worker.success.connect(lambda d: self._on_command_success(d, "Перемещение в пиксель"))
                worker.failed.connect(lambda e: self._on_command_failed(e, "Перемещение в пиксель"))
            self._check_homing_and_execute(do_move)

    def _update_placement_buttons_state(self):
        has_pcb = (self._last_result is not None)
        has_blank = any(str(det.get("name", "")).lower() == "blank" for det in self.placement_detections)
        self.btn_start_placement.setEnabled(has_pcb and has_blank)
        self.btn_auto_place.setEnabled(has_pcb and has_blank)
        self.btn_fabricate.setEnabled(has_pcb and len(self.placed_pcbs) > 0)

    def _validate_placement_geometry(self, pos: QPointF, angle: float) -> str | None:
        pcb_exact = self._get_pcb_combined_path(pos, angle, margin=0.0)
        if pcb_exact.isEmpty(): return None
        bbox = pcb_exact.boundingRect()
        
        limits = self.machine_limits
        min_x, max_x = limits.get("x", {}).get("min", 0.0), limits.get("x", {}).get("max", 260.0)
        min_y, max_y = limits.get("y", {}).get("min", -160.0), limits.get("y", {}).get("max", 20.0)
        
        if bbox.left() < min_x or bbox.right() > max_x or bbox.top() < min_y or bbox.bottom() > max_y: return "Плата выходит за пределы рабочей области станка!"
            
        pcb_blank_check = self._get_pcb_combined_path(pos, angle, margin=1.0)
        blank_path = QPainterPath()
        for i, det in enumerate(self.placement_detections):
            if str(det.get("name", "")).lower() == "blank" and i < len(self.placement_segments_mm):
                blank_path.addPolygon(QPolygonF([QPointF(pt[0], pt[1]) for pt in self.placement_segments_mm[i]]))
                
        if blank_path.isEmpty(): return "Не найден заготовочный материал (класс 'blank')!"
        
        huge_rect = QPainterPath()
        huge_rect.addRect(-1000, -1000, 3000, 3000)
        if huge_rect.subtracted(blank_path).intersects(pcb_blank_check): return "Элементы платы выходят за границы заготовки!"
            
        pcb_danger_check = self._get_pcb_combined_path(pos, angle, margin=3.0)
        danger_path = QPainterPath()
        for i, det in enumerate(self.placement_detections):
            if str(det.get("name", "")).lower() == "danger" and i < len(self.placement_segments_mm):
                danger_path.addPolygon(QPolygonF([QPointF(pt[0], pt[1]) for pt in self.placement_segments_mm[i]]))
                
        if not danger_path.isEmpty() and danger_path.intersects(pcb_danger_check): return "Плата пересекает запретную зону (danger)!"
        return None

    def _on_placement_toggled(self, checked: bool) -> None:
        if checked: self._start_placement_mode()
        else: self._stop_placement_mode()

    def _start_placement_mode(self) -> None:
        if not self._last_result:
            self.log_human(False, "Сперва создайте плату (Создать SVG)!")
            self.btn_start_placement.setChecked(False)
            return
        
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
        self._update_placement_preview_item()
        
    def _stop_placement_mode(self) -> None:
        self.placement_active = False
        self.view_map.set_selection_mode(False)
        try:
            self.view_map.mouseMovedInScene.disconnect(self._on_placement_mouse_move)
            self.view_map.rightClickedInScene.disconnect(self._on_placement_right_click)
            self.view_map.keyPressed.disconnect(self._on_placement_key_press)
        except Exception: pass
            
        if self.placement_preview_item:
            try: self.view_map.scene().removeItem(self.placement_preview_item)
            except RuntimeError: pass
            self.placement_preview_item = None

    def _on_placement_mouse_move(self, scene_x, scene_y) -> None:
        if not self.placement_active: return
        self.placed_pcb_pos = QPointF(scene_x, -scene_y)
        self._update_placement_preview_item()

    def _on_placement_key_press(self, key: int) -> None:
        if not self.placement_active: return
        if key in (Qt.Key_Minus, 1045, 45):
            self.placed_pcb_angle = (self.placed_pcb_angle - 5.0) % 360
            self._update_placement_preview_item()
        elif key in (Qt.Key_Equal, Qt.Key_Plus, 61, 43):
            self.placed_pcb_angle = (self.placed_pcb_angle + 5.0) % 360
            self._update_placement_preview_item()

    def _update_placement_preview_item(self) -> None:
        if self.placement_preview_item:
            try: self.view_map.scene().removeItem(self.placement_preview_item)
            except RuntimeError: pass
        self.placement_preview_item = self._create_pcb_graphics_group(self.placed_pcb_pos, self.placed_pcb_angle)
        self.view_map.scene().addItem(self.placement_preview_item)

    def _on_placement_right_click(self, scene_x, scene_y) -> None:
        if not self.placement_active: return
        
        err = self._validate_placement_geometry(self.placed_pcb_pos, self.placed_pcb_angle)
        if err is not None:
            self.log_human(False, f"Ошибка размещения: {err}")
            return
            
        self.placed_pcbs.append({'pos': self.placed_pcb_pos, 'angle': self.placed_pcb_angle})
        self.btn_start_placement.setChecked(False) 
        self._update_placement_views()
        self._update_placement_buttons_state()
        self.log_human(True, f"Плата зафиксирована (X: {self.placed_pcb_pos.x():.2f}, Y: {self.placed_pcb_pos.y():.2f}, Угол: {self.placed_pcb_angle:.1f}°)")

    def _set_placement_ui_locked(self, locked: bool):
        self.btn_fetch_photo.setEnabled(not locked)
        self.btn_select_pixel.setEnabled(not locked and bool(self.placement_segments_mm))
        self.btn_start_placement.setEnabled(not locked and self._last_result is not None)
        # Кнопка ручного пробинга удалена, её состояние регулируется FAB-конвейером
        self.combo_model.setEnabled(not locked)

    def _on_auto_placement_clicked(self) -> None:
        if getattr(self, "auto_worker", None) and self.auto_worker.isRunning():
            self.log_human(False, "Авто-размещение отменяется...")
            self.auto_worker.cancel()
            return

        self.placed_pcbs = []
        self._update_placement_views()
        self._set_placement_ui_locked(True)
        
        self.btn_auto_place.setText("⏳ Отменить вычисления...")
        self.btn_auto_place.setStyleSheet("background-color: #FFA500; font-weight: bold;")
        self.btn_auto_place.setChecked(True)
        self.log_human(True, "Запуск расчета авторазмещения (4 потока)...")
        
        self.find_auto_placement_async()

    def find_auto_placement_async(self) -> None:
        blank_path = QPainterPath()
        danger_path = QPainterPath()
        
        for i, det in enumerate(self.placement_detections):
            poly = None
            if i < len(self.placement_segments_mm):
                poly = QPolygonF([QPointF(pt[0], pt[1]) for pt in self.placement_segments_mm[i]])
            if not poly: continue
            name = str(det.get("name", "")).lower()
            if name == "blank": blank_path.addPolygon(poly)
            elif name == "danger": danger_path.addPolygon(poly)
                
        if blank_path.isEmpty():
            self._on_auto_placement_failed("Не найдена зона Blank")
            return

        bbox = blank_path.boundingRect()
        b_xmin, b_xmax, b_ymin, b_ymax = bbox.left(), bbox.right(), bbox.top(), bbox.bottom()
        
        candidates = []
        for x in np.arange(b_xmin, b_xmax, 2.0):
            for y in np.arange(b_ymin, b_ymax, 2.0):
                pt = QPointF(x, y)
                if blank_path.contains(pt) and not danger_path.contains(pt):
                    candidates.append(pt)
                    
        candidates.sort(key=lambda pt: math.hypot(pt.x() - b_xmin, b_ymax - pt.y()))
        
        angles = [0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0]
        precalc_pcb = {}
        for angle in angles:
            pb = self._get_pcb_combined_path(QPointF(0,0), angle, margin=1.0)
            pd = self._get_pcb_combined_path(QPointF(0,0), angle, margin=3.0)
            precalc_pcb[angle] = (pb, pb.boundingRect(), pd, pd.boundingRect())
            
        huge_rect = QPainterPath()
        huge_rect.addRect(-1000, -1000, 3000, 3000)
        inverted_blank = huge_rect.subtracted(blank_path)
        
        gb_bbox = blank_path.boundingRect()
        gd_bbox = danger_path.boundingRect() if not danger_path.isEmpty() else None

        self.auto_worker = AutoPlacementWorker(
            candidates, angles, precalc_pcb, 
            self.machine_limits, inverted_blank, danger_path, 
            gb_bbox, gd_bbox
        )
        self.auto_worker.finished_success.connect(self._on_auto_placement_success)
        self.auto_worker.finished_failed.connect(self._on_auto_placement_failed)
        self.auto_worker.start()

    def _on_auto_placement_success(self, pos, angle):
        self.placed_pcbs.append({'pos': pos, 'angle': angle})
        self._reset_auto_placement_btn()
        
        self.placement_tabs.setCurrentWidget(self.proj_tab)
        self._update_placement_views()
        self.log_human(True, f"Авто-размещение успешно (X: {pos.x():.2f}, Y: {pos.y():.2f}, Угол: {angle:.1f}°)")

    def _on_auto_placement_failed(self, reason):
        self._reset_auto_placement_btn()
        self.log_human(False, f"Авторазмещение не удалось: {reason}")

    def _reset_auto_placement_btn(self):
        self.btn_auto_place.setText("🤖 Авто-размещение")
        self.btn_auto_place.setStyleSheet("")
        self.btn_auto_place.setChecked(False)
        self._set_placement_ui_locked(False)

    def _get_blank_safe_path_physical(self, margin: float = 2.0) -> QPainterPath:
        blank_path = QPainterPath()
        for i, det in enumerate(self.placement_detections):
            if str(det.get("name", "")).lower() == "blank" and i < len(self.placement_segments_mm):
                blank_path.addPolygon(QPolygonF([QPointF(pt[0], pt[1]) for pt in self.placement_segments_mm[i]]))
        if blank_path.isEmpty(): return blank_path
        stroker = QPainterPathStroker()
        stroker.setWidth(margin * 2) 
        return blank_path.subtracted(stroker.createStroke(blank_path))

    def _generate_mill_probe_points(self, pos, angle, scale, cx, cy, drill_groups) -> list[dict]:
        pcb_hull = self._get_pcb_convex_hull_physical(pos, angle)
        if pcb_hull.isEmpty(): return []

        stroker = QPainterPathStroker()
        stroker.setWidth(10.0)
        stroker.setJoinStyle(Qt.RoundJoin)
        stroker.setCapStyle(Qt.RoundCap)
        pcb_expanded = QPainterPath()
        pcb_expanded.addPath(pcb_hull)
        pcb_expanded.addPath(stroker.createStroke(pcb_hull))

        blank_safe = self._get_blank_safe_path_physical(margin=2.0)
        danger_zone = self._get_danger_path_physical()
        limits = self.machine_limits
        l_min_x, l_max_x = limits.get("x", {}).get("min", 0.0), limits.get("x", {}).get("max", 260.0)
        l_min_y, l_max_y = limits.get("y", {}).get("min", -160.0), limits.get("y", {}).get("max", 20.0)

        # Сбор зон отверстий для их исключения из карты высот гравировки
        hole_exclusion_areas = []
        for d_mm, holes in drill_groups.items():
            r_mm = (d_mm / 2.0) * 1.01
            for h in holes:
                mx, my = self._map_px_to_physical(h.x_px, h.y_px, pos, angle, scale, cx, cy)
                hole_exclusion_areas.append((mx, my, r_mm))

        bbox = pcb_expanded.boundingRect()
        grid_step = 10.0 
        points = []

        for x in np.arange(bbox.left(), bbox.right() + grid_step, grid_step):
            for y in np.arange(bbox.top(), bbox.bottom() + grid_step, grid_step):
                if not (l_min_x <= x <= l_max_x and l_min_y <= y <= l_max_y): continue
                pt = QPointF(x, y)
                if pcb_expanded.contains(pt) and blank_safe.contains(pt) and not danger_zone.contains(pt):
                    in_hole = False
                    for hx, hy, hr in hole_exclusion_areas:
                        if math.hypot(x - hx, y - hy) <= hr:
                            in_hole = True
                            break
                    if not in_hole:
                        points.append({"x": round(float(x), 3), "y": round(float(y), 3)})
        return points

    def _generate_probe_points(self) -> list[dict]:
        if not self.placed_pcbs: return []
        placed = self.placed_pcbs[-1]
        scale = self._get_scale()
        cx, cy = self._get_center_px()
        
        # Для тестовой визуализации сетки пробинга в UI не вырезаем отверстия
        return self._generate_mill_probe_points(placed['pos'], placed['angle'], scale, cx, cy, {})

    def _map_px_to_physical(self, px, py, pcb_pos, pcb_angle, scale, cx, cy):
        rad = math.radians(pcb_angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        rel_x = (px - cx) * scale
        rel_y = -(py - cy) * scale
        rx = rel_x * cos_a - rel_y * sin_a
        ry = rel_x * sin_a + rel_y * cos_a
        return pcb_pos.x() + rx, pcb_pos.y() + ry

    # ================= ИЗГОТОВЛЕНИЕ ПЛАТЫ (FABRICATION WORKFLOW) =================
    def _start_fabrication_workflow(self):
        if not self.placed_pcbs or not self._last_result:
            self.log_human(False, "Плата не размещена!")
            return
            
        def start():
            self.btn_fabricate.setEnabled(False)
            self._fab_plan = []
            
            pcb_pos = self.placed_pcbs[0]['pos']
            pcb_angle = self.placed_pcbs[0]['angle']
            scale = self._get_scale()
            cx, cy = self._get_center_px()
            
            # Группировка отверстий по диаметру
            holes = self._last_result.holes
            drill_groups = {}
            for h in holes:
                d_mm = round((h.radius_px * 2 * scale), 2)
                drill_groups.setdefault(d_mm, []).append(h)
                
            sorted_diams = sorted(drill_groups.keys())
            
            # 1. Добавление этапов сверления под каждый диаметр
            for diam in sorted_diams:
                pts_px = [(h.x_px, h.y_px) for h in drill_groups[diam]]
                pts_mm = []
                for px, py in pts_px:
                    mx, my = self._map_px_to_physical(px, py, pcb_pos, pcb_angle, scale, cx, cy)
                    pts_mm.append({'x': round(mx, 3), 'y': round(my, 3)})
                    
                self._fab_plan.append(lambda d=diam, pts=pts_mm: self._fab_drill_phase(d, pts))
                
            # 2. Добавление этапа фрезеровки дорожек и контуров
            self._fab_plan.append(lambda: self._fab_mill_phase(pcb_pos, pcb_angle, scale, cx, cy, drill_groups))
            self._next_fab_step()
            
        self._check_homing_and_execute(start)

    def _next_fab_step(self):
        if not getattr(self, '_fab_plan', []):
            self._safe_travel(0.0, 0.0)
            self.log_human(True, "🎉 ПРОЦЕСС ИЗГОТОВЛЕНИЯ ПОЛНОСТЬЮ ЗАВЕРШЕН!")
            self.btn_fabricate.setEnabled(True)
            return
        step_func = self._fab_plan.pop(0)
        step_func()

    def _safe_travel(self, x, y, on_success=None):
        """Безопасный переезд с предварительным подъемом по Z"""
        ip = self.resolve_server_ip()
        worker1 = ApiWorker(f"http://{ip}:8000/api/v0/move_absolute", "POST", {"z": 0.0, "diagonal": True})
        worker1.failed.connect(lambda e: self.log_human(False, f"Ошибка подъема Z: {e}"))
        
        def move_xy():
            worker2 = ApiWorker(f"http://{ip}:8000/api/v0/move_absolute", "POST", {"x": x, "y": y, "diagonal": True})
            if on_success: worker2.success.connect(lambda _: on_success())
            worker2.failed.connect(lambda e: self.log_human(False, f"Ошибка переезда XY: {e}"))
            self._track_worker(worker2)
            
        worker1.success.connect(lambda _: move_xy())
        self._track_worker(worker1)

    def _prompt_user(self, title, text, on_yes):
        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setIcon(QMessageBox.Information)
        btn_ok = msg.addButton("✅ Выполнено", QMessageBox.AcceptRole)
        btn_cancel = msg.addButton("❌ Отмена", QMessageBox.RejectRole)
        msg.exec_()
        if msg.clickedButton() == btn_ok: on_yes()
        else:
            self.log_human(False, "Процесс прерван пользователем.")
            self.btn_fabricate.setEnabled(True)

    def _fab_drill_phase(self, diam_mm, points_mm):
        self.log_human(True, f"Этап сверления: диаметр {diam_mm} мм")
        def step2():
            self._prompt_user("Смена инструмента", f"Установите СВЕРЛО {diam_mm} мм.\nПодключите крокодил автоуровня.", step3)
        def step3():
            self.log_human(True, f"Снятие карты высот для отверстий ({len(points_mm)} точек)...")
            ip = self.resolve_server_ip()
            payload = {"points": points_mm, "V_MAX": float(self.cut_speed_steps_s.value()), "A_MAX": float(self.max_accel_steps_s2.value())}
            worker = ApiWorker(f"http://{ip}:8000/api/v0/z_probe", "POST", payload, timeout=None)
            worker.success.connect(step4)
            worker.failed.connect(lambda e: self.log_human(False, f"Ошибка Z-пробинга: {e}"))
            self._track_worker(worker)
        def step4(data):
            probed_pts = data.get("points", [])
            self._current_drill_z_map = { (p['x'], p['y']): p['z'] for p in probed_pts }
            self._safe_travel(0, 20, step5)
        def step5():
            self._prompt_user("Запуск обработки", "ВНИМАНИЕ: Снимите крокодил автоуровня!\nНачать сверление?", step6)
        def step6():
            gc = GCodeBuilder(feedrate=float(self.cut_speed_steps_s.value()))
            gc.motor_on()
            for p in points_mm:
                x, y = p['x'], p['y']
                z_surface = self._current_drill_z_map.get((x, y), 0.0)
                safe_z = z_surface - 2.0  
                target_z = z_surface + float(self.stock_thickness_mm.value()) + 0.5 
                
                gc.rapid(z=0)
                gc.rapid(x=x, y=y)
                gc.rapid(z=safe_z)
                gc.feed(z=target_z, f=60.0)
                gc.rapid(z=0)
                
            gc.motor_off()
            ip = self.resolve_server_ip()
            payload = {"commands": gc.lines, "V_MAX": float(self.cut_speed_steps_s.value()), "A_MAX": float(self.max_accel_steps_s2.value())}
            worker = ApiWorker(f"http://{ip}:8000/api/v0/run_gcode", "POST", payload, timeout=None)
            worker.success.connect(lambda _: self._next_fab_step())
            worker.failed.connect(lambda e: self.log_human(False, f"Ошибка сверления: {e}"))
            self._track_worker(worker)
            
        self._safe_travel(0, 20, step2)

    def _fab_mill_phase(self, pos, angle, scale, cx, cy, drill_groups):
        self.log_human(True, "Этап фрезеровки контуров и проводников")
        def step2():
            self._prompt_user("Смена инструмента", "Установите ГРАВЕР/ФРЕЗУ.\nПодключите крокодил автоуровня.", step3)
        def step3():
            probe_pts = self._generate_mill_probe_points(pos, angle, scale, cx, cy, drill_groups)
            if not probe_pts:
                self.log_human(False, "Не сгенерированы точки пробинга.")
                return
            
            ip = self.resolve_server_ip()
            payload = {"points": probe_pts, "V_MAX": float(self.cut_speed_steps_s.value()), "A_MAX": float(self.max_accel_steps_s2.value())}
            worker = ApiWorker(f"http://{ip}:8000/api/v0/z_probe", "POST", payload, timeout=None)
            worker.success.connect(step4)
            worker.failed.connect(lambda e: self.log_human(False, f"Ошибка Z-пробинга: {e}"))
            self._track_worker(worker)
        def step4(data):
            pts = data.get("points", [])
            self.tabs.setCurrentWidget(self.spline_tab)
            self.spline_widget.plot_points(pts)
            
            arr_pts = np.array([[p['x'], p['y']] for p in pts])
            arr_zs = np.array([p['z'] for p in pts])
            lin = LinearNDInterpolator(arr_pts, arr_zs)
            near = NearestNDInterpolator(arr_pts, arr_zs)
            def get_z(x, y):
                z = lin(x, y)
                if np.isnan(z): z = near(x, y)
                return float(z)
            self._z_interpolator = get_z
            self._safe_travel(0, 20, step5)
            
        def step5():
            self._prompt_user("Запуск фрезеровки", "ВНИМАНИЕ: Снимите крокодил автоуровня!\nНачать фрезеровку?", step6)
            
        def step6():
            self.log_human(True, "Экспорт векторных путей в G-код и запуск фрезеровки...")
            gc = GCodeBuilder(feedrate=float(self.cut_speed_steps_s.value()))
            gc.motor_on()
            
            def add_paths(paths_px, depth):
                for poly in paths_px:
                    if len(poly) < 2: continue
                    mx, my = self._map_px_to_physical(poly[0][0], poly[0][1], pos, angle, scale, cx, cy)
                    z_surf = self._z_interpolator(mx, my)
                    gc.rapid(z=0)
                    gc.rapid(x=mx, y=my)
                    gc.feed(z=z_surf + depth, f=100.0) 
                    
                    for px, py in poly[1:]:
                        nx, ny = self._map_px_to_physical(px, py, pos, angle, scale, cx, cy)
                        nz_surf = self._z_interpolator(nx, ny)
                        gc.feed(x=nx, y=ny, z=nz_surf + depth, f=float(self.cut_speed_steps_s.value()))
                    gc.rapid(z=0)

            # 1. Изоляция дорожек (Black)
            depth_traces = float(self.trace_depth_mm.value()) * 1.01
            add_paths(self._last_result.black_paths, depth_traces)
            
            # 2. Текст / маркировка (Green)
            depth_silk = float(self.green_depth_mm.value())
            add_paths(self._last_result.green_paths, depth_silk)
            
            # 3. Обрезка контура (Red) с распределением по слоям заглубления
            stock = float(self.stock_thickness_mm.value())
            max_pass = float(self.trace_depth_mm.value()) * 1.25
            passes = math.ceil(stock / max_pass)
            pass_depths = [(i + 1) * (stock / passes) for i in range(passes)]
            
            for depth_cut in pass_depths:
                add_paths(self._last_result.cut_paths, depth_cut)
                
            gc.motor_off()
            ip = self.resolve_server_ip()
            payload = {"commands": gc.lines, "V_MAX": float(self.cut_speed_steps_s.value()), "A_MAX": float(self.max_accel_steps_s2.value())}
            worker = ApiWorker(f"http://{ip}:8000/api/v0/run_gcode", "POST", payload, timeout=None)
            worker.success.connect(lambda _: self._next_fab_step())
            worker.failed.connect(lambda e: self.log_human(False, f"Ошибка фрезеровки: {e}"))
            self._track_worker(worker)

        self._safe_travel(0, 20, step2)

    def _init_layer_list(self) -> None:
        self.layer_list.clear()
        layers = [("source", "Оригинал", True), ("holes", "Отверстия", True), ("green", "Текст", True), ("paths", "Дорожки", True), ("cut", "Контур", True)]
        self.layer_list.blockSignals(True)
        for key, title, checked in layers:
            item = QListWidgetItem(title)
            item.setData(Qt.UserRole, key) 
            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsDragEnabled | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            self.layer_list.addItem(item)
        self.layer_list.blockSignals(False)
        self._update_preview()
        self._update_placement_buttons_state()

    def _build_source_group(self) -> QGroupBox:
        group = QGroupBox("Входы")
        layout = QGridLayout(group)
        layout.addWidget(QPushButton("Выбрать картинку", clicked=self._select_image), 0, 0)
        self.image_path_edit = QLineEdit()
        self.image_path_edit.setReadOnly(True)
        layout.addWidget(self.image_path_edit, 0, 1)
        layout.addWidget(QPushButton("Выбрать слои (собрать PNG)", clicked=self._show_layer_composer), 1, 0)
        layout.addWidget(QPushButton("Выбрать Gerber", clicked=self._show_gerber_stub), 2, 0)
        layout.addWidget(QLabel("Поддерживаемые форматы: JPG, PNG, BMP"), 1, 1, 2, 1) 
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
        self.width_mm.setValue(100.0)
        layout.addWidget(self.width_mm, 0, 1)
        self.height_radio = QRadioButton("Высота (мм)")
        self.height_radio.toggled.connect(self._sync_scale_mode)
        layout.addWidget(self.height_radio, 1, 0)
        self.height_mm = QDoubleSpinBox()
        self.height_mm.setRange(0.001, 1_000_000.0)
        self.height_mm.setEnabled(False)
        layout.addWidget(self.height_mm, 1, 1)
        return group

    @staticmethod
    def _spin(minimum: float, maximum: float, value: float, suffix: str = "", decimals: int = 3) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setValue(value)
        if suffix: spin.setSuffix(suffix)
        return spin

    def _build_drills_group(self) -> QGroupBox:
        group = QGroupBox("Диаметры свёрл")
        layout = QGridLayout(group)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setHorizontalSpacing(15)
        layout.setVerticalSpacing(4)
        self.drill_spins = []
        
        hole_colors = [
            ((0, 0, 255), "Синий"),
            ((0, 255, 255), "Голубой"),
            ((255, 0, 255), "Пурпурный"),
            ((255, 255, 0), "Желтый"),
            ((255, 165, 0), "Оранжевый")
        ]
        default_diams = [0.8, 1.0, 1.2, 1.5, 2.0]
        
        for idx, (rgb, name) in enumerate(hole_colors):
            row_layout = QHBoxLayout()
            
            swatch = QLabel()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); border: 1px solid #000;")
            row_layout.addWidget(swatch)
            
            lbl = QLabel(f"RGB{rgb}")
            lbl.setFixedWidth(100)
            row_layout.addWidget(lbl)
            
            spin = self._spin(0.0, 1000.0, default_diams[idx], suffix=" мм", decimals=2)
            self.drill_spins.append(spin)
            row_layout.addWidget(spin)
            
            row_layout.addStretch()
            
            # Первые 3 элемента в колонку 0, остальные в колонку 1
            col = 0 if idx < 3 else 1
            row = idx if idx < 3 else idx - 3
            
            layout.addLayout(row_layout, row, col)
            
        return group

    def _build_tool_group(self) -> QGroupBox:
        group = QGroupBox("Инструменты и векторные пути")
        form = QFormLayout(group)
        self.tool_angle_deg = self._spin(0.1, 179.0, 30.0, suffix=" °", decimals=1)
        self.stepover_percent = self._spin(1.0, 100.0, 50.0, suffix=" %", decimals=1)
        self.simplify_tolerance_mm = self._spin(0.0, 100.0, 0.0, suffix=" мм", decimals=4)
        
        self.green_mode = QComboBox()
        self.green_mode.addItems(["По центру линий (Centerline)", "Змейка (scanline)", "Контуры (contour-offset)"])
        
        form.addRow("Угол фрезы:", self.tool_angle_deg)
        form.addRow("Разряженность:", self.stepover_percent)
        form.addRow("Симплификация:", self.simplify_tolerance_mm)
        form.addRow("Режим гравировки:", self.green_mode)
        return group

    def _validate_depths(self) -> None:
        copper = self.trace_depth_mm.value()
        green = self.green_depth_mm.value()
        if green >= copper:
            self.green_depth_mm.blockSignals(True)
            new_green = max(0.001, copper - 0.01)
            self.green_depth_mm.setValue(new_green)
            self.green_depth_mm.blockSignals(False)
            logger.info(f"⚠️ Глубина гравировки автоматически скорректирована до {new_green:.3f} мм.")

    def _build_future_gcode_group(self) -> QGroupBox:
        group = QGroupBox("Параметры для создания G-кода")
        form = QFormLayout(group)
        
        self.stock_thickness_mm = self._spin(0.001, 1000.0, 1.5, suffix=" мм")
        self.cut_speed_steps_s = self._spin(0.001, 100000.0, 15000.0, suffix=" шаг/с", decimals=0)
        self.max_accel_steps_s2 = self._spin(0.001, 1000000.0, 10000.0, suffix=" шаг/с²", decimals=0)
        
        self.trace_depth_mm = self._spin(0.001, 1000.0, 0.2, suffix=" мм")
        self.green_depth_mm = self._spin(0.001, 1000.0, 0.1, suffix=" мм")
        
        self.trace_depth_mm.valueChanged.connect(lambda: self._validate_depths())
        self.green_depth_mm.valueChanged.connect(lambda: self._validate_depths())
        
        motor_layout = QHBoxLayout()
        self.btn_motor_on = QPushButton("🟢 Включить")
        self.btn_motor_off = QPushButton("🔴 Выключить")
        self.btn_motor_on.clicked.connect(lambda: self._set_motor_state(True))
        self.btn_motor_off.clicked.connect(lambda: self._set_motor_state(False))
        motor_layout.addWidget(self.btn_motor_on)
        motor_layout.addWidget(self.btn_motor_off)
        
        form.addRow("Толщина заготовки:", self.stock_thickness_mm)
        form.addRow("Скорость реза:", self.cut_speed_steps_s)
        form.addRow("Ускорение:", self.max_accel_steps_s2)
        form.addRow("Толщина меди:", self.trace_depth_mm)
        form.addRow("Глубина гравировки:", self.green_depth_mm)
        form.addRow("Шпиндель:", motor_layout)
        return group

    def _sync_scale_mode(self) -> None:
        self.width_mm.setEnabled(self.width_radio.isChecked())
        self.height_mm.setEnabled(not self.width_radio.isChecked())

    def _select_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Исходная картинка", str(self._base_dir), "Форматы (*.jpg *.png *.bmp)")
        if path: self._load_image_state(Path(path))

    def _show_layer_composer(self) -> None:
        dialog = LayerComposerDialog(self)
        if dialog.exec_() and (paths := dialog.get_paths()):
            self._compose_and_load_layers(paths)

    def _compose_and_load_layers(self, paths_dict: dict[str, str]) -> None:
        resolved_paths = {k: Path(v) for k, v in paths_dict.items() if v}
        out_path = self._base_dir / "composed_source.png"
        compose_layers_to_image(resolved_paths, out_path)
        self._load_image_state(out_path)

    def _show_gerber_stub(self) -> None:
        dialog = GerberComposerDialog(self)
        if dialog.exec_() and (paths := dialog.get_paths()):
            layers_paths = {k: Path(v) for k, v in paths.items()}
            
            # Теперь функция возвращает 4 значения, включая массив диаметров сверл
            composed, w_mm, h_mm, drill_diams = render_gerber_and_excellon(layers_paths)
            
            # Обновляем значения в спинбоксах интерфейса
            for i, spin in enumerate(self.drill_spins):
                if i < len(drill_diams):
                    spin.setValue(drill_diams[i])
            
            out_path = self._base_dir / "composed_source.png"
            is_success, buffer = cv2.imencode(".png", cv2.cvtColor(composed, cv2.COLOR_RGBA2BGRA))
            if is_success: buffer.tofile(str(out_path))
            self._load_image_state(out_path)
            self.width_radio.setChecked(True)
            self.width_mm.setValue(w_mm)
            self._sync_scale_mode()
    
    def _load_image_state(self, path: Path) -> None:
        self._image_path = path
        self.image_path_edit.setText(str(path))
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
            
            for i, val in enumerate(config.drill_diameters_mm):
                if i < len(self.drill_spins):
                    self.drill_spins[i].setValue(val)
        except Exception: pass

    def _build_config(self) -> PipelineConfig:
        if self._image_path is None: raise PipelineError("Сперва выберите картинку.")
        w = float(self.width_mm.value()) if self.width_radio.isChecked() else None
        h = float(self.height_mm.value()) if not self.width_radio.isChecked() else None
        mode = "centerline" if "Centerline" in self.green_mode.currentText() else ("scanline" if "scanline" in self.green_mode.currentText() else "contour-offset")
        
        diams = [float(spin.value()) for spin in self.drill_spins]
        
        return PipelineConfig(
            image_path=self._image_path, width_mm=w, height_mm=h,
            tool_angle_deg=float(self.tool_angle_deg.value()), 
            drill_diameters_mm=diams,
            stepover_percent=float(self.stepover_percent.value()), simplify_tolerance_mm=float(self.simplify_tolerance_mm.value()),
            green_mode=mode, stock_thickness_mm=float(self.stock_thickness_mm.value()),
            cut_speed_mm_s=float(self.cut_speed_steps_s.value()), max_accel_mm_s2=float(self.max_accel_steps_s2.value()),
            trace_depth_mm=float(self.trace_depth_mm.value()), green_depth_mm=float(self.green_depth_mm.value()),
            skip_validation=self._initial_skip_validation
        )

    def _generate(self) -> None:
        try:
            config = self._build_config()
            out_dir = self._base_dir / "outputs"
            if out_dir.exists(): shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            result = run_pipeline(config)
            files = export_svg_bundle(result, config, out_dir)
            
            self._last_result = result
            self.placed_pcbs = [] 
            self._update_preview()
            self._update_placement_views()
            self._update_placement_buttons_state()
            self.log_human(True, f"Генерация SVG успешно завершена. Создано файлов: {len(files)}.")
        except Exception as exc:
            self.log_human(False, f"Ошибка генерации: {exc}")

    def _update_preview(self) -> None:
        scene = self.preview_label.scene()
        scene.clear()  
        if self._source_qimage is None and self._last_result is None: return

        for layer_key, enabled in reversed([(str(self.layer_list.item(i).data(Qt.UserRole)), self.layer_list.item(i).checkState() == Qt.Checked) for i in range(self.layer_list.count())]):
            if not enabled: continue
            if layer_key == "source" and self._source_qimage is not None:
                item = scene.addPixmap(QPixmap.fromImage(self._source_qimage))
                item.setOpacity(0.3)
                continue
            if self._last_result is None: continue
            
            if layer_key == "holes":
                for hole in self._last_result.holes:
                    c = QColor(*hole.color, 220)
                    scene.addEllipse(hole.x_px - hole.radius_px, hole.y_px - hole.radius_px, hole.radius_px * 2, hole.radius_px * 2, QPen(Qt.NoPen), c)
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
        pen_thick.setCapStyle(Qt.RoundCap); pen_thick.setJoinStyle(Qt.RoundJoin)
        scene.addPath(qpath, pen_thick)
        pen_thin = QPen(color); pen_thin.setWidthF(0); pen_thin.setCosmetic(True)
        scene.addPath(qpath, pen_thin)
        
    def find_auto_placement_sync(self) -> tuple[QPointF, float] | None:
        blank_path = QPainterPath()
        danger_path = QPainterPath()
        
        for i, det in enumerate(self.placement_detections):
            poly = None
            if i < len(self.placement_segments_mm):
                poly = QPolygonF([QPointF(pt[0], pt[1]) for pt in self.placement_segments_mm[i]])
            if not poly: continue
            name = str(det.get("name", "")).lower()
            if name == "blank": blank_path.addPolygon(poly)
            elif name == "danger": danger_path.addPolygon(poly)
                
        if blank_path.isEmpty():
            return None

        bbox = blank_path.boundingRect()
        b_xmin, b_xmax, b_ymin, b_ymax = bbox.left(), bbox.right(), bbox.top(), bbox.bottom()
        
        candidates = []
        for x in np.arange(b_xmin, b_xmax, 2.0):
            for y in np.arange(b_ymin, b_ymax, 2.0):
                pt = QPointF(x, y)
                if blank_path.contains(pt) and not danger_path.contains(pt):
                    candidates.append(pt)
                    
        candidates.sort(key=lambda pt: math.hypot(pt.x() - b_xmin, b_ymax - pt.y()))
        
        angles = [0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0]
        precalc_pcb = {}
        for angle in angles:
            pb = self._get_pcb_combined_path(QPointF(0,0), angle, margin=1.0)
            pd = self._get_pcb_combined_path(QPointF(0,0), angle, margin=3.0)
            precalc_pcb[angle] = (pb, pb.boundingRect(), pd, pd.boundingRect())
            
        huge_rect = QPainterPath()
        huge_rect.addRect(-1000, -1000, 3000, 3000)
        inverted_blank = huge_rect.subtracted(blank_path)
        
        limits = self.machine_limits
        min_x, max_x = limits.get("x", {}).get("min", 0.0), limits.get("x", {}).get("max", 260.0)
        min_y, max_y = limits.get("y", {}).get("min", -160.0), limits.get("y", {}).get("max", 20.0)
        
        for pt in candidates:
            for angle in angles:
                pb, pb_rect, pd, pd_rect = precalc_pcb[angle]
                
                trans_pb_rect = pb_rect.translated(pt)
                if trans_pb_rect.left() < min_x or trans_pb_rect.right() > max_x or \
                   trans_pb_rect.top() < min_y or trans_pb_rect.bottom() > max_y:
                    continue
                    
                trans_pb = pb.translated(pt)
                if inverted_blank.intersects(trans_pb):
                    continue
                    
                if not danger_path.isEmpty():
                    trans_pd = pd.translated(pt)
                    if danger_path.intersects(trans_pd):
                        continue
                        
                return pt, angle
        return None
    
    def _set_motor_state(self, is_on: bool) -> None:
            if not self.is_homed:
                self.log_human(False, "Ошибка: Невозможно управлять шпинделем, сперва выполните паркинг!")
                return

            ip = self.resolve_server_ip()
            if not ip: 
                return

            payload = {"state": is_on}
            worker = ApiWorker(f"http://{ip}:8000/api/v0/set_motor_state", "POST", payload, timeout=3.0)
            
            action_name = "Включение мотора" if is_on else "Выключение мотора"
            
            worker.success.connect(lambda d: self.log_human(True, f"{action_name} выполнено успешно."))
            worker.failed.connect(lambda e: self.log_human(False, f"Ошибка сети при попытке изменить состояние мотора: {e}"))
            self._track_worker(worker)