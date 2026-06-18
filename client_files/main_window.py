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
from PyQt5.QtGui import QColor, QImage, QPainterPath, QPen, QBrush, QPixmap, QPolygonF, QFont, QPainterPathStroker
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidgetItem, QGraphicsScene, QMainWindow, QMessageBox, QPushButton, QRadioButton, QTabWidget,
    QVBoxLayout, QWidget, QCheckBox, QFrame, QSlider, QTextEdit, QGraphicsItemGroup, QGraphicsPathItem
)
from pipeline import PipelineConfig, PipelineError, PipelineResult, run_pipeline, compose_layers_to_image
from svg_export import export_svg_bundle
from gerber_importer import render_gerber_and_excellon

from common import COLOR_MAP, DEFAULT_COLOR, QLogHandler, QLogSignal, logger
from workers import ApiWorker, FetchWorker
from widgets import GerberComposerDialog, LayerComposerDialog, LayerListWidget, PreviewView, Spline3DWidget
from ip_by_mac import get_ip_by_mac

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
        self.placed_pcbs = []  # Содержит dict {'pos': QPointF, 'angle': float}

        self._build_ui()

        # Инициализация терминала логов
        self.log_emitter = QLogSignal()
        self.log_emitter.log_msg.connect(self.terminal.append)
        gui_logger = QLogHandler(self.log_emitter)
        logging.getLogger().addHandler(gui_logger)

        self._init_layer_list()
        
        QTimer.singleShot(1000, self._api_get_coords)

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(8)

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

        # Терминал
        term_group = QGroupBox("Терминал станка (Логи)")
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
        self.combo_model.setCurrentText("fast")
        top_controls.addWidget(self.combo_model)

        self.btn_fetch_photo = QPushButton("Получить фото")
        self.btn_fetch_photo.clicked.connect(self._fetch_placement_photo)
        top_controls.addWidget(self.btn_fetch_photo)

        self.btn_select_pixel = QPushButton("🎯 Ехать в пиксель")
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

        # Кнопка пробинга
        self.btn_run_probing = QPushButton("📏 Сделать пробинг")
        self.btn_run_probing.setEnabled(False)
        self.btn_run_probing.setStyleSheet("background-color: #32a852; color: white;")
        self.btn_run_probing.clicked.connect(self._on_run_probing_clicked)
        top_controls.addWidget(self.btn_run_probing)

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

    def _build_spline_tab(self, parent: QWidget) -> None:
        layout = QVBoxLayout(parent)
        self.spline_widget = Spline3DWidget()
        layout.addWidget(self.spline_widget)

    def show_error(self, title, message):
        logger.error(f"UI Error [{title}]: {message}")
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
        logger.info("Запуск процедуры паркинга (Homing)...")
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
        logger.warning("ЭКСТРЕННАЯ ОСТАНОВКА АКТИВИРОВАНА В ГУИ")
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
            self.show_error("Ошибка лимитов", f"Координата X ({tx:.2f} мм) выходит за пределы [{min_x:.2f}, {max_x:.2f}].")
            return
        if not (min_y <= ty <= max_y):
            self.show_error("Ошибка лимитов", f"Координата Y ({ty:.2f} мм) выходит за пределы [{min_y:.2f}, {max_y:.2f}].")
            return
        if not (min_z <= tz <= max_z):
            self.show_error("Ошибка лимитов", f"Координата Z ({tz:.2f} мм) выходит за пределы [{min_z:.2f}, {max_z:.2f}].")
            return

        ip = self.resolve_server_ip()
        if not ip: return

        payload = {"x": float(tx), "y": float(ty), "z": float(tz), "diagonal": self.manual_diag_cb.isChecked()}
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
                "Выполнить паркинг перед снимком?"
            )
            msg_box.setStyleSheet("QLabel{ min-width: 350px; font-size: 11px; } QPushButton{ min-height: 25px; }")
            
            btn_park = msg_box.addButton("🏠 Да, выполнить паркинг", QMessageBox.AcceptRole)
            btn_photo = msg_box.addButton("📸 Нет, сделать фото как есть", QMessageBox.DestructiveRole)
            btn_cancel = msg_box.addButton("❌ Отменить снимок", QMessageBox.RejectRole)
            
            msg_box.exec_()
            clicked = msg_box.clickedButton()
            
            if clicked == btn_cancel:
                logger.info("Пользователь отменил получение фото из-за неподходящих координат.")
                return
            elif clicked == btn_park:
                logger.info("Пользователь выбрал парковку перед получением фото.")
                self.run_homing()
                return
            else:
                logger.warning("Пользователь проигнорировал предупреждение и делает снимок как есть.")

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
            self.btn_run_probing.setEnabled(False)

            self.placement_status.setText("Успешно.")
            self._update_placement_buttons_state()
        except Exception as e:
            self.placement_status.setText(f"Ошибка парсинга: {e}")
            logger.error(f"Fetch Parse Error: {e}")
        finally:
            self.btn_fetch_photo.setEnabled(True)

    def _on_fetch_failed(self, err):
        self.placement_status.setText(f"Ошибка сети: {err}")
        self.btn_fetch_photo.setEnabled(True)
        self.show_error("Ошибка", f"Не удалось получить данные:\n{err}")

    # --- ВСПОМОГАТЕЛЬНАЯ ГЕОМЕТРИЯ ДЛЯ ПЛАТЫ (МАСШТАБЫ И ЦЕНТРЫ) ---
    def _get_scale(self) -> float:
        """Возвращает масштаб: миллиметры на один пиксель."""
        if not self._source_qimage: return 1.0
        config = self._build_config()
        if config.width_mm is not None:
            return config.width_mm / self._source_qimage.width()
        elif config.height_mm is not None:
            return config.height_mm / self._source_qimage.height()
        return 1.0

    def _get_center_px(self) -> tuple[float, float]:
        """Возвращает центр платы в исходных пикселях."""
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
        """
        Строит единый QPainterPath, описывающий точную площадь (в физических мм), 
        которую занимают ВЕСЬ отрисованный текст, пути, контуры и отверстия с учетом отступов.
        Пустые пространства внутри контуров игнорируются (не считаются коллизией).
        """
        res = QPainterPath()
        if not self._last_result: return res
        
        scale = self._get_scale()
        cx, cy = self._get_center_px()
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        
        def map_pt(x, y):
            rel_x = (x - cx) * scale
            rel_y = -(y - cy) * scale # Перевод в декартову СК (Y вверх)
            rx = rel_x * cos_a - rel_y * sin_a
            ry = rel_x * sin_a + rel_y * cos_a
            return QPointF(pos.x() + rx, pos.y() + ry)
            
        # Линейные слои (Дорожки, Шелкография, Контур)
        layers = [
            (self._last_result.cut_paths, self._last_result.cut_tool_radius_px),
            (self._last_result.black_paths, self._last_result.black_tool_radius_px),
            (self._last_result.green_paths, self._last_result.green_tool_radius_px)
        ]
        
        for polys, r_px in layers:
            if not polys: continue
            dia_mm = (r_px * 2 * scale) + (margin * 2)
            stroker = QPainterPathStroker()
            stroker.setWidth(max(0.01, dia_mm))
            stroker.setCapStyle(Qt.RoundCap)
            stroker.setJoinStyle(Qt.RoundJoin)
            
            layer_path = QPainterPath()
            for poly in polys:
                if len(poly) < 2: continue
                layer_path.moveTo(map_pt(*poly[0]))
                for pt in poly[1:]:
                    layer_path.lineTo(map_pt(*pt))
            # Добавляем расширенный путь (обведенный фрезой и отступом)
            res.addPath(stroker.createStroke(layer_path))
            
        # Сверловка (Отверстия)
        for hole in self._last_result.holes:
            center = map_pt(hole.x_px, hole.y_px)
            r_mm = (hole.radius_px * scale) + margin
            res.addEllipse(center, r_mm, r_mm)
            
        return res

    def _get_pcb_convex_hull_physical(self, pos: QPointF, angle: float) -> QPainterPath:
        """
        Возвращает Convex Hull (выпуклую оболочку) платы по всем ее самым 
        выпирающим частям (контуры, текст, дороги, отверстия) в физических миллиметрах.
        """
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
                for pt in poly:
                    all_pts.append(map_pt_phys(*pt))
        for hole in self._last_result.holes:
            p_center = map_pt_phys(hole.x_px, hole.y_px)
            r_mm = hole.radius_px * scale
            all_pts.extend([
                QPointF(p_center.x() + r_mm, p_center.y()),
                QPointF(p_center.x() - r_mm, p_center.y()),
                QPointF(p_center.x(), p_center.y() + r_mm),
                QPointF(p_center.x(), p_center.y() - r_mm)
            ])

        if not all_pts: return QPainterPath()

        from scipy.spatial import ConvexHull
        pts_array = np.array([[p.x(), p.y()] for p in all_pts])
        hull_path = QPainterPath()
        try:
            hull = ConvexHull(pts_array)
            first = True
            for vertex_idx in hull.vertices:
                pt = pts_array[vertex_idx]
                if first:
                    hull_path.moveTo(pt[0], pt[1])
                    first = False
                else:
                    hull_path.lineTo(pt[0], pt[1])
            hull_path.closeSubpath()
        except:
            pass
        return hull_path

    def _get_danger_path_physical(self) -> QPainterPath:
        """Возвращает QPainterPath запретной зоны в физ. координатах с запасом 3мм."""
        danger_path = QPainterPath()
        for i, det in enumerate(self.placement_detections):
            if str(det.get("name", "")).lower() == "danger" and i < len(self.placement_segments_mm):
                poly = QPolygonF([QPointF(pt[0], pt[1]) for pt in self.placement_segments_mm[i]])
                danger_path.addPolygon(poly)

        stroker = QPainterPathStroker()
        stroker.setWidth(6.0) # Отступ 3мм во все стороны = диаметр 6мм
        margined = stroker.createStroke(danger_path)

        res = QPainterPath()
        res.addPath(danger_path)
        res.addPath(margined)
        return res

    def _create_pcb_graphics_group(self, pos: QPointF, angle: float) -> QGraphicsItemGroup:
        """Создает группу графических элементов для отображения ВСЕХ слоев платы."""
        group = QGraphicsItemGroup()
        if not self._last_result: return group
        
        scale = self._get_scale()
        cx, cy = self._get_center_px()
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        
        def map_pt_scene(x, y):
            # В scene_map координата Y инвертирована относительно физики
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
            pen.setWidthF(dia_mm)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            item.setPen(pen)
            group.addToGroup(item)
            
            # Тонкая линия по центру для красоты
            thin_pen = QPen(QColor(color_rgba[0], color_rgba[1], color_rgba[2], 255))
            thin_pen.setWidthF(0)
            thin_pen.setCosmetic(True)
            item_thin = QGraphicsPathItem(path)
            item_thin.setPen(thin_pen)
            group.addToGroup(item_thin)
            
        # Цвета как в предпросмотре: Контур (Red), Дорожки (Black), Текст (Green)
        add_layer(self._last_result.cut_paths, self._last_result.cut_tool_radius_px, (255, 0, 0, 140))
        add_layer(self._last_result.black_paths, self._last_result.black_tool_radius_px, (20, 20, 20, 140))
        add_layer(self._last_result.green_paths, self._last_result.green_tool_radius_px, (0, 200, 0, 140))
        
        # Отверстия (Blue)
        for hole in self._last_result.holes:
            center = map_pt_scene(hole.x_px, hole.y_px)
            r_mm = hole.radius_px * scale
            path = QPainterPath()
            path.addEllipse(center, r_mm, r_mm)
            item = QGraphicsPathItem(path)
            item.setPen(QPen(Qt.NoPen))
            item.setBrush(QBrush(QColor(0, 80, 255, 220)))
            group.addToGroup(item)
            
        return group


    def _update_placement_views(self):
        if not self.placement_img: return

        # 1. ОТРИСОВКА ВЕРХА (ФОТО)
        scene_yolo = self.view_yolo.scene()
        scene_yolo.clear()
        pixmap = QPixmap.fromImage(self.placement_img)
        scene_yolo.addPixmap(pixmap)

        show_flags = {"workspace": self.chk_workspace.isChecked(), "blank": self.chk_blank.isChecked(), "danger": self.chk_danger.isChecked()}

        for i, det in enumerate(self.placement_detections):
            c_name = str(det.get("name", "")).lower()
            if not show_flags.get(c_name, True): continue
                
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
            if c_name == "workspace": continue 
            
            if i < len(self.placement_segments_mm) and self.placement_segments_mm[i]:
                priority = 0 if c_name == "blank" else 1 
                draw_order.append((priority, c_name, self.placement_segments_mm[i]))

        draw_order.sort(key=lambda x: x[0])
        for _, c_name, seg_mm in draw_order:
            color = COLOR_MAP.get(c_name, DEFAULT_COLOR)
            polygon = QPolygonF([QPointF(pt[0], -pt[1]) for pt in seg_mm])
            pen = QPen(color, 1)
            scene_map.addPolygon(polygon, pen, QBrush(color))

        # Отрисовка размещенных плат со ВСЕМИ слоями
        for placement in self.placed_pcbs:
            group = self._create_pcb_graphics_group(placement['pos'], placement['angle'])
            scene_map.addItem(group)

    def _on_pixel_clicked(self, x, y):
        self.btn_select_pixel.setChecked(False)

        if not self.placement_img: return
        w, h = self.placement_img.width(), self.placement_img.height()
        if not (0 <= x < w and 0 <= y < h): return 
            
        if not self.is_homed or self.current_z is None:
            reply = QMessageBox.question(self, "Требуется паркинг", "Координаты не определены.\nВыполнить паркинг прямо сейчас?", QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes: self.run_homing()
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
            payload = {"px_x": float(x), "px_y": float(y), "z": z_spin.value(), "diagonal": diag_cb.isChecked()}
            self.api_worker = ApiWorker(f"http://{ip}:8000/api/v0/move_pixel", "POST", payload)
            self.api_worker.success.connect(self._on_command_success)
            self.api_worker.failed.connect(self._on_command_failed)
            self.api_worker.start()

    # --- ПРОВЕРКИ И АВТОМАТИКА ---
    def _update_placement_buttons_state(self):
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

    def _validate_placement_geometry(self, pos: QPointF, angle: float) -> str | None:
        """
        Проверка коллизий через точные булевы операции QPainterPath.
        Проверяется только фактическое пересечение элементов платы (с учетом толщины фрезы и отступа)
        с границами зон. Пустоты внутри платы не вызывают ложных срабатываний.
        """
        # 1. Лимиты станка (без отступов)
        pcb_exact = self._get_pcb_combined_path(pos, angle, margin=0.0)
        if pcb_exact.isEmpty(): return None # Нет данных
        bbox = pcb_exact.boundingRect()
        
        limits = self.placement_limits
        min_x = limits.get("x", {}).get("min", 0.0)
        max_x = limits.get("x", {}).get("max", 260.0)
        min_y = limits.get("y", {}).get("min", -160.0)
        max_y = limits.get("y", {}).get("max", 20.0)
        
        if bbox.left() < min_x or bbox.right() > max_x or bbox.top() < min_y or bbox.bottom() > max_y:
            return f"Плата выходит за пределы рабочей области станка! X: [{min_x}, {max_x}], Y: [{min_y}, {max_y}]"
            
        # 2. Проверка заготовки (с запасом 1.0 мм)
        pcb_blank_check = self._get_pcb_combined_path(pos, angle, margin=1.0)
        blank_path = QPainterPath()
        for i, det in enumerate(self.placement_detections):
            if str(det.get("name", "")).lower() == "blank" and i < len(self.placement_segments_mm):
                poly = QPolygonF([QPointF(pt[0], pt[1]) for pt in self.placement_segments_mm[i]])
                blank_path.addPolygon(poly)
                
        if blank_path.isEmpty(): return "Не найден заготовочный материал (класс 'blank')!"
        
        # Создаем негатив бланка, чтобы проверить, вылезло ли что-то "за пределы"
        huge_rect = QPainterPath()
        huge_rect.addRect(-1000, -1000, 3000, 3000)
        inverted_blank = huge_rect.subtracted(blank_path)
        if inverted_blank.intersects(pcb_blank_check):
            return "Фактические элементы платы выходят за границы заготовки (с учетом запаса 1.0 мм)!"
            
        # 3. Проверка опасных зон (с запасом 3.0 мм)
        pcb_danger_check = self._get_pcb_combined_path(pos, angle, margin=3.0)
        danger_path = QPainterPath()
        for i, det in enumerate(self.placement_detections):
            if str(det.get("name", "")).lower() == "danger" and i < len(self.placement_segments_mm):
                poly = QPolygonF([QPointF(pt[0], pt[1]) for pt in self.placement_segments_mm[i]])
                danger_path.addPolygon(poly)
                
        if not danger_path.isEmpty() and danger_path.intersects(pcb_danger_check):
            return "Фактические элементы платы пересекают запретную зону (класс 'danger') с учетом запаса 3.0 мм!"
            
        return None

    def _start_placement_mode(self) -> None:
        if not self._last_result:
            self.show_error("Ошибка", "Сперва создайте плату (нажмите 'Создать SVG' на первой вкладке)!")
            return
        
        self.placed_pcbs = []
        self._update_placement_views()
        self.btn_run_probing.setEnabled(False)
        
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
            try: self.view_map.scene().removeItem(self.placement_preview_item)
            except RuntimeError: pass
            self.placement_preview_item = None
            
        self.placement_status.setText("Размещение завершено.")

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
            self.show_error("Ошибка размещения", err)
            return
            
        placement_data = {'pos': self.placed_pcb_pos, 'angle': self.placed_pcb_angle}
        self.placed_pcbs.append(placement_data)
        
        self._stop_placement_mode()
        self._update_placement_views()
        self.btn_run_probing.setEnabled(True)
        
        logger.info(f"Ручное размещение зафиксировано (X: {self.placed_pcb_pos.x():.2f}, Y: {self.placed_pcb_pos.y():.2f}, Угол: {self.placed_pcb_angle:.1f}°)")
        QMessageBox.information(
            self, "Успех", 
            f"Плата зафиксирована:\nX: {self.placed_pcb_pos.x():.2f} мм, Y: {self.placed_pcb_pos.y():.2f} мм\nУгол: {self.placed_pcb_angle:.1f}°"
        )

    def find_auto_placement(self) -> tuple[QPointF, float] | None:
        blanks = []
        for i, det in enumerate(self.placement_detections):
            if str(det.get("name", "")).lower() == "blank":
                if i < len(self.placement_segments_mm):
                    seg = self.placement_segments_mm[i]
                    blanks.append(QPolygonF([QPointF(pt[0], pt[1]) for pt in seg]))
        if not blanks: return None
        
        all_pts = [pt for b in blanks for pt in b]
        b_xmin = min(pt.x() for pt in all_pts)
        b_xmax = max(pt.x() for pt in all_pts)
        b_ymin = min(pt.y() for pt in all_pts)
        b_ymax = max(pt.y() for pt in all_pts)
        
        step_x = 4.0
        step_y = 4.0
        angles = [0.0, 90.0, 180.0, 270.0, 45.0, 135.0, 225.0, 315.0]
        
        for angle in angles:
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
                        err = self._validate_placement_geometry(pt_center, angle)
                        if err is None:
                            return pt_center, angle
                    curr_y += step_y
                curr_x += step_x
        return None

    def _on_auto_placement_clicked(self) -> None:
        self.placed_pcbs = []
        self._update_placement_views()
        self.btn_run_probing.setEnabled(False)
        
        res = self.find_auto_placement()
        if res is None:
            self.show_error("Ошибка авто-размещения", "Невозможно разместить автоматически: нет безопасного свободного места.")
            return
            
        pos, angle = res
        self.placed_pcbs.append({'pos': pos, 'angle': angle})
        
        self.placement_tabs.setCurrentWidget(self.proj_tab)
        self._update_placement_views()
        self.btn_run_probing.setEnabled(True)
        
        logger.info(f"Авто-размещение успешно (X: {pos.x():.2f}, Y: {pos.y():.2f}, Угол: {angle:.1f}°)")
        QMessageBox.information(
            self, "Авто-размещение выполнено", 
            f"Координаты:\nX: {pos.x():.2f} мм, Y: {pos.y():.2f} мм\nУгол: {angle:.1f}°"
        )

    def _get_blank_safe_path_physical(self, margin: float = 2.0) -> QPainterPath:
        """Возвращает безопасную зону заготовки (суженную на margin мм от краев)."""
        blank_path = QPainterPath()
        for i, det in enumerate(self.placement_detections):
            if str(det.get("name", "")).lower() == "blank" and i < len(self.placement_segments_mm):
                poly = QPolygonF([QPointF(pt[0], pt[1]) for pt in self.placement_segments_mm[i]])
                blank_path.addPolygon(poly)
                
        if blank_path.isEmpty(): 
            return blank_path
            
        # Чтобы "сузить" бланк, вычитаем из него его же обводку (stroke)
        stroker = QPainterPathStroker()
        stroker.setWidth(margin * 2) # Умножаем на 2, т.к. stroke строится в обе стороны от линии контура
        outline = stroker.createStroke(blank_path)
        
        return blank_path.subtracted(outline)

    def _on_run_probing_clicked(self) -> None:
        """Метод генерации умной сетки и запуска процесса пробинга."""
        if not self.placed_pcbs:
            self.show_error("Ошибка", "Сперва нужно разместить плату!")
            return

        # --- ОКНО ПРОВЕРКИ КРОКОДИЛОВ ---
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setWindowTitle("Важная проверка перед пробингом")
        msg_box.setText("Вы точно прикрепили крокодилы к шпинделю и плате?")
        
        # Создаем свои кнопки с нужным текстом
        btn_yes = msg_box.addButton("Да, прикрепил, начать пробинг", QMessageBox.AcceptRole)
        btn_no = msg_box.addButton("Нет, не прикрепил, отменить", QMessageBox.RejectRole)
        
        msg_box.exec_()
        
        # Если пользователь нажал "Нет" или просто закрыл окно крестиком
        if msg_box.clickedButton() != btn_yes:
            logger.info("Пробинг отменен: пользователь не прикрепил крокодилы.")
            return
        # --------------------------------
        
        if not self.is_homed:
            reply = QMessageBox.question(
                self, "Требуется паркинг",
                "Станок не имеет точных координат.\nВыполнить паркинг перед началом пробинга?",
                QMessageBox.Yes | QMessageBox.Cancel
            )
            if reply == QMessageBox.Yes:
                self.run_homing()
            return

        placed = self.placed_pcbs[-1]
        pcb_hull = self._get_pcb_convex_hull_physical(placed['pos'], placed['angle'])

        if pcb_hull.isEmpty():
            self.show_error("Ошибка", "Контур платы пуст, невозможно рассчитать сетку.")
            return

        # 1. Расширяем зону пробинга на 5мм вокруг платы для лучшей интерполяции краев
        stroker = QPainterPathStroker()
        stroker.setWidth(10.0)  # Диаметр 10 = радиус (отступ) 5мм во все стороны
        stroker.setJoinStyle(Qt.RoundJoin)
        stroker.setCapStyle(Qt.RoundCap)
        pcb_expanded = QPainterPath()
        pcb_expanded.addPath(pcb_hull)
        pcb_expanded.addPath(stroker.createStroke(pcb_hull))

        # 2. Получаем зоны строгих ограничений
        blank_safe = self._get_blank_safe_path_physical(margin=2.0) # Запас 2 мм от края бланка
        danger_zone = self._get_danger_path_physical()              # Запас 3 мм от Danger (заложен внутри метода)

        # ПОЛУЧАЕМ ФИЗИЧЕСКИЕ ЛИМИТЫ СТАНКА ИЗ API
        limits = self.placement_limits
        limit_min_x = limits.get("x", {}).get("min", 0.0)
        limit_max_x = limits.get("x", {}).get("max", 260.0)
        limit_min_y = limits.get("y", {}).get("min", -160.0)
        limit_max_y = limits.get("y", {}).get("max", 20.0)

        # Находим границы РАСШИРЕННОЙ оболочки платы
        bbox = pcb_expanded.boundingRect()
        min_x, max_x = bbox.left(), bbox.right()
        min_y, max_y = bbox.top(), bbox.bottom()

        grid_step = 10.0 # 10 мм = 1 см
        points = []

        # Создаем сетку
        for x in np.arange(min_x, max_x + grid_step, grid_step):
            for y in np.arange(min_y, max_y + grid_step, grid_step):
                # ЖЕСТКАЯ ПРОВЕРКА: Точка не должна выходить за пределы станка
                if not (limit_min_x <= x <= limit_max_x and limit_min_y <= y <= limit_max_y):
                    continue

                pt = QPointF(x, y)
                # УСЛОВИЯ ДОБАВЛЕНИЯ ТОЧКИ:
                # 1. Точка внутри расширенной на 5мм области платы
                # 2. Точка строго внутри заготовки (с отступом от краев)
                # 3. Точка строго вне опасных зон (с отступом от краев)
                if pcb_expanded.contains(pt) and blank_safe.contains(pt) and not danger_zone.contains(pt):
                    points.append({"x": round(float(x), 3), "y": round(float(y), 3)})

        # Если плата крошечная и сетка пустая, пробуем хотя бы центр
        if not points:
            center = bbox.center()
            cx, cy = center.x(), center.y()
            if (limit_min_x <= cx <= limit_max_x and limit_min_y <= cy <= limit_max_y):
                if blank_safe.contains(center) and not danger_zone.contains(center):
                    points.append({"x": round(float(cx), 3), "y": round(float(cy), 3)})
                    
        if not points:
            self.show_error("Внимание", "Не найдено безопасных точек для пробинга!\nВозможно, плата слишком близко к зоне Danger или вылезает за Blank.")
            return

        ip = self.resolve_server_ip()
        if not ip: return

        # Берем скорости из настроек G-кода
        payload = {
            "points": points,
            "V_MAX": float(self.cut_speed_steps_s.value()),
            "A_MAX": float(self.max_accel_steps_s2.value())
        }

        self.btn_run_probing.setEnabled(False)
        self.status_label.setText(f"Запуск Z-пробинга (точек: {len(points)})...")
        
        self.api_worker = ApiWorker(f"http://{ip}:8000/api/v0/z_probe", "POST", payload)
        self.api_worker.success.connect(self._on_probing_success)
        self.api_worker.failed.connect(self._on_probing_failed)
        self.api_worker.start()

    def _on_probing_success(self, data):
        self.btn_run_probing.setEnabled(True)
        self.status_label.setText("Z-Пробинг завершен.")
        self._api_get_coords()

        points = data.get("points", [])
        if not points:
            self.show_error("Ошибка", "Станок вернул пустой список точек.")
            return

        # Переключаемся на вкладку со сплайном и отрисовываем
        self.tabs.setCurrentWidget(self.spline_tab)
        self.spline_widget.plot_points(points)
        QMessageBox.information(self, "Успех", f"Успешно снята карта высот ({len(points)} точек).")

    def _on_probing_failed(self, err):
        self.btn_run_probing.setEnabled(True)
        self.status_label.setText("Ошибка Z-пробинга.")
        self._api_get_coords()
        self.show_error("Ошибка пробинга", err)


    # --- МЕТОДЫ ОСНОВНОЙ ГЕНЕРАЦИИ (БЕЗ ИЗМЕНЕНИЙ) ---
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
            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsDragEnabled | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
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
        if suffix: spin.setSuffix(suffix)
        return spin

    def _sync_scale_mode(self) -> None:
        by_width = self.width_radio.isChecked()
        self.width_mm.setEnabled(by_width)
        self.height_mm.setEnabled(not by_width)

    def _select_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Выберите исходную картинку", str(self._base_dir), "Растровые форматы (*.jpg *.jpeg *.png *.bmp)")
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

            logger.info("Генерация векторных путей успешна.")
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