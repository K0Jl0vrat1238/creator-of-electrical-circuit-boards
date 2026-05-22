# main.py
from __future__ import annotations
import shutil
import sys
import math
import logging
import argparse  # Парсер аргументов командной строки
from pathlib import Path
import cv2
import numpy as np
import yaml  # Импортируем PyYAML для загрузки конфигурации
from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal, QMimeData
from PyQt5.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QGraphicsPixmapItem,
    QGraphicsScene, QGraphicsView, QMainWindow, QMessageBox, QPushButton,
    QRadioButton, QStyleFactory, QTabWidget, QVBoxLayout, QWidget, QCheckBox
)
from pipeline import PipelineConfig, PipelineError, PipelineResult, run_pipeline, compose_layers_to_image
from svg_export import export_svg_bundle
from gerber_importer import render_gerber_and_excellon

logger = logging.getLogger("gui")

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
    def __init__(self) -> None:
        super().__init__(QGraphicsScene())
        self.setViewportUpdateMode(QGraphicsView.SmartViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(QColor("#efefef"))
        self.setMinimumSize(640, 480)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setStyleSheet("border: 1px solid #808080;")
        self._zoom = 1.0
        self._panning = False
        self._pan_start = None

        self._lod_timer = QTimer(self)
        self._lod_timer.setSingleShot(True)
        self._lod_timer.timeout.connect(self._set_high_quality)

        self._set_high_quality()

    def _set_low_quality(self) -> None:
        self.setRenderHint(QPainter.Antialiasing, False)
        self.setRenderHint(QPainter.SmoothPixmapTransform, False)

    def _set_high_quality(self) -> None:
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.viewport().update()

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

        self._set_low_quality()
        self.scale(factor, factor)
        self._zoom = next_zoom
        self._lod_timer.start(500)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.scene().items():
            self._panning = True
            self._pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            self._set_low_quality()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._panning and self._pan_start is not None:
            delta = event.pos() - self._pan_start
            self._pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y()) 
            self._lod_timer.start(500)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._panning:
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.ArrowCursor)
            self._lod_timer.start(500)
            event.accept()
            return
        super().mouseReleaseEvent(event)


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

    def dragLeaveEvent(self, event) -> None:
        self._dragged_row = None
        super().dragLeaveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._dragged_row = None
        super().mouseReleaseEvent(event)


class MainWindow(QMainWindow):
    def __init__(self, skip_validation: bool = False) -> None:
        super().__init__()
        self.setWindowTitle("Фрезеровщик печатных плат")
        self.setMinimumSize(1140, 760)
        self._base_dir = Path(__file__).resolve().parent
        self._image_path: Path | None = None
        self._source_rgba: np.ndarray | None = None
        self._source_qimage: QImage | None = None
        self._last_result: PipelineResult | None = None
        
        self._initial_skip_validation = skip_validation

        self._build_ui()
        self._init_layer_list()

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

    def _build_controls_tab(self, parent: QWidget) -> None:
        outer = QVBoxLayout(parent)
        outer.setSpacing(10)
        outer.addWidget(self._build_source_group())
        outer.addWidget(self._build_scale_group())
        outer.addWidget(self._build_tool_group())
        outer.addWidget(self._build_future_gcode_group())

        self.status_label = QLabel("Готов.")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        outer.addWidget(self.status_label)

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

        layout.addLayout(left, 1)

        self.preview_label = PreviewView() 
        layout.addWidget(self.preview_label, 4)

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
        
        self.skip_validation_checkbox = QCheckBox("Игнорировать ошибки трассировки и коллизии")
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

        self.cut_speed_mm_s = self._spin(0.001, 100000.0, 20.0, suffix=" мм/с")
        form.addRow("Скорость реза:", self.cut_speed_mm_s)

        self.max_accel_mm_s2 = self._spin(0.001, 1_000_000.0, 500.0, suffix=" мм/с²")
        form.addRow("Максимальное ускорение:", self.max_accel_mm_s2)

        self.trace_depth_mm = self._spin(0.001, 1000.0, 0.2, suffix=" мм")
        form.addRow("Глубина прореза дорожки:", self.trace_depth_mm)

        self.green_depth_mm = self._spin(0.001, 1000.0, 0.1, suffix=" мм")
        form.addRow("Глубина гравировки:", self.green_depth_mm)
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
        if not path:
            return
        logger.info(f"User selected raw image: {path}")
        self._load_image_state(Path(path))

    def _show_layer_composer(self) -> None:
        dialog = LayerComposerDialog(self)
        if dialog.exec_():
            paths = dialog.get_paths()
            if paths:
                self.status_label.setText("Склеиваем слои...")
                QApplication.processEvents()
                try:
                    logger.info(f"Assembling image layers from list: {paths}")
                    self._compose_and_load_layers(paths)
                except Exception as e:
                    logger.error("Layer assembly error", exc_info=True)
                    QMessageBox.critical(self, "Ошибка сборки", f"Не удалось склеить слои:\n{e}")
                    self.status_label.setText("Ошибка сборки слоев.")

    def _compose_and_load_layers(self, paths_dict: dict[str, str]) -> None:
        resolved_paths = {k: Path(v) for k, v in paths_dict.items() if v}
        out_path = self._base_dir / "composed_source.png"
        try:
            # Используем общую функцию сборки слоев из pipeline.py
            compose_layers_to_image(resolved_paths, out_path)
            self._load_image_state(out_path)
            self.status_label.setText("Слои успешно собраны.")
        except Exception as e:
            logger.error("Layer assembly error", exc_info=True)
            QMessageBox.critical(self, "Ошибка сборки", f"Не удалось склеить слои:\n{e}")
            self.status_label.setText("Ошибка сборки слоев.")

    def _show_gerber_stub(self) -> None:
        dialog = GerberComposerDialog(self)
        if dialog.exec_():
            paths = dialog.get_paths()
            if not paths:
                return
            
            self.status_label.setText("Импортируем Gerber-файлы...")
            QApplication.processEvents()
            try:
                logger.info(f"Initiating Gerber import with file paths: {paths}")
                layers_paths = {k: Path(v) for k, v in paths.items()}
                composed, width_mm, height_mm = render_gerber_and_excellon(layers_paths)
                
                # Сохраняем временный PNG
                out_path = self._base_dir / "composed_source.png"
                is_success, buffer = cv2.imencode(".png", cv2.cvtColor(composed, cv2.COLOR_RGBA2BGRA))
                if is_success:
                    buffer.tofile(str(out_path))
                else:
                    raise ValueError("Ошибка кодирования временного PNG.")
                
                self._load_image_state(out_path)
                self.width_radio.setChecked(True)
                self.width_mm.setValue(width_mm)
                self._sync_scale_mode()
                
                logger.info(f"Gerber vectors rasterized and loaded: size={width_mm:.2f}x{height_mm:.2f} mm")
                self.status_label.setText(f"Gerber успешно импортирован. Размер: {width_mm:.2f} x {height_mm:.2f} мм")
            except Exception as e:
                logger.error("Gerber/Excellon import error", exc_info=True)
                QMessageBox.critical(self, "Ошибка импорта Gerber", f"Не удалось прочитать векторные файлы:\n{e}")
                self.status_label.setText("Ошибка импорта Gerber.")
    
    def _load_image_state(self, path: Path) -> None:
        self._image_path = path
        self.image_path_edit.setText(str(path))
        self.status_label.setText("Картинка загружена.")
        self._source_rgba = self._load_rgba_image(self._image_path)
        self._source_qimage = self._numpy_rgba_to_qimage(self._source_rgba)
        self._last_result = None  
        self._update_preview()
        self.preview_label.fit_to_scene() 

    def load_config_from_yaml(self, path: Path) -> None:
        """Считывает настройки из YAML-файла и безопасно заполняет ими GUI."""
        logger.info(f"Loading UI settings from YAML: {path.name}")
        try:
            config = PipelineConfig.from_yaml(path, skip_validation=self._initial_skip_validation)
            
            # 1. Загрузка исходного изображения
            if config.image_path.exists():
                self._load_image_state(config.image_path)
                logger.info(f"Config loaded image: {config.image_path}")
            else:
                logger.warning(f"Image path from config does not exist: {config.image_path}")

            # 2. Физические размеры
            if config.width_mm is not None:
                self.width_radio.setChecked(True)
                self.width_mm.setValue(config.width_mm)
            elif config.height_mm is not None:
                self.height_radio.setChecked(True)
                self.height_mm.setValue(config.height_mm)
            self._sync_scale_mode()

            # 3. Параметры инструментов
            self.tool_angle_deg.setValue(config.tool_angle_deg)
            self.drill_diameter_mm.setValue(config.drill_diameter_mm)
            self.stepover_percent.setValue(config.stepover_percent)
            self.simplify_tolerance_mm.setValue(config.simplify_tolerance_mm)
            
            idx = self.green_mode.findData(config.green_mode)
            if idx != -1:
                self.green_mode.setCurrentIndex(idx)

            # 4. Параметры G-кода
            self.stock_thickness_mm.setValue(config.stock_thickness_mm)
            self.cut_speed_mm_s.setValue(config.cut_speed_mm_s)
            self.max_accel_mm_s2.setValue(config.max_accel_mm_s2)
            self.trace_depth_mm.setValue(config.trace_depth_mm)
            self.green_depth_mm.setValue(config.green_depth_mm)
            self.skip_validation_checkbox.setChecked(config.skip_validation)

            logger.info("Successfully populated UI fields from YAML.")
            self.status_label.setText(f"Конфиг {path.name} загружен.")
            
        except Exception as e:
            logger.error(f"Failed to parse configuration YAML: {e}", exc_info=True)
            self.status_label.setText("Ошибка разбора YAML.")

    def _build_config(self) -> PipelineConfig:
        if self._image_path is None:
            raise PipelineError("Сперва выберите исходную картинку.")

        width_mm = float(self.width_mm.value()) if self.width_radio.isChecked() else None
        height_mm = float(self.height_mm.value()) if not self.width_radio.isChecked() else None

        return PipelineConfig(
            image_path=self._image_path,
            width_mm=width_mm,
            height_mm=height_mm,
            tool_angle_deg=float(self.tool_angle_deg.value()),
            drill_diameter_mm=float(self.drill_diameter_mm.value()),
            stepover_percent=float(self.stepover_percent.value()),
            simplify_tolerance_mm=float(self.simplify_tolerance_mm.value()),
            green_mode=self.green_mode.currentData(),
            stock_thickness_mm=float(self.stock_thickness_mm.value()),
            cut_speed_mm_s=float(self.cut_speed_mm_s.value()),
            max_accel_mm_s2=float(self.max_accel_mm_s2.value()),
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
            
            logger.info(f"Clearing output directory: {out_dir}")
            self._clear_output_dir(out_dir)

            logger.info("Starting processing pipeline configuration...")
            result = run_pipeline(config)
            
            logger.info("Exporting paths into SVG bundles...")
            files = export_svg_bundle(result, config, out_dir)
            
            self._last_result = result
            self._update_preview()

            self.status_label.setText("SVG сгенерировано.")
            logger.info(f"SVG bundle generation finalized. Created files count: {len(files)}")
            
            # Формирование отчета
            details = "Созданные файлы:\n" + "\n".join(f"- {path.name}" for path in files.values())
            
            # Добавление предупреждений
            if result.warnings:
                warnings_text = "\n\n⚠️ Предупреждения / Игнорируемые ошибки:\n" + "\n".join(f"• {w}" for w in result.warnings)
                logger.warning(f"Pipeline finished with warnings: {result.warnings}")
            else:
                warnings_text = ""

            QMessageBox.information(self, "Успешно завершено", details + warnings_text)
            
        except PipelineError as exc:
            logger.warning(f"Pipeline domain validation failed: {exc}")
            self.status_label.setText("Ошибка генерации.")
            QMessageBox.critical(self, "Ошибка", str(exc))
        except Exception as exc:
            logger.error("Critical error during pipeline execution", exc_info=True)
            self.status_label.setText("Ошибка.")
            QMessageBox.critical(self, "Ошибка", f"Непредвиденная ошибка: {exc}")

    def _update_preview(self) -> None:
        scene = self.preview_label.scene()
        scene.clear()  

        source_img = self._source_qimage
        result = self._last_result

        if source_img is None and result is None:
            return

        for layer_key, enabled in reversed(self._layer_state()):
            if not enabled:
                continue
            
            if layer_key == "source" and source_img is not None:
                pixmap = QPixmap.fromImage(source_img)
                item = scene.addPixmap(pixmap)
                item.setTransformationMode(Qt.SmoothTransformation)
                item.setOpacity(0.3)
                continue
                
            if result is None:
                continue
                
            if layer_key == "holes":
                brush = QColor(0, 80, 255, 220)
                pen = QPen(Qt.NoPen)
                for hole in result.holes:
                    r = hole.radius_px
                    scene.addEllipse(hole.x_px - r, hole.y_px - r, r * 2, r * 2, pen, brush)
                    
            elif layer_key == "green":
                self._add_paths_to_scene(scene, result.green_paths, QColor(0, 200, 0, 255), result.green_tool_radius_px * 2)
            elif layer_key == "paths":
                self._add_paths_to_scene(scene, result.black_paths, QColor(20, 20, 20, 255), result.black_tool_radius_px * 2)
            elif layer_key == "cut":
                self._add_paths_to_scene(scene, result.cut_paths, QColor(255, 0, 0, 255), result.cut_tool_radius_px * 2)

        if scene.items():
            scene.setSceneRect(scene.itemsBoundingRect())

    def _add_paths_to_scene(
        self,
        scene: QGraphicsScene,
        paths: list[list[tuple[float, float]]],
        color: QColor,
        width: float,
    ) -> None:
        """Отрисовка в 2 слоя: толстая полупрозрачная линия + тонкая направляющая."""
        if not paths:
            return
            
        qpath = QPainterPath()
        for polyline in paths:
            if len(polyline) < 2:
                continue
            qpath.moveTo(QPointF(polyline[0][0], polyline[0][1]))
            for x, y in polyline[1:]:
                qpath.lineTo(QPointF(x, y))

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
        state: list[tuple[str, bool]] = []
        for idx in range(self.layer_list.count()):
            item = self.layer_list.item(idx)
            key = str(item.data(Qt.UserRole))
            enabled = item.checkState() == Qt.Checked
            state.append((key, enabled))
        return state

    def _clear_output_dir(self, output_dir: Path) -> None:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _load_rgba_image(path: Path) -> np.ndarray:
        raw = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise PipelineError("Не удалось загрузить картинку.")
        if image.ndim == 2: return cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
        if image.ndim == 3 and image.shape[2] == 3: return cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        if image.ndim == 3 and image.shape[2] == 4: return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        raise PipelineError("Неподдерживаемый формат.")

    @staticmethod
    def _numpy_rgba_to_qimage(rgba: np.ndarray) -> QImage:
        arr = np.ascontiguousarray(rgba)
        h, w = arr.shape[:2]
        return QImage(arr.data, w, h, w * 4, QImage.Format_RGBA8888).copy()


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