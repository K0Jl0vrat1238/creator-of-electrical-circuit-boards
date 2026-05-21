# main.py
from __future__ import annotations
import shutil
import sys
import math
import logging
from pathlib import Path
import cv2
import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal, QMimeData
from PyQt5.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QGraphicsPixmapItem,
    QGraphicsScene, QGraphicsView, QMainWindow, QMessageBox, QPushButton,
    QRadioButton, QStyleFactory, QTabWidget, QVBoxLayout, QWidget
)
from pipeline import PipelineConfig, PipelineError, PipelineResult, run_pipeline
from svg_export import export_svg_bundle
from gerber_importer import render_gerber_and_excellon

# Инициализируем логгер для графического интерфейса
logger = logging.getLogger("gui")

def setup_logging(log_file_path: Path) -> None:
    """Настройка глобальной конфигурации логирования (Консоль + Файл)."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Вывод в консоль (INFO и выше)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Запись в файл (DEBUG и выше)
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
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Фрезеровщик печатных плат")
        self.setMinimumSize(1140, 760)
        self._base_dir = Path(__file__).resolve().parent
        self._image_path: Path | None = None
        self._source_rgba: np.ndarray | None = None
        self._source_qimage: QImage | None = None
        self._last_result: PipelineResult | None = None

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
        images = {}
        max_w, max_h = 0, 0
        for layer_name, path in paths_dict.items():
            raw = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            corners = [int(img[0, 0]), int(img[0, -1]), int(img[-1, 0]), int(img[-1, -1])]
            if sum(corners) / 4.0 < 128:
                img = cv2.bitwise_not(img)
            _, mask = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            h, w = mask.shape 
            if w > max_w: max_w = w
            if h > max_h: max_h = h
            images[layer_name] = mask

        if not images:
            raise ValueError("Не удалось прочитать ни один слой.")

        composed = np.full((max_h, max_w, 4), 255, dtype=np.uint8)
        colors = {
            'green': [0, 255, 0, 255],
            'paths': [0, 0, 0, 255],
            'cut':   [255, 0, 0, 255],
            'holes': [0, 0, 255, 255]
        }
        for layer_name in ['green', 'paths', 'cut', 'holes']:
            if layer_name in images:
                mask = images[layer_name]
                h, w = mask.shape
                off_y, off_x = (max_h - h) // 2, (max_w - w) // 2
                roi = composed[off_y:off_y+h, off_x:off_x+w]
                roi[mask == 255] = np.array(colors[layer_name], dtype=np.uint8)

        out_path = self._base_dir / "composed_source.png"
        is_success, buffer = cv2.imencode(".png", cv2.cvtColor(composed, cv2.COLOR_RGBA2BGRA))
        if is_success:
            buffer.tofile(str(out_path))
        else:
            raise ValueError("Ошибка кодирования PNG.")
        
        logger.info(f"Temporary composed PNG saved to: {out_path}")
        self._load_image_state(out_path)
        self.status_label.setText("Слои успешно собраны.")

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
        )

    def _generate(self) -> None:
        try:
            self.status_label.setText("Обработка...")
            QApplication.processEvents()

            config = self._build_config()
            out_dir = self._base_dir / "outputs"
            self._clear_output_dir(out_dir)

            result = run_pipeline(config)
            
            # Передаем объект config вторым аргументом для записи YAML-файла
            files = export_svg_bundle(result, config, out_dir) 
            self._last_result = result
            self._update_preview()

            self.status_label.setText("SVG сгенерировано.")
            
            # Формирование отчета
            details = "Созданные файлы:\n" + "\n".join(f"- {path.name}" for path in files.values())
            
            # Добавление предупреждений, если они есть
            if result.warnings:
                warnings_text = "\n\n⚠️ Внимание (некритично):\n" + "\n".join(f"• {w}" for w in result.warnings)
            else:
                warnings_text = ""

            QMessageBox.information(self, "Успешно завершено", details + warnings_text)
            
        except PipelineError as exc:
            self.status_label.setText("Ошибка генерации.")
            QMessageBox.critical(self, "Ошибка", str(exc))
        except Exception as exc:
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
                item.setOpacity(0.3)  # Снизили яркость оригинала, чтобы векторы были виднее
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

        # 1. ТОЛСТАЯ линия (физический рез фрезы) - полупрозрачная
        thick_color = QColor(color)
        thick_color.setAlpha(90)  # прозрачность 
        pen_thick = QPen(thick_color)
        pen_thick.setWidthF(max(1.0, float(width)))
        pen_thick.setCosmetic(False) # Отключаем косметику - масштабируется при зуме!
        pen_thick.setCapStyle(Qt.RoundCap)
        pen_thick.setJoinStyle(Qt.RoundJoin)
        scene.addPath(qpath, pen_thick)

        # 2. ТОНКАЯ линия (центр фрезы / вектор SVG) - яркая
        pen_thin = QPen(color)
        pen_thin.setWidthF(0)  # ширина 0 = ровно 1 пиксель всегда (косметически)
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
QLineEdit, QDoubleSpinBox, QComboBox, QListWidget { background-color: #ffffff; border: 1px solid #808080; padding: 2px 4px; }
QPushButton { background-color: #c0c0c0; border-top: 2px solid #ffffff; border-left: 2px solid #ffffff; border-right: 2px solid #404040; border-bottom: 2px solid #404040; min-height: 20px; padding: 2px 10px; }
QPushButton:pressed { border-top: 2px solid #404040; border-left: 2px solid #404040; border-right: 2px solid #ffffff; border-bottom: 2px solid #ffffff; }
"""

def main() -> int:
    app = QApplication(sys.argv)
    
    # Инициализация логирования (консоль + файл app.log)
    base_dir = Path(__file__).resolve().parent
    setup_logging(base_dir / "app.log")
    
    if "windows" in {name.lower() for name in QStyleFactory.keys()}: app.setStyle("windows")
    app.setStyleSheet(build_stylesheet())
    window = MainWindow()
    window.show()
    return app.exec_()

if __name__ == "__main__":
    raise SystemExit(main())