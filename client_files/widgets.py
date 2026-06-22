# widgets.py
from __future__ import annotations

import logging
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal, QMimeData
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAbstractItemView, QDialog, QDialogButtonBox,
    QFileDialog, QGridLayout, QLabel, QLineEdit, QListWidget, QGraphicsScene, QGraphicsView, 
    QPushButton, QVBoxLayout, QWidget, QHBoxLayout, QSlider
)

import numpy as np
import scipy.interpolate as interp
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

logger = logging.getLogger("gui")

COLOR_MAP = {
    "danger": QColor(255, 50, 50),
    "workspace": QColor(0, 191, 255),
    "blank": QColor(50, 205, 50),
}
DEFAULT_COLOR = QColor(200, 200, 200)

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

class Spline3DWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.fig.patch.set_facecolor('#c0c0c0')
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor('#c0c0c0')
        self.layout.addWidget(self.canvas, stretch=1)

        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("Увеличение неровностей (Масштаб Z):"))
        self.z_scale_slider = QSlider(Qt.Horizontal)
        self.z_scale_slider.setRange(1, 100)
        self.z_scale_slider.setValue(5)  
        self.z_scale_slider.valueChanged.connect(self._update_z_scale)
        ctrl_layout.addWidget(self.z_scale_slider)
        
        # Кнопка сохранения в PNG
        self.btn_save_png = QPushButton("💾 Сохранить PNG")
        self.btn_save_png.clicked.connect(self._save_png)
        ctrl_layout.addWidget(self.btn_save_png)
        
        self.layout.addLayout(ctrl_layout)
        self.current_points = []

    def plot_points(self, points: list[dict]):
        self.current_points = points
        self.ax.clear()

        if not points or len(points) < 4:
            self.ax.text2D(0.5, 0.5, "Недостаточно точек для построения поверхности (минимум 4)",
                           transform=self.ax.transAxes, ha="center")
            self.canvas.draw()
            return

        xs = np.array([p['x'] for p in points])
        ys = np.array([p['y'] for p in points])
        zs = np.array([p['z'] for p in points])

        margin = 1.0
        grid_x, grid_y = np.mgrid[min(xs)-margin:max(xs)+margin:60j, min(ys)-margin:max(ys)+margin:60j]

        try:
            grid_z = interp.griddata((xs, ys), zs, (grid_x, grid_y), method='cubic')
            grid_z_linear = interp.griddata((xs, ys), zs, (grid_x, grid_y), method='linear')
            nans = np.isnan(grid_z)
            grid_z[nans] = grid_z_linear[nans]
            
            grid_z_nearest = interp.griddata((xs, ys), zs, (grid_x, grid_y), method='nearest')
            nans = np.isnan(grid_z)
            grid_z[nans] = grid_z_nearest[nans]

            self.ax.plot_surface(grid_x, grid_y, grid_z, cmap='coolwarm_r', edgecolor='none', alpha=0.85)
            self.ax.scatter(xs, ys, zs, color='black', s=15, label='Точки Z-пробинга', depthshade=False)

            self.ax.set_xlabel('X (мм)')
            self.ax.set_ylabel('Y (мм)')
            self.ax.set_zlabel('Z (мм)')
            self.ax.set_title("Карта высот платы (Сплайн)")
            self.ax.legend()
                
            if not self.ax.zaxis_inverted():
                self.ax.invert_zaxis()

            self.ax.view_init(elev=25, azim=-65)
            self._update_z_scale()
        except Exception as e:
            self.ax.text2D(0.5, 0.5, f"Ошибка построения сплайна:\n{e}",
                           transform=self.ax.transAxes, ha="center")
            self.canvas.draw()

    def _update_z_scale(self):
        if not self.current_points: return
        scale = self.z_scale_slider.value() / 10.0
        self.ax.set_box_aspect((1, 1, scale))
        self.canvas.draw_idle()

    def _save_png(self):
        try:
            out_dir = Path(__file__).resolve().parent / "outputs"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "spline.png"
            self.fig.savefig(out_path, dpi=150)
            logger.info(f"✅ Сплайн успешно сохранен в: {out_path}")
        except Exception as e:
            logger.error(f"❌ Ошибка при сохранении сплайна: {e}")