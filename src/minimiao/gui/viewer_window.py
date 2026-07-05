# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSplitter

try:
    from . import custom_widgets as cw
except ImportError:
    from minimiao.gui import custom_widgets as cw


class LiveViewer(QWidget):

    def __init__(self, logg, parent=None):
        super().__init__(parent)
        self.logg = logg
        self._setup_ui()

        self.live_worker = None

        self._target_img = None
        self._picking_enabled = False
        self._picking_n = None
        self.target_points = []

        self.image_plot.scene().sigMouseClicked.connect(self._on_target_mouse_clicked)
        self.image_plot.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.image_plot.installEventFilter(self)

        self.target_spots_item = pg.ScatterPlotItem(size=8, pen=pg.mkPen(width=2))
        self.image_plot.addItem(self.target_spots_item)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Vertical)

        image_widget = self._create_image_widgets()
        splitter.addWidget(image_widget)

        table_widget = self._create_table_widget()
        splitter.addWidget(table_widget)

        layout.addWidget(splitter)
        self.setLayout(layout)

    def _create_image_widgets(self):
        layout_image = QVBoxLayout()

        self.image_plot = pg.PlotWidget()
        self.image_plot.setAspectLocked(True)
        self.image_plot.getPlotItem().hideAxis("left")
        self.image_plot.getPlotItem().hideAxis("bottom")

        self.graph_img_item = pg.ImageItem(axisOrder="row-major")
        self.image_plot.addItem(self.graph_img_item)
        self.image_plot.invertY(True)

        layout_image.addWidget(self.image_plot, stretch=1)

        image_widget = QWidget()
        image_widget.setLayout(layout_image)

        return image_widget

    def _create_table_widget(self):
        layout_table = QVBoxLayout()

        self.spot_table = cw.TableWidget(headers=['X', 'Y', 'Z', 'Intensity'], n_rows=2)

        layout_table.addWidget(cw.LabelWidget("Spot Coordinates"))
        layout_table.addWidget(self.spot_table)

        table_widget = QWidget()
        table_widget.setLayout(layout_table)

        return table_widget

    def set_graph_image(self, img: np.ndarray, levels=None):
        old_shape = None if self._target_img is None else self._target_img.shape
        self._target_img = img
        self.graph_img_item.setImage(img, autoLevels=(levels is None))

        if old_shape != img.shape:
            self.image_plot.getViewBox().autoRange()

    def on_camera_update_from_thread(self, frame: np.ndarray):
        """Runs in camera thread. Do NOT touch Qt widgets here."""
        if frame is None:
            return
        else:
            self.set_graph_image(frame)

    def start_target_picking(self):
        self.logg.info("Start picking...")
        self._picking_enabled = True
        self._picking_enabled = True

        self.target_points = []
        self._update_target_spots_overlay()

        self.image_plot.setFocus()
        self.image_plot.setCursor(Qt.CursorShape.CrossCursor)

        self._update_pick_status()

    def finish_target_picking(self):
        self._picking_enabled = False
        self.image_plot.unsetCursor()
        self._update_pick_status(done=True)

    def cancel_target_picking(self):
        self._picking_enabled = False
        self.image_plot.unsetCursor()
        self._update_pick_status(cancelled=True)

    def get_target_spots(self):
        return self.spot_table.get_row(-1)

    def _update_pick_status(self, done=False, cancelled=False):
        n = len(getattr(self, "target_points", []))
        if cancelled:
            title = "Picking cancelled"
        elif done:
            title = f"Picked {n} spot(s)"
        elif self._picking_enabled:
            title = f"Picking: {n} spot(s)  |  Left-click add, Backspace undo, Enter finish, Esc cancel"
        else:
            title = f"Spots: {n}"
        self.image_plot.getPlotItem().setTitle(title)

    def _on_target_mouse_clicked(self, ev):
        if not self._picking_enabled or self._target_img is None:
            return

        # right-click = finish (optional)
        if ev.button() == Qt.MouseButton.RightButton:
            self.finish_target_picking()
            return

        if ev.button() != Qt.MouseButton.LeftButton:
            return

        vb = self.image_plot.getPlotItem().vb
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return

        p = vb.mapSceneToView(ev.scenePos())
        x, y = float(p.x()), float(p.y())

        h, w = self._target_img.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return

        u = int(np.clip(np.floor(x), 0, w - 1))
        v = int(np.clip(np.floor(y), 0, h - 1))
        self.target_points.append((u, v))
        self.spot_table.fill_row(len(self.target_points) - 1, [u, v, 0.0, 1.0])
        self._update_target_spots_overlay()
        self._update_pick_status()

    def _update_target_spots_overlay(self):
        if not self.target_points:
            self.target_spots_item.setData([], [])
            return
        xs = [p[0] for p in self.target_points]
        ys = [p[1] for p in self.target_points]
        self.target_spots_item.setData(xs, ys)

    def eventFilter(self, obj, event):
        if obj is self.image_plot and self._picking_enabled:
            if event.type() == QEvent.Type.KeyPress:
                key = event.key()

                if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                    self.finish_target_picking()
                    return True

                if key == Qt.Key.Key_Backspace:
                    if self.target_points:
                        self.target_points.pop()
                        self._update_target_spots_overlay()
                    return True

                if key == Qt.Key.Key_Escape:
                    self.cancel_target_picking()
                    return True

        return super().eventFilter(obj, event)
