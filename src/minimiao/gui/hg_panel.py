# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.

import json

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QEvent, pyqtSlot, pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSplitter, QHBoxLayout, QSpinBox, QDoubleSpinBox

from minimiao import logger

try:
    from . import custom_widgets as cw
except ImportError:
    from minimiao.gui import custom_widgets as cw


class HGPanel(QWidget):
    Signal_load_target = pyqtSignal()
    Signal_pick_spot = pyqtSignal()
    Signal_compute_cgh = pyqtSignal()
    Signal_save_pattern = pyqtSignal()

    def __init__(self, logg=None, parent=None):
        super().__init__(parent)
        self.config = {"HGWidget Path": ""}
        self.logg = logg or logger.setup_logging()
        self._setup_ui()
        self._target_img = None
        self._picking_enabled = False
        self._picking_n = None
        self.target_points = []

        self.target_plot.scene().sigMouseClicked.connect(self._on_target_mouse_clicked)
        self.target_plot.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.target_plot.installEventFilter(self)

        self.target_spots_item = pg.ScatterPlotItem(size=10, pen=pg.mkPen(width=2))
        self.target_plot.addItem(self.target_spots_item)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Vertical)

        control_widget = self._create_cgh_panel()
        splitter.addWidget(control_widget)

        plot_widget = QWidget()
        plot_layout = self._create_plot_widgets()
        plot_widget.setLayout(plot_layout)
        splitter.addWidget(plot_widget)

        table_widget = QWidget()
        table_layout = self._create_table_widget()
        table_widget.setLayout(table_layout)
        splitter.addWidget(table_widget)

        layout.addWidget(splitter)
        self.setLayout(layout)

    def _create_cgh_panel(self):
        group = cw.GroupWidget()
        cgh_scroll_area, cgh_scroll_layout = cw.create_scroll_area("G")

        self.QPushButton_CGH_Load = cw.PushButtonWidget('Load Target')
        self.QPushButton_CGH_Pick = cw.PushButtonWidget('Pick Spots')
        self.QSpinBox_CGH_Iteration = cw.SpinBoxWidget(0, 1024, 1, 32)
        self.QDoubleSpinBox_CGH_Magnification = cw.DoubleSpinBoxWidget(0, 128, 1, 1, 7)
        self.QSpinBox_CGH_CenterX = cw.SpinBoxWidget(0, 2048, 1, 600)
        self.QSpinBox_CGH_CenterY = cw.SpinBoxWidget(0, 2048, 1, 395)
        self.QDoubleSpinBox_SLM_Focal = cw.DoubleSpinBoxWidget(0, 2000, 1, 2, 320)
        self.QPushButton_CGH_Compute = cw.PushButtonWidget('Compute CGH')
        self.QPushButton_CGH_Save = cw.PushButtonWidget('Save CGH')

        cgh_scroll_layout.addWidget(cw.LabelWidget(str('CGH Computation')), 0, 0, 1, 1)
        cgh_scroll_layout.addWidget(cw.FrameWidget(), 1, 0, 1, 3)
        cgh_scroll_layout.addWidget(self.QPushButton_CGH_Load, 2, 0, 1, 1)
        cgh_scroll_layout.addWidget(self.QPushButton_CGH_Pick, 3, 0, 1, 1)
        cgh_scroll_layout.addWidget(self.QPushButton_CGH_Compute, 4, 0, 1, 1)
        cgh_scroll_layout.addWidget(self.QPushButton_CGH_Save, 5, 0, 1, 1)
        cgh_scroll_layout.addWidget(cw.LabelWidget(str('Iterations')), 2, 1, 1, 1)
        cgh_scroll_layout.addWidget(self.QSpinBox_CGH_Iteration, 2, 2, 1, 1)
        cgh_scroll_layout.addWidget(cw.LabelWidget(str('Magnification')), 3, 1, 1, 1)
        cgh_scroll_layout.addWidget(self.QDoubleSpinBox_CGH_Magnification, 3, 2, 1, 1)
        cgh_scroll_layout.addWidget(cw.LabelWidget(str('Center-X')), 4, 1, 1, 1)
        cgh_scroll_layout.addWidget(self.QSpinBox_CGH_CenterX, 4, 2, 1, 1)
        cgh_scroll_layout.addWidget(cw.LabelWidget(str('Center-Y')), 5, 1, 1, 1)
        cgh_scroll_layout.addWidget(self.QSpinBox_CGH_CenterY, 5, 2, 1, 1)
        cgh_scroll_layout.addWidget(cw.LabelWidget(str('Focal Length')), 6, 1, 1, 1)
        cgh_scroll_layout.addWidget(self.QDoubleSpinBox_SLM_Focal, 6, 2, 1, 1)

        group_layout = QHBoxLayout(group)
        group_layout.addWidget(cgh_scroll_area)
        group.setLayout(group_layout)
        return group

    def _create_plot_widgets(self):
        layout_plot = QVBoxLayout()

        self.target_plot = pg.PlotWidget()
        self.target_plot.setAspectLocked(True)
        self.target_plot.getPlotItem().hideAxis("left")
        self.target_plot.getPlotItem().hideAxis("bottom")

        self.target_img_item = pg.ImageItem(axisOrder="row-major")  # numpy (H,W)
        self.target_plot.addItem(self.target_img_item)
        self.target_plot.invertY(True)

        self.pattern_plot = pg.PlotWidget()
        self.pattern_plot.setAspectLocked(True)
        self.pattern_plot.getPlotItem().hideAxis("left")
        self.pattern_plot.getPlotItem().hideAxis("bottom")

        self.pattern_img_item = pg.ImageItem(axisOrder="row-major")  # numpy (H,W)
        self.pattern_plot.addItem(self.pattern_img_item)
        self.pattern_plot.invertY(True)

        layout_plot.addWidget(self.target_plot, stretch=1)
        layout_plot.addWidget(self.pattern_plot, stretch=1)
        return layout_plot

    def _create_table_widget(self):
        layout_table = QVBoxLayout()

        self.spot_table = cw.TableWidget(headers=['X', 'Y', 'Z', 'Intensity'], n_rows=2)

        layout_table.addWidget(cw.LabelWidget("Spot Coordinates"))
        layout_table.addWidget(self.spot_table)
        return layout_table

    def _set_signal_connections(self):
        self.QPushButton_CGH_Load.clicked.connect(self.load_target)
        self.QPushButton_CGH_Pick.clicked.connect(self.pick_spot)
        self.QPushButton_CGH_Compute.clicked.connect(self.compute_cgh)
        self.QPushButton_CGH_Save.clicked.connect(self.save_pattern)

    @pyqtSlot()
    def load_target(self):
        self.Signal_load_target.emit()

    @pyqtSlot()
    def pick_spot(self):
        self.Signal_pick_spot.emit()
        self.logg.info("Start target picking")

    @pyqtSlot()
    def compute_cgh(self):
        self.Signal_compute_cgh.emit()

    @pyqtSlot()
    def save_pattern(self):
        self.Signal_save_pattern.emit()

    def get_cgh_parameters(self):
        n  = self.QSpinBox_CGH_Iteration.value()
        m = self.QDoubleSpinBox_CGH_Magnification.value()
        f = self.QDoubleSpinBox_SLM_Focal.value()
        c0 = self.QSpinBox_CGH_CenterX.value()
        c1 = self.QSpinBox_CGH_CenterY.value()
        return n, m, (c0, c1)

    def set_target_image(self, img2d: np.ndarray, levels=None):
        self._target_img = img2d
        self.target_img_item.setImage(img2d, autoLevels=(levels is None))
        if levels is not None:
            self.target_img_item.setLevels(levels)
        h, w = img2d.shape[:2]
        self.target_plot.setLimits(xMin=0, xMax=w, yMin=0, yMax=h)
        self.target_plot.setRange(xRange=(0, w), yRange=(0, h), padding=0)

    def set_pattern_image(self, img2d: np.ndarray, levels=None):
        self.pattern_img_item.clear()
        self.pattern_img_item.setImage(img2d, autoLevels=(levels is None))
        if levels is not None:
            self.pattern_img_item.setLevels(levels)
        self.pattern_plot.update()

    def start_target_picking(self):
        self.logg.info("Start picking...")
        self._picking_enabled = True
        self._picking_enabled = True

        self.target_points = []
        self._update_target_spots_overlay()

        self.target_plot.setFocus()
        self.target_plot.setCursor(Qt.CursorShape.CrossCursor)

        self._update_pick_status()

    def finish_target_picking(self):
        self._picking_enabled = False
        self.target_plot.unsetCursor()
        self._update_pick_status(done=True)

    def cancel_target_picking(self):
        self._picking_enabled = False
        self.target_plot.unsetCursor()
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

        self.target_plot.getPlotItem().setTitle(title)

    def _on_target_mouse_clicked(self, ev):
        if not self._picking_enabled or self._target_img is None:
            return

        # right-click = finish (optional)
        if ev.button() == Qt.MouseButton.RightButton:
            self.finish_target_picking()
            return

        if ev.button() != Qt.MouseButton.LeftButton:
            return

        vb = self.target_plot.getPlotItem().vb
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return

        p = vb.mapSceneToView(ev.scenePos())
        x, y = float(p.x()), float(p.y())

        h, w = self._target_img.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return

        u = int(round(x))
        v = int(round(y))
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
        if obj is self.target_plot and self._picking_enabled:
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

    def save_spinbox_values(self):
        values = {}
        for name in dir(self):
            obj = getattr(self, name)
            if isinstance(obj, (QSpinBox, QDoubleSpinBox)):
                values[name] = obj.value()
        with open(self.config["ConWidget Path"], 'w') as f:
            json.dump(values, f, indent=4)

    def load_spinbox_values(self):
        try:
            with open(self.config["ConWidget Path"], 'r') as f:
                values = json.load(f)
            for name, value in values.items():
                widget = getattr(self, name, None)
                if widget is not None:
                    widget.setValue(value)
        except FileNotFoundError:
            pass


if __name__ == '__main__':
    import sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    win = HGPanel()
    win.show()
    sys.exit(app.exec())
