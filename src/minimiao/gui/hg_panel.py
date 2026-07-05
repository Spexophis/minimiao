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
        self._set_signal_connections()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Vertical)

        control_widget = self._create_cgh_panel()
        splitter.addWidget(control_widget)

        plot_widget = QWidget()
        plot_layout = self._create_plot_widgets()
        plot_widget.setLayout(plot_layout)
        splitter.addWidget(plot_widget)

        layout.addWidget(splitter)
        self.setLayout(layout)

    def _create_cgh_panel(self):
        group = cw.GroupWidget()
        cgh_scroll_area, cgh_scroll_layout = cw.create_scroll_area("G")

        self.QPushButton_CGH_Load = cw.PushButtonWidget('Load Target')
        self.QPushButton_CGH_Pick = cw.PushButtonWidget('Pick Spots')
        self.QSpinBox_CGH_Iteration = cw.SpinBoxWidget(0, 1024, 1, 32)
        self.QDoubleSpinBox_CGH_Magnification = cw.DoubleSpinBoxWidget(0, 50, 1, 2, 0.55)
        self.QSpinBox_CGH_CenterX = cw.SpinBoxWidget(0, 2048, 1, 617)
        self.QSpinBox_CGH_CenterY = cw.SpinBoxWidget(0, 2048, 1, 445)
        self.QDoubleSpinBox_SLM_Focal = cw.DoubleSpinBoxWidget(0, 2000, 1, 2, 510)
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

        self.pattern_plot = pg.PlotWidget()
        self.pattern_plot.setAspectLocked(True)
        self.pattern_plot.getPlotItem().hideAxis("left")
        self.pattern_plot.getPlotItem().hideAxis("bottom")

        self.pattern_img_item = pg.ImageItem(axisOrder="row-major")  # numpy (H,W)
        self.pattern_plot.addItem(self.pattern_img_item)

        layout_plot.addWidget(self.pattern_plot, stretch=1)
        return layout_plot

    def _set_signal_connections(self):
        self.QPushButton_CGH_Load.clicked.connect(self.load_target)
        self.QPushButton_CGH_Pick.clicked.connect(self.pick_spots)
        self.QPushButton_CGH_Compute.clicked.connect(self.compute_cgh)
        self.QPushButton_CGH_Save.clicked.connect(self.save_pattern)

    @pyqtSlot()
    def load_target(self):
        self.Signal_load_target.emit()

    @pyqtSlot()
    def pick_spots(self):
        self.Signal_pick_spot.emit()

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
        return n, m, f, (c0, c1)

    def set_pattern_image(self, img2d: np.ndarray, levels=None):
        self.pattern_img_item.clear()
        self.pattern_img_item.setImage(img2d, autoLevels=(levels is None))
        if levels is not None:
            self.pattern_img_item.setLevels(levels)
        self.pattern_plot.update()

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
