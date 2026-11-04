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
    Signal_display_slot = pyqtSignal(int)
    Signal_calib_generate = pyqtSignal(int, float)
    Signal_calib_measure = pyqtSignal(int, int)
    Signal_calib_load = pyqtSignal()
    Signal_calib_use = pyqtSignal(bool)
    Signal_calib_auto = pyqtSignal(dict)

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

        calib_widget = self._create_calibration_panel()
        splitter.addWidget(calib_widget)

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
        self.QDoubleSpinBox_CGH_Magnification = cw.DoubleSpinBoxWidget(0, 50, 1, 3, 0.332)
        self.QSpinBox_CGH_CenterX = cw.SpinBoxWidget(-4096, 8192, 1, 1201)
        self.QSpinBox_CGH_CenterY = cw.SpinBoxWidget(-4096, 8192, 1, 1208)
        self.QDoubleSpinBox_SLM_Focal = cw.DoubleSpinBoxWidget(0, 2000, 1, 2, 455)
        self.QPushButton_CGH_Compute = cw.PushButtonWidget('Compute CGH')
        self.QPushButton_CGH_Save = cw.PushButtonWidget('Save CGH')
        self.QComboBox_CGH_Mode = cw.ComboBoxWidget(list_items=["Sequential", "Simultaneous", "Hybrid"])
        self.QComboBox_CGH_Mode.setCurrentText("Simultaneous")
        self.QPushButton_SLM_Display = cw.PushButtonWidget('Display Slot')
        self.QSpinBox_SLM_Display_Slot = cw.SpinBoxWidget(0, 1024, 1, 0)

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
        cgh_scroll_layout.addWidget(cw.LabelWidget(str('Mode')), 7, 1, 1, 1)
        cgh_scroll_layout.addWidget(self.QComboBox_CGH_Mode, 7, 2, 1, 1)
        cgh_scroll_layout.addWidget(self.QPushButton_SLM_Display, 8, 0, 1, 1)
        cgh_scroll_layout.addWidget(cw.LabelWidget(str('Slot')), 8, 1, 1, 1)
        cgh_scroll_layout.addWidget(self.QSpinBox_SLM_Display_Slot, 8, 2, 1, 1)

        group_layout = QHBoxLayout(group)
        group_layout.addWidget(cgh_scroll_area)
        group.setLayout(group_layout)
        return group

    def _create_calibration_panel(self):
        group = cw.GroupWidget()
        calib_scroll_area, calib_scroll_layout = cw.create_scroll_area("G")

        self.QSpinBox_Calib_Grid = cw.SpinBoxWidget(2, 10, 1, 4)
        self.QDoubleSpinBox_Calib_Span = cw.DoubleSpinBoxWidget(0.05, 0.95, 0.05, 2, 0.5)
        self.QSpinBox_Calib_Settle = cw.SpinBoxWidget(0, 5000, 10, 300)
        self.QSpinBox_Calib_Average = cw.SpinBoxWidget(1, 20, 1, 3)
        self.QPushButton_Calib_Generate = cw.PushButtonWidget('Calib Holograms')
        self.QPushButton_Calib_Measure = cw.PushButtonWidget('Calib Measure')
        self.QPushButton_Calib_Load = cw.PushButtonWidget('Load Calib')
        self.QCheckBox_Calib_Use = cw.CheckBoxWidget('Use Calib')
        self.QSpinBox_Calib_Refine = cw.SpinBoxWidget(0, 10, 1, 5)
        self.QPushButton_Calib_Auto = cw.PushButtonWidget('Auto Calibrate')
        self.QLabel_Calib_Status = cw.LabelWidget('Calibration: none')

        calib_scroll_layout.addWidget(cw.LabelWidget(str('Camera-SLM Affine Calibration')), 0, 0, 1, 3)
        calib_scroll_layout.addWidget(cw.FrameWidget(), 1, 0, 1, 3)
        calib_scroll_layout.addWidget(self.QPushButton_Calib_Generate, 2, 0, 1, 1)
        calib_scroll_layout.addWidget(self.QPushButton_Calib_Measure, 3, 0, 1, 1)
        calib_scroll_layout.addWidget(self.QPushButton_Calib_Load, 4, 0, 1, 1)
        calib_scroll_layout.addWidget(self.QCheckBox_Calib_Use, 5, 0, 1, 1)
        calib_scroll_layout.addWidget(cw.LabelWidget(str('Grid N x N')), 2, 1, 1, 1)
        calib_scroll_layout.addWidget(self.QSpinBox_Calib_Grid, 2, 2, 1, 1)
        calib_scroll_layout.addWidget(cw.LabelWidget(str('Span (field)')), 3, 1, 1, 1)
        calib_scroll_layout.addWidget(self.QDoubleSpinBox_Calib_Span, 3, 2, 1, 1)
        calib_scroll_layout.addWidget(cw.LabelWidget(str('Settle / ms')), 4, 1, 1, 1)
        calib_scroll_layout.addWidget(self.QSpinBox_Calib_Settle, 4, 2, 1, 1)
        calib_scroll_layout.addWidget(cw.LabelWidget(str('Avg frames')), 5, 1, 1, 1)
        calib_scroll_layout.addWidget(self.QSpinBox_Calib_Average, 5, 2, 1, 1)
        calib_scroll_layout.addWidget(self.QPushButton_Calib_Auto, 6, 0, 1, 1)
        calib_scroll_layout.addWidget(cw.LabelWidget(str('Refine grid')), 6, 1, 1, 1)
        calib_scroll_layout.addWidget(self.QSpinBox_Calib_Refine, 6, 2, 1, 1)
        calib_scroll_layout.addWidget(self.QLabel_Calib_Status, 7, 0, 1, 3)

        group_layout = QHBoxLayout(group)
        group_layout.addWidget(calib_scroll_area)
        group.setLayout(group_layout)
        return group

    def get_calibration_settings(self):
        return {"grid": self.QSpinBox_Calib_Grid.value(), "span": self.QDoubleSpinBox_Calib_Span.value(),
                "settle_ms": self.QSpinBox_Calib_Settle.value(), "n_avg": self.QSpinBox_Calib_Average.value(),
                "refine": self.QSpinBox_Calib_Refine.value()}

    def set_simple_mapping(self, magnification: float, center_x: float, center_y: float):
        self.QDoubleSpinBox_CGH_Magnification.setValue(magnification)
        self.QSpinBox_CGH_CenterX.setValue(int(round(center_x)))
        self.QSpinBox_CGH_CenterY.setValue(int(round(center_y)))

    def set_calibration_status(self, text: str, use: bool = None):
        self.QLabel_Calib_Status.setText(text)
        if use is not None:
            self.QCheckBox_Calib_Use.setChecked(use)

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
        self.QPushButton_SLM_Display.clicked.connect(self.display_slot)
        self.QPushButton_Calib_Generate.clicked.connect(
            lambda: self.Signal_calib_generate.emit(self.QSpinBox_Calib_Grid.value(),
                                                    self.QDoubleSpinBox_Calib_Span.value()))
        self.QPushButton_Calib_Measure.clicked.connect(
            lambda: self.Signal_calib_measure.emit(self.QSpinBox_Calib_Settle.value(),
                                                   self.QSpinBox_Calib_Average.value()))
        self.QPushButton_Calib_Load.clicked.connect(self.Signal_calib_load.emit)
        self.QPushButton_Calib_Auto.clicked.connect(
            lambda: self.Signal_calib_auto.emit(self.get_calibration_settings()))
        self.QCheckBox_Calib_Use.toggled.connect(self.Signal_calib_use.emit)

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

    @pyqtSlot()
    def display_slot(self):
        self.Signal_display_slot.emit(self.QSpinBox_SLM_Display_Slot.value())

    def get_cgh_mode(self):
        return self.QComboBox_CGH_Mode.currentText()

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
