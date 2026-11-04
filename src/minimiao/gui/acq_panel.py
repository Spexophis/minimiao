# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.

"""Acquisition block panel: one block, a loop over SLM slots, a loop over laser power."""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout

from minimiao import logger

try:
    from . import custom_widgets as cw
except ImportError:
    from minimiao.gui import custom_widgets as cw


class AcqPanel(QWidget):
    Signal_run_block = pyqtSignal(dict)
    Signal_run_slot_loop = pyqtSignal(dict, str, float)
    Signal_run_power_loop = pyqtSignal(dict, float, float, int, float)

    CAMERA_TRIGGERS = {"Internal": "internal", "TriggerBox": "external"}

    def __init__(self, logg=None, parent=None):
        super().__init__(parent)
        self.logg = logg or logger.setup_logging()
        layout = QVBoxLayout(self)
        layout.addWidget(self._create_block_group())
        layout.addWidget(self._create_loop_group())
        self.setLayout(layout)
        self._connect()

    def _create_block_group(self):
        group = cw.GroupWidget()
        area, g = cw.create_scroll_area("G")

        self.QSpinBox_slot = cw.SpinBoxWidget(0, 1024, 1, 1)
        self.QDoubleSpinBox_power = cw.DoubleSpinBoxWidget(0, 200, 0.1, 2, 1.0)
        self.QSpinBox_pulses = cw.SpinBoxWidget(1, 10000, 1, 5)
        self.QDoubleSpinBox_width = cw.DoubleSpinBoxWidget(0.1, 10000, 0.5, 1, 10.0)
        self.QDoubleSpinBox_period = cw.DoubleSpinBoxWidget(1, 100000, 1, 1, 100.0)
        self.QSpinBox_laser_ch = cw.SpinBoxWidget(1, 5, 1, 1)
        self.QDoubleSpinBox_pre = cw.DoubleSpinBoxWidget(0, 600, 0.1, 2, 1.0)
        self.QDoubleSpinBox_post = cw.DoubleSpinBoxWidget(0, 600, 0.1, 2, 1.0)
        self.QSpinBox_settle = cw.SpinBoxWidget(0, 10000, 10, 100)
        self.QComboBox_cam_trigger = cw.ComboBoxWidget(list(self.CAMERA_TRIGGERS))
        self.QSpinBox_fps = cw.SpinBoxWidget(1, 5000, 1, 100)
        self.QSpinBox_cam_ch = cw.SpinBoxWidget(1, 5, 1, 2)
        self.QLineEdit_name = cw.LineEditWidget()
        self.QPushButton_run_block = cw.PushButtonWidget("Run Block")

        rows = [("Acquisition Block", None, None, None),
                ("SLM slot", self.QSpinBox_slot, "Laser power / mW", self.QDoubleSpinBox_power),
                ("Pulses", self.QSpinBox_pulses, "Laser TB channel", self.QSpinBox_laser_ch),
                ("Pulse width / ms", self.QDoubleSpinBox_width, "Pulse period / ms", self.QDoubleSpinBox_period),
                ("Pre delay / s", self.QDoubleSpinBox_pre, "Post delay / s", self.QDoubleSpinBox_post),
                ("SLM settle / ms", self.QSpinBox_settle, "Name", self.QLineEdit_name),
                ("Camera trigger", self.QComboBox_cam_trigger, "Camera TB channel", self.QSpinBox_cam_ch),
                ("Frame rate / Hz", self.QSpinBox_fps, None, self.QPushButton_run_block)]
        for r, (l1, w1, l2, w2) in enumerate(rows):
            g.addWidget(cw.LabelWidget(l1), r, 0)
            if w1 is not None:
                g.addWidget(w1, r, 1)
            if l2 is not None:
                g.addWidget(cw.LabelWidget(l2), r, 2)
            if w2 is not None:
                g.addWidget(w2, r, 3)
        g.addWidget(cw.LabelWidget("Exposure / ROI / binning: camera settings in the Ctrl Panel. "
                                   "Frame rate applies to TriggerBox camera trigger only. Esc aborts."),
                    len(rows), 0, 1, 4)
        hl = QHBoxLayout(group)
        hl.addWidget(area)
        return group

    def _create_loop_group(self):
        group = cw.GroupWidget()
        area, g = cw.create_scroll_area("G")
        self.QLineEdit_slots = cw.LineEditWidget()
        self.QLineEdit_slots.setText("1-5")
        self.QPushButton_slot_loop = cw.PushButtonWidget("Run Slot Loop")
        self.QDoubleSpinBox_p_start = cw.DoubleSpinBoxWidget(0, 200, 0.1, 2, 0.5)
        self.QDoubleSpinBox_p_step = cw.DoubleSpinBoxWidget(-200, 200, 0.1, 2, 0.5)
        self.QSpinBox_p_n = cw.SpinBoxWidget(1, 1000, 1, 5)
        self.QPushButton_power_loop = cw.PushButtonWidget("Run Power Loop")
        self.QDoubleSpinBox_interval = cw.DoubleSpinBoxWidget(0, 3600, 0.5, 1, 2.0)

        g.addWidget(cw.LabelWidget("Loops (use block settings above)"), 0, 0, 1, 4)
        g.addWidget(cw.LabelWidget("Slots (e.g. 1-5, 8)"), 1, 0)
        g.addWidget(self.QLineEdit_slots, 1, 1, 1, 2)
        g.addWidget(self.QPushButton_slot_loop, 1, 3)
        g.addWidget(cw.LabelWidget("Power start / mW"), 2, 0)
        g.addWidget(self.QDoubleSpinBox_p_start, 2, 1)
        g.addWidget(cw.LabelWidget("Step / mW"), 2, 2)
        g.addWidget(self.QDoubleSpinBox_p_step, 2, 3)
        g.addWidget(cw.LabelWidget("Steps"), 3, 0)
        g.addWidget(self.QSpinBox_p_n, 3, 1)
        g.addWidget(self.QPushButton_power_loop, 3, 3)
        g.addWidget(cw.LabelWidget("Interval between blocks / s"), 4, 0, 1, 2)
        g.addWidget(self.QDoubleSpinBox_interval, 4, 2)
        hl = QHBoxLayout(group)
        hl.addWidget(area)
        return group

    def _connect(self):
        self.QPushButton_run_block.clicked.connect(lambda: self.Signal_run_block.emit(self.get_block_settings()))
        self.QPushButton_slot_loop.clicked.connect(
            lambda: self.Signal_run_slot_loop.emit(self.get_block_settings(), self.QLineEdit_slots.text(),
                                                   self.QDoubleSpinBox_interval.value()))
        self.QPushButton_power_loop.clicked.connect(
            lambda: self.Signal_run_power_loop.emit(self.get_block_settings(), self.QDoubleSpinBox_p_start.value(),
                                                    self.QDoubleSpinBox_p_step.value(), self.QSpinBox_p_n.value(),
                                                    self.QDoubleSpinBox_interval.value()))

    def get_block_settings(self) -> dict:
        """Keys match experiments.acquisition.BlockSettings."""
        return {
            "slot": self.QSpinBox_slot.value(),
            "laser": "473",
            "laser_power_mw": self.QDoubleSpinBox_power.value(),
            "n_pulses": self.QSpinBox_pulses.value(),
            "pulse_width_ms": self.QDoubleSpinBox_width.value(),
            "pulse_period_ms": self.QDoubleSpinBox_period.value(),
            "laser_channel": self.QSpinBox_laser_ch.value(),
            "pre_delay_s": self.QDoubleSpinBox_pre.value(),
            "post_delay_s": self.QDoubleSpinBox_post.value(),
            "slm_settle_ms": float(self.QSpinBox_settle.value()),
            "camera_trigger": self.CAMERA_TRIGGERS[self.QComboBox_cam_trigger.currentText()],
            "frame_rate_hz": self.QSpinBox_fps.value(),
            "camera_channel": self.QSpinBox_cam_ch.value(),
            "name": self.QLineEdit_name.text().strip(),
        }
