# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QSplitter, QSpinBox, QDoubleSpinBox

try:
    from . import custom_widgets as cw
except:
    from minimiao.gui import custom_widgets as cw


class ControlPanel(QWidget):
    Signal_stage_move_usb = pyqtSignal(str, float, float, float)
    Signal_stage_move = pyqtSignal(str, float, float, float)
    Signal_slm_correction = pyqtSignal()
    Signal_slm_load = pyqtSignal()
    Signal_slm_set = pyqtSignal()
    Signal_slm_base = pyqtSignal(bool, bool)  # lens on, cross on -> rewrite slot 0
    Signal_slm_display = pyqtSignal(int)
    Signal_set_laser = pyqtSignal(list, bool, float)
    Signal_trigger_reset = pyqtSignal()
    Signal_trigger_fire = pyqtSignal(list)
    N_TTL = 8
    TTL_FIELDS = ("start", "pulse", "interval", "n")
    Signal_video = pyqtSignal(bool)
    Signal_data_acquire = pyqtSignal(bool, int)
    Signal_save_file = pyqtSignal(str)

    def __init__(self, logg=None, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.config = {"ConWidget Path": "", "Digital Timing Presets": ""}
        self.logg = logg
        self._setup_ui()
        self._set_signal_connections()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Vertical)

        self.camera_slm_panel = self._create_camera_slm_panel()
        self.stage_trigger_panel = self._create_stage_trigger_panel()
        self.laser_panel = self._create_laser_panel()

        splitter.addWidget(self.camera_slm_panel)
        splitter.addWidget(self.stage_trigger_panel)
        splitter.addWidget(self.laser_panel)

        layout.addWidget(splitter)
        self.setLayout(layout)

    def _create_camera_slm_panel(self):
        group = cw.GroupWidget()
        scmos_scroll_area, scmos_scroll_layout = cw.create_scroll_area()
        slm_scroll_area, slm_scroll_layout = cw.create_scroll_area()

        self.QSpinBox_scmos_coordinate_x = cw.SpinBoxWidget(0, 2400, 1, 0)
        self.QSpinBox_scmos_coordinate_y = cw.SpinBoxWidget(0, 2400, 1, 0)
        self.QSpinBox_scmos_coordinate_nx = cw.SpinBoxWidget(0, 2400, 1, 2400)
        self.QSpinBox_scmos_coordinate_ny = cw.SpinBoxWidget(0, 2400, 1, 2400)
        self.QSpinBox_scmos_coordinate_bin = cw.SpinBoxWidget(0, 512, 1, 1)
        # ms with 0.001 ms (1 us) steps; applied with 1 us resolution when the camera supports it
        self.QDoubleSpinBox_scmos_t_exposure = cw.DoubleSpinBoxWidget(0.001, 10000, 0.1, 3, 20)
        self.QComboBox_scmos_readout = cw.ComboBoxWidget(list_items=[], maxlen=400)
        self.set_readout_options([])
        self.QLCDNumber_scmos_frame_rate = cw.LCDNumberWidget(0, 6)
        self.QPushButton_video = cw.PushButtonWidget("Video", checkable=True)
        # 0 = record until Acquire is clicked off; N > 0 = exactly N frames (camera sequence mode)
        self.QSpinBox_acquisition_number = cw.SpinBoxWidget(0, 10000000, 100, 0)
        self.QPushButton_acquire = cw.PushButtonWidget('Acquire', checkable=True)

        scmos_scroll_layout.addRow(cw.LabelWidget(str('Kinetix Camera')))
        scmos_scroll_layout.addRow(cw.FrameWidget())
        scmos_scroll_layout.addRow(cw.LabelWidget(str('X')), self.QSpinBox_scmos_coordinate_x)
        scmos_scroll_layout.addRow(cw.LabelWidget(str('Y')), self.QSpinBox_scmos_coordinate_y)
        scmos_scroll_layout.addRow(cw.LabelWidget(str('Nx')), self.QSpinBox_scmos_coordinate_nx)
        scmos_scroll_layout.addRow(cw.LabelWidget(str('Ny')), self.QSpinBox_scmos_coordinate_ny)
        scmos_scroll_layout.addRow(cw.LabelWidget(str('Bin')), self.QSpinBox_scmos_coordinate_bin)
        scmos_scroll_layout.addRow(cw.LabelWidget(str('Exposure / ms')), self.QDoubleSpinBox_scmos_t_exposure)
        scmos_scroll_layout.addRow(cw.LabelWidget(str('Readout')), self.QComboBox_scmos_readout)
        scmos_scroll_layout.addRow(cw.LabelWidget(str('FPS')), self.QLCDNumber_scmos_frame_rate)
        scmos_scroll_layout.addRow(self.QPushButton_video)
        scmos_scroll_layout.addRow(cw.LabelWidget(str('Frames (0 = until stop)')), self.QSpinBox_acquisition_number)
        scmos_scroll_layout.addRow(self.QPushButton_acquire)

        self.QPushButton_SLM_Correction = cw.PushButtonWidget('Load Correction')
        self.QPushButton_SLM_Load = cw.PushButtonWidget('Load Pattern')
        self.QLineEdit_SLM_Pattern = cw.LineEditWidget(True)
        self.QSpinBox_SLM_Slot = cw.SpinBoxWidget(0, 1024, 1, 1)
        self.QPushButton_SLM_Display = cw.PushButtonWidget('Display Pattern')
        self.QSpinBox_SLM_OffsetX = cw.SpinBoxWidget(-512, 512, 1, -100)
        self.QSpinBox_SLM_OffsetY = cw.SpinBoxWidget(-512, 512, 1, -80)
        self.QDoubleSpinBox_SLM_Focal = cw.DoubleSpinBoxWidget(0, 2000, 1, 2, 455)
        self.QPushButton_SLM_Lens = cw.PushButtonWidget('Add Lens', checkable=True)
        self.QPushButton_SLM_Cross = cw.PushButtonWidget('Add Cross', checkable=True)
        self.QLabel_SLM_Lens = cw.LabelWidget('Lens D: -')

        slm_scroll_layout.addRow(cw.LabelWidget(str('Hamamatsu SLM')))
        slm_scroll_layout.addRow(cw.FrameWidget())
        slm_scroll_layout.addRow(self.QPushButton_SLM_Correction)
        slm_scroll_layout.addRow(self.QPushButton_SLM_Load, self.QPushButton_SLM_Display)
        slm_scroll_layout.addRow(cw.LabelWidget(str('Slot')), self.QSpinBox_SLM_Slot)
        slm_scroll_layout.addRow(self.QLineEdit_SLM_Pattern)
        slm_scroll_layout.addRow(cw.LabelWidget(str('Offset X')), self.QSpinBox_SLM_OffsetX)
        slm_scroll_layout.addRow(cw.LabelWidget(str('Offset Y')), self.QSpinBox_SLM_OffsetY)
        slm_scroll_layout.addRow(cw.LabelWidget(str('Fresnel f / mm')), self.QDoubleSpinBox_SLM_Focal)
        slm_scroll_layout.addRow(self.QPushButton_SLM_Lens, self.QPushButton_SLM_Cross)
        slm_scroll_layout.addRow(self.QLabel_SLM_Lens)

        group_layout = QHBoxLayout(group)
        group_layout.addWidget(scmos_scroll_area)
        group_layout.addWidget(slm_scroll_area)
        group.setLayout(group_layout)
        return group

    def _create_stage_trigger_panel(self):
        group = cw.GroupWidget()
        trigger_scroll_area, trigger_scroll_layout = cw.create_scroll_area("G")

        self.QPushButton_reset_trigger = cw.PushButtonWidget("Stop / Reset")
        self.QPushButton_fire_trigger = cw.PushButtonWidget("Fire")
        self.QLabel_trigger_status = cw.LabelWidget("Not fired")
        trigger_scroll_layout.addWidget(cw.LabelWidget(str('Trigger Box')), 0, 0)
        trigger_scroll_layout.addWidget(self.QPushButton_fire_trigger, 0, 1, 1, 2)
        trigger_scroll_layout.addWidget(self.QPushButton_reset_trigger, 0, 3, 1, 2)
        trigger_scroll_layout.addWidget(cw.FrameWidget(), 1, 0, 1, 5)
        for col, text in enumerate(('', 'Start / ms', 'Pulse / ms', 'Interval / ms', 'N')):
            if text:
                trigger_scroll_layout.addWidget(cw.LabelWidget(text), 2, col, 1, 1)
        for n in range(1, self.N_TTL + 1):
            start = cw.DoubleSpinBoxWidget(0, 600000, 1, 0, 0)
            pulse = cw.DoubleSpinBoxWidget(0.01, 600000, 1, 2, 10.0)
            interval = cw.DoubleSpinBoxWidget(0.01, 600000, 10, 2, 100.0)
            count = cw.SpinBoxWidget(0, 100000, 1, 5 if n == 1 else 0)
            setattr(self, f"QDoubleSpinBox_ttl_start_{n}", start)
            setattr(self, f"QDoubleSpinBox_ttl_pulse_{n}", pulse)
            setattr(self, f"QDoubleSpinBox_ttl_interval_{n}", interval)
            setattr(self, f"QSpinBox_ttl_n_{n}", count)
            row = 2 + n
            trigger_scroll_layout.addWidget(cw.LabelWidget(f'DO#{n}'), row, 0, 1, 1)
            trigger_scroll_layout.addWidget(start, row, 1, 1, 1)
            trigger_scroll_layout.addWidget(pulse, row, 2, 1, 1)
            trigger_scroll_layout.addWidget(interval, row, 3, 1, 1)
            trigger_scroll_layout.addWidget(count, row, 4, 1, 1)
        trigger_scroll_layout.addWidget(self.QLabel_trigger_status, 3 + self.N_TTL, 0, 1, 5)

        group_layout = QHBoxLayout(group)
        group_layout.addWidget(trigger_scroll_area)
        group.setLayout(group_layout)
        return group

    def _create_laser_panel(self):
        group = cw.GroupWidget()
        laser_473_scroll_area, laser_473_scroll_layout = cw.create_scroll_area()

        self.QDoubleSpinBox_laserpower_473 = cw.DoubleSpinBoxWidget(0, 200, 0.1, 1, 0.0)
        self.QPushButton_laser_473 = cw.PushButtonWidget('ON', checkable=True)

        laser_473_scroll_layout.addRow(cw.LabelWidget(str('Laser 473')), self.QDoubleSpinBox_laserpower_473)
        laser_473_scroll_layout.addRow(self.QPushButton_laser_473)

        group_layout = QHBoxLayout(group)
        group_layout.addWidget(laser_473_scroll_area)
        group.setLayout(group_layout)
        return group

    def _set_signal_connections(self):
        self.QPushButton_SLM_Correction.clicked.connect(self.load_slm_correction)
        self.QPushButton_SLM_Load.clicked.connect(self.load_slm_pattern)
        self.QPushButton_SLM_Lens.clicked.connect(self.emit_slm_base)
        self.QPushButton_SLM_Cross.clicked.connect(self.emit_slm_base)
        self.QDoubleSpinBox_SLM_Focal.editingFinished.connect(
            lambda: self.emit_slm_base() if self.QPushButton_SLM_Lens.isChecked() else None)
        self.QPushButton_SLM_Display.clicked.connect(
            lambda: self.Signal_slm_display.emit(self.QSpinBox_SLM_Slot.value()))
        self.QPushButton_laser_473.clicked.connect(self.set_laser_473)
        self.QDoubleSpinBox_laserpower_473.valueChanged.connect(self.pw_laser_473)
        self.QPushButton_reset_trigger.clicked.connect(self.reset_trigger)
        self.QPushButton_fire_trigger.clicked.connect(lambda: self.Signal_trigger_fire.emit(self.get_digital_parameters()))
        self.QPushButton_video.clicked.connect(self.run_video)
        self.QPushButton_acquire.clicked.connect(self.run_acquisition)

    def get_scmos_roi(self):
        return [self.QSpinBox_scmos_coordinate_x.value(), self.QSpinBox_scmos_coordinate_y.value(),
                self.QSpinBox_scmos_coordinate_nx.value(), self.QSpinBox_scmos_coordinate_ny.value(),
                self.QSpinBox_scmos_coordinate_bin.value()]

    def get_scmos_exposure(self):
        return self.QDoubleSpinBox_scmos_t_exposure.value()

    DEFAULT_READOUT = (1, 0, 1)  # port, speed, gain: what live/recording used before the selector existed

    def set_readout_options(self, options):
        """Fill the readout selector from the camera's port/speed/gain table
        (list of dicts from KinetixCamera.get_readout_options); keeps a default entry if empty."""
        combo = self.QComboBox_scmos_readout
        combo.clear()
        if not options:
            p, sp, g = self.DEFAULT_READOUT
            combo.addItem(f"Port {p} / Speed {sp} / Gain {g} (default)", (p, sp, g))
            return
        default_idx = 0
        for i, o in enumerate(options):
            bits = f", {o['bit_depth']}-bit" if o.get("bit_depth") else ""
            pix = f", {o['pixel_time_ns']} ns/px" if o.get("pixel_time_ns") else ""
            gain = f", gain {o.get('gain_name') or o['gain']}"
            combo.addItem(f"{o.get('port_name', 'Port ' + str(o['port']))} / speed {o['speed']}{bits}{pix}{gain}",
                          (o["port"], o["speed"], o["gain"]))
            if (o["port"], o["speed"]) == self.DEFAULT_READOUT[:2]:
                default_idx = i
        combo.setCurrentIndex(default_idx)

    def get_scmos_readout(self):
        """(port, speed, gain) indices of the selected readout mode."""
        data = self.QComboBox_scmos_readout.currentData()
        return tuple(data) if data else self.DEFAULT_READOUT

    def display_frame_rate(self, fps: float):
        """Measured camera frame rate (whole numbers from 100 fps, one decimal below)."""
        self.QLCDNumber_scmos_frame_rate.display(round(fps) if fps >= 100 else round(fps, 1))

    def display_scmos_timings(self, exposure_time=None, kinetic_time=None):
        if exposure_time is not None:
            self.QDoubleSpinBox_scmos_t_exposure.setValue(exposure_time)
        if kinetic_time is not None:
            fps = 1.0 / kinetic_time
            self.QLCDNumber_scmos_frame_rate.display(fps)

    @pyqtSlot()
    def load_slm_correction(self):
        self.Signal_slm_correction.emit()

    @pyqtSlot()
    def load_slm_pattern(self):
        self.Signal_slm_load.emit()

    def display_loaded_pattern(self, filename=None):
        if filename is not None:
            self.QLineEdit_SLM_Pattern.setText(str(filename))

    def emit_slm_base(self):
        self.Signal_slm_base.emit(self.QPushButton_SLM_Lens.isChecked(), self.QPushButton_SLM_Cross.isChecked())

    def set_slm_base_buttons(self, lens: bool, cross: bool):
        """Reflect the state actually applied (e.g. after an error) without re-emitting."""
        for btn, on in ((self.QPushButton_SLM_Lens, lens), (self.QPushButton_SLM_Cross, cross)):
            btn.blockSignals(True)
            btn.setChecked(on)
            btn.blockSignals(False)

    def display_lens_aperture(self, diameter_px: float, diameter_mm: float):
        self.QLabel_SLM_Lens.setText(f"Lens D: {diameter_px:.0f} px ({diameter_mm:.2f} mm)"
                                     if diameter_px > 0 else "Lens D: - (no lens)")

    def get_slm_focal_mm(self):
        """Fresnel lens focal length for slot 0 (mm; 0 = no lens)."""
        return self.QDoubleSpinBox_SLM_Focal.value()

    def get_slm_parameters(self):
        ox = self.QSpinBox_SLM_OffsetX.value()
        oy = self.QSpinBox_SLM_OffsetY.value()
        n = self.QSpinBox_SLM_Slot.value()
        return (ox, oy), n

    @pyqtSlot(bool)
    def set_laser_473(self, checked: bool):
        power = self.QDoubleSpinBox_laserpower_473.value()
        self.Signal_set_laser.emit(["473"], checked, power)

    @pyqtSlot(float)
    def pw_laser_473(self, power: float):
        checked = self.QPushButton_laser_473.isChecked()
        power = self.QDoubleSpinBox_laserpower_473.value()
        self.Signal_set_laser.emit(["473"], checked, power)

    @pyqtSlot()
    def reset_trigger(self):
        self.Signal_trigger_reset.emit()

    def _ttl_widget(self, field, n):
        prefix = "QSpinBox" if field == "n" else "QDoubleSpinBox"
        return getattr(self, f"{prefix}_ttl_{field}_{n}")

    def get_digital_parameters(self):
        """Per-output pulse settings for TriggerBox DO#1..8 (all rows; n_pulses = 0 means off)."""
        return [{"channel": n,
                 "start_ms": self._ttl_widget("start", n).value(),
                 "width_ms": self._ttl_widget("pulse", n).value(),
                 "interval_ms": self._ttl_widget("interval", n).value(),
                 "n_pulses": self._ttl_widget("n", n).value()} for n in range(1, self.N_TTL + 1)]

    def display_trigger_status(self, text: str):
        self.QLabel_trigger_status.setText(text)

    @pyqtSlot()
    def run_video(self):
        if self.QPushButton_video.isChecked():
            self.Signal_video.emit(True)
        else:
            self.Signal_video.emit(False)

    @pyqtSlot()
    def run_acquisition(self):
        if self.QPushButton_acquire.isChecked():
            acq_num = self.QSpinBox_acquisition_number.value()
            self.Signal_data_acquire.emit(True, acq_num)
        else:
            self.Signal_data_acquire.emit(False, 0)


if __name__ == '__main__':
    import sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    win = ControlPanel()
    win.show()
    sys.exit(app.exec())
