# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import json

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
    Signal_video = pyqtSignal(bool, str)
    Signal_data_acquire = pyqtSignal(bool, str, int)
    Signal_save_file = pyqtSignal(str)

    def __init__(self, logg=None, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.config = {"ConWidget Path": "", "Digital Timing Presets": ""}
        self.logg = logg
        self._setup_ui()
        # self.digital_timing_presets = self.load_digital_timing_presets()
        self._set_signal_connections()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Vertical)

        self.camera_slm_panel = self._create_camera_slm_panel()
        self.stage_trigger_panel = self._create_stage_trigger_panel()
        self.laser_panel = self._create_laser_panel()
        self.acq_panel = self._create_acquisition_panel()

        splitter.addWidget(self.camera_slm_panel)
        splitter.addWidget(self.stage_trigger_panel)
        splitter.addWidget(self.laser_panel)
        splitter.addWidget(self.acq_panel)

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
        self.QDoubleSpinBox_scmos_t_exposure = cw.DoubleSpinBoxWidget(0, 1000, 1, 3, 20)
        self.QLCDNumber_scmos_frame_rate = cw.LCDNumberWidget(25, 4)

        scmos_scroll_layout.addRow(cw.LabelWidget(str('sCMOS')))
        scmos_scroll_layout.addRow(cw.FrameWidget())
        scmos_scroll_layout.addRow(cw.LabelWidget(str('X')), self.QSpinBox_scmos_coordinate_x)
        scmos_scroll_layout.addRow(cw.LabelWidget(str('Y')), self.QSpinBox_scmos_coordinate_y)
        scmos_scroll_layout.addRow(cw.LabelWidget(str('Nx')), self.QSpinBox_scmos_coordinate_nx)
        scmos_scroll_layout.addRow(cw.LabelWidget(str('Ny')), self.QSpinBox_scmos_coordinate_ny)
        scmos_scroll_layout.addRow(cw.LabelWidget(str('Bin')), self.QSpinBox_scmos_coordinate_bin)
        scmos_scroll_layout.addRow(cw.LabelWidget(str('Exposure / ms')), self.QDoubleSpinBox_scmos_t_exposure)
        scmos_scroll_layout.addRow(cw.LabelWidget(str('FPS')), self.QLCDNumber_scmos_frame_rate)

        self.QPushButton_SLM_Correction = cw.PushButtonWidget('Load Correction')
        self.QPushButton_SLM_Load = cw.PushButtonWidget('Load Pattern')
        self.QLineEdit_SLM_Pattern = cw.LineEditWidget(True)
        # Load Pattern refuses slot 0 (correction); Display Pattern can show any slot including 0
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
        slm_scroll_layout.addRow(self.QPushButton_SLM_Load)
        slm_scroll_layout.addRow(self.QLineEdit_SLM_Pattern)
        slm_scroll_layout.addRow(cw.LabelWidget(str('Slot')), self.QSpinBox_SLM_Slot)
        slm_scroll_layout.addRow(self.QPushButton_SLM_Display)
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
        stage_scroll_area, stage_scroll_layout = cw.create_scroll_area("G")
        trigger_scroll_area, trigger_scroll_layout = cw.create_scroll_area("G")

        self.QDoubleSpinBox_stage_x_usb = cw.DoubleSpinBoxWidget(0, 100, 0.020, 3, 20.000)
        self.QLCDNumber_stage_position_x = cw.LCDNumberWidget()
        self.QDoubleSpinBox_stage_x = cw.DoubleSpinBoxWidget(0, 100, 0.020, 3, 30.000)
        self.QDoubleSpinBox_step_x = cw.DoubleSpinBoxWidget(0, 50, 0.001, 4, 0.030)
        self.QDoubleSpinBox_range_x = cw.DoubleSpinBoxWidget(0, 50, 0.001, 4, 0.780)
        self.QDoubleSpinBox_stage_y_usb = cw.DoubleSpinBoxWidget(0, 100, 0.020, 3, 20.000)
        self.QLCDNumber_stage_position_y = cw.LCDNumberWidget()
        self.QDoubleSpinBox_stage_y = cw.DoubleSpinBoxWidget(0, 100, 0.020, 3, 30.000)
        self.QDoubleSpinBox_step_y = cw.DoubleSpinBoxWidget(0, 50, 0.001, 4, 0.030)
        self.QDoubleSpinBox_range_y = cw.DoubleSpinBoxWidget(0, 50, 0.001, 4, 0.780)
        self.QDoubleSpinBox_stage_z_usb = cw.DoubleSpinBoxWidget(0, 100, 0.04, 2, 20.00)
        self.QLCDNumber_stage_position_z = cw.LCDNumberWidget()
        self.QDoubleSpinBox_stage_z = cw.DoubleSpinBoxWidget(0, 100, 0.04, 2, 30.00)
        self.QDoubleSpinBox_step_z = cw.DoubleSpinBoxWidget(0, 50, 0.001, 4, 0.160)
        self.QDoubleSpinBox_range_z = cw.DoubleSpinBoxWidget(0, 50, 0.001, 4, 4.80)

        stage_scroll_layout.addWidget(cw.LabelWidget(str('Stage')), 0, 0)
        stage_scroll_layout.addWidget(cw.FrameWidget(), 1, 0, 1, 3)
        stage_scroll_layout.addWidget(cw.LabelWidget(str('X (um)')), 2, 0)
        stage_scroll_layout.addWidget(self.QDoubleSpinBox_stage_x_usb, 2, 1)
        stage_scroll_layout.addWidget(self.QLCDNumber_stage_position_x, 2, 2)
        stage_scroll_layout.addWidget(cw.LabelWidget(str('Origin / um')), 3, 0)
        stage_scroll_layout.addWidget(cw.LabelWidget(str('Step / um')), 3, 1)
        stage_scroll_layout.addWidget(cw.LabelWidget(str('Range / um')), 3, 2)
        stage_scroll_layout.addWidget(self.QDoubleSpinBox_stage_x, 4, 0)
        stage_scroll_layout.addWidget(self.QDoubleSpinBox_step_x, 4, 1)
        stage_scroll_layout.addWidget(self.QDoubleSpinBox_range_x, 4, 2)
        stage_scroll_layout.addWidget(cw.FrameWidget(), 5, 0, 1, 3)
        stage_scroll_layout.addWidget(cw.LabelWidget(str('Y (um)')), 6, 0)
        stage_scroll_layout.addWidget(self.QDoubleSpinBox_stage_y_usb, 6, 1)
        stage_scroll_layout.addWidget(self.QLCDNumber_stage_position_y, 6, 2)
        stage_scroll_layout.addWidget(cw.LabelWidget(str('Origin / um')), 7, 0)
        stage_scroll_layout.addWidget(cw.LabelWidget(str('Step / um')), 7, 1)
        stage_scroll_layout.addWidget(cw.LabelWidget(str('Range / um')), 7, 2)
        stage_scroll_layout.addWidget(self.QDoubleSpinBox_stage_y, 8, 0)
        stage_scroll_layout.addWidget(self.QDoubleSpinBox_step_y, 8, 1)
        stage_scroll_layout.addWidget(self.QDoubleSpinBox_range_y, 8, 2)
        stage_scroll_layout.addWidget(cw.FrameWidget(), 9, 0, 1, 3)
        stage_scroll_layout.addWidget(cw.LabelWidget(str('Z (um)')), 10, 0)
        stage_scroll_layout.addWidget(self.QDoubleSpinBox_stage_z_usb, 10, 1)
        stage_scroll_layout.addWidget(self.QLCDNumber_stage_position_z, 10, 2)
        stage_scroll_layout.addWidget(cw.LabelWidget(str('Origin / um')), 11, 0)
        stage_scroll_layout.addWidget(cw.LabelWidget(str('Step / um')), 11, 1)
        stage_scroll_layout.addWidget(cw.LabelWidget(str('Range / um')), 11, 2)
        stage_scroll_layout.addWidget(self.QDoubleSpinBox_stage_z, 12, 0)
        stage_scroll_layout.addWidget(self.QDoubleSpinBox_step_z, 12, 1)
        stage_scroll_layout.addWidget(self.QDoubleSpinBox_range_z, 12, 2)

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
        # DO#n -> TriggerBox DIGn -> OUTn. N = 0: output off. N >= 2 needs DO#1-5 (PWM channels).
        # Default DO#1: Exp1 laser train, 5 x 10 ms pulses, 100 ms apart.
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
        group_layout.addWidget(stage_scroll_area)
        group_layout.addWidget(trigger_scroll_area)
        group.setLayout(group_layout)
        return group

    def _create_laser_panel(self):
        group = cw.GroupWidget()
        laser_473_scroll_area, laser_473_scroll_layout = cw.create_scroll_area()

        self.QDoubleSpinBox_laserpower_473 = cw.DoubleSpinBoxWidget(0, 200, 0.1, 1, 0.0)
        self.QPushButton_laser_473 = cw.PushButtonWidget('ON', checkable=True)

        laser_473_scroll_layout.addRow(self.QDoubleSpinBox_laserpower_473)
        laser_473_scroll_layout.addRow(self.QPushButton_laser_473)

        group_layout = QHBoxLayout(group)
        group_layout.addWidget(laser_473_scroll_area)
        group.setLayout(group_layout)
        return group

    def _create_acquisition_panel(self):
        group = cw.GroupWidget()
        acq_scroll_area, acq_scroll_layout = cw.create_scroll_area("G")

        self.QComboBox_imaging_camera_selection = cw.ComboBoxWidget(list_items=["sCMOS"])
        self.QComboBox_live_modes = cw.ComboBoxWidget(list_items=["Wide Field"])
        self.QPushButton_video = cw.PushButtonWidget("Video", checkable=True)
        self.QPushButton_save_live_timing_presets = cw.PushButtonWidget("Save Live TTLs")
        self.QComboBox_acquisition_modes = cw.ComboBoxWidget(list_items=["2D_WideField", "3D_WideField"])
        self.QSpinBox_acquisition_number = cw.SpinBoxWidget(1, 999, 1, 1)
        self.QPushButton_acquire = cw.PushButtonWidget('Acquire', checkable=True)
        self.QPushButton_save_acquisition_timing_presets = cw.PushButtonWidget("Save Acq TTLs")

        acq_scroll_layout.addWidget(cw.LabelWidget(str('Camera')), 0, 0, 1, 1)
        acq_scroll_layout.addWidget(self.QComboBox_imaging_camera_selection, 1, 0, 1, 1)
        acq_scroll_layout.addWidget(cw.LabelWidget(str('Live Modes')), 0, 1, 1, 1)
        acq_scroll_layout.addWidget(self.QComboBox_live_modes, 1, 1, 1, 1)
        acq_scroll_layout.addWidget(self.QPushButton_save_live_timing_presets, 2, 1, 1, 1)
        acq_scroll_layout.addWidget(self.QPushButton_video, 1, 2, 1, 1)
        acq_scroll_layout.addWidget(cw.LabelWidget(str('Acq Modes')), 0, 3, 1, 1)
        acq_scroll_layout.addWidget(self.QComboBox_acquisition_modes, 1, 3, 1, 1)
        acq_scroll_layout.addWidget(self.QPushButton_save_acquisition_timing_presets, 2, 3, 1, 1)
        acq_scroll_layout.addWidget(cw.LabelWidget(str('Acq Number')), 0, 4, 1, 1)
        acq_scroll_layout.addWidget(self.QSpinBox_acquisition_number, 1, 4, 1, 1)
        acq_scroll_layout.addWidget(self.QPushButton_acquire, 2, 4, 1, 1)

        group_layout = QVBoxLayout(group)
        group_layout.addWidget(acq_scroll_area)
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
        self.QDoubleSpinBox_stage_x.valueChanged.connect(self.set_stage_x)
        self.QDoubleSpinBox_stage_y.valueChanged.connect(self.set_stage_y)
        self.QDoubleSpinBox_stage_z.valueChanged.connect(self.set_stage_z)
        self.QDoubleSpinBox_stage_x_usb.valueChanged.connect(self.set_stage_x_usb)
        self.QDoubleSpinBox_stage_y_usb.valueChanged.connect(self.set_stage_y_usb)
        self.QDoubleSpinBox_stage_z_usb.valueChanged.connect(self.set_stage_z_usb)
        self.QPushButton_laser_473.clicked.connect(self.set_laser_473)
        self.QDoubleSpinBox_laserpower_473.valueChanged.connect(self.pw_laser_473)
        self.QPushButton_reset_trigger.clicked.connect(self.reset_trigger)
        self.QPushButton_fire_trigger.clicked.connect(lambda: self.Signal_trigger_fire.emit(self.get_digital_parameters()))
        self.QPushButton_video.clicked.connect(self.run_video)
        self.QPushButton_acquire.clicked.connect(self.run_acquisition)
        self.QComboBox_live_modes.currentIndexChanged.connect(self.load_selected_digital_timing_presets)
        self.QComboBox_acquisition_modes.currentIndexChanged.connect(self.load_selected_digital_timing_presets)
        self.QPushButton_save_live_timing_presets.clicked.connect(lambda: self.save_digital_timing_preset("live"))
        self.QPushButton_save_acquisition_timing_presets.clicked.connect(
            lambda: self.save_digital_timing_preset("acquisition"))

    def get_scmos_roi(self):
        return [self.QSpinBox_scmos_coordinate_x.value(), self.QSpinBox_scmos_coordinate_y.value(),
                self.QSpinBox_scmos_coordinate_nx.value(), self.QSpinBox_scmos_coordinate_ny.value(),
                self.QSpinBox_scmos_coordinate_bin.value()]

    def get_scmos_exposure(self):
        return self.QDoubleSpinBox_scmos_t_exposure.value()

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

    @pyqtSlot(float)
    def set_stage_x(self, pos_x: float):
        pos_y = self.QDoubleSpinBox_stage_y.value()
        pos_z = self.QDoubleSpinBox_stage_z.value()
        self.Signal_stage_move.emit("x", pos_x, pos_y, pos_z)

    @pyqtSlot(float)
    def set_stage_y(self, pos_y: float):
        pos_x = self.QDoubleSpinBox_stage_x.value()
        pos_z = self.QDoubleSpinBox_stage_z.value()
        self.Signal_stage_move.emit("y", pos_x, pos_y, pos_z)

    @pyqtSlot(float)
    def set_stage_z(self, pos_z: float):
        pos_x = self.QDoubleSpinBox_stage_x.value()
        pos_y = self.QDoubleSpinBox_stage_y.value()
        self.Signal_stage_move.emit("z", pos_x, pos_y, pos_z)

    @pyqtSlot(float)
    def set_stage_x_usb(self, pos_x: float):
        pos_y = self.QDoubleSpinBox_stage_y.value()
        pos_z = self.QDoubleSpinBox_stage_z.value()
        self.Signal_stage_move_usb.emit("x", pos_x, pos_y, pos_z)

    @pyqtSlot(float)
    def set_stage_y_usb(self, pos_y: float):
        pos_x = self.QDoubleSpinBox_stage_x.value()
        pos_z = self.QDoubleSpinBox_stage_z.value()
        self.Signal_stage_move_usb.emit("y", pos_x, pos_y, pos_z)

    @pyqtSlot(float)
    def set_stage_z_usb(self, pos_z: float):
        pos_x = self.QDoubleSpinBox_stage_x.value()
        pos_y = self.QDoubleSpinBox_stage_y.value()
        self.Signal_stage_move_usb.emit("z", pos_x, pos_y, pos_z)

    def get_stage_positions(self):
        return [[self.QDoubleSpinBox_stage_x_usb.value(), self.QDoubleSpinBox_stage_x.value()],
                [self.QDoubleSpinBox_stage_y_usb.value(), self.QDoubleSpinBox_stage_y.value()],
                [self.QDoubleSpinBox_stage_z_usb.value(), self.QDoubleSpinBox_stage_z.value()]]

    def get_stage_scan_parameters(self):
        axis_lengths = [self.QDoubleSpinBox_range_x.value(), self.QDoubleSpinBox_range_y.value(),
                        self.QDoubleSpinBox_range_z.value()]
        step_sizes = [self.QDoubleSpinBox_step_x.value(), self.QDoubleSpinBox_step_y.value(),
                      self.QDoubleSpinBox_step_z.value()]
        return axis_lengths, step_sizes

    def display_stage_position_x(self, ps):
        self.QLCDNumber_stage_position_x.display(ps)

    def display_stage_position_y(self, ps):
        self.QLCDNumber_stage_position_y.display(ps)

    def display_stage_position_z(self, ps):
        self.QLCDNumber_stage_position_z.display(ps)

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
        vm = self.QComboBox_live_modes.currentText()
        if self.QPushButton_video.isChecked():
            self.Signal_video.emit(True, vm)
        else:
            self.Signal_video.emit(False, vm)

    @pyqtSlot()
    def run_acquisition(self):
        if self.QPushButton_acquire.isChecked():
            acq_mode = self.QComboBox_acquisition_modes.currentText()
            acq_num = self.QSpinBox_acquisition_number.value()
            self.Signal_data_acquire.emit(True, acq_mode, acq_num)
        else:
            self.Signal_data_acquire.emit(False, "None", 0)

    @pyqtSlot()
    def load_selected_digital_timing_presets(self):
        text = self.QComboBox_live_modes.currentText()
        values = getattr(self, "digital_timing_presets", {}).get(text, {})
        for n in range(1, self.N_TTL + 1):
            for field in self.TTL_FIELDS:
                key = f"ttl_{field}_{n}"
                if key in values:
                    self._ttl_widget(field, n).setValue(values[key])

    @pyqtSlot(str)
    def save_digital_timing_preset(self, m: str):
        if m == "live":
            set_name = self.QComboBox_live_modes.currentText()
        elif m == "acquisition":
            set_name = self.QComboBox_acquisition_modes.currentText()
        else:
            set_name = None
        if set_name:
            if not getattr(self, "digital_timing_presets", None):
                self.digital_timing_presets = {}
            if not self.config.get("Digital Timing Presets"):
                self.logg.error("No file configured for digital timing presets; preset not saved")
                return
            self.digital_timing_presets[set_name] = {
                f"ttl_{field}_{n}": self._ttl_widget(field, n).value()
                for n in range(1, self.N_TTL + 1) for field in self.TTL_FIELDS}
            with open(self.config["Digital Timing Presets"], 'w') as f:
                json.dump(self.digital_timing_presets, f, indent=4)
        else:
            return

    def load_digital_timing_presets(self):
        try:
            with open(self.config["Digital Timing Presets"], 'r') as f:
                presets = json.load(f)
            return presets
        except FileNotFoundError:
            return {}

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
    win = ControlPanel()
    win.show()
    sys.exit(app.exec())
