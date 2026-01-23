# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import json

from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt6.QtWidgets import QSplitter, QWidget, QHBoxLayout, QVBoxLayout, QSpinBox, QDoubleSpinBox

from . import custom_widgets as cw


class ControlPanel(QWidget):
    Signal_galvo_set = pyqtSignal(float, float)
    Signal_piezo_move = pyqtSignal(str, float)
    Signal_set_laser = pyqtSignal(list, bool, float)
    Signal_daq_update = pyqtSignal(int)
    Signal_daq_reset = pyqtSignal()
    Signal_plot_trigger = pyqtSignal()
    Signal_video = pyqtSignal(bool, str)
    Signal_data_acquire = pyqtSignal(str, int)
    Signal_save_file = pyqtSignal(str)

    def __init__(self, config, logg, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.config = config
        self.logg = logg
        self._setup_ui()
        self.load_spinbox_values()
        self.galvo_scan_presets = self.load_galvo_scan_presets()
        self.digital_timing_presets = self.load_digital_timing_presets()
        self._set_signal_connections()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        self.position_panel = self._create_position_panel()
        self.laser_panel = self._create_laser_panel()
        self.daq_panel = self._create_daq_panel()
        self.acq_panel = self._create_acquisition_panel()

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.position_panel)
        splitter.addWidget(self.laser_panel)
        splitter.addWidget(self.daq_panel)
        splitter.addWidget(self.acq_panel)

        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def _create_position_panel(self):
        group = cw.GroupWidget()
        piezo_scroll_area, piezo_scroll_layout = cw.create_scroll_area("G")
        galvo_scroll_area, galvo_scroll_layout = cw.create_scroll_area("G")

        self.QDoubleSpinBox_stage_z = cw.DoubleSpinBoxWidget(0, 100, 0.04, 2, 30.00)
        self.QDoubleSpinBox_step_z = cw.DoubleSpinBoxWidget(0, 50, 0.001, 4, 0.160)
        self.QDoubleSpinBox_range_z = cw.DoubleSpinBoxWidget(0, 50, 0.001, 4, 4.80)
        self.QDoubleSpinBox_piezo_return_time = cw.DoubleSpinBoxWidget(0, 50, 0.01, 2, 0.05)
        self.QPushButton_focus_finding = cw.PushButtonWidget('Find Focus')

        piezo_scroll_layout.addWidget(cw.LabelWidget(str('Piezo')), 0, 0)
        piezo_scroll_layout.addWidget(cw.FrameWidget(), 1, 0, 1, 2)
        piezo_scroll_layout.addWidget(cw.LabelWidget(str('Z (um)')), 2, 0)
        piezo_scroll_layout.addWidget(cw.LabelWidget(str('Origin / um')), 3, 0)
        piezo_scroll_layout.addWidget(cw.LabelWidget(str('Step / um')), 3, 1)
        piezo_scroll_layout.addWidget(cw.LabelWidget(str('Range / um')), 3, 2)
        piezo_scroll_layout.addWidget(self.QDoubleSpinBox_stage_z, 4, 0)
        piezo_scroll_layout.addWidget(self.QDoubleSpinBox_step_z, 4, 1)
        piezo_scroll_layout.addWidget(self.QDoubleSpinBox_range_z, 4, 2)
        piezo_scroll_layout.addWidget(cw.FrameWidget(), 5, 0, 1, 3)
        piezo_scroll_layout.addWidget(cw.LabelWidget(str('Piezo Return / s')), 6, 0)
        piezo_scroll_layout.addWidget(self.QDoubleSpinBox_piezo_return_time, 6, 1)
        piezo_scroll_layout.addWidget(cw.FrameWidget(), 7, 0, 1, 3)
        piezo_scroll_layout.addWidget(self.QPushButton_focus_finding, 8, 0)

        self.QSpinBox_galvo_frequency = cw.SpinBoxWidget(0, 300, 1, 100)
        self.QLCDNumber_galvo_frequency = cw.LCDNumberWidget(0, 3)
        self.QDoubleSpinBox_galvo_x = cw.DoubleSpinBoxWidget(-10, 10, 0.0001, 5, 0)
        self.QDoubleSpinBox_galvo_y = cw.DoubleSpinBoxWidget(-10, 10, 0.0001, 5, 0)
        self.QDoubleSpinBox_galvo_range_x = cw.DoubleSpinBoxWidget(-10, 10, 0.0001, 5, 0.4)
        self.QDoubleSpinBox_galvo_range_y = cw.DoubleSpinBoxWidget(-10, 10, 0.0001, 5, 0.4)
        self.QDoubleSpinBox_dot_range_x = cw.DoubleSpinBoxWidget(0, 10, 0.0001, 5, 0.2)
        self.QDoubleSpinBox_dot_range_y = cw.DoubleSpinBoxWidget(0, 10, 0.0001, 5, 0.2)
        self.QDoubleSpinBox_dot_step_x = cw.DoubleSpinBoxWidget(0, 10, 0.0001, 5, 0.01720)
        self.QDoubleSpinBox_dot_step_y = cw.DoubleSpinBoxWidget(0, 10, 0.0001, 5, 0.01720)
        self.QDoubleSpinBox_galvo_offset_x = cw.DoubleSpinBoxWidget(-10, 10, 0.0001, 5, 0.0)
        self.QDoubleSpinBox_galvo_offset_y = cw.DoubleSpinBoxWidget(-10, 10, 0.0001, 5, 0.0)
        self.QSpinBox_galvo_step_response = cw.SpinBoxWidget(0, 2000, 1, 320)
        self.QSpinBox_galvo_return_time = cw.SpinBoxWidget(0, 2000, 1, 800)
        self.QComboBox_galvo_scan_presets = cw.ComboBoxWidget(list_items=[])
        self.QPushButton_save_galvo_scan_presets = cw.PushButtonWidget("Save Scan")
        self.QLineEdit_new_galvo_scan_preset = cw.LineEditWidget()
        self.QPushButton_save_new_galvo_scan_preset = cw.PushButtonWidget("New Scan")

        galvo_scroll_layout.addWidget(cw.LabelWidget(str('Galvo Scanner')), 0, 0)
        galvo_scroll_layout.addWidget(cw.LabelWidget(str('X Origin / v')), 1, 0)
        galvo_scroll_layout.addWidget(self.QDoubleSpinBox_galvo_x, 1, 1)
        galvo_scroll_layout.addWidget(cw.LabelWidget(str('Scan Range / V')), 2, 0)
        galvo_scroll_layout.addWidget(self.QDoubleSpinBox_galvo_range_x, 2, 1)
        galvo_scroll_layout.addWidget(cw.LabelWidget(str('Dot Range / V')), 3, 0)
        galvo_scroll_layout.addWidget(self.QDoubleSpinBox_dot_range_x, 3, 1)
        galvo_scroll_layout.addWidget(cw.LabelWidget(str('Dot Step / V')), 4, 0)
        galvo_scroll_layout.addWidget(self.QDoubleSpinBox_dot_step_x, 4, 1)
        galvo_scroll_layout.addWidget(cw.LabelWidget(str('Offset X / V')), 5, 0)
        galvo_scroll_layout.addWidget(self.QDoubleSpinBox_galvo_offset_x, 5, 1)
        galvo_scroll_layout.addWidget(cw.LabelWidget(str('Y Origin / v')), 1, 2)
        galvo_scroll_layout.addWidget(self.QDoubleSpinBox_galvo_y, 1, 3)
        galvo_scroll_layout.addWidget(cw.LabelWidget(str('Scan Range / V')), 2, 2)
        galvo_scroll_layout.addWidget(self.QDoubleSpinBox_galvo_range_y, 2, 3)
        galvo_scroll_layout.addWidget(cw.LabelWidget(str('Dot Range / V')), 3, 2)
        galvo_scroll_layout.addWidget(self.QDoubleSpinBox_dot_range_y, 3, 3)
        galvo_scroll_layout.addWidget(cw.LabelWidget(str('Dot Step / V')), 4, 2)
        galvo_scroll_layout.addWidget(self.QDoubleSpinBox_dot_step_y, 4, 3)
        galvo_scroll_layout.addWidget(cw.LabelWidget(str('Offset Y / V')), 5, 2)
        galvo_scroll_layout.addWidget(self.QDoubleSpinBox_galvo_offset_y, 5, 3)
        galvo_scroll_layout.addWidget(cw.LabelWidget(str('Galvo Return / us')), 6, 0)
        galvo_scroll_layout.addWidget(self.QSpinBox_galvo_return_time, 6, 1)
        galvo_scroll_layout.addWidget(cw.LabelWidget(str('Galvo StpResp / us')), 7, 0)
        galvo_scroll_layout.addWidget(self.QSpinBox_galvo_step_response, 7, 1)
        galvo_scroll_layout.addWidget(self.QComboBox_galvo_scan_presets, 8, 0, 1, 2)
        galvo_scroll_layout.addWidget(self.QPushButton_save_galvo_scan_presets, 8, 2)
        galvo_scroll_layout.addWidget(self.QLineEdit_new_galvo_scan_preset, 9, 0, 1, 2)
        galvo_scroll_layout.addWidget(self.QPushButton_save_new_galvo_scan_preset, 9, 2)

        group_layout = QHBoxLayout(group)
        group_layout.addWidget(piezo_scroll_area)
        group_layout.addWidget(galvo_scroll_area)
        group.setLayout(group_layout)
        return group

    def _create_laser_panel(self):
        group = cw.GroupWidget()
        laser_405_scroll_area, laser_405_scroll_layout = cw.create_scroll_area()
        laser_488_0_scroll_area, laser_488_0_scroll_layout = cw.create_scroll_area()
        laser_488_1_scroll_area, laser_488_1_scroll_layout = cw.create_scroll_area()

        self.QRadioButton_laser_405 = cw.RadioButtonWidget('405 nm')
        self.QDoubleSpinBox_laserpower_405 = cw.DoubleSpinBoxWidget(0, 200, 0.1, 1, 0.0)
        self.QPushButton_laser_405 = cw.PushButtonWidget('ON', checkable=True)
        self.QRadioButton_laser_488_0 = cw.RadioButtonWidget('488 nm #0')
        self.QDoubleSpinBox_laserpower_488_0 = cw.DoubleSpinBoxWidget(0, 200, 0.1, 1, 0.0)
        self.QPushButton_laser_488_0 = cw.PushButtonWidget('ON', checkable=True)
        self.QRadioButton_laser_488_1 = cw.RadioButtonWidget('488 nm #1')
        self.QDoubleSpinBox_laserpower_488_1 = cw.DoubleSpinBoxWidget(0, 200, 0.1, 1, 0.0)
        self.QPushButton_laser_488_1 = cw.PushButtonWidget('ON', checkable=True)

        laser_405_scroll_layout.addRow(self.QRadioButton_laser_405, self.QDoubleSpinBox_laserpower_405)
        laser_405_scroll_layout.addRow(self.QPushButton_laser_405)
        laser_488_0_scroll_layout.addRow(self.QRadioButton_laser_488_0, self.QDoubleSpinBox_laserpower_488_0)
        laser_488_0_scroll_layout.addRow(self.QPushButton_laser_488_0)
        laser_488_1_scroll_layout.addRow(self.QRadioButton_laser_488_1, self.QDoubleSpinBox_laserpower_488_1)
        laser_488_1_scroll_layout.addRow(self.QPushButton_laser_488_1)

        group_layout = QHBoxLayout(group)
        group_layout.addWidget(laser_405_scroll_area)
        group_layout.addWidget(laser_488_0_scroll_area)
        group_layout.addWidget(laser_488_1_scroll_area)
        group.setLayout(group_layout)
        return group

    def _create_daq_panel(self):
        group = cw.GroupWidget()
        daq_scroll_area, daq_scroll_layout = cw.create_scroll_area("G")

        self.QSpinBox_daq_sample_rate = cw.SpinBoxWidget(0, 80, 1, 80)
        self.QPushButton_plot_trigger = cw.PushButtonWidget("Plot Triggers")
        self.QPushButton_reset_daq = cw.PushButtonWidget("Reset")
        self.QDoubleSpinBox_ttl_start_on_405 = cw.DoubleSpinBoxWidget(0, 50, 0.001, 5, 0.008)
        self.QDoubleSpinBox_ttl_stop_on_405 = cw.DoubleSpinBoxWidget(0, 50, 0.001, 5, 0.032)
        self.QDoubleSpinBox_ttl_start_off_488 = cw.DoubleSpinBoxWidget(0, 50, 0.001, 5, 0.008)
        self.QDoubleSpinBox_ttl_stop_off_488 = cw.DoubleSpinBoxWidget(0, 50, 0.001, 5, 0.032)
        self.QDoubleSpinBox_ttl_start_read_488 = cw.DoubleSpinBoxWidget(0, 50, 0.001, 5, 0.008)
        self.QDoubleSpinBox_ttl_stop_read_488 = cw.DoubleSpinBoxWidget(0, 50, 0.001, 5, 0.032)
        self.QDoubleSpinBox_ttl_start_cmos = cw.DoubleSpinBoxWidget(0, 50, 0.001, 5, 0.008)
        self.QDoubleSpinBox_ttl_stop_cmos = cw.DoubleSpinBoxWidget(0, 50, 0.001, 5, 0.032)
        self.QDoubleSpinBox_ttl_start_mpd_h = cw.DoubleSpinBoxWidget(0, 50, 0.001, 5, 0.008)
        self.QDoubleSpinBox_ttl_stop_mpd_h = cw.DoubleSpinBoxWidget(0, 50, 0.001, 5, 0.032)
        self.QDoubleSpinBox_ttl_start_mpd_v = cw.DoubleSpinBoxWidget(0, 50, 0.001, 5, 0.008)
        self.QDoubleSpinBox_ttl_stop_mpd_v = cw.DoubleSpinBoxWidget(0, 50, 0.001, 5, 0.032)

        daq_scroll_layout.addWidget(cw.LabelWidget(str('Sample Rate / KS/s')), 0, 0, 1, 1)
        daq_scroll_layout.addWidget(self.QPushButton_reset_daq, 0, 1, 1, 1)
        daq_scroll_layout.addWidget(self.QSpinBox_daq_sample_rate, 1, 0, 1, 1)
        daq_scroll_layout.addWidget(self.QPushButton_plot_trigger, 2, 0, 1, 1)
        daq_scroll_layout.addWidget(cw.LabelWidget(str('From / s')), 1, 1, 1, 1)
        daq_scroll_layout.addWidget(cw.LabelWidget(str('To / s')), 2, 1, 1, 1)
        daq_scroll_layout.addWidget(cw.LabelWidget(str('DO#0 - L405')), 0, 2, 1, 1)
        daq_scroll_layout.addWidget(self.QDoubleSpinBox_ttl_start_on_405, 1, 2, 1, 1)
        daq_scroll_layout.addWidget(self.QDoubleSpinBox_ttl_stop_on_405, 2, 2, 1, 1)
        daq_scroll_layout.addWidget(cw.LabelWidget(str('DO#1 - L488')), 0, 3, 1, 1)
        daq_scroll_layout.addWidget(self.QDoubleSpinBox_ttl_start_off_488, 1, 3, 1, 1)
        daq_scroll_layout.addWidget(self.QDoubleSpinBox_ttl_stop_off_488, 2, 3, 1, 1)
        daq_scroll_layout.addWidget(cw.LabelWidget(str('DO#3 - L488')), 0, 5, 1, 1)
        daq_scroll_layout.addWidget(self.QDoubleSpinBox_ttl_start_read_488, 1, 5, 1, 1)
        daq_scroll_layout.addWidget(self.QDoubleSpinBox_ttl_stop_read_488, 2, 5, 1, 1)
        daq_scroll_layout.addWidget(cw.LabelWidget(str('DO#4 - MPD_0')), 0, 6, 1, 1)
        daq_scroll_layout.addWidget(self.QDoubleSpinBox_ttl_start_mpd_h, 1, 6, 1, 1)
        daq_scroll_layout.addWidget(self.QDoubleSpinBox_ttl_stop_mpd_h, 2, 6, 1, 1)
        daq_scroll_layout.addWidget(cw.LabelWidget(str('DO#5 - MPD_1')), 0, 8, 1, 1)
        daq_scroll_layout.addWidget(self.QDoubleSpinBox_ttl_start_mpd_v, 1, 8, 1, 1)
        daq_scroll_layout.addWidget(self.QDoubleSpinBox_ttl_stop_mpd_v, 2, 8, 1, 1)
        daq_scroll_layout.addWidget(cw.LabelWidget(str('DO#6 - CMOS')), 0, 9, 1, 1)
        daq_scroll_layout.addWidget(self.QDoubleSpinBox_ttl_start_cmos, 1, 9, 1, 1)
        daq_scroll_layout.addWidget(self.QDoubleSpinBox_ttl_stop_cmos, 2, 9, 1, 1)

        group_layout = QVBoxLayout(group)
        group_layout.addWidget(daq_scroll_area)
        group.setLayout(group_layout)
        return group

    def _create_acquisition_panel(self):
        group = cw.GroupWidget()
        acq_scroll_area, acq_scroll_layout = cw.create_scroll_area("G")

        self.QComboBox_imaging_detector_selection = cw.ComboBoxWidget(list_items=["MPD_0 + MPD_1", "MPD_0 + PMT"])
        self.QComboBox_live_modes = cw.ComboBoxWidget(list_items=["Point Scan", "Static Point"])
        self.QPushButton_video = cw.PushButtonWidget("Video", checkable=True)
        self.QPushButton_save_live_timing_presets = cw.PushButtonWidget("Save Live TTLs")
        self.QComboBox_acquisition_modes = cw.ComboBoxWidget(list_items=["Point Scan 2D", "Static Point 2D"])
        self.QSpinBox_acquisition_number = cw.SpinBoxWidget(1, 999, 1, 1)
        self.QPushButton_acquire = cw.PushButtonWidget('Acquire')
        self.QPushButton_save_acquisition_timing_presets = cw.PushButtonWidget("Save Acq TTLs")

        acq_scroll_layout.addWidget(cw.LabelWidget(str('Detector')), 0, 0, 1, 1)
        acq_scroll_layout.addWidget(self.QComboBox_imaging_detector_selection, 1, 0, 1, 1)
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
        self.QDoubleSpinBox_galvo_x.valueChanged.connect(self.set_galvo_x)
        self.QDoubleSpinBox_galvo_y.valueChanged.connect(self.set_galvo_y)
        self.QComboBox_galvo_scan_presets.currentTextChanged.connect(self.load_selected_preset)
        self.QPushButton_save_galvo_scan_presets.clicked.connect(self.save_galvo_scan_preset)
        self.QPushButton_save_new_galvo_scan_preset.clicked.connect(self.create_new_galvo_preset)
        self.QDoubleSpinBox_stage_z.valueChanged.connect(self.set_piezo_z)
        self.QPushButton_laser_488_0.clicked.connect(self.set_laser_488_0)
        self.QPushButton_laser_488_1.clicked.connect(self.set_laser_488_1)
        self.QPushButton_laser_405.clicked.connect(self.set_laser_405)
        self.QSpinBox_daq_sample_rate.valueChanged.connect(self.update_daq)
        self.QPushButton_reset_daq.clicked.connect(self.reset_daq)
        self.QPushButton_plot_trigger.clicked.connect(self.plot_trigger_sequence)
        self.QPushButton_video.clicked.connect(self.run_video)
        self.QPushButton_acquire.clicked.connect(self.run_acquisition)
        self.QComboBox_live_modes.currentIndexChanged.connect(self.load_selected_digital_timing_presets)
        self.QComboBox_acquisition_modes.currentIndexChanged.connect(self.load_selected_digital_timing_presets)
        self.QPushButton_save_live_timing_presets.clicked.connect(lambda: self.save_digital_timing_preset("live"))
        self.QPushButton_save_acquisition_timing_presets.clicked.connect(lambda: self.save_digital_timing_preset("acquisition"))

    @pyqtSlot(float)
    def set_galvo_x(self, value: float):
        vy = self.QDoubleSpinBox_galvo_y.value()
        self.Signal_galvo_set.emit(value, vy)

    @pyqtSlot(float)
    def set_galvo_y(self, value: float):
        vx = self.QDoubleSpinBox_galvo_x.value()
        self.Signal_galvo_set.emit(vx, value)

    def get_galvo_positions(self):
        return [self.QDoubleSpinBox_galvo_x.value(), self.QDoubleSpinBox_galvo_y.value()]

    def get_galvo_scan_parameters(self):
        galvo_positions = [self.QDoubleSpinBox_galvo_x.value(), self.QDoubleSpinBox_galvo_y.value()]
        galvo_ranges = [[self.QDoubleSpinBox_galvo_range_x.value(), self.QDoubleSpinBox_galvo_range_y.value()],
                        [self.QDoubleSpinBox_dot_range_x.value(), self.QDoubleSpinBox_dot_range_y.value()]]
        dot_pos = [self.QDoubleSpinBox_dot_step_x.value(), self.QDoubleSpinBox_dot_step_y.value()]
        offsets = [self.QDoubleSpinBox_galvo_offset_x.value(), self.QDoubleSpinBox_galvo_offset_y.value()]
        returns = [self.QSpinBox_galvo_return_time.value(), self.QSpinBox_galvo_step_response.value()]
        return galvo_positions, galvo_ranges, dot_pos, offsets, returns

    @pyqtSlot(float)
    def set_piezo_z(self, pos_z: float):
        self.Signal_piezo_move.emit("z", pos_z)

    def get_piezo_positions(self):
        return self.QDoubleSpinBox_stage_z.value()

    def get_piezo_scan_time(self):
        return self.QDoubleSpinBox_piezo_return_time.value()

    def get_piezo_scan_parameters(self):
        axis_origins = self.QDoubleSpinBox_stage_z.value()
        axis_lengths = self.QDoubleSpinBox_range_z.value()
        step_sizes = self.QDoubleSpinBox_step_z.value()
        return [axis_origins], [axis_lengths], [step_sizes]

    @pyqtSlot(bool)
    def set_laser_488_0(self, checked: bool):
        power = self.QDoubleSpinBox_laserpower_488_0.value()
        self.Signal_set_laser.emit(["488_0"], checked, power)

    @pyqtSlot(bool)
    def set_laser_488_1(self, checked: bool):
        power = self.QDoubleSpinBox_laserpower_488_1.value()
        self.Signal_set_laser.emit(["488_1"], checked, power)

    @pyqtSlot(bool)
    def set_laser_405(self, checked: bool):
        power = self.QDoubleSpinBox_laserpower_405.value()
        self.Signal_set_laser.emit(["405"], checked, power)

    def get_lasers(self):
        lasers = []
        if self.QRadioButton_laser_405.isChecked():
            lasers.append(0)
        if self.QRadioButton_laser_488_0.isChecked():
            lasers.append(1)
        if self.QRadioButton_laser_488_1.isChecked():
            lasers.append(2)
        return lasers

    def get_cobolt_laser_power(self, laser):
        if laser == "405":
            return [self.QDoubleSpinBox_laserpower_405.value()]
        if laser == "488_0":
            return [self.QDoubleSpinBox_laserpower_488_0.value()]
        if laser == "488_1":
            return [self.QDoubleSpinBox_laserpower_488_1.value()]
        if "all" == laser:
            return [self.QDoubleSpinBox_laserpower_405.value(), self.QDoubleSpinBox_laserpower_488_0.value(),
                    self.QDoubleSpinBox_laserpower_488_1.value()]
        return None

    @pyqtSlot(int)
    def update_daq(self, sample_rate: int):
        self.Signal_daq_update.emit(sample_rate)

    @pyqtSlot()
    def reset_daq(self):
        self.Signal_daq_reset.emit()

    def get_detector(self):
        return self.QComboBox_imaging_detector_selection.currentIndex()

    def get_digital_parameters(self):
        digital_starts = [self.QDoubleSpinBox_ttl_start_on_405.value(),
                          self.QDoubleSpinBox_ttl_start_off_488.value(),
                          self.QDoubleSpinBox_ttl_start_read_488.value(),
                          self.QDoubleSpinBox_ttl_start_mpd_h.value(),
                          self.QDoubleSpinBox_ttl_start_mpd_v.value(),
                          self.QDoubleSpinBox_ttl_start_cmos.value()]
        digital_ends = [self.QDoubleSpinBox_ttl_stop_on_405.value(),
                        self.QDoubleSpinBox_ttl_stop_off_488.value(),
                        self.QDoubleSpinBox_ttl_stop_read_488.value(),
                        self.QDoubleSpinBox_ttl_stop_mpd_h.value(),
                        self.QDoubleSpinBox_ttl_stop_mpd_v.value(),
                        self.QDoubleSpinBox_ttl_stop_cmos.value()]
        return digital_starts, digital_ends

    @pyqtSlot()
    def plot_trigger_sequence(self):
        self.Signal_plot_trigger.emit()

    @pyqtSlot()
    def run_video(self):
        vm = self.QComboBox_live_modes.currentText()
        if self.QPushButton_video.isChecked():
            self.Signal_video.emit(True, vm)
        else:
            self.Signal_video.emit(False, vm)

    @pyqtSlot()
    def run_acquisition(self):
        acq_mode = self.QComboBox_acquisition_modes.currentText()
        acq_num = self.QSpinBox_acquisition_number.value()
        self.Signal_data_acquire.emit(acq_mode, acq_num)

    @pyqtSlot()
    def save_galvo_scan_preset(self):
        set_name = self.QComboBox_galvo_scan_presets.currentText()
        if not set_name:
            return
        self.galvo_scan_presets[set_name] = {
            "QDoubleSpinBox_galvo_x": self.QDoubleSpinBox_galvo_x.value(),
            "QDoubleSpinBox_galvo_y": self.QDoubleSpinBox_galvo_y.value(),
            "QDoubleSpinBox_galvo_range_x": self.QDoubleSpinBox_galvo_range_x.value(),
            "QDoubleSpinBox_galvo_range_y": self.QDoubleSpinBox_galvo_range_y.value(),
            "QDoubleSpinBox_dot_range_x": self.QDoubleSpinBox_dot_range_x.value(),
            "QDoubleSpinBox_dot_range_y": self.QDoubleSpinBox_dot_range_y.value(),
            "QDoubleSpinBox_dot_step_x": self.QDoubleSpinBox_dot_step_x.value(),
            "QDoubleSpinBox_dot_step_y": self.QDoubleSpinBox_dot_step_y.value(),
            "QDoubleSpinBox_galvo_offset_x": self.QDoubleSpinBox_galvo_offset_x.value(),
            "QDoubleSpinBox_galvo_offset_y": self.QDoubleSpinBox_galvo_offset_y.value()
        }
        with open(self.config["Galvo Scan Presets"], 'w') as f:
            json.dump(self.galvo_scan_presets, f, indent=4)

    @pyqtSlot(str)
    def load_selected_preset(self, set_name: str):
        values = self.galvo_scan_presets.get(set_name, {})
        self.QDoubleSpinBox_galvo_x.setValue(values.get("QDoubleSpinBox_galvo_x", 0))
        self.QDoubleSpinBox_galvo_y.setValue(values.get("QDoubleSpinBox_galvo_y", 0))
        self.QDoubleSpinBox_galvo_range_x.setValue(values.get("QDoubleSpinBox_galvo_range_x", 0))
        self.QDoubleSpinBox_galvo_range_y.setValue(values.get("QDoubleSpinBox_galvo_range_y", 0))
        self.QDoubleSpinBox_dot_range_x.setValue(values.get("QDoubleSpinBox_dot_range_x", 0))
        self.QDoubleSpinBox_dot_range_y.setValue(values.get("QDoubleSpinBox_dot_range_y", 0))
        self.QDoubleSpinBox_dot_step_x.setValue(values.get("QDoubleSpinBox_dot_step_x", 0))
        self.QDoubleSpinBox_dot_step_y.setValue(values.get("QDoubleSpinBox_dot_step_y", 0))
        self.QDoubleSpinBox_galvo_offset_x.setValue(values.get("QDoubleSpinBox_galvo_offset_x", 0))
        self.QDoubleSpinBox_galvo_offset_y.setValue(values.get("QDoubleSpinBox_galvo_offset_y", 0))

    @pyqtSlot()
    def create_new_galvo_preset(self):
        new_preset_name = self.QLineEdit_new_galvo_scan_preset.text().strip()
        if new_preset_name and new_preset_name not in self.galvo_scan_presets:
            self.galvo_scan_presets[new_preset_name] = {
                "QDoubleSpinBox_galvo_x": self.QDoubleSpinBox_galvo_x.value(),
                "QDoubleSpinBox_galvo_y": self.QDoubleSpinBox_galvo_y.value(),
                "QDoubleSpinBox_galvo_range_x": self.QDoubleSpinBox_galvo_range_x.value(),
                "QDoubleSpinBox_galvo_range_y": self.QDoubleSpinBox_galvo_range_y.value(),
                "QDoubleSpinBox_dot_range_x": self.QDoubleSpinBox_dot_range_x.value(),
                "QDoubleSpinBox_dot_range_y": self.QDoubleSpinBox_dot_range_y.value(),
                "QDoubleSpinBox_dot_step_x": self.QDoubleSpinBox_dot_step_x.value(),
                "QDoubleSpinBox_dot_step_y": self.QDoubleSpinBox_dot_step_y.value(),
                "QDoubleSpinBox_galvo_offset_x": self.QDoubleSpinBox_galvo_offset_x.value(),
                "QDoubleSpinBox_galvo_offset_y": self.QDoubleSpinBox_galvo_offset_y.value()
            }
            with open(self.config["Galvo Scan Presets"], 'w') as f:
                json.dump(self.galvo_scan_presets, f, indent=4)
            self.QComboBox_galvo_scan_presets.addItem(new_preset_name)
            self.QComboBox_galvo_scan_presets.setCurrentText(new_preset_name)
            self.QLineEdit_new_galvo_scan_preset.clear()

    @pyqtSlot()
    def load_selected_digital_timing_presets(self):
        text = self.QComboBox_live_modes.currentText()
        values = self.digital_timing_presets.get(text, {})
        self.QDoubleSpinBox_step_z.setValue(values.get("QDoubleSpinBox_step_z", 0))
        self.QDoubleSpinBox_range_z.setValue(values.get("QDoubleSpinBox_range_z", 0))
        self.QDoubleSpinBox_ttl_start_on_405.setValue(values.get("QDoubleSpinBox_ttl_start_on_405", 0))
        self.QDoubleSpinBox_ttl_stop_on_405.setValue(values.get("QDoubleSpinBox_ttl_stop_on_405", 0))
        self.QDoubleSpinBox_ttl_start_off_488.setValue(values.get("QDoubleSpinBox_ttl_start_off_488", 0))
        self.QDoubleSpinBox_ttl_stop_off_488.setValue(values.get("QDoubleSpinBox_ttl_stop_off_488", 0))
        self.QDoubleSpinBox_ttl_start_read_488.setValue(values.get("QDoubleSpinBox_ttl_start_read_488", 0))
        self.QDoubleSpinBox_ttl_stop_read_488.setValue(values.get("QDoubleSpinBox_ttl_stop_read_488", 0))
        self.QDoubleSpinBox_ttl_start_cmos.setValue(values.get("QDoubleSpinBox_ttl_start_cmos", 0))
        self.QDoubleSpinBox_ttl_stop_cmos.setValue(values.get("QDoubleSpinBox_ttl_stop_cmos", 0))
        self.QDoubleSpinBox_ttl_start_mpd_h.setValue(values.get("QDoubleSpinBox_ttl_start_mpd_h", 0))
        self.QDoubleSpinBox_ttl_stop_mpd_h.setValue(values.get("QDoubleSpinBox_ttl_stop_mpd_h", 0))
        self.QDoubleSpinBox_ttl_start_mpd_v.setValue(values.get("QDoubleSpinBox_ttl_start_mpd_v", 0))
        self.QDoubleSpinBox_ttl_stop_mpd_v.setValue(values.get("QDoubleSpinBox_ttl_stop_mpd_v", 0))

    @pyqtSlot(str)
    def save_digital_timing_preset(self, m: str):
        if m == "live":
            set_name = self.QComboBox_live_modes.currentText()
        elif m == "acquisition":
            set_name = self.QComboBox_acquisition_modes.currentText()
        else:
            set_name = None
        if set_name:
            self.digital_timing_presets[set_name] = {
                "QDoubleSpinBox_step_z": self.QDoubleSpinBox_step_z.value(),
                "QDoubleSpinBox_range_z": self.QDoubleSpinBox_range_z.value(),
                "QDoubleSpinBox_ttl_start_on_405": self.QDoubleSpinBox_ttl_start_on_405.value(),
                "QDoubleSpinBox_ttl_stop_on_405": self.QDoubleSpinBox_ttl_stop_on_405.value(),
                "QDoubleSpinBox_ttl_start_off_488": self.QDoubleSpinBox_ttl_start_off_488.value(),
                "QDoubleSpinBox_ttl_stop_off_488": self.QDoubleSpinBox_ttl_stop_off_488.value(),
                "QDoubleSpinBox_ttl_start_read_488": self.QDoubleSpinBox_ttl_start_read_488.value(),
                "QDoubleSpinBox_ttl_stop_read_488": self.QDoubleSpinBox_ttl_stop_read_488.value(),
                "QDoubleSpinBox_ttl_start_cmos": self.QDoubleSpinBox_ttl_start_cmos.value(),
                "QDoubleSpinBox_ttl_stop_cmos": self.QDoubleSpinBox_ttl_stop_cmos.value(),
                "QDoubleSpinBox_ttl_start_mpd_h": self.QDoubleSpinBox_ttl_start_mpd_h.value(),
                "QDoubleSpinBox_ttl_stop_mpd_h": self.QDoubleSpinBox_ttl_stop_mpd_h.value(),
                "QDoubleSpinBox_ttl_start_mpd_v": self.QDoubleSpinBox_ttl_start_mpd_v.value(),
                "QDoubleSpinBox_ttl_stop_mpd_v": self.QDoubleSpinBox_ttl_stop_mpd_v.value(),
            }
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

    def load_galvo_scan_presets(self):
        try:
            with open(self.config["Galvo Scan Presets"], 'r') as f:
                presets = json.load(f)
            for name, value in presets.items():
                self.QComboBox_galvo_scan_presets.addItem(name)
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

    @staticmethod
    def write_config(dataframe, dfd):
        with open(dfd, 'w') as f:
            json.dump(dataframe, f, indent=4)