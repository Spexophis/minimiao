# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import json

from PyQt6.QtCore import pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QSpinBox, QDoubleSpinBox

from . import custom_widgets as cw


class ControlPanel(QWidget):
    Signal_check_scmos_temperature = pyqtSignal()
    Signal_switch_scmos_cooler = pyqtSignal(bool)
    Signal_deck_read_position = pyqtSignal()
    Signal_deck_zero_position = pyqtSignal()
    Signal_deck_move_single_step = pyqtSignal(bool)
    Signal_deck_move_continuous = pyqtSignal(bool, int, float)
    Signal_piezo_move_usb = pyqtSignal(str, float, float, float)
    Signal_piezo_move = pyqtSignal(str, float, float, float)
    Signal_set_laser = pyqtSignal(list, bool, float)
    Signal_daq_update = pyqtSignal(int)
    Signal_daq_reset = pyqtSignal()
    Signal_plot_trigger = pyqtSignal()
    Signal_focus_finding = pyqtSignal()
    Signal_focus_locking = pyqtSignal(bool)
    Signal_video = pyqtSignal(bool, str)
    Signal_fft = pyqtSignal(bool)
    Signal_plot_profile = pyqtSignal()
    Signal_add_profile = pyqtSignal()
    Signal_data_acquire = pyqtSignal(bool, str, int)
    Signal_save_file = pyqtSignal(str)

    def __init__(self, logg, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.config = {"ConWidget Path": "", "Digital Timing Presets": ""}
        self.logg = logg
        self._setup_ui()
        self.load_spinbox_values()
        self.digital_timing_presets = self.load_digital_timing_presets()
        self._set_signal_connections()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        self.camera_panel = self._create_camera_panel()
        self.position_panel = self._create_position_panel()
        self.laser_panel = self._create_laser_panel()
        self.daq_panel = self._create_daq_panel()
        self.slm_panel = self._create_slm_panel()
        self.acq_panel = self._create_acquisition_panel()

        main_layout.addWidget(self.camera_panel)
        main_layout.addWidget(self.position_panel)
        main_layout.addWidget(self.laser_panel)
        main_layout.addWidget(self.daq_panel)
        main_layout.addWidget(self.slm_panel)
        main_layout.addWidget(self.acq_panel)

        main_layout.addStretch(1)
        self.setLayout(main_layout)

    def _create_camera_panel(self):
        group = cw.GroupWidget()
        scmos_scroll_area, scmos_scroll_layout = cw.create_scroll_area()

        self.QLCDNumber_scmos_tempetature = cw.LCDNumberWidget(0, 3)
        self.QPushButton_scmos_cooler_check = cw.PushButtonWidget('Check', False, True)
        self.QPushButton_scmos_cooler_switch = cw.PushButtonWidget('Cooler OFF', True, True, True)
        self.QSpinBox_scmos_coordinate_x = cw.SpinBoxWidget(0, 1024, 1, 1)
        self.QSpinBox_scmos_coordinate_y = cw.SpinBoxWidget(0, 1024, 1, 1)
        self.QSpinBox_scmos_coordinate_nx = cw.SpinBoxWidget(0, 1024, 1, 1024)
        self.QSpinBox_scmos_coordinate_ny = cw.SpinBoxWidget(0, 1024, 1, 1024)
        self.QSpinBox_scmos_coordinate_bin = cw.SpinBoxWidget(0, 1024, 1, 1)
        self.QSpinBox_scmos_gain = cw.SpinBoxWidget(0, 300, 1, 0)
        self.QDoubleSpinBox_scmos_t_exposure = cw.DoubleSpinBoxWidget(0, 10, 0.001, 5, 0.001)
        self.QLCDNumber_scmos_frame_rate = cw.LCDNumberWidget(25, 4)

        scmos_scroll_layout.addRow(cw.LabelWidget(str('sCMOS')))
        scmos_scroll_layout.addRow(cw.FrameWidget())
        scmos_scroll_layout.addRow(cw.LabelWidget(str('Temperature')), self.QLCDNumber_scmos_tempetature)
        scmos_scroll_layout.addRow(cw.LabelWidget(str('X')), self.QSpinBox_scmos_coordinate_x)
        scmos_scroll_layout.addRow(cw.LabelWidget(str('Y')), self.QSpinBox_scmos_coordinate_y)
        scmos_scroll_layout.addRow(cw.LabelWidget(str('Nx')), self.QSpinBox_scmos_coordinate_nx)
        scmos_scroll_layout.addRow(cw.LabelWidget(str('Ny')), self.QSpinBox_scmos_coordinate_ny)
        scmos_scroll_layout.addRow(cw.LabelWidget(str('Bin')), self.QSpinBox_scmos_coordinate_bin)
        scmos_scroll_layout.addRow(cw.LabelWidget(str('Gain')), self.QSpinBox_scmos_gain)
        scmos_scroll_layout.addRow(cw.LabelWidget(str('Exposure / s')), self.QDoubleSpinBox_scmos_t_exposure)
        scmos_scroll_layout.addRow(cw.LabelWidget(str('FPS')), self.QLCDNumber_scmos_frame_rate)

        group_layout = QHBoxLayout(group)
        group_layout.addWidget(scmos_scroll_area)
        group.setLayout(group_layout)
        return group

    def _create_position_panel(self):
        group = cw.GroupWidget()
        mcl_piezo_scroll_area, mcl_piezo_scroll_layout = cw.create_scroll_area("G")

        self.QDoubleSpinBox_stage_x_usb = cw.DoubleSpinBoxWidget(0, 100, 0.020, 3, 20.000)
        self.QLCDNumber_piezo_position_x = cw.LCDNumberWidget()
        self.QDoubleSpinBox_stage_x = cw.DoubleSpinBoxWidget(0, 100, 0.020, 3, 30.000)
        self.QDoubleSpinBox_step_x = cw.DoubleSpinBoxWidget(0, 50, 0.001, 4, 0.030)
        self.QDoubleSpinBox_range_x = cw.DoubleSpinBoxWidget(0, 50, 0.001, 4, 0.780)
        self.QDoubleSpinBox_stage_y_usb = cw.DoubleSpinBoxWidget(0, 100, 0.020, 3, 20.000)
        self.QLCDNumber_piezo_position_y = cw.LCDNumberWidget()
        self.QDoubleSpinBox_stage_y = cw.DoubleSpinBoxWidget(0, 100, 0.020, 3, 30.000)
        self.QDoubleSpinBox_step_y = cw.DoubleSpinBoxWidget(0, 50, 0.001, 4, 0.030)
        self.QDoubleSpinBox_range_y = cw.DoubleSpinBoxWidget(0, 50, 0.001, 4, 0.780)
        self.QDoubleSpinBox_stage_z_usb = cw.DoubleSpinBoxWidget(0, 100, 0.04, 2, 20.00)
        self.QLCDNumber_piezo_position_z = cw.LCDNumberWidget()
        self.QDoubleSpinBox_stage_z = cw.DoubleSpinBoxWidget(0, 100, 0.04, 2, 30.00)
        self.QDoubleSpinBox_step_z = cw.DoubleSpinBoxWidget(0, 50, 0.001, 4, 0.160)
        self.QDoubleSpinBox_range_z = cw.DoubleSpinBoxWidget(0, 50, 0.001, 4, 4.80)

        mcl_piezo_scroll_layout.addWidget(cw.LabelWidget(str('MCL Piezo')), 0, 0)
        mcl_piezo_scroll_layout.addWidget(cw.FrameWidget(), 1, 0, 1, 3)
        mcl_piezo_scroll_layout.addWidget(cw.LabelWidget(str('X (um)')), 2, 0)
        mcl_piezo_scroll_layout.addWidget(self.QDoubleSpinBox_stage_x_usb, 2, 1)
        mcl_piezo_scroll_layout.addWidget(self.QLCDNumber_piezo_position_x, 2, 2)
        mcl_piezo_scroll_layout.addWidget(cw.LabelWidget(str('Origin / um')), 3, 0)
        mcl_piezo_scroll_layout.addWidget(cw.LabelWidget(str('Step / um')), 3, 1)
        mcl_piezo_scroll_layout.addWidget(cw.LabelWidget(str('Range / um')), 3, 2)
        mcl_piezo_scroll_layout.addWidget(self.QDoubleSpinBox_stage_x, 4, 0)
        mcl_piezo_scroll_layout.addWidget(self.QDoubleSpinBox_step_x, 4, 1)
        mcl_piezo_scroll_layout.addWidget(self.QDoubleSpinBox_range_x, 4, 2)
        mcl_piezo_scroll_layout.addWidget(cw.FrameWidget(), 5, 0, 1, 3)
        mcl_piezo_scroll_layout.addWidget(cw.LabelWidget(str('Y (um)')), 6, 0)
        mcl_piezo_scroll_layout.addWidget(self.QDoubleSpinBox_stage_y_usb, 6, 1)
        mcl_piezo_scroll_layout.addWidget(self.QLCDNumber_piezo_position_y, 6, 2)
        mcl_piezo_scroll_layout.addWidget(cw.LabelWidget(str('Origin / um')), 7, 0)
        mcl_piezo_scroll_layout.addWidget(cw.LabelWidget(str('Step / um')), 7, 1)
        mcl_piezo_scroll_layout.addWidget(cw.LabelWidget(str('Range / um')), 7, 2)
        mcl_piezo_scroll_layout.addWidget(self.QDoubleSpinBox_stage_y, 8, 0)
        mcl_piezo_scroll_layout.addWidget(self.QDoubleSpinBox_step_y, 8, 1)
        mcl_piezo_scroll_layout.addWidget(self.QDoubleSpinBox_range_y, 8, 2)
        mcl_piezo_scroll_layout.addWidget(cw.FrameWidget(), 9, 0, 1, 3)
        mcl_piezo_scroll_layout.addWidget(cw.LabelWidget(str('Z (um)')), 10, 0)
        mcl_piezo_scroll_layout.addWidget(self.QDoubleSpinBox_stage_z_usb, 10, 1)
        mcl_piezo_scroll_layout.addWidget(self.QLCDNumber_piezo_position_z, 10, 2)
        mcl_piezo_scroll_layout.addWidget(cw.LabelWidget(str('Origin / um')), 11, 0)
        mcl_piezo_scroll_layout.addWidget(cw.LabelWidget(str('Step / um')), 11, 1)
        mcl_piezo_scroll_layout.addWidget(cw.LabelWidget(str('Range / um')), 11, 2)
        mcl_piezo_scroll_layout.addWidget(self.QDoubleSpinBox_stage_z, 12, 0)
        mcl_piezo_scroll_layout.addWidget(self.QDoubleSpinBox_step_z, 12, 1)
        mcl_piezo_scroll_layout.addWidget(self.QDoubleSpinBox_range_z, 12, 2)

        group_layout = QHBoxLayout(group)
        group_layout.addWidget(mcl_piezo_scroll_area)
        group.setLayout(group_layout)
        return group

    def _create_laser_panel(self):
        group = cw.GroupWidget()
        laser_473_scroll_area, laser_473_scroll_layout = cw.create_scroll_area()

        self.QRadioButton_laser_473 = cw.RadioButtonWidget('473 nm')
        self.QDoubleSpinBox_laserpower_473 = cw.DoubleSpinBoxWidget(0, 200, 0.1, 1, 0.0)
        self.QPushButton_laser_473 = cw.PushButtonWidget('ON', checkable=True)

        laser_473_scroll_layout.addRow(self.QRadioButton_laser_473, self.QDoubleSpinBox_laserpower_473)
        laser_473_scroll_layout.addRow(self.QPushButton_laser_473)

        group_layout = QHBoxLayout(group)
        group_layout.addWidget(laser_473_scroll_area)
        group.setLayout(group_layout)
        return group

    def _create_daq_panel(self):
        group = cw.GroupWidget()
        daq_scroll_area, daq_scroll_layout = cw.create_scroll_area("G")

        self.QSpinBox_daq_sample_rate = cw.SpinBoxWidget(100, 1250, 1, 250)
        self.QPushButton_plot_trigger = cw.PushButtonWidget("Plot Triggers")
        self.QPushButton_reset_daq = cw.PushButtonWidget("Reset")
        self.QDoubleSpinBox_ttl_start_on_473 = cw.DoubleSpinBoxWidget(0, 50, 0.001, 5, 0.008)
        self.QDoubleSpinBox_ttl_stop_on_473 = cw.DoubleSpinBoxWidget(0, 50, 0.001, 5, 0.032)
        self.QDoubleSpinBox_ttl_start_off_488 = cw.DoubleSpinBoxWidget(0, 50, 0.001, 5, 0.008)
        self.QDoubleSpinBox_ttl_stop_off_488 = cw.DoubleSpinBoxWidget(0, 50, 0.001, 5, 0.032)
        self.QDoubleSpinBox_ttl_start_read_488 = cw.DoubleSpinBoxWidget(0, 50, 0.001, 5, 0.008)
        self.QDoubleSpinBox_ttl_stop_read_488 = cw.DoubleSpinBoxWidget(0, 50, 0.001, 5, 0.032)
        self.QDoubleSpinBox_ttl_start_scmos = cw.DoubleSpinBoxWidget(0, 50, 0.001, 5, 0.008)
        self.QDoubleSpinBox_ttl_stop_scmos = cw.DoubleSpinBoxWidget(0, 50, 0.001, 5, 0.032)
        self.QDoubleSpinBox_ttl_start_scmos = cw.DoubleSpinBoxWidget(0, 50, 0.001, 5, 0.008)
        self.QDoubleSpinBox_ttl_stop_scmos = cw.DoubleSpinBoxWidget(0, 50, 0.001, 5, 0.032)

        daq_scroll_layout.addWidget(cw.LabelWidget(str('Sample Rate / KS/s')), 0, 0, 1, 1)
        daq_scroll_layout.addWidget(self.QPushButton_reset_daq, 0, 1, 1, 1)
        daq_scroll_layout.addWidget(self.QSpinBox_daq_sample_rate, 1, 0, 1, 1)
        daq_scroll_layout.addWidget(self.QPushButton_plot_trigger, 2, 0, 1, 1)
        daq_scroll_layout.addWidget(cw.LabelWidget(str('From / s')), 1, 1, 1, 1)
        daq_scroll_layout.addWidget(cw.LabelWidget(str('To / s')), 2, 1, 1, 1)
        daq_scroll_layout.addWidget(cw.LabelWidget(str('DO#0 - L473')), 0, 2, 1, 1)
        daq_scroll_layout.addWidget(self.QDoubleSpinBox_ttl_start_on_473, 1, 2, 1, 1)
        daq_scroll_layout.addWidget(self.QDoubleSpinBox_ttl_stop_on_473, 2, 2, 1, 1)
        daq_scroll_layout.addWidget(cw.LabelWidget(str('DO#1 - L488')), 0, 3, 1, 1)
        daq_scroll_layout.addWidget(self.QDoubleSpinBox_ttl_start_off_488, 1, 3, 1, 1)
        daq_scroll_layout.addWidget(self.QDoubleSpinBox_ttl_stop_off_488, 2, 3, 1, 1)
        daq_scroll_layout.addWidget(cw.LabelWidget(str('DO#3 - L488')), 0, 5, 1, 1)
        daq_scroll_layout.addWidget(self.QDoubleSpinBox_ttl_start_read_488, 1, 5, 1, 1)
        daq_scroll_layout.addWidget(self.QDoubleSpinBox_ttl_stop_read_488, 2, 5, 1, 1)
        daq_scroll_layout.addWidget(cw.LabelWidget(str('DO#4 - iXon')), 0, 6, 1, 1)
        daq_scroll_layout.addWidget(self.QDoubleSpinBox_ttl_start_scmos, 1, 6, 1, 1)
        daq_scroll_layout.addWidget(self.QDoubleSpinBox_ttl_stop_scmos, 2, 6, 1, 1)
        daq_scroll_layout.addWidget(cw.LabelWidget(str('DO#5 - Kira')), 0, 8, 1, 1)
        daq_scroll_layout.addWidget(self.QDoubleSpinBox_ttl_start_scmos, 1, 8, 1, 1)
        daq_scroll_layout.addWidget(self.QDoubleSpinBox_ttl_stop_scmos, 2, 8, 1, 1)

        group_layout = QVBoxLayout(group)
        group_layout.addWidget(daq_scroll_area)
        group.setLayout(group_layout)
        return group

    def _create_slm_panel(self):
        group = cw.GroupWidget()
        slm_scroll_area, slm_scroll_layout = cw.create_scroll_area("G")

        self.QPushButton_SLM_Correction = cw.PushButtonWidget('Load Correction')
        self.QPushButton_SLM_Load = cw.PushButtonWidget('Load Pattern')
        self.QSpinBox_SLM_OffsetX = cw.SpinBoxWidget(0, 1024, 1, 0)
        self.QSpinBox_SLM_OffsetY = cw.SpinBoxWidget(0, 1024, 1, 0)
        self.QDoubleSpinBox_SLM_Focal = cw.DoubleSpinBoxWidget(0, 2000, 1, 2, 180)

        slm_scroll_layout.addWidget(cw.LabelWidget(str('Hamamatsu SLM')), 0, 0, 1, 1)
        slm_scroll_layout.addWidget(cw.FrameWidget(), 1, 0, 1, 3)
        slm_scroll_layout.addWidget(self.QPushButton_SLM_Correction, 2, 0, 1, 1)
        slm_scroll_layout.addWidget(self.QPushButton_SLM_Load, 3, 0, 1, 1)
        slm_scroll_layout.addWidget(cw.LabelWidget(str('Offset X')), 2, 1, 1, 1)
        slm_scroll_layout.addWidget(self.QSpinBox_SLM_OffsetX, 2, 2, 1, 1)
        slm_scroll_layout.addWidget(cw.LabelWidget(str('Offset Y')), 3, 1, 1, 1)
        slm_scroll_layout.addWidget(self.QSpinBox_SLM_OffsetY, 3, 2, 1, 1)
        slm_scroll_layout.addWidget(cw.LabelWidget(str('Focal Length')), 4, 1, 1, 1)
        slm_scroll_layout.addWidget(self.QDoubleSpinBox_SLM_Focal, 4, 2, 1, 1)

        group_layout = QHBoxLayout(group)
        group_layout.addWidget(slm_scroll_area)
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
        self.QPushButton_scmos_cooler_check.clicked.connect(self.check_scmos_temperature)
        self.QDoubleSpinBox_stage_x.valueChanged.connect(self.set_piezo_x)
        self.QDoubleSpinBox_stage_y.valueChanged.connect(self.set_piezo_y)
        self.QDoubleSpinBox_stage_z.valueChanged.connect(self.set_piezo_z)
        self.QDoubleSpinBox_stage_x_usb.valueChanged.connect(self.set_piezo_x_usb)
        self.QDoubleSpinBox_stage_y_usb.valueChanged.connect(self.set_piezo_y_usb)
        self.QDoubleSpinBox_stage_z_usb.valueChanged.connect(self.set_piezo_z_usb)
        self.QPushButton_laser_473.clicked.connect(self.set_laser_473)
        self.QSpinBox_daq_sample_rate.valueChanged.connect(self.update_daq)
        self.QPushButton_reset_daq.clicked.connect(self.reset_daq)
        self.QPushButton_plot_trigger.clicked.connect(self.plot_trigger_sequence)
        self.QPushButton_video.clicked.connect(self.run_video)
        self.QPushButton_acquire.clicked.connect(self.run_acquisition)
        self.QComboBox_live_modes.currentIndexChanged.connect(self.load_selected_digital_timing_presets)
        self.QComboBox_acquisition_modes.currentIndexChanged.connect(self.load_selected_digital_timing_presets)
        self.QPushButton_save_live_timing_presets.clicked.connect(lambda: self.save_digital_timing_preset("live"))
        self.QPushButton_save_acquisition_timing_presets.clicked.connect(
            lambda: self.save_digital_timing_preset("acquisition"))

    @pyqtSlot()
    def check_scmos_temperature(self):
        self.Signal_check_scmos_temperature.emit()

    def display_scmos_temperature(self, temperature):
        self.QLCDNumber_scmos_tempetature.display(temperature)

    def get_scmos_roi(self):
        return [self.QSpinBox_scmos_coordinate_x.value(), self.QSpinBox_scmos_coordinate_y.value(),
                self.QSpinBox_scmos_coordinate_nx.value(), self.QSpinBox_scmos_coordinate_ny.value(),
                self.QSpinBox_scmos_coordinate_bin.value()]

    def get_scmos_exposure(self):
        return self.QDoubleSpinBox_scmos_t_exposure.value()

    def get_scmos_gain(self):
        return self.QSpinBox_scmos_gain.value()

    def display_scmos_timings(self, exposure_time=None, kinetic_time=None):
        if exposure_time is not None:
            self.QDoubleSpinBox_scmos_t_exposure.setValue(exposure_time)
        if kinetic_time is not None:
            fps = 1.0 / kinetic_time
            self.QLCDNumber_scmos_frame_rate.display(fps)

    @pyqtSlot(float)
    def set_piezo_x(self, pos_x: float):
        pos_y = self.QDoubleSpinBox_stage_y.value()
        pos_z = self.QDoubleSpinBox_stage_z.value()
        self.Signal_piezo_move.emit("x", pos_x, pos_y, pos_z)

    @pyqtSlot(float)
    def set_piezo_y(self, pos_y: float):
        pos_x = self.QDoubleSpinBox_stage_x.value()
        pos_z = self.QDoubleSpinBox_stage_z.value()
        self.Signal_piezo_move.emit("y", pos_x, pos_y, pos_z)

    @pyqtSlot(float)
    def set_piezo_z(self, pos_z: float):
        pos_x = self.QDoubleSpinBox_stage_x.value()
        pos_y = self.QDoubleSpinBox_stage_y.value()
        self.Signal_piezo_move.emit("z", pos_x, pos_y, pos_z)

    @pyqtSlot(float)
    def set_piezo_x_usb(self, pos_x: float):
        pos_y = self.QDoubleSpinBox_stage_y.value()
        pos_z = self.QDoubleSpinBox_stage_z.value()
        self.Signal_piezo_move_usb.emit("x", pos_x, pos_y, pos_z)

    @pyqtSlot(float)
    def set_piezo_y_usb(self, pos_y: float):
        pos_x = self.QDoubleSpinBox_stage_x.value()
        pos_z = self.QDoubleSpinBox_stage_z.value()
        self.Signal_piezo_move_usb.emit("y", pos_x, pos_y, pos_z)

    @pyqtSlot(float)
    def set_piezo_z_usb(self, pos_z: float):
        pos_x = self.QDoubleSpinBox_stage_x.value()
        pos_y = self.QDoubleSpinBox_stage_y.value()
        self.Signal_piezo_move_usb.emit("z", pos_x, pos_y, pos_z)

    def get_piezo_positions(self):
        return [[self.QDoubleSpinBox_stage_x_usb.value(), self.QDoubleSpinBox_stage_x.value()],
                [self.QDoubleSpinBox_stage_y_usb.value(), self.QDoubleSpinBox_stage_y.value()],
                [self.QDoubleSpinBox_stage_z_usb.value(), self.QDoubleSpinBox_stage_z.value()]]

    def get_piezo_scan_parameters(self):
        axis_lengths = [self.QDoubleSpinBox_range_x.value(), self.QDoubleSpinBox_range_y.value(),
                        self.QDoubleSpinBox_range_z.value()]
        step_sizes = [self.QDoubleSpinBox_step_x.value(), self.QDoubleSpinBox_step_y.value(),
                      self.QDoubleSpinBox_step_z.value()]
        return axis_lengths, step_sizes

    def display_piezo_position_x(self, ps):
        self.QLCDNumber_piezo_position_x.display(ps)

    def display_piezo_position_y(self, ps):
        self.QLCDNumber_piezo_position_y.display(ps)

    def display_piezo_position_z(self, ps):
        self.QLCDNumber_piezo_position_z.display(ps)

    @pyqtSlot(bool)
    def set_laser_473(self, checked: bool):
        power = self.QDoubleSpinBox_laserpower_473.value()
        self.Signal_set_laser.emit(["473"], checked, power)

    def get_lasers(self):
        lasers = []
        if self.QRadioButton_laser_473.isChecked():
            lasers.append(0)
        return lasers

    def get_cobolt_laser_power(self, laser):
        if laser == "473":
            return [self.QDoubleSpinBox_laserpower_473.value()]
        if "all" == laser:
            return [self.QDoubleSpinBox_laserpower_473.value()]
        return None

    @pyqtSlot(int)
    def update_daq(self, sample_rate: int):
        self.Signal_daq_update.emit(sample_rate)

    @pyqtSlot()
    def reset_daq(self):
        self.Signal_daq_reset.emit()

    def get_digital_parameters(self):
        digital_starts = [self.QDoubleSpinBox_ttl_start_on_473.value(),
                          self.QDoubleSpinBox_ttl_start_off_488.value(),
                          self.QDoubleSpinBox_ttl_start_read_488.value(),
                          self.QDoubleSpinBox_ttl_start_scmos.value(),
                          self.QDoubleSpinBox_ttl_start_scmos.value()]
        digital_ends = [self.QDoubleSpinBox_ttl_stop_on_473.value(),
                        self.QDoubleSpinBox_ttl_stop_off_488.value(),
                        self.QDoubleSpinBox_ttl_stop_read_488.value(),
                        self.QDoubleSpinBox_ttl_stop_scmos.value(),
                        self.QDoubleSpinBox_ttl_stop_scmos.value()]
        return digital_starts, digital_ends

    @pyqtSlot()
    def plot_trigger_sequence(self):
        self.Signal_plot_trigger.emit()

    @pyqtSlot()
    def run_focus_finding(self):
        self.Signal_focus_finding.emit()

    @pyqtSlot()
    def run_focus_locking(self):
        if self.QPushButton_focus_locking.isChecked():
            self.Signal_focus_locking.emit(True)
        else:
            self.Signal_focus_locking.emit(False)

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
        values = self.digital_timing_presets.get(text, {})
        self.QDoubleSpinBox_step_x.setValue(values.get("QDoubleSpinBox_step_x", 0))
        self.QDoubleSpinBox_step_y.setValue(values.get("QDoubleSpinBox_step_y", 0))
        self.QDoubleSpinBox_step_z.setValue(values.get("QDoubleSpinBox_step_z", 0))
        self.QDoubleSpinBox_range_x.setValue(values.get("QDoubleSpinBox_range_x", 0))
        self.QDoubleSpinBox_range_y.setValue(values.get("QDoubleSpinBox_range_y", 0))
        self.QDoubleSpinBox_range_z.setValue(values.get("QDoubleSpinBox_range_z", 0))
        self.QDoubleSpinBox_ttl_start_on_473.setValue(values.get("QDoubleSpinBox_ttl_start_on_473", 0))
        self.QDoubleSpinBox_ttl_stop_on_473.setValue(values.get("QDoubleSpinBox_ttl_stop_on_473", 0))
        self.QDoubleSpinBox_ttl_start_off_488.setValue(values.get("QDoubleSpinBox_ttl_start_off_488", 0))
        self.QDoubleSpinBox_ttl_stop_off_488.setValue(values.get("QDoubleSpinBox_ttl_stop_off_488", 0))
        self.QDoubleSpinBox_ttl_start_read_488.setValue(values.get("QDoubleSpinBox_ttl_start_read_488", 0))
        self.QDoubleSpinBox_ttl_stop_read_488.setValue(values.get("QDoubleSpinBox_ttl_stop_read_488", 0))
        self.QDoubleSpinBox_ttl_start_scmos.setValue(values.get("QDoubleSpinBox_ttl_start_scmos", 0))
        self.QDoubleSpinBox_ttl_stop_scmos.setValue(values.get("QDoubleSpinBox_ttl_stop_scmos", 0))
        self.QDoubleSpinBox_ttl_start_scmos.setValue(values.get("QDoubleSpinBox_ttl_start_scmos", 0))
        self.QDoubleSpinBox_ttl_stop_scmos.setValue(values.get("QDoubleSpinBox_ttl_stop_scmos", 0))

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
                "QDoubleSpinBox_step_x": self.QDoubleSpinBox_step_x.value(),
                "QDoubleSpinBox_step_y": self.QDoubleSpinBox_step_y.value(),
                "QDoubleSpinBox_step_z": self.QDoubleSpinBox_step_z.value(),
                "QDoubleSpinBox_range_x": self.QDoubleSpinBox_range_x.value(),
                "QDoubleSpinBox_range_y": self.QDoubleSpinBox_range_y.value(),
                "QDoubleSpinBox_range_z": self.QDoubleSpinBox_range_z.value(),
                "QDoubleSpinBox_ttl_start_on_473": self.QDoubleSpinBox_ttl_start_on_473.value(),
                "QDoubleSpinBox_ttl_stop_on_473": self.QDoubleSpinBox_ttl_stop_on_473.value(),
                "QDoubleSpinBox_ttl_start_off_488": self.QDoubleSpinBox_ttl_start_off_488.value(),
                "QDoubleSpinBox_ttl_stop_off_488": self.QDoubleSpinBox_ttl_stop_off_488.value(),
                "QDoubleSpinBox_ttl_start_read_488": self.QDoubleSpinBox_ttl_start_read_488.value(),
                "QDoubleSpinBox_ttl_stop_read_488": self.QDoubleSpinBox_ttl_stop_read_488.value(),
                "QDoubleSpinBox_ttl_start_scmos": self.QDoubleSpinBox_ttl_start_scmos.value(),
                "QDoubleSpinBox_ttl_stop_scmos": self.QDoubleSpinBox_ttl_stop_scmos.value(),
                "QDoubleSpinBox_ttl_start_scmos": self.QDoubleSpinBox_ttl_start_scmos.value(),
                "QDoubleSpinBox_ttl_stop_scmos": self.QDoubleSpinBox_ttl_stop_scmos.value()
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
