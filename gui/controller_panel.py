import json

from PyQt6.QtCore import pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QSpinBox, QDoubleSpinBox

from gui import custom_widgets as cw


class ControlPanel(QWidget):
    Signal_check_emccd_temperature = pyqtSignal()
    Signal_switch_emccd_cooler = pyqtSignal(bool)
    Signal_piezo_move_usb = pyqtSignal(str, float, float, float)
    Signal_piezo_move = pyqtSignal(str, float, float, float)
    Signal_deck_read_position = pyqtSignal()
    Signal_deck_zero_position = pyqtSignal()
    Signal_deck_move_single_step = pyqtSignal(bool)
    Signal_deck_move_continuous = pyqtSignal(bool, int, float)
    Signal_set_laser = pyqtSignal(list, bool, float)
    Signal_daq_update = pyqtSignal(int)
    Signal_daq_reset = pyqtSignal()
    Signal_plot_trigger = pyqtSignal()
    Signal_focus_finding = pyqtSignal()
    Signal_focus_locking = pyqtSignal(bool)
    Signal_video = pyqtSignal(bool, str)
    Signal_fft = pyqtSignal(bool)
    Signal_plot_profile = pyqtSignal(bool)
    Signal_add_profile = pyqtSignal()
    Signal_data_acquire = pyqtSignal(str, int)
    Signal_save_file = pyqtSignal(str)

    def __init__(self, config, logg, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.config = config
        self.logg = logg
        self._setup_ui()
        self._load_spinbox_values()
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
        emccd_scroll_area, emccd_scroll_layout = cw.create_scroll_area()
        scmos_scroll_area, scmos_scroll_layout = cw.create_scroll_area()
        cmos_scroll_area, cmos_scroll_layout = cw.create_scroll_area()

        self.QLCDNumber_ccd_tempetature = cw.LCDNumberWidget(0, 3)
        self.QPushButton_emccd_cooler_check = cw.PushButtonWidget('Check', False, True)
        self.QPushButton_emccd_cooler_switch = cw.PushButtonWidget('Cooler OFF', True, True, True)
        self.QSpinBox_emccd_coordinate_x = cw.SpinBoxWidget(0, 1024, 1, 1)
        self.QSpinBox_emccd_coordinate_y = cw.SpinBoxWidget(0, 1024, 1, 1)
        self.QSpinBox_emccd_coordinate_nx = cw.SpinBoxWidget(0, 1024, 1, 1024)
        self.QSpinBox_emccd_coordinate_ny = cw.SpinBoxWidget(0, 1024, 1, 1024)
        self.QSpinBox_emccd_coordinate_bin = cw.SpinBoxWidget(0, 1024, 1, 1)
        self.QSpinBox_emccd_gain = cw.SpinBoxWidget(0, 300, 1, 0)
        self.QDoubleSpinBox_emccd_t_clean = cw.DoubleSpinBoxWidget(0, 10, 0.001, 5, 0.009)
        self.QDoubleSpinBox_emccd_t_exposure = cw.DoubleSpinBoxWidget(0, 10, 0.001, 5, 0.001)
        self.QDoubleSpinBox_emccd_t_standby = cw.DoubleSpinBoxWidget(0, 10, 0.001, 5, 0.050)

        emccd_scroll_layout.addRow(cw.LabelWidget(str('EMCCD')))
        emccd_scroll_layout.addRow(cw.FrameWidget())
        emccd_scroll_layout.addRow(cw.LabelWidget(str('Temperature')), self.QLCDNumber_ccd_tempetature)
        emccd_scroll_layout.addRow(self.QPushButton_emccd_cooler_check, self.QPushButton_emccd_cooler_switch)
        emccd_scroll_layout.addRow(cw.LabelWidget(str('X')), self.QSpinBox_emccd_coordinate_x)
        emccd_scroll_layout.addRow(cw.LabelWidget(str('Y')), self.QSpinBox_emccd_coordinate_y)
        emccd_scroll_layout.addRow(cw.LabelWidget(str('Nx')), self.QSpinBox_emccd_coordinate_nx)
        emccd_scroll_layout.addRow(cw.LabelWidget(str('Ny')), self.QSpinBox_emccd_coordinate_ny)
        emccd_scroll_layout.addRow(cw.LabelWidget(str('Bin')), self.QSpinBox_emccd_coordinate_bin)
        emccd_scroll_layout.addRow(cw.LabelWidget(str('EMGain')), self.QSpinBox_emccd_gain)
        emccd_scroll_layout.addRow(cw.LabelWidget(str('Clean / s')), self.QDoubleSpinBox_emccd_t_clean)
        emccd_scroll_layout.addRow(cw.LabelWidget(str('Exposure / s')), self.QDoubleSpinBox_emccd_t_exposure)
        emccd_scroll_layout.addRow(cw.LabelWidget(str('Standby / s')), self.QDoubleSpinBox_emccd_t_standby)

        self.QSpinBox_scmos_coordinate_x = cw.SpinBoxWidget(0, 2048, 1, 0)
        self.QSpinBox_scmos_coordinate_y = cw.SpinBoxWidget(0, 2048, 1, 0)
        self.QSpinBox_scmos_coordinate_nx = cw.SpinBoxWidget(0, 2048, 1, 2048)
        self.QSpinBox_scmos_coordinate_ny = cw.SpinBoxWidget(0, 2048, 1, 2048)
        self.QSpinBox_scmos_coordinate_bin = cw.SpinBoxWidget(0, 2048, 1, 1)
        self.QSpinBox_scmos_gain = cw.SpinBoxWidget(0, 300, 1, 0)
        self.QDoubleSpinBox_scmos_t_clean = cw.DoubleSpinBoxWidget(0, 10, 0.001, 5, 0.009)
        self.QDoubleSpinBox_scmos_t_exposure = cw.DoubleSpinBoxWidget(0, 10, 0.001, 5, 0.001)
        self.QDoubleSpinBox_scmos_t_standby = cw.DoubleSpinBoxWidget(0, 10, 0.001, 5, 0.050)

        scmos_scroll_layout.addRow(cw.LabelWidget(str('sCMOS')))
        scmos_scroll_layout.addRow(cw.FrameWidget())
        scmos_scroll_layout.addRow(cw.LabelWidget(str('X')), self.QSpinBox_scmos_coordinate_x)
        scmos_scroll_layout.addRow(cw.LabelWidget(str('Y')), self.QSpinBox_scmos_coordinate_y)
        scmos_scroll_layout.addRow(cw.LabelWidget(str('Nx')), self.QSpinBox_scmos_coordinate_nx)
        scmos_scroll_layout.addRow(cw.LabelWidget(str('Ny')), self.QSpinBox_scmos_coordinate_ny)
        scmos_scroll_layout.addRow(cw.LabelWidget(str('Binx')), self.QSpinBox_scmos_coordinate_bin)
        scmos_scroll_layout.addRow(cw.LabelWidget(str('Gain')), self.QSpinBox_scmos_gain)
        scmos_scroll_layout.addRow(cw.LabelWidget(str('Clean / s')), self.QDoubleSpinBox_scmos_t_clean)
        scmos_scroll_layout.addRow(cw.LabelWidget(str('Exposure / s')), self.QDoubleSpinBox_scmos_t_exposure)
        scmos_scroll_layout.addRow(cw.LabelWidget(str('Standby / s')), self.QDoubleSpinBox_scmos_t_standby)

        self.QSpinBox_cmos_coordinate_x = cw.SpinBoxWidget(0, 2048, 1, 0)
        self.QSpinBox_cmos_coordinate_y = cw.SpinBoxWidget(0, 2048, 1, 0)
        self.QSpinBox_cmos_coordinate_nx = cw.SpinBoxWidget(0, 2048, 1, 2048)
        self.QSpinBox_cmos_coordinate_ny = cw.SpinBoxWidget(0, 2048, 1, 2048)
        self.QSpinBox_cmos_coordinate_bin = cw.SpinBoxWidget(0, 2048, 1, 1)
        self.QSpinBox_cmos_gain = cw.SpinBoxWidget(0, 300, 1, 0)
        self.QDoubleSpinBox_cmos_t_clean = cw.DoubleSpinBoxWidget(0, 10, 0.001, 5, 0.009)
        self.QDoubleSpinBox_cmos_t_exposure = cw.DoubleSpinBoxWidget(0, 10, 0.001, 5, 0.001)
        self.QDoubleSpinBox_cmos_t_standby = cw.DoubleSpinBoxWidget(0, 10, 0.001, 5, 0.050)

        cmos_scroll_layout.addRow(cw.LabelWidget(str('CMOS')))
        cmos_scroll_layout.addRow(cw.FrameWidget())
        cmos_scroll_layout.addRow(cw.LabelWidget(str('X')), self.QSpinBox_cmos_coordinate_x)
        cmos_scroll_layout.addRow(cw.LabelWidget(str('Y')), self.QSpinBox_cmos_coordinate_y)
        cmos_scroll_layout.addRow(cw.LabelWidget(str('Nx')), self.QSpinBox_cmos_coordinate_nx)
        cmos_scroll_layout.addRow(cw.LabelWidget(str('Ny')), self.QSpinBox_cmos_coordinate_ny)
        cmos_scroll_layout.addRow(cw.LabelWidget(str('Binx')), self.QSpinBox_cmos_coordinate_bin)
        cmos_scroll_layout.addRow(cw.LabelWidget(str('Gain')), self.QSpinBox_cmos_gain)
        cmos_scroll_layout.addRow(cw.LabelWidget(str('Clean / s')), self.QDoubleSpinBox_cmos_t_clean)
        cmos_scroll_layout.addRow(cw.LabelWidget(str('Exposure / s')), self.QDoubleSpinBox_cmos_t_exposure)
        cmos_scroll_layout.addRow(cw.LabelWidget(str('Standby / s')), self.QDoubleSpinBox_cmos_t_standby)

        group_layout = QHBoxLayout(group)
        group_layout.addWidget(emccd_scroll_area)
        group_layout.addWidget(scmos_scroll_area)
        group_layout.addWidget(cmos_scroll_area)
        group.setLayout(group_layout)
        return group

    def _create_position_panel(self):
        group = cw.GroupWidget()
        mad_deck_scroll_area, mad_deck_scroll_layout = cw.create_scroll_area()
        mcl_piezo_scroll_area, mcl_piezo_scroll_layout = cw.create_scroll_area("G")

        self.QLCDNumber_deck_position = cw.LCDNumberWidget()
        self.QPushButton_deck_position = cw.PushButtonWidget('Read')
        self.QPushButton_deck_position_zero = cw.PushButtonWidget('Zero')
        self.QPushButton_move_deck_up = cw.PushButtonWidget('Up')
        self.QPushButton_move_deck_down = cw.PushButtonWidget('Down')
        self.QSpinBox_deck_direction = cw.SpinBoxWidget(-1, 1, 2, 1)
        self.QDoubleSpinBox_deck_velocity = cw.DoubleSpinBoxWidget(0.02, 1.50, 0.02, 2, 0.02)
        self.QPushButton_move_deck = cw.PushButtonWidget('Move', checkable=True)

        mad_deck_scroll_layout.addRow(cw.LabelWidget(str('Mad Deck')))
        mad_deck_scroll_layout.addRow(cw.FrameWidget())
        mad_deck_scroll_layout.addRow(cw.LabelWidget(str('Position (mm)')), self.QLCDNumber_deck_position)
        mad_deck_scroll_layout.addRow(self.QPushButton_deck_position, self.QPushButton_deck_position_zero)
        mad_deck_scroll_layout.addRow(cw.LabelWidget(str('Direction (+up)')), self.QSpinBox_deck_direction)
        mad_deck_scroll_layout.addRow(cw.LabelWidget(str('Velocity (mm)')), self.QDoubleSpinBox_deck_velocity)
        mad_deck_scroll_layout.addRow(self.QPushButton_move_deck)
        mad_deck_scroll_layout.addRow(cw.LabelWidget(str('Single step')))
        mad_deck_scroll_layout.addRow(self.QPushButton_move_deck_up, self.QPushButton_move_deck_down)

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
        self.QDoubleSpinBox_piezo_return_time = cw.DoubleSpinBoxWidget(0, 50, 0.01, 2, 0.06)
        self.QPushButton_focus_finding = cw.PushButtonWidget('Find Focus')
        self.QPushButton_focus_locking = cw.PushButtonWidget('Lock Focus', checkable=True)
        self.QDoubleSpinBox_pid_kp = cw.DoubleSpinBoxWidget(0, 100, 0.01, 2, 0.5)
        self.QDoubleSpinBox_pid_ki = cw.DoubleSpinBoxWidget(0, 100, 0.01, 2, 0.5)
        self.QDoubleSpinBox_pid_kd = cw.DoubleSpinBoxWidget(0, 100, 0.01, 2, 0.0)

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
        mcl_piezo_scroll_layout.addWidget(cw.FrameWidget(), 13, 0, 1, 3)
        mcl_piezo_scroll_layout.addWidget(cw.LabelWidget(str('Piezo Return / s')), 14, 0)
        mcl_piezo_scroll_layout.addWidget(self.QDoubleSpinBox_piezo_return_time, 14, 1)
        mcl_piezo_scroll_layout.addWidget(cw.FrameWidget(), 15, 0, 1, 3)
        mcl_piezo_scroll_layout.addWidget(self.QPushButton_focus_finding, 16, 0)
        mcl_piezo_scroll_layout.addWidget(self.QPushButton_focus_locking, 16, 1)
        mcl_piezo_scroll_layout.addWidget(cw.LabelWidget(str('PID - kP')), 17, 0)
        mcl_piezo_scroll_layout.addWidget(cw.LabelWidget(str('PID - kI')), 17, 1)
        mcl_piezo_scroll_layout.addWidget(cw.LabelWidget(str('PID - kD')), 17, 2)
        mcl_piezo_scroll_layout.addWidget(self.QDoubleSpinBox_pid_kp, 18, 0)
        mcl_piezo_scroll_layout.addWidget(self.QDoubleSpinBox_pid_ki, 18, 1)
        mcl_piezo_scroll_layout.addWidget(self.QDoubleSpinBox_pid_kd, 18, 2)

        group_layout = QHBoxLayout(group)
        group_layout.addWidget(mad_deck_scroll_area)
        group_layout.addWidget(mcl_piezo_scroll_area)
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

        self.QSpinBox_daq_sample_rate = cw.SpinBoxWidget(100, 1250, 1, 250)
        self.QPushButton_plot_trigger = cw.PushButtonWidget("Plot Triggers")
        self.QPushButton_reset_daq = cw.PushButtonWidget("Reset")
        self.QDoubleSpinBox_ttl_start_on_405 = cw.DoubleSpinBoxWidget(0, 50, 0.001, 5, 0.008)
        self.QDoubleSpinBox_ttl_stop_on_405 = cw.DoubleSpinBoxWidget(0, 50, 0.001, 5, 0.032)
        self.QDoubleSpinBox_ttl_start_off_488 = cw.DoubleSpinBoxWidget(0, 50, 0.001, 5, 0.008)
        self.QDoubleSpinBox_ttl_stop_off_488 = cw.DoubleSpinBoxWidget(0, 50, 0.001, 5, 0.032)
        self.QDoubleSpinBox_ttl_start_read_488 = cw.DoubleSpinBoxWidget(0, 50, 0.001, 5, 0.008)
        self.QDoubleSpinBox_ttl_stop_read_488 = cw.DoubleSpinBoxWidget(0, 50, 0.001, 5, 0.032)
        self.QDoubleSpinBox_ttl_start_emccd = cw.DoubleSpinBoxWidget(0, 50, 0.001, 5, 0.008)
        self.QDoubleSpinBox_ttl_stop_emccd = cw.DoubleSpinBoxWidget(0, 50, 0.001, 5, 0.032)
        self.QDoubleSpinBox_ttl_start_scmos = cw.DoubleSpinBoxWidget(0, 50, 0.001, 5, 0.008)
        self.QDoubleSpinBox_ttl_stop_scmos = cw.DoubleSpinBoxWidget(0, 50, 0.001, 5, 0.032)
        self.QDoubleSpinBox_ttl_start_cmos = cw.DoubleSpinBoxWidget(0, 50, 0.001, 5, 0.008)
        self.QDoubleSpinBox_ttl_stop_cmos = cw.DoubleSpinBoxWidget(0, 50, 0.001, 5, 0.032)

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
        daq_scroll_layout.addWidget(cw.LabelWidget(str('DO#4 - iXon')), 0, 6, 1, 1)
        daq_scroll_layout.addWidget(self.QDoubleSpinBox_ttl_start_emccd, 1, 6, 1, 1)
        daq_scroll_layout.addWidget(self.QDoubleSpinBox_ttl_stop_emccd, 2, 6, 1, 1)
        daq_scroll_layout.addWidget(cw.LabelWidget(str('DO#6 - Kira')), 0, 8, 1, 1)
        daq_scroll_layout.addWidget(self.QDoubleSpinBox_ttl_start_scmos, 1, 8, 1, 1)
        daq_scroll_layout.addWidget(self.QDoubleSpinBox_ttl_stop_scmos, 2, 8, 1, 1)
        daq_scroll_layout.addWidget(cw.LabelWidget(str('DO#7 - FLIR')), 0, 9, 1, 1)
        daq_scroll_layout.addWidget(self.QDoubleSpinBox_ttl_start_cmos, 1, 9, 1, 1)
        daq_scroll_layout.addWidget(self.QDoubleSpinBox_ttl_stop_cmos, 2, 9, 1, 1)

        group_layout = QVBoxLayout(group)
        group_layout.addWidget(daq_scroll_area)
        group.setLayout(group_layout)
        return group

    def _create_slm_panel(self):
        group = cw.GroupWidget()
        slm_scroll_area, slm_scroll_layout = cw.create_scroll_area()

        self.QComboBox_slm_sequence = cw.ComboBoxWidget(list_items=[])

        slm_scroll_layout.addRow(cw.LabelWidget(str('SLM')), self.QComboBox_slm_sequence)

        group_layout = QHBoxLayout(group)
        group_layout.addWidget(slm_scroll_area)
        group.setLayout(group_layout)
        return group

    def _create_acquisition_panel(self):
        group = cw.GroupWidget()
        acq_scroll_area, acq_scroll_layout = cw.create_scroll_area("G")

        self.QComboBox_imaging_camera_selection = cw.ComboBoxWidget(list_items=["EMCCD", "sCMOS", "CMOS"])
        self.QComboBox_live_modes = cw.ComboBoxWidget(list_items=["Wide Field", "Point Scan", "Focus Lock"])
        self.QPushButton_video = cw.PushButtonWidget("Video", checkable=True)
        self.QPushButton_fft = cw.PushButtonWidget("FFT", checkable=True, enable=False)
        self.QComboBox_profile_axis = cw.ComboBoxWidget(list_items=["X", "Y"])
        self.QPushButton_plot_profile = cw.PushButtonWidget("Live Profile", checkable=True, enable=False)
        self.QPushButton_add_profile = cw.PushButtonWidget("Plot Profile")
        self.QPushButton_save_live_timing_presets = cw.PushButtonWidget("Save Live TTLs")
        self.QComboBox_acquisition_modes = cw.ComboBoxWidget(list_items=["Wide Field 2D", "Wide Field 3D",
                                                                         "Point Scan 2D", "Point Scan 3D"])
        self.QSpinBox_acquisition_number = cw.SpinBoxWidget(1, 999, 1, 1)
        self.QPushButton_acquire = cw.PushButtonWidget('Acquire')
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
        self.QPushButton_emccd_cooler_check.clicked.connect(self.check_emccd_temperature)
        self.QPushButton_emccd_cooler_switch.clicked.connect(self.switch_emccd_cooler)
        self.QDoubleSpinBox_stage_x.valueChanged.connect(self.set_piezo_x)
        self.QDoubleSpinBox_stage_y.valueChanged.connect(self.set_piezo_y)
        self.QDoubleSpinBox_stage_z.valueChanged.connect(self.set_piezo_z)
        self.QDoubleSpinBox_stage_x_usb.valueChanged.connect(self.set_piezo_x_usb)
        self.QDoubleSpinBox_stage_y_usb.valueChanged.connect(self.set_piezo_y_usb)
        self.QDoubleSpinBox_stage_z_usb.valueChanged.connect(self.set_piezo_z_usb)
        self.QPushButton_deck_position.clicked.connect(self.read_deck)
        self.QPushButton_deck_position_zero.clicked.connect(self.zero_deck)
        self.QPushButton_move_deck_up.clicked.connect(self.deck_move_up)
        self.QPushButton_move_deck_down.clicked.connect(self.deck_move_down)
        self.QPushButton_move_deck.clicked.connect(self.deck_move_range)
        self.QPushButton_laser_488_0.clicked.connect(self.set_laser_488_0)
        self.QPushButton_laser_488_1.clicked.connect(self.set_laser_488_1)
        self.QPushButton_laser_405.clicked.connect(self.set_laser_405)
        self.QSpinBox_daq_sample_rate.valueChanged.connect(self.update_daq)
        self.QPushButton_reset_daq.clicked.connect(self.reset_daq)
        self.QPushButton_plot_trigger.clicked.connect(self.plot_trigger_sequence)
        self.QPushButton_focus_finding.clicked.connect(self.run_focus_finding)
        self.QPushButton_focus_locking.clicked.connect(self.run_focus_locking)
        self.QPushButton_video.clicked.connect(self.run_video)
        self.QPushButton_fft.clicked.connect(self.run_fft)
        self.QPushButton_plot_profile.clicked.connect(self.run_plot_profile)
        self.QPushButton_add_profile.clicked.connect(self.run_add_profile)
        self.QPushButton_acquire.clicked.connect(self.run_acquisition)
        self.QComboBox_live_modes.currentIndexChanged.connect(self.load_selected_digital_timing_presets)
        self.QComboBox_acquisition_modes.currentIndexChanged.connect(self.load_selected_digital_timing_presets)
        self.QPushButton_save_live_timing_presets.clicked.connect(lambda: self.save_digital_timing_preset("live"))
        self.QPushButton_save_acquisition_timing_presets.clicked.connect(
            lambda: self.save_digital_timing_preset("acquisition"))

    @pyqtSlot()
    def check_emccd_temperature(self):
        self.Signal_check_emccd_temperature.emit()

    @pyqtSlot(bool)
    def switch_emccd_cooler(self, checked: bool):
        self.Signal_switch_emccd_cooler.emit(checked)
        if checked:
            self.QPushButton_emccd_cooler_switch.setText("Cooler ON")
        else:
            self.QPushButton_emccd_cooler_switch.setText("Cooler OFF")

    def get_emccd_roi(self):
        return [self.QSpinBox_emccd_coordinate_x.value(), self.QSpinBox_emccd_coordinate_y.value(),
                self.QSpinBox_emccd_coordinate_nx.value(), self.QSpinBox_emccd_coordinate_ny.value(),
                self.QSpinBox_emccd_coordinate_bin.value()]

    def get_emccd_gain(self):
        return self.QSpinBox_emccd_gain.value()

    def display_camera_temperature(self, temperature):
        self.QLCDNumber_ccd_tempetature.display(temperature)

    def display_camera_timings(self, clean=None, exposure=None, standby=None):
        if clean is not None:
            self.QDoubleSpinBox_emccd_t_clean.setValue(clean)
        if exposure is not None:
            self.QDoubleSpinBox_emccd_t_exposure.setValue(exposure)
        if standby is not None:
            self.QDoubleSpinBox_emccd_t_standby.setValue(standby)

    def get_scmos_roi(self):
        return [self.QSpinBox_scmos_coordinate_x.value(), self.QSpinBox_scmos_coordinate_y.value(),
                self.QSpinBox_scmos_coordinate_nx.value(), self.QSpinBox_scmos_coordinate_ny.value(),
                self.QSpinBox_scmos_coordinate_bin.value()]

    def get_scmos_gain(self):
        return self.QSpinBox_scmos_gain.value()

    def get_cmos_roi(self):
        return [self.QSpinBox_cmos_coordinate_x.value(), self.QSpinBox_cmos_coordinate_y.value(),
                self.QSpinBox_cmos_coordinate_nx.value(), self.QSpinBox_cmos_coordinate_ny.value(),
                self.QSpinBox_cmos_coordinate_bin.value()]

    def get_cmos_gain(self):
        return self.QSpinBox_cmos_gain.value()

    def get_imaging_camera(self):
        detection_device = self.QComboBox_imaging_camera_selection.currentIndex()
        return detection_device

    @pyqtSlot()
    def read_deck(self):
        self.Signal_deck_read_position.emit()

    @pyqtSlot()
    def zero_deck(self):
        self.Signal_deck_zero_position.emit()

    @pyqtSlot()
    def deck_move_up(self):
        self.Signal_deck_move_single_step.emit(True)

    @pyqtSlot()
    def deck_move_down(self):
        self.Signal_deck_move_single_step.emit(False)

    @pyqtSlot(bool)
    def deck_move_range(self, checked: bool):
        distance = self.QSpinBox_deck_direction.value()
        velocity = self.QDoubleSpinBox_deck_velocity.value()
        self.Signal_deck_move_continuous.emit(checked, distance, velocity)

    def display_deck_position(self, mdposz):
        self.QLCDNumber_deck_position.display(mdposz)

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

    def get_piezo_return_time(self):
        return self.QDoubleSpinBox_piezo_return_time.value()

    def get_piezo_scan_parameters(self):
        axis_lengths = [self.QDoubleSpinBox_range_x.value(), self.QDoubleSpinBox_range_y.value(),
                        self.QDoubleSpinBox_range_z.value()]
        step_sizes = [self.QDoubleSpinBox_step_x.value(), self.QDoubleSpinBox_step_y.value(),
                      self.QDoubleSpinBox_step_z.value()]
        return axis_lengths, step_sizes

    def get_pid_parameters(self):
        return (self.QDoubleSpinBox_pid_kp.value(),
                self.QDoubleSpinBox_pid_ki.value(),
                self.QDoubleSpinBox_pid_kd.value())

    def display_piezo_position_x(self, ps):
        self.QLCDNumber_piezo_position_x.display(ps)

    def display_piezo_position_y(self, ps):
        self.QLCDNumber_piezo_position_y.display(ps)

    def display_piezo_position_z(self, ps):
        self.QLCDNumber_piezo_position_z.display(ps)

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
            self.QPushButton_fft.setEnabled(True)
            self.QPushButton_plot_profile.setEnabled(True)
        else:
            self.Signal_video.emit(False, vm)
            if self.QPushButton_fft.isChecked():
                self.Signal_fft.emit(False)
            self.QPushButton_fft.setEnabled(False)
            self.QPushButton_fft.setChecked(False)
            if self.QPushButton_plot_profile.isChecked():
                self.Signal_plot_profile.emit(False)
            self.QPushButton_plot_profile.setEnabled(False)
            self.QPushButton_plot_profile.setChecked(False)

    @pyqtSlot()
    def run_fft(self):
        if self.QPushButton_fft.isChecked():
            self.Signal_fft.emit(True)
        else:
            self.Signal_fft.emit(False)

    @pyqtSlot(bool)
    def run_plot_profile(self, checked: bool):
        self.Signal_plot_profile.emit(checked)

    @pyqtSlot()
    def run_add_profile(self):
        self.Signal_add_profile.emit()

    @pyqtSlot()
    def run_acquisition(self):
        acq_mode = self.QComboBox_acquisition_modes.currentText()
        acq_num = self.QSpinBox_acquisition_number.value()
        self.Signal_data_acquire.emit(acq_mode, acq_num)

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
        self.QDoubleSpinBox_ttl_start_on_405.setValue(values.get("QDoubleSpinBox_ttl_start_on_405", 0))
        self.QDoubleSpinBox_ttl_stop_on_405.setValue(values.get("QDoubleSpinBox_ttl_stop_on_405", 0))
        self.QDoubleSpinBox_ttl_start_off_488.setValue(values.get("QDoubleSpinBox_ttl_start_off_488", 0))
        self.QDoubleSpinBox_ttl_stop_off_488.setValue(values.get("QDoubleSpinBox_ttl_stop_off_488", 0))
        self.QDoubleSpinBox_ttl_start_read_488.setValue(values.get("QDoubleSpinBox_ttl_start_read_488", 0))
        self.QDoubleSpinBox_ttl_stop_read_488.setValue(values.get("QDoubleSpinBox_ttl_stop_read_488", 0))
        self.QDoubleSpinBox_ttl_start_emccd.setValue(values.get("QDoubleSpinBox_ttl_start_emccd", 0))
        self.QDoubleSpinBox_ttl_stop_emccd.setValue(values.get("QDoubleSpinBox_ttl_stop_emccd", 0))
        self.QDoubleSpinBox_ttl_start_scmos.setValue(values.get("QDoubleSpinBox_ttl_start_scmos", 0))
        self.QDoubleSpinBox_ttl_stop_scmos.setValue(values.get("QDoubleSpinBox_ttl_stop_scmos", 0))
        self.QDoubleSpinBox_ttl_start_cmos.setValue(values.get("QDoubleSpinBox_ttl_start_cmos", 0))
        self.QDoubleSpinBox_ttl_stop_cmos.setValue(values.get("QDoubleSpinBox_ttl_stop_cmos", 0))

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
                "QDoubleSpinBox_ttl_start_on_405": self.QDoubleSpinBox_ttl_start_on_405.value(),
                "QDoubleSpinBox_ttl_stop_on_405": self.QDoubleSpinBox_ttl_stop_on_405.value(),
                "QDoubleSpinBox_ttl_start_off_488": self.QDoubleSpinBox_ttl_start_off_488.value(),
                "QDoubleSpinBox_ttl_stop_off_488": self.QDoubleSpinBox_ttl_stop_off_488.value(),
                "QDoubleSpinBox_ttl_start_read_488": self.QDoubleSpinBox_ttl_start_read_488.value(),
                "QDoubleSpinBox_ttl_stop_read_488": self.QDoubleSpinBox_ttl_stop_read_488.value(),
                "QDoubleSpinBox_ttl_start_emccd": self.QDoubleSpinBox_ttl_start_emccd.value(),
                "QDoubleSpinBox_ttl_stop_emccd": self.QDoubleSpinBox_ttl_stop_emccd.value(),
                "QDoubleSpinBox_ttl_start_scmos": self.QDoubleSpinBox_ttl_start_scmos.value(),
                "QDoubleSpinBox_ttl_stop_scmos": self.QDoubleSpinBox_ttl_stop_scmos.value(),
                "QDoubleSpinBox_ttl_start_cmos": self.QDoubleSpinBox_ttl_start_cmos.value(),
                "QDoubleSpinBox_ttl_stop_cmos": self.QDoubleSpinBox_ttl_stop_cmos.value(),
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

    def _save_spinbox_values(self):
        values = {}
        for name in dir(self):
            obj = getattr(self, name)
            if isinstance(obj, (QSpinBox, QDoubleSpinBox)):
                values[name] = obj.value()
        with open(self.config["ConWidget Path"], 'w') as f:
            json.dump(values, f, indent=4)

    def _load_spinbox_values(self):
        try:
            with open(self.config["ConWidget Path"], 'r') as f:
                values = json.load(f)
            for name, value in values.items():
                widget = getattr(self, name, None)
                if widget is not None:
                    widget.setValue(value)
        except FileNotFoundError:
            pass
