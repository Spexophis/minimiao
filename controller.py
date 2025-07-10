from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget, QFormLayout, QHBoxLayout, QVBoxLayout, QGridLayout, QSpinBox, QDoubleSpinBox

import custom_widgets as cw


class ControlPanel(QWidget):
    startClicked = pyqtSignal()
    stopClicked = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_ui()
        self._set_signal_connections()
        # self._load_spinbox_values()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        self.camera_panel = self._create_camera_panel()
        # self.position_panel = self._create_position_panel()
        # self.laser_panel = self._create_laser_panel()
        # self.daq_panel = self._create_daq_panel()

        main_layout.addWidget(self.camera_panel)
        # main_layout.addWidget(self.position_panel)
        # main_layout.addWidget(self.laser_panel)
        # main_layout.addWidget(self.daq_panel)

        main_layout.addStretch(1)
        self.setLayout(main_layout)

    def _create_camera_panel(self):
        group = cw.GroupWidget()
        emccd_scroll_area, emccd_scroll_layout = cw.create_scroll_area()

        self.QLCDNumber_ccd_tempetature = cw.LCDNumberWidget(0, 3)
        self.QPushButton_emccd_cooler_check = cw.PushButtonWidget('Check', False, True)
        self.QPushButton_emccd_cooler_switch = cw.PushButtonWidget('Cooler OFF', True, True, True)
        self.QSpinBox_emccd_coordinate_x = cw.SpinBoxWidget(0, 1024, 1, 1)
        self.QSpinBox_emccd_coordinate_y = cw.SpinBoxWidget(0, 1024, 1, 1)
        self.QSpinBox_emccd_coordinate_nx = cw.SpinBoxWidget(0, 1024, 1, 1024)
        self.QSpinBox_emccd_coordinate_ny = cw.SpinBoxWidget(0, 1024, 1, 1024)
        self.QSpinBox_emccd_coordinate_binx = cw.SpinBoxWidget(0, 1024, 1, 1)
        self.QSpinBox_emccd_coordinate_biny = cw.SpinBoxWidget(0, 1024, 1, 1)
        self.QSpinBox_emccd_gain = cw.SpinBoxWidget(0, 300, 1, 0)
        self.QDoubleSpinBox_emccd_t_clean = cw.DoubleSpinBoxWidget(0, 10, 0.001, 5, 0.009)
        self.QDoubleSpinBox_emccd_exposure_time = cw.DoubleSpinBoxWidget(0, 10, 0.001, 5, 0.001)
        self.QDoubleSpinBox_emccd_t_standby = cw.DoubleSpinBoxWidget(0, 10, 0.001, 5, 0.050)
        self.QDoubleSpinBox_emccd_gvs = cw.DoubleSpinBoxWidget(-5., 5., 0.01, 2, 5.)

        emccd_scroll_layout.addRow(cw.LabelWidget(str('EMCCD')))
        emccd_scroll_layout.addRow(cw.FrameWidget())
        emccd_scroll_layout.addRow(cw.LabelWidget(str('Temperature')), self.QLCDNumber_ccd_tempetature)
        emccd_scroll_layout.addRow(self.QPushButton_emccd_cooler_check, self.QPushButton_emccd_cooler_switch)
        emccd_scroll_layout.addRow(cw.LabelWidget(str('X')), self.QSpinBox_emccd_coordinate_x)
        emccd_scroll_layout.addRow(cw.LabelWidget(str('Y')), self.QSpinBox_emccd_coordinate_y)
        emccd_scroll_layout.addRow(cw.LabelWidget(str('Nx')), self.QSpinBox_emccd_coordinate_nx)
        emccd_scroll_layout.addRow(cw.LabelWidget(str('Ny')), self.QSpinBox_emccd_coordinate_ny)
        emccd_scroll_layout.addRow(cw.LabelWidget(str('Binx')), self.QSpinBox_emccd_coordinate_binx)
        emccd_scroll_layout.addRow(cw.LabelWidget(str('Biny')), self.QSpinBox_emccd_coordinate_biny)
        emccd_scroll_layout.addRow(cw.LabelWidget(str('EMGain')), self.QSpinBox_emccd_gain)
        emccd_scroll_layout.addRow(cw.LabelWidget(str('Clean / s')), self.QDoubleSpinBox_emccd_t_clean)
        emccd_scroll_layout.addRow(cw.LabelWidget(str('Exposure / s')), self.QDoubleSpinBox_emccd_exposure_time)
        emccd_scroll_layout.addRow(cw.LabelWidget(str('Standby / s')), self.QDoubleSpinBox_emccd_t_standby)
        emccd_scroll_layout.addRow(cw.LabelWidget(str('GalvoSW')), self.QDoubleSpinBox_emccd_gvs)

        # Set the scroll area as the only widget in the group
        group_layout = QHBoxLayout(group)
        group_layout.addWidget(emccd_scroll_area)
        group.setLayout(group_layout)
        return group

    # def _create_position_panel(self):
    #     group = cw.GroupWidget("Position")
    #     layout = QVBoxLayout(group)
    #
    #
    #
    #     group.setLayout(layout)
    #     return group
    #
    # def _create_laser_panel(self):
    #     group = cw.GroupWidget("Laser")
    #     layout = QVBoxLayout(group)
    #
    #
    #
    #     group.setLayout(layout)
    #     return group
    #
    # def _create_daq_panel(self):
    #     group = cw.GroupWidget("DAQ")
    #     layout = QVBoxLayout(group)
    #
    #
    #
    #     group.setLayout(layout)
    #     return group

    def _set_signal_connections(self):
        pass

    # def _save_spinbox_values(self):
    #     values = {}
    #     for name in dir(self):
    #         obj = getattr(self, name)
    #         if isinstance(obj, (QSpinBox, QDoubleSpinBox)):
    #             values[name] = obj.value()
    #     self.config.write_config(values, self.config.configs["ControlPanel Path"])
    #
    # def _load_spinbox_values(self):
    #     try:
    #         values = self.config.load_config(self.config.configs["ControlPanel Path"])
    #         for name, value in values.items():
    #             widget = getattr(self, name, None)
    #             if widget is not None:
    #                 widget.setValue(value)
    #     except FileNotFoundError:
    #         pass
