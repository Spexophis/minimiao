# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import json

from PyQt6.QtCore import pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QSpinBox, QDoubleSpinBox

from . import custom_widgets as cw


class AOPanel(QWidget):
    Signal_set_zernike = pyqtSignal(int, float)
    Signal_set_dm = pyqtSignal(int)
    Signal_set_dm_flat = pyqtSignal()
    Signal_update_cmd = pyqtSignal()
    Signal_load_dm = pyqtSignal()
    Signal_save_dm = pyqtSignal()
    Signal_sensorlessAO = pyqtSignal()

    def __init__(self, config, logg, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.config = config
        self.logg = logg
        self._setup_ui()
        self.load_spinbox_values()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        self.dm_panel = self._create_dm_panel()
        self.sensorless_panel = self._create_sensorless_panel()

        main_layout.addWidget(self.dm_panel)
        main_layout.addWidget(self.sensorless_panel)

        main_layout.addStretch(1)
        self.setLayout(main_layout)

    def _create_dm_panel(self):
        group = cw.GroupWidget()
        dm_scroll_area, dm_scroll_layout = cw.create_scroll_area("G")

        self.QComboBox_wfsmd = cw.ComboBoxWidget(list_items=['zonal', 'modal'])
        self.QSpinBox_zernike_mode = cw.SpinBoxWidget(0, 100, 1, 0)
        self.QDoubleSpinBox_zernike_mode_amp = cw.DoubleSpinBoxWidget(-20, 20, 0.01, 2, 0)
        self.QPushButton_set_zernike_mode = cw.PushButtonWidget('Set Zernike')
        self.QComboBox_cmd = cw.ComboBoxWidget(list_items=['0', '1'])
        self.QComboBox_cmd.setCurrentIndex(1)
        self.QPushButton_set_dm = cw.PushButtonWidget('Set DM')
        self.QPushButton_load_dm = cw.PushButtonWidget('Load DM')
        self.QPushButton_update_cmd = cw.PushButtonWidget('Add DM')
        self.QPushButton_save_dm = cw.PushButtonWidget('Save DM')
        self.QPushButton_change_dm_flat = cw.PushButtonWidget('Save Flat')

        dm_scroll_layout.addWidget(cw.LabelWidget(str('Deformable Device')), 0, 0, 1, 2)
        dm_scroll_layout.addWidget(cw.LabelWidget(str('CMDs')), 1, 0, 1, 1)
        dm_scroll_layout.addWidget(self.QComboBox_cmd, 1, 1, 1, 1)
        dm_scroll_layout.addWidget(self.QPushButton_set_dm, 2, 1, 1, 1)
        dm_scroll_layout.addWidget(self.QPushButton_load_dm, 3, 0, 1, 1)
        dm_scroll_layout.addWidget(self.QPushButton_change_dm_flat, 3, 1, 1, 1)
        dm_scroll_layout.addWidget(cw.LabelWidget(str('Method')), 0, 2, 1, 1)
        dm_scroll_layout.addWidget(self.QComboBox_wfsmd, 0, 3, 1, 1)
        dm_scroll_layout.addWidget(cw.LabelWidget(str('Zernike Mode')), 1, 2, 1, 1)
        dm_scroll_layout.addWidget(self.QSpinBox_zernike_mode, 1, 3, 1, 1)
        dm_scroll_layout.addWidget(cw.LabelWidget(str('Amplitude')), 2, 2, 1, 1)
        dm_scroll_layout.addWidget(self.QDoubleSpinBox_zernike_mode_amp, 2, 3, 1, 1)
        dm_scroll_layout.addWidget(self.QPushButton_set_zernike_mode, 3, 2, 1, 1)
        dm_scroll_layout.addWidget(self.QPushButton_update_cmd, 3, 3, 1, 1)

        group_layout = QHBoxLayout(group)
        group_layout.addWidget(dm_scroll_area)
        group.setLayout(group_layout)
        return group

    def _create_sensorless_panel(self):
        group = cw.GroupWidget()
        sensorless_scroll_area, sensorless_scroll_layout = cw.create_scroll_area("G")

        self.QSpinBox_zernike_mode_start = cw.SpinBoxWidget(1, 64, 1, 4)
        self.QSpinBox_zernike_mode_stop = cw.SpinBoxWidget(1, 64, 1, 10)
        self.QDoubleSpinBox_zernike_mode_amps_start = cw.DoubleSpinBoxWidget(-50, 50, 0.005, 3, -0.01)
        self.QSpinBox_zernike_mode_amps_stepnum = cw.SpinBoxWidget(0, 50, 2, 3)
        self.QDoubleSpinBox_zernike_mode_amps_step = cw.DoubleSpinBoxWidget(-50, 50, 0.005, 3, 0.01)
        self.QDoubleSpinBox_lpf = cw.DoubleSpinBoxWidget(0, 1, 0.05, 2, 0.1)
        self.QDoubleSpinBox_hpf = cw.DoubleSpinBoxWidget(0, 1, 0.05, 2, 0.6)
        self.QComboBox_metric = cw.ComboBoxWidget(list_items=['Max(Intensity)', 'Sum(Intensity)', 'Mask(Intensity)',
                                                              'SNR(FFT)', 'HighPass(FFT)', 'Selected(FFT)'])
        self.QDoubleSpinBox_select_frequency = cw.DoubleSpinBoxWidget(0, 50, 0.001, 3, 1.410)
        self.QPushButton_sensorless_run = cw.PushButtonWidget('Run AO')
        self.QRadioButton_sensorless_error = cw.RadioButtonWidget('ErrorIn')

        sensorless_scroll_layout.addWidget(cw.LabelWidget(str('Sensorless AO')), 0, 0, 1, 2)
        sensorless_scroll_layout.addWidget(cw.LabelWidget(str('Zernike Modes')), 1, 0, 1, 2)
        sensorless_scroll_layout.addWidget(cw.LabelWidget(str('From')), 2, 0, 1, 1)
        sensorless_scroll_layout.addWidget(self.QSpinBox_zernike_mode_start, 2, 1, 1, 1)
        sensorless_scroll_layout.addWidget(cw.LabelWidget(str('To')), 3, 0, 1, 1)
        sensorless_scroll_layout.addWidget(self.QSpinBox_zernike_mode_stop, 3, 1, 1, 1)
        sensorless_scroll_layout.addWidget(self.QRadioButton_sensorless_error, 4, 0, 1, 1)
        sensorless_scroll_layout.addWidget(self.QPushButton_sensorless_run, 5, 0, 1, 1)
        sensorless_scroll_layout.addWidget(cw.LabelWidget(str('Amplitudes')), 0, 2, 1, 2)
        sensorless_scroll_layout.addWidget(cw.LabelWidget(str('From')), 1, 2, 1, 1)
        sensorless_scroll_layout.addWidget(self.QDoubleSpinBox_zernike_mode_amps_start, 1, 3, 1, 1)
        sensorless_scroll_layout.addWidget(cw.LabelWidget(str('StepNum')), 2, 2, 1, 1)
        sensorless_scroll_layout.addWidget(self.QSpinBox_zernike_mode_amps_stepnum, 2, 3, 1, 1)
        sensorless_scroll_layout.addWidget(cw.LabelWidget(str('StepSize')), 3, 2, 1, 1)
        sensorless_scroll_layout.addWidget(self.QDoubleSpinBox_zernike_mode_amps_step, 3, 3, 1, 1)
        sensorless_scroll_layout.addWidget(cw.LabelWidget(str('Image Metric')), 4, 2, 1, 1)
        sensorless_scroll_layout.addWidget(self.QComboBox_metric, 5, 2, 1, 1)
        sensorless_scroll_layout.addWidget(cw.LabelWidget(str('LPF')), 0, 4, 1, 1)
        sensorless_scroll_layout.addWidget(self.QDoubleSpinBox_lpf, 1, 4, 1, 1)
        sensorless_scroll_layout.addWidget(cw.LabelWidget(str('HPF')), 2, 4, 1, 1)
        sensorless_scroll_layout.addWidget(self.QDoubleSpinBox_hpf, 3, 4, 1, 1)
        sensorless_scroll_layout.addWidget(cw.LabelWidget(str('Select')), 4, 4, 1, 1)
        sensorless_scroll_layout.addWidget(self.QDoubleSpinBox_select_frequency, 5, 4, 1, 1)

        group_layout = QHBoxLayout(group)
        group_layout.addWidget(sensorless_scroll_area)
        group.setLayout(group_layout)
        return group

    def _set_signal_connections(self):
        self.QPushButton_set_dm.clicked.connect(self.set_dm_cmd)
        self.QPushButton_set_zernike_mode.clicked.connect(self.set_dm_zernike)
        self.QPushButton_update_cmd.clicked.connect(self.update_dm_cmd)
        self.QPushButton_load_dm.clicked.connect(self.load_dm_file)
        self.QPushButton_save_dm.clicked.connect(self.save_dm_cmd)
        self.QPushButton_change_dm_flat.clicked.connect(self.change_dm_flat)
        self.QPushButton_sensorless_run.clicked.connect(self.run_sensorless_correction)

    @pyqtSlot()
    def set_dm_zernike(self):
        mode, amp = self.get_zernike_mode()
        self.Signal_set_zernike.emit(mode, amp)

    @pyqtSlot()
    def set_dm_cmd(self):
        i = self.get_cmd_index()
        self.Signal_set_dm.emit(i)
        
    @pyqtSlot()
    def update_dm_cmd(self):
        self.Signal_update_cmd.emit()

    @pyqtSlot()
    def change_dm_flat(self):
        self.Signal_set_dm_flat.emit()

    @pyqtSlot()
    def load_dm_file(self):
        self.Signal_load_dm.emit()

    @pyqtSlot()
    def save_dm_cmd(self):
        self.Signal_save_dm.emit()

    def get_zernike_mode(self):
        return self.QSpinBox_zernike_mode.value(), self.QDoubleSpinBox_zernike_mode_amp.value()

    def get_cmd_index(self):
        return self.QComboBox_cmd.currentIndex()

    def update_cmd_index(self, wst=True):
        item = '{}'.format(self.QComboBox_cmd.count())
        self.QComboBox_cmd.addItem(item)
        if wst:
            self.QComboBox_cmd.setCurrentIndex(self.QComboBox_cmd.count() - 1)

    @pyqtSlot()
    def run_sensorless_correction(self):
        self.Signal_sensorlessAO.emit()

    def get_sensorless_iteration(self):
        return (self.QSpinBox_zernike_mode_start.value(), self.QSpinBox_zernike_mode_stop.value(),
                self.QDoubleSpinBox_zernike_mode_amps_start.value(), self.QDoubleSpinBox_zernike_mode_amps_step.value(),
                self.QSpinBox_zernike_mode_amps_stepnum.value())

    def get_sensorless_parameters(self):
        return (self.QDoubleSpinBox_lpf.value(), self.QDoubleSpinBox_hpf.value(),
                self.QDoubleSpinBox_select_frequency.value(), self.QComboBox_metric.currentText(),
                self.QRadioButton_sensorless_error.isChecked())

    def save_spinbox_values(self):
        values = {}
        for name in dir(self):
            obj = getattr(self, name)
            if isinstance(obj, (QSpinBox, QDoubleSpinBox)):
                values[name] = obj.value()
        with open(self.config["AOWidget Path"], 'w') as f:
            json.dump(values, f, indent=4)

    def load_spinbox_values(self):
        try:
            with open(self.config["AOWidget Path"], 'r') as f:
                values = json.load(f)
            for name, value in values.items():
                widget = getattr(self, name, None)
                if widget is not None:
                    widget.setValue(value)
        except FileNotFoundError:
            pass
