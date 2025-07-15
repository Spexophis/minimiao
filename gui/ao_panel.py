import json

from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt6.QtWidgets import QWidget, QSplitter, QHBoxLayout, QVBoxLayout, QSpinBox, QDoubleSpinBox

from gui import custom_widgets as cw


class AOPanel(QWidget):
    Signal_img_shwfs_base = pyqtSignal()
    Signal_img_wfs = pyqtSignal(bool)
    Signal_img_shwfr_run = pyqtSignal()
    Signal_img_shwfs_compute_wf = pyqtSignal()
    Signal_img_shwfs_correct_wf = pyqtSignal(int)
    Signal_img_shwfs_save_wf = pyqtSignal()
    Signal_img_shwfs_acquisition = pyqtSignal()
    Signal_dm_selection = pyqtSignal(str)
    Signal_push_actuator = pyqtSignal(int, float)
    Signal_influence_function = pyqtSignal()
    Signal_set_zernike = pyqtSignal()
    Signal_set_dm = pyqtSignal()
    Signal_set_dm_flat = pyqtSignal()
    Signal_update_cmd = pyqtSignal()
    Signal_load_dm = pyqtSignal()
    Signal_save_dm = pyqtSignal()
    Signal_sensorlessAO_run = pyqtSignal()
    Signal_sensorlessAO_auto = pyqtSignal()
    Signal_sensorlessAO_metric_acquisition = pyqtSignal()
    Signal_sensorlessAO_ml_acquisition = pyqtSignal()

    def __init__(self, bus, config, logg, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.bus = bus
        self.config = config
        self.logg = logg
        self._setup_ui()
        self.QComboBox_wfs_camera_selection.setCurrentIndex(1)
        self.load_spinbox_values()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        self.image_panel = self._create_image_panel()
        self.parameter_panel = self._create_parameter_panel()
        self.shwfs_panel = self._create_shwfs_panel()
        self.dwfs_panel = self._create_dwfs_panel()
        self.dm_panel = self._create_dm_panel()
        self.sensorless_panel = self._create_sensorless_panel()

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.image_panel)
        splitter.addWidget(self.parameter_panel)
        splitter.addWidget(self.shwfs_panel)
        splitter.addWidget(self.dwfs_panel)
        splitter.addWidget(self.dm_panel)
        splitter.addWidget(self.sensorless_panel)

        main_layout.addWidget(splitter)
        main_layout.addStretch(1)
        self.setLayout(main_layout)

    def _create_image_panel(self):
        group = cw.GroupWidget()
        image_shwfs_scroll_area, image_shwfs_scroll_layout = cw.create_scroll_area()

        self.lcdNumber_wfmax_img = cw.LCDNumberWidget()
        self.lcdNumber_wfmin_img = cw.LCDNumberWidget()
        self.lcdNumber_wfrms_img = cw.LCDNumberWidget()

        image_shwfs_scroll_layout.addRow(cw.LabelWidget(str('Wavefront MAX')), self.lcdNumber_wfmax_img)
        image_shwfs_scroll_layout.addRow(cw.LabelWidget(str('Wavefront MIN')), self.lcdNumber_wfmin_img)
        image_shwfs_scroll_layout.addRow(cw.LabelWidget(str('Wavefront RMS')), self.lcdNumber_wfrms_img)

        group_layout = QHBoxLayout(group)
        group_layout.addWidget(image_shwfs_scroll_area)
        group.setLayout(group_layout)
        return group

    def _create_parameter_panel(self):
        group = cw.GroupWidget()
        confocal_shwfs_parameters_scroll_area, confocal_shwfs_parameters_scroll_layout = cw.create_scroll_area()

        self.QLabel_wfrmd_foc = cw.LabelWidget(str('Method'))
        self.QComboBox_wfrmd_foc = cw.ComboBoxWidget(list_items=['correlation', 'centerofmass'])
        self.QSpinBox_base_xcenter_foc = cw.SpinBoxWidget(0, 2048, 1, 1024)
        self.QSpinBox_base_ycenter_foc = cw.SpinBoxWidget(0, 2048, 1, 1024)
        self.QSpinBox_offset_xcenter_foc = cw.SpinBoxWidget(0, 2048, 1, 1024)
        self.QSpinBox_offset_ycenter_foc = cw.SpinBoxWidget(0, 2048, 1, 1024)
        self.QSpinBox_n_lenslets_x_foc = cw.SpinBoxWidget(0, 64, 1, 14)
        self.QSpinBox_n_lenslets_y_foc = cw.SpinBoxWidget(0, 64, 1, 14)
        self.QSpinBox_spacing_foc = cw.SpinBoxWidget(0, 64, 1, 26)
        self.QSpinBox_radius_foc = cw.SpinBoxWidget(0, 64, 1, 12)
        self.QDoubleSpinBox_foc_background = cw.DoubleSpinBoxWidget(0, 1, 0.01, 2, 0.1)

        confocal_shwfs_parameters_scroll_layout.addRow(cw.LabelWidget(str('Illumination')))
        confocal_shwfs_parameters_scroll_layout.addRow(cw.FrameWidget())
        confocal_shwfs_parameters_scroll_layout.addRow(cw.LabelWidget(str('Method')), self.QComboBox_wfrmd_foc)
        confocal_shwfs_parameters_scroll_layout.addRow(cw.LabelWidget(str('X_center (Base)')),
                                                       self.QSpinBox_base_xcenter_foc)
        confocal_shwfs_parameters_scroll_layout.addRow(cw.LabelWidget(str('Y_center (Base)')),
                                                       self.QSpinBox_base_ycenter_foc)
        confocal_shwfs_parameters_scroll_layout.addRow(cw.LabelWidget(str('X_center (Offset)')),
                                                       self.QSpinBox_offset_xcenter_foc)
        confocal_shwfs_parameters_scroll_layout.addRow(cw.LabelWidget(str('Y_center (Offset)')),
                                                       self.QSpinBox_offset_ycenter_foc)
        confocal_shwfs_parameters_scroll_layout.addRow(cw.LabelWidget(str('Lenslet X')),
                                                       self.QSpinBox_n_lenslets_x_foc)
        confocal_shwfs_parameters_scroll_layout.addRow(cw.LabelWidget(str('Lenslet Y')),
                                                       self.QSpinBox_n_lenslets_y_foc)
        confocal_shwfs_parameters_scroll_layout.addRow(cw.LabelWidget(str('Spacing')), self.QSpinBox_spacing_foc)
        confocal_shwfs_parameters_scroll_layout.addRow(cw.LabelWidget(str('Radius')), self.QSpinBox_radius_foc)
        confocal_shwfs_parameters_scroll_layout.addRow(cw.LabelWidget(str('Background')),
                                                       self.QDoubleSpinBox_foc_background)

        group_layout = QHBoxLayout(group)
        group_layout.addWidget(confocal_shwfs_parameters_scroll_area)
        group.setLayout(group_layout)
        return group

    def _create_shwfs_panel(self):
        group = cw.GroupWidget()
        image_shwfs_scroll_area, image_shwfs_scroll_layout = cw.create_scroll_area()

        self.QComboBox_wfs_camera_selection = cw.ComboBoxWidget(list_items=["EMCCD", "SCMOS", "Thorlabs"])
        self.QPushButton_img_shwfs_base = cw.PushButtonWidget('SetBase', enable=True)
        self.QPushButton_run_img_wfs = cw.PushButtonWidget('RunWFS', checkable=True)
        self.QPushButton_run_img_wfr = cw.PushButtonWidget('RunWFR', enable=True)
        self.QPushButton_img_shwfs_compute_wf = cw.PushButtonWidget('ComputeWF', enable=True)
        self.QPushButton_img_shwfs_save_wf = cw.PushButtonWidget('SaveWF', enable=True)
        self.QPushButton_img_shwfs_acquisition = cw.PushButtonWidget('ACQ')

        image_shwfs_scroll_layout.addRow(cw.LabelWidget(str('Camera')), self.QComboBox_wfs_camera_selection)
        image_shwfs_scroll_layout.addRow(self.QPushButton_run_img_wfs, self.QPushButton_img_shwfs_base)
        image_shwfs_scroll_layout.addRow(self.QPushButton_run_img_wfr, self.QPushButton_img_shwfs_acquisition)
        image_shwfs_scroll_layout.addRow(self.QPushButton_img_shwfs_compute_wf, self.QPushButton_img_shwfs_save_wf)

        group_layout = QHBoxLayout(group)
        group_layout.addWidget(image_shwfs_scroll_area)
        group.setLayout(group_layout)
        return group

    def _create_dwfs_panel(self):
        group = cw.GroupWidget()
        dwfs_scroll_area, dwfs_scroll_layout = cw.create_scroll_area("G")
        
        self.QSpinBox_close_loop_number = cw.SpinBoxWidget(0, 100, 1, 1)
        self.QPushButton_dwfs_cl_correction = cw.PushButtonWidget('Close Loop Correction')

        dwfs_scroll_layout.addWidget(cw.LabelWidget(str('Loop #   (0 - infinite)')), 0, 0, 1, 1)
        dwfs_scroll_layout.addWidget(self.QSpinBox_close_loop_number, 0, 1, 1, 1)
        dwfs_scroll_layout.addWidget(self.QPushButton_dwfs_cl_correction, 0, 2, 1, 1)

        group_layout = QHBoxLayout(group)
        group_layout.addWidget(dwfs_scroll_area)
        group.setLayout(group_layout)
        return group

    def _create_dm_panel(self):
        group = cw.GroupWidget()
        dm_scroll_area, dm_scroll_layout = cw.create_scroll_area("G")

        self.QComboBox_dms = cw.ComboBoxWidget(list_items=[])
        self.QComboBox_wfsmd = cw.ComboBoxWidget(list_items=['modal', 'phase', 'zonal'])
        self.QSpinBox_actuator = cw.SpinBoxWidget(0, 96, 1, 0)
        self.QDoubleSpinBox_actuator_push = cw.DoubleSpinBoxWidget(-1, 1, 0.005, 3, 0)
        self.QPushButton_push_actuator = cw.PushButtonWidget('Push')
        self.QPushButton_influence_fuction_laser = cw.PushButtonWidget('InfluFunc')
        self.QSpinBox_zernike_mode = cw.SpinBoxWidget(0, 100, 1, 0)
        self.QDoubleSpinBox_zernike_mode_amp = cw.DoubleSpinBoxWidget(-20, 20, 0.01, 2, 0)
        self.QPushButton_set_zernike_mode = cw.PushButtonWidget('Set Zernike')
        self.QComboBox_cmd = cw.ComboBoxWidget(list_items=['0', '1'])
        self.QComboBox_cmd.setCurrentIndex(1)
        self.QPushButton_setDM = cw.PushButtonWidget('Set DM')
        self.QPushButton_load_dm = cw.PushButtonWidget('Load DM')
        self.QPushButton_update_cmd = cw.PushButtonWidget('Add DM')
        self.QPushButton_save_dm = cw.PushButtonWidget('Save DM')
        self.QPushButton_change_dm_flat = cw.PushButtonWidget('Save Flat')

        dm_scroll_layout.addWidget(cw.LabelWidget(str('DM')), 0, 0, 1, 1)
        dm_scroll_layout.addWidget(self.QComboBox_dms, 0, 1, 1, 1)
        dm_scroll_layout.addWidget(cw.LabelWidget(str('Method')), 0, 2, 1, 1)
        dm_scroll_layout.addWidget(self.QComboBox_wfsmd, 0, 3, 1, 1)
        dm_scroll_layout.addWidget(cw.LabelWidget(str('Actuator')), 1, 0, 1, 1)
        dm_scroll_layout.addWidget(self.QSpinBox_actuator, 1, 1, 1, 1)
        dm_scroll_layout.addWidget(cw.LabelWidget(str('Push')), 2, 0, 1, 1)
        dm_scroll_layout.addWidget(self.QDoubleSpinBox_actuator_push, 2, 1, 1, 1)
        dm_scroll_layout.addWidget(self.QPushButton_push_actuator, 3, 0, 1, 1)
        dm_scroll_layout.addWidget(self.QPushButton_influence_fuction_laser, 3, 1, 1, 1)
        dm_scroll_layout.addWidget(cw.LabelWidget(str('Zernike Mode')), 1, 2, 1, 1)
        dm_scroll_layout.addWidget(self.QSpinBox_zernike_mode, 1, 3, 1, 1)
        dm_scroll_layout.addWidget(cw.LabelWidget(str('Amplitude')), 2, 2, 1, 1)
        dm_scroll_layout.addWidget(self.QDoubleSpinBox_zernike_mode_amp, 2, 3, 1, 1)
        dm_scroll_layout.addWidget(self.QPushButton_set_zernike_mode, 3, 2, 1, 1)
        dm_scroll_layout.addWidget(self.QComboBox_cmd, 4, 0, 1, 1)
        dm_scroll_layout.addWidget(self.QPushButton_setDM, 4, 1, 1, 1)
        dm_scroll_layout.addWidget(self.QPushButton_load_dm, 3, 3, 1, 1)
        dm_scroll_layout.addWidget(self.QPushButton_update_cmd, 4, 2, 1, 1)
        dm_scroll_layout.addWidget(self.QPushButton_change_dm_flat, 4, 3, 1, 1)
        
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
        self.QPushButton_sensorless_auto = cw.PushButtonWidget('Auto AO')
        self.QPushButton_sensorless_metric_acqs = cw.PushButtonWidget('Run MFACQs')
        self.QPushButton_sensorless_ml_acqs = cw.PushButtonWidget('Run MLACQs')
        self.QRadioButton_sensorless_error = cw.RadioButtonWidget('ErrorIn')

        sensorless_scroll_layout.addWidget(cw.LabelWidget(str('Zernike Modes')), 0, 0, 1, 2)
        sensorless_scroll_layout.addWidget(cw.LabelWidget(str('From')), 1, 0, 1, 1)
        sensorless_scroll_layout.addWidget(self.QSpinBox_zernike_mode_start, 1, 1, 1, 1)
        sensorless_scroll_layout.addWidget(cw.LabelWidget(str('To')), 2, 0, 1, 1)
        sensorless_scroll_layout.addWidget(self.QSpinBox_zernike_mode_stop, 2, 1, 1, 1)
        sensorless_scroll_layout.addWidget(cw.LabelWidget(str('Amplitudes')), 0, 2, 1, 2)
        sensorless_scroll_layout.addWidget(cw.LabelWidget(str('From')), 1, 2, 1, 1)
        sensorless_scroll_layout.addWidget(self.QDoubleSpinBox_zernike_mode_amps_start, 1, 3, 1, 1)
        sensorless_scroll_layout.addWidget(cw.LabelWidget(str('StepNum')), 2, 2, 1, 1)
        sensorless_scroll_layout.addWidget(self.QSpinBox_zernike_mode_amps_stepnum, 2, 3, 1, 1)
        sensorless_scroll_layout.addWidget(cw.LabelWidget(str('StepSize')), 3, 2, 1, 1)
        sensorless_scroll_layout.addWidget(self.QDoubleSpinBox_zernike_mode_amps_step, 3, 3, 1, 1)
        sensorless_scroll_layout.addWidget(cw.LabelWidget(str('LPF')), 0, 4, 1, 1)
        sensorless_scroll_layout.addWidget(self.QDoubleSpinBox_lpf, 1, 4, 1, 1)
        sensorless_scroll_layout.addWidget(cw.LabelWidget(str('HPF')), 2, 4, 1, 1)
        sensorless_scroll_layout.addWidget(self.QDoubleSpinBox_hpf, 3, 4, 1, 1)
        sensorless_scroll_layout.addWidget(cw.LabelWidget(str('Select')), 4, 4, 1, 1)
        sensorless_scroll_layout.addWidget(self.QDoubleSpinBox_select_frequency, 5, 4, 1, 1)
        sensorless_scroll_layout.addWidget(cw.LabelWidget(str('Image Metric')), 0, 5, 1, 1)
        sensorless_scroll_layout.addWidget(self.QComboBox_metric, 1, 5, 1, 1)
        sensorless_scroll_layout.addWidget(self.QRadioButton_sensorless_error, 2, 5, 1, 1)
        sensorless_scroll_layout.addWidget(self.QPushButton_sensorless_run, 3, 5, 1, 1)
        sensorless_scroll_layout.addWidget(self.QPushButton_sensorless_auto, 4, 5, 1, 1)
        sensorless_scroll_layout.addWidget(self.QPushButton_sensorless_metric_acqs, 3, 0, 1, 2)
        sensorless_scroll_layout.addWidget(self.QPushButton_sensorless_ml_acqs, 4, 0, 1, 2)

        group_layout = QHBoxLayout(group)
        group_layout.addWidget(sensorless_scroll_area)
        group.setLayout(group_layout)
        return group

    def _set_signal_connections(self):
        self.QPushButton_img_shwfs_base.clicked.connect(self.img_wfs_base)
        self.QPushButton_run_img_wfs.clicked.connect(self.run_img_wfs)
        self.QPushButton_run_img_wfr.clicked.connect(self.run_img_wfr)
        self.QPushButton_img_shwfs_compute_wf.clicked.connect(self.compute_img_wf)
        self.QPushButton_img_shwfs_save_wf.clicked.connect(self.save_img_wf)
        self.QPushButton_img_shwfs_acquisition.clicked.connect(self.wfs_acq)
        self.QComboBox_dms.currentIndexChanged.connect(self.select_dm)
        self.QPushButton_push_actuator.clicked.connect(self.push_dm_actuator)
        self.QPushButton_influence_fuction_laser.clicked.connect(self.run_influence_function)
        self.QPushButton_set_zernike_mode.clicked.connect(self.set_dm_zernike)
        self.QPushButton_setDM.clicked.connect(self.set_dm_acts)
        self.QPushButton_update_cmd.clicked.connect(self.update_dm_cmd)
        self.QPushButton_load_dm.clicked.connect(self.load_dm_file)
        self.QPushButton_save_dm.clicked.connect(self.save_dm_cmd)
        self.QPushButton_change_dm_flat.clicked.connect(self.change_dm_flat)
        self.QPushButton_dwfs_cl_correction.clicked.connect(self.run_close_loop_correction)
        self.QPushButton_sensorless_run.clicked.connect(self.run_sensorless_correction)
        self.QPushButton_sensorless_auto.clicked.connect(self.run_sensorless_auto)
        self.QPushButton_sensorless_metric_acqs.clicked.connect(self.run_sensorless_metric_acquisition)
        self.QPushButton_sensorless_ml_acqs.clicked.connect(self.run_sensorless_ml_acquisition)       

    def display_img_wf_properties(self, properties):
        self.lcdNumber_wfmin_img.display(properties[0])
        self.lcdNumber_wfmax_img.display(properties[1])
        self.lcdNumber_wfrms_img.display(properties[2])

    def get_parameters_foc(self):
        return (self.QSpinBox_base_xcenter_foc.value(), self.QSpinBox_base_ycenter_foc.value(),
                self.QSpinBox_offset_xcenter_foc.value(), self.QSpinBox_offset_ycenter_foc.value(),
                self.QSpinBox_n_lenslets_x_foc.value(), self.QSpinBox_n_lenslets_y_foc.value(),
                self.QSpinBox_spacing_foc.value(), self.QSpinBox_radius_foc.value(),
                self.QDoubleSpinBox_foc_background.value())

    def get_gradient_method_img(self):
        return self.QComboBox_wfrmd_foc.currentText()

    def get_img_wfs_method(self):
        return self.QComboBox_wfsmd.currentText()

    def get_wfs_camera(self):
        return self.QComboBox_wfs_camera_selection.currentIndex()

    @pyqtSlot()
    def img_wfs_base(self):
        self.Signal_img_shwfs_base.emit()

    @pyqtSlot()
    def run_img_wfs(self):
        if self.QPushButton_run_img_wfs.isChecked():
            self.Signal_img_wfs.emit(True)
        else:
            self.Signal_img_wfs.emit(False)

    @pyqtSlot()
    def run_img_wfr(self):
        self.Signal_img_shwfr_run.emit()

    @pyqtSlot()
    def compute_img_wf(self):
        self.Signal_img_shwfs_compute_wf.emit()

    @pyqtSlot()
    def save_img_wf(self):
        self.Signal_img_shwfs_save_wf.emit()

    @pyqtSlot()
    def wfs_acq(self):
        self.Signal_img_shwfs_acquisition.emit()

    @pyqtSlot()
    def select_dm(self):
        dn = self.QComboBox_dms.currentText()
        self.Signal_dm_selection.emit(dn)

    @pyqtSlot()
    def push_dm_actuator(self):
        n = self.QSpinBox_actuator.value()
        a = self.QDoubleSpinBox_actuator_push.value()
        self.Signal_push_actuator.emit(n, a)

    @pyqtSlot()
    def run_influence_function(self):
        self.Signal_influence_function.emit()

    @pyqtSlot()
    def set_dm_zernike(self):
        self.Signal_set_zernike.emit()

    @pyqtSlot()
    def set_dm_acts(self):
        self.Signal_set_dm.emit()

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

    def get_actuator(self):
        return self.QSpinBox_actuator.value(), self.QDoubleSpinBox_actuator_push.value()

    def get_zernike_mode(self):
        return self.QSpinBox_zernike_mode.value(), self.QDoubleSpinBox_zernike_mode_amp.value()

    def get_dm_selection(self):
        return self.QComboBox_dms.currentText()

    def get_cmd_index(self):
        return self.QComboBox_cmd.currentText()

    def update_cmd_index(self, wst=True):
        item = '{}'.format(self.QComboBox_cmd.count())
        self.QComboBox_cmd.addItem(item)
        if wst:
            self.QComboBox_cmd.setCurrentIndex(self.QComboBox_cmd.count() - 1)

    @pyqtSlot()
    def run_close_loop_correction(self):
        n = self.QSpinBox_close_loop_number.value()
        self.Signal_img_shwfs_correct_wf.emit(n)

    @pyqtSlot()
    def run_sensorless_correction(self):
        self.Signal_sensorlessAO_run.emit()

    @pyqtSlot()
    def run_sensorless_auto(self):
        self.Signal_sensorlessAO_auto.emit()

    @pyqtSlot()
    def run_sensorless_metric_acquisition(self):
        self.Signal_sensorlessAO_metric_acquisition.emit()

    @pyqtSlot()
    def run_sensorless_ml_acquisition(self):
        self.Signal_sensorlessAO_ml_acquisition.emit()

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
