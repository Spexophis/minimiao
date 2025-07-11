import json

from PyQt6.QtCore import pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QSpinBox, QDoubleSpinBox

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

    def __init__(self, config, logg, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.config = config
        self.logg = logg
        self._setup_ui()
        # self._load_spinbox_values()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        self.image_panel = self._create_image_panel()
        # self.position_panel = self._create_position_panel()
        # self.laser_panel = self._create_laser_panel()
        # self.daq_panel = self._create_daq_panel()
        # self.live_panel = self._create_live_panel()
        # self.acq_panel = self._create_acquisition_panel()

        main_layout.addWidget(self.image_panel)
        # main_layout.addWidget(self.position_panel)
        # main_layout.addWidget(self.laser_panel)
        # main_layout.addWidget(self.daq_panel)
        # main_layout.addWidget(self.live_panel)
        # main_layout.addWidget(self.acq_panel)

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

