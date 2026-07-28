# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import json

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QSpinBox, QDoubleSpinBox

try:
    from . import custom_widgets as cw
except ImportError as e:
    from minimiao.gui import custom_widgets as cw

try:
    from ..utilities.zernike_generator import zernike_basis as _zg_basis, noll_to_nm
except ImportError as e:
    from minimiao.utilities.zernike_generator import zernike_basis as _zg_basis, noll_to_nm


_ZMS = [
    (-1, 1), (1, 1),
    (0, 2), (-2, 2), (2, 2),
    (-1, 3), (1, 3), (-3, 3), (3, 3),
    (0, 4), (-2, 4), (2, 4), (-4, 4), (4, 4),
    (-1, 5), (1, 5), (-3, 5), (3, 5), (-5, 5), (5, 5),
    (0, 6), (-2, 6), (2, 6), (-4, 6), (4, 6), (-4, 6), (4, 6),
    (-1, 7), (1, 7), (-3, 7), (3, 7), (-5, 7), (5, 7), (-7, 7), (7, 7),
]
_ZM_LABELS = [f"{m},{n}" for m, n in _ZMS]

_BRUSH_POS = pg.mkBrush(70, 130, 180, 220)  # steel blue  – positive
_BRUSH_NEG = pg.mkBrush(210, 70, 60, 220)  # tomato red  – negative

# Diverging blue–white–red LUT for the wavefront image
_WF_LUT = pg.ColorMap(
    pos=np.array([0.0, 0.5, 1.0]),
    color=np.array([[0, 80, 220, 255], [245, 245, 245, 255], [220, 30, 30, 255]], dtype=np.uint8),
).getLookupTable(nPts=256)


def _build_zernike_basis(size=128):
    """Return (basis, mask) where basis[i] is the Zernike polynomial for
    _ZMS[i] evaluated on a (size×size) unit-disk grid, using zernike_generator."""
    # Build reverse lookup (n, m) → Noll j using the reliable noll_to_nm direction
    noll_map = {}
    for j in range(1, 300):
        n, m = noll_to_nm(j)
        noll_map[(n, m)] = j
    # Each DPP entry is (m_dpp, n_dpp); zernike_generator uses (n, m) convention
    noll_indices = [noll_map[(n_dpp, m_dpp)] for m_dpp, n_dpp in _ZMS]
    nz_max = max(noll_indices)
    Z, _, _ = _zg_basis(nx=size, ny=size, nz=nz_max)
    # Z.shape = (nz_max, size, size), Noll j=1 at index 0
    basis = np.array([Z[j - 1] for j in noll_indices], dtype=np.float32)
    # Reconstruct the pupil mask using the same coords as zernike_basis internals
    x = np.arange(size, dtype=np.float64)
    x_norm = (x - (size - 1) / 2.0) / (size / 2.0)
    xx, yy = np.meshgrid(x_norm, x_norm)
    mask = (xx ** 2 + yy ** 2) <= 1.0
    return basis, mask


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
        self._set_signal_connections()
        self.load_spinbox_values()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        self.dm_viz_panel = self._create_dm_viz_panel()
        self.dm_panel = self._create_dm_panel()
        self.sensorless_panel = self._create_sensorless_panel()

        main_layout.addWidget(self.dm_viz_panel)
        main_layout.addWidget(self.dm_panel)
        main_layout.addWidget(self.sensorless_panel)

        main_layout.addStretch(1)
        self.setLayout(main_layout)

    def _create_dm_viz_panel(self):
        self._zernike_basis, self._pupil_mask = _build_zernike_basis(size=128)

        group = cw.GroupWidget()
        layout = QVBoxLayout(group)

        # ── Wavefront image ──────────────────────────────────────
        self._wf_glw = pg.GraphicsLayoutWidget()

        self._wf_glw.addLabel("Wavefront", row=0, col=0)
        self._wf_vb = self._wf_glw.addViewBox(row=1, col=0, lockAspect=True)
        self._wf_vb.setMenuEnabled(False)

        self._wf_image = pg.ImageItem()
        self._wf_image.setLookupTable(_WF_LUT)
        self._wf_vb.addItem(self._wf_image)

        # Seed with a flat (zero) wavefront so the pupil outline is visible
        flat = np.zeros((128, 128), dtype=np.float32)
        flat[~self._pupil_mask] = np.nan
        self._wf_image.setImage(flat.T, levels=(-1.0, 1.0))
        self._wf_vb.autoRange()

        # ── Bar chart ─────────────────────────────────────────────
        nz = len(_ZMS)
        self._zernike_plot = pg.PlotWidget(title="Zernike Amplitudes")
        self._zernike_plot.showGrid(x=False, y=True, alpha=0.3)
        self._zernike_plot.setLabel('left', 'Amplitude')
        self._zernike_plot.addLine(y=0, pen=pg.mkPen('w', width=1))

        ticks = [[(i, lbl) for i, lbl in enumerate(_ZM_LABELS)]]
        ax = self._zernike_plot.getAxis('bottom')
        ax.setTicks(ticks)
        ax.setStyle(tickFont=QFont("Arial", 7))

        self._zernike_bars = pg.BarGraphItem(x=np.arange(nz), height=np.zeros(nz), width=0.7, brushes=[_BRUSH_POS] * nz)
        self._zernike_plot.addItem(self._zernike_bars)

        layout.addWidget(self._wf_glw, stretch=1)
        layout.addWidget(self._zernike_plot, stretch=1)
        group.setLayout(layout)
        return group

    def _create_dm_panel(self):
        group = cw.GroupWidget()
        dm_scroll_area, dm_scroll_layout = cw.create_scroll_area("G")

        self.QSpinBox_zernike_mode = cw.SpinBoxWidget(0, 34, 1, 0)
        self.QDoubleSpinBox_zernike_mode_amp = cw.DoubleSpinBoxWidget(-2, 2, 0.001, 3, 0)
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
        self.QComboBox_metric = cw.ComboBoxWidget(list_items=['Max(Intensity)', 'Sum(Intensity)',
                                                              'SNR(FFT)', 'HighPass(FFT)'])
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

    def update_dm_display(self, amps):
        nz = len(_ZMS)
        heights = list(amps[:nz]) if len(amps) >= nz else list(amps) + [0.0] * (nz - len(amps))

        # Bar chart
        brushes = [_BRUSH_POS if h >= 0 else _BRUSH_NEG for h in heights]
        self._zernike_bars.setOpts(height=heights, brushes=brushes)
        lo = min(min(heights), 0)
        hi = max(max(heights), 0)
        margin = max(abs(hi - lo) * 0.15, 0.05)
        self._zernike_plot.setYRange(lo - margin, hi + margin)

        # Wavefront reconstruction: W = sum_i(amp_i * Z_i)
        amp_arr = np.array(heights, dtype=np.float32)
        wavefront = np.einsum('i,ihw->hw', amp_arr, self._zernike_basis)
        wavefront[~self._pupil_mask] = np.nan

        valid = wavefront[self._pupil_mask]
        absmax = float(np.max(np.abs(valid))) if valid.size > 0 else 1.0
        if absmax == 0 or not np.isfinite(absmax):
            absmax = 1.0
        self._wf_image.setImage(wavefront.T, levels=(-absmax, absmax))
        self._wf_vb.autoRange()

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
                self.QComboBox_metric.currentText(), self.QRadioButton_sensorless_error.isChecked())

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
