# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, pyqtSignal, Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSplitter, QGridLayout, QHBoxLayout, QButtonGroup

from . import custom_widgets as cw
from minimiao.utilities import image_processor as ipr


class PhotonPool(QObject):
    def __init__(self, max_len=2 ** 16, dt_s=4e-6, px=(64, 64), parent=None):
        super().__init__(parent)
        self.max_len = int(max_len)
        self.buf_0 = deque(np.zeros(self.max_len, dtype=np.int64), maxlen=self.max_len)
        self.buf_1 = deque(np.zeros(self.max_len, dtype=np.int64), maxlen=self.max_len)
        self.dt_s = dt_s
        self.xt = np.arange(self.max_len) * float(self.dt_s)
        self.img_0 = np.zeros(px, dtype=np.float64)
        self.img_1 = np.zeros(px, dtype=np.float64)

    def new_acquire(self, recon_img, counts):
        self.buf_0.extend(counts[0])
        self.buf_1.extend(counts[1])
        self.img_0 = recon_img[0]
        self.img_1 = recon_img[1]

    def reset_buffer(self, max_len: int | None = None, dt_s: float | None = None, px: tuple | None = None):
        if max_len is not None:
            self.max_len = min(int(max_len), int(2 ** 16))
        self.buf_0 = deque(np.zeros(self.max_len, dtype=np.int64), maxlen=self.max_len)
        self.buf_1 = deque(np.zeros(self.max_len, dtype=np.int64), maxlen=self.max_len)
        if dt_s is not None:
            self.dt_s = float(dt_s)
        self.xt = np.arange(self.max_len) * float(self.dt_s)
        if px is not None:
            self.img_0 = np.zeros(px, dtype=np.float64)
            self.img_1 = np.zeros(px, dtype=np.float64)


class LiveViewer(QWidget):
    frame_idx_signal = pyqtSignal(int)

    def __init__(self, config, logg, parent=None):
        super().__init__(parent)
        self.config = config
        self.logg = logg
        pg.setConfigOptions(useOpenGL=True, antialias=False)
        self._setup_ui()
        self._overlay_n = 0
        self.photon_pool = PhotonPool()
        self.data_curve_0 = None
        self.data_curve_1 = None
        self.psr_mode = False
        self.psr_fn = 1
        self.x_min, self.x_max = None, None
        self.y_min, self.y_max = None, None
        self._setup_signal_connections()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Vertical)

        plot_0_widget = self._create_plot_0_widget()
        splitter.addWidget(plot_0_widget)

        metric_widget = self._create_metric_widget()
        splitter.addWidget(metric_widget)

        plot_1_widget = self._create_plot_1_widget()
        splitter.addWidget(plot_1_widget)

        layout.addWidget(splitter)
        self.setLayout(layout)

    def _setup_signal_connections(self):
        self.btn_metrics_enable.toggled.connect(self._on_metrics_toggled)
        self.btn_profile_0.toggled.connect(self._on_profile_toggled_0)
        self.btn_profile_1.toggled.connect(self._on_profile_toggled_1)
        self.rb_axis_x_0.toggled.connect(lambda _: self._refresh_profile(0))
        self.rb_axis_x_1.toggled.connect(lambda _: self._refresh_profile(1))
        self.spin_center_0.valueChanged.connect(lambda _: self._refresh_profile(0))
        self.spin_center_1.valueChanged.connect(lambda _: self._refresh_profile(1))
        self.spin_width_0.valueChanged.connect(lambda _: self._refresh_profile(0))
        self.spin_width_1.valueChanged.connect(lambda _: self._refresh_profile(1))

    def _create_plot_0_widget(self):
        layout_plot_0 = QGridLayout()

        self.QComboBox_plot_selection_0 = cw.ComboBoxWidget(list_items=["MPD #0", "PMT", "Empty"], length=80)
        self.btn_profile_0 = cw.PushButtonWidget("Photon Trace", checkable=True, checked=False)

        top_bar_0 = QWidget()
        top_bar_layout_0 = QHBoxLayout(top_bar_0)
        top_bar_layout_0.setContentsMargins(0, 0, 0, 0)
        top_bar_layout_0.addWidget(self.QComboBox_plot_selection_0)
        top_bar_layout_0.addWidget(self.btn_profile_0)
        top_bar_layout_0.addStretch()

        self.graph_plot_0 = pg.PlotWidget()
        self.graph_plot_0.setAspectLocked(True)
        self.graph_plot_0.setLabel('left', 'Y Position', units='v')
        self.graph_plot_0.setLabel('bottom', 'X Position', units='v')
        self.graph_plot_0.getAxis('left').enableAutoSIPrefix(False)
        self.graph_plot_0.getAxis('bottom').enableAutoSIPrefix(False)

        self.graph_img_item_0 = pg.ImageItem(axisOrder="row-major")
        self.graph_plot_0.addItem(self.graph_img_item_0)
        self.graph_plot_0.invertY(True)

        self.color_bar_0 = pg.ColorBarItem(interactive=False, colorMap=pg.ColorMap([0.0, 1.0], [[0,0,0,255],[255,255,255,255]]))
        self.color_bar_0.setImageItem(self.graph_img_item_0, insert_in=self.graph_plot_0.getPlotItem())

        self.data_plot_0 = pg.PlotWidget()
        self.data_plot_0.showGrid(x=True, y=True)

        self.data_curve_0 = self.data_plot_0.plot()
        self.data_curve_0.setDownsampling(auto=True, method="peak")
        self.data_curve_0.setSkipFiniteCheck(True)

        pi_0 = self.data_plot_0.getPlotItem()
        pi_0.setClipToView(True)
        pi_0.enableAutoRange(x=False)

        self.profile_controls_0 = self._create_profile_controls(0)
        self.profile_controls_0.setVisible(False)

        layout_plot_0.addWidget(top_bar_0, 0, 0, 1, 2)
        layout_plot_0.addWidget(self.graph_plot_0, 1, 0)
        layout_plot_0.addWidget(self.data_plot_0, 1, 1)
        layout_plot_0.addWidget(self.profile_controls_0, 2, 1)

        plot_0_widget = QWidget()
        plot_0_widget.setLayout(layout_plot_0)

        return plot_0_widget

    def _create_plot_1_widget(self):
        layout_plot_1 = QGridLayout()

        self.QComboBox_plot_selection_1 = cw.ComboBoxWidget(list_items=["MPD #1", "PMT", "Empty"], length=80)
        self.btn_profile_1 = cw.PushButtonWidget("Photon Trace", checkable=True, checked=False)

        top_bar_1 = QWidget()
        top_bar_layout_1 = QHBoxLayout(top_bar_1)
        top_bar_layout_1.setContentsMargins(0, 0, 0, 0)
        top_bar_layout_1.addWidget(self.QComboBox_plot_selection_1)
        top_bar_layout_1.addWidget(self.btn_profile_1)
        top_bar_layout_1.addStretch()

        self.graph_plot_1 = pg.PlotWidget()
        self.graph_plot_1.setAspectLocked(True)
        self.graph_plot_1.setLabel('left', 'Y Position', units='v')
        self.graph_plot_1.setLabel('bottom', 'X Position', units='v')
        self.graph_plot_1.getAxis('left').enableAutoSIPrefix(False)
        self.graph_plot_1.getAxis('bottom').enableAutoSIPrefix(False)

        self.graph_img_item_1 = pg.ImageItem(axisOrder="row-major")
        self.graph_plot_1.addItem(self.graph_img_item_1)
        self.graph_plot_1.invertY(True)

        self.color_bar_1 = pg.ColorBarItem(interactive=False, colorMap=pg.ColorMap([0.0, 1.0], [[0,0,0,255],[255,255,255,255]]))
        self.color_bar_1.setImageItem(self.graph_img_item_1, insert_in=self.graph_plot_1.getPlotItem())

        self.data_plot_1 = pg.PlotWidget()
        self.data_plot_1.showGrid(x=True, y=True)

        self.data_curve_1 = self.data_plot_1.plot()

        self.data_curve_1.setDownsampling(auto=True, method="peak")
        self.data_curve_1.setSkipFiniteCheck(True)

        pi_1 = self.data_plot_1.getPlotItem()
        pi_1.setClipToView(True)
        pi_1.enableAutoRange(x=False)

        self.profile_controls_1 = self._create_profile_controls(1)
        self.profile_controls_1.setVisible(False)

        layout_plot_1.addWidget(top_bar_1, 0, 0, 1, 2)
        layout_plot_1.addWidget(self.graph_plot_1, 1, 0)
        layout_plot_1.addWidget(self.data_plot_1, 1, 1)
        layout_plot_1.addWidget(self.profile_controls_1, 2, 1)

        plot_1_widget = QWidget()
        plot_1_widget.setLayout(layout_plot_1)

        return plot_1_widget

    def _create_profile_controls(self, idx):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        rb_x = cw.RadioButtonWidget("X axis", autoex=False)
        rb_y = cw.RadioButtonWidget("Y axis", autoex=False)
        rb_x.setChecked(True)

        grp = QButtonGroup(container)
        grp.addButton(rb_x)
        grp.addButton(rb_y)
        grp.setExclusive(True)

        spin_center = cw.SpinBoxWidget(range_min=0, range_max=1024, step=1, value=1)
        spin_width = cw.SpinBoxWidget(range_min=1, range_max=32, step=1, value=1)

        layout.addWidget(cw.LabelWidget("Axis:"))
        layout.addWidget(rb_x)
        layout.addWidget(rb_y)
        layout.addWidget(cw.LabelWidget("Center:"))
        layout.addWidget(spin_center)
        layout.addWidget(cw.LabelWidget("Width:"))
        layout.addWidget(spin_width)
        layout.addStretch()

        if idx == 0:
            self.rb_axis_x_0 = rb_x
            self.rb_axis_y_0 = rb_y
            self._btn_group_axis_0 = grp
            self.spin_center_0 = spin_center
            self.spin_width_0 = spin_width
        else:
            self.rb_axis_x_1 = rb_x
            self.rb_axis_y_1 = rb_y
            self._btn_group_axis_1 = grp
            self.spin_center_1 = spin_center
            self.spin_width_1 = spin_width

        return container

    def _create_metric_widget(self):
        layout_metric = QGridLayout()

        self.lcdNumber_img_0_sobel = cw.LCDNumberWidget()
        self.lcdNumber_img_0_laplacian = cw.LCDNumberWidget()
        self.lcdNumber_img_1_sobel = cw.LCDNumberWidget()
        self.lcdNumber_img_1_laplacian = cw.LCDNumberWidget()
        self.btn_metrics_enable = cw.PushButtonWidget("Metrics ON", checkable=True, checked=False)

        layout_metric.addWidget(cw.LabelWidget('Sobel', align=2), 0, 0)
        layout_metric.addWidget(self.lcdNumber_img_0_sobel, 0, 1)
        layout_metric.addWidget(self.lcdNumber_img_1_sobel, 1, 1)
        layout_metric.addWidget(cw.LabelWidget('Laplacian', align=2), 0, 2)
        layout_metric.addWidget(self.lcdNumber_img_0_laplacian, 0, 3)
        layout_metric.addWidget(self.lcdNumber_img_1_laplacian, 1, 3)
        layout_metric.addWidget(self.btn_metrics_enable, 0, 4, 2, 1)

        metric_widget = QWidget()
        metric_widget.setLayout(layout_metric)

        return metric_widget

    def _on_metrics_toggled(self, checked):
        self.btn_metrics_enable.setText("Metrics ON" if checked else "Metrics OFF")

    def _on_profile_toggled_0(self, checked):
        self.btn_profile_0.setText("Line Profile" if checked else "Photon Trace")
        self.profile_controls_0.setVisible(checked)
        if not checked:
            self._restore_trace(0)

    def _on_profile_toggled_1(self, checked):
        self.btn_profile_1.setText("Line Profile" if checked else "Photon Trace")
        self.profile_controls_1.setVisible(checked)
        if not checked:
            self._restore_trace(1)

    def _restore_trace(self, idx):
        """Rebuild the photon-count trace curve and immediately populate it with current data."""
        xt = self.photon_pool.xt
        counts = np.array(self.photon_pool.buf_0 if idx == 0 else self.photon_pool.buf_1)
        plot_widget = self.data_plot_0 if idx == 0 else self.data_plot_1

        plot_widget.clear()
        plot_widget.showGrid(x=True, y=True)
        plot_widget.setLabel('bottom', '')
        plot_widget.setLabel('left', '')

        curve = plot_widget.plot()
        curve.setDownsampling(auto=True, method="peak")
        curve.setSkipFiniteCheck(True)

        pi = plot_widget.getPlotItem()
        pi.setClipToView(True)
        pi.enableAutoRange(x=False)
        # Set the x range to the full time axis so the axis is not frozen
        pi.setXRange(float(xt[0]), float(xt[-1]), padding=0)

        curve.setData(xt, counts)

        if idx == 0:
            self.data_curve_0 = curve
        else:
            self.data_curve_1 = curve

    def _compute_line_profile(self, img, axis_x, center, width):
        """Return (positions, profile) for the selected axis and averaging window."""
        h, w = img.shape
        half = width // 2
        if axis_x:
            # Profile along X dimension (columns), averaged over rows near center
            r0 = max(0, center - half)
            r1 = min(h, r0 + width)
            profile = img[r0:r1, :].mean(axis=0)
            positions = np.arange(profile.shape[0])
        else:
            # Profile along Y dimension (rows), averaged over columns near center
            c0 = max(0, center - half)
            c1 = min(w, c0 + width)
            profile = img[:, c0:c1].mean(axis=1)
            positions = np.arange(profile.shape[0])
        return positions, profile

    def _refresh_profile(self, idx):
        """Recompute and redisplay the line profile for the given panel index."""
        if idx == 0 and self.btn_profile_0.isChecked():
            self._draw_profile(self.photon_pool.img_0, self.data_plot_0,
                               self.rb_axis_x_0.isChecked(),
                               self.spin_center_0.value(),
                               self.spin_width_0.value())
        elif idx == 1 and self.btn_profile_1.isChecked():
            self._draw_profile(self.photon_pool.img_1, self.data_plot_1,
                               self.rb_axis_x_1.isChecked(),
                               self.spin_center_1.value(),
                               self.spin_width_1.value())

    def _draw_profile(self, img, plot_widget, axis_x, center, width):
        positions, profile = self._compute_line_profile(img, axis_x, center, width)
        plot_widget.clear()
        plot_widget.showGrid(x=True, y=True)
        plot_widget.getPlotItem().enableAutoRange(x=True, y=True)
        axis_label = "X pixel" if axis_x else "Y pixel"
        plot_widget.setLabel('bottom', axis_label)
        plot_widget.setLabel('left', 'Intensity')
        plot_widget.plot(positions, profile, pen=pg.mkPen(color='c', width=1))

    def set_plots(self, ps):
        self.QComboBox_plot_selection_0.setCurrentIndex(ps[0])
        self.QComboBox_plot_selection_1.setCurrentIndex(ps[1])

    def plot_trace(self, y, x=None, overlay=False):
        y = np.asarray(y)
        if y.size == 0:
            return
        if not overlay:
            self.data_plot_0.clear()
            self._overlay_n = 0
        if x is None:
            x = np.arange(y.size)
        self.data_plot_0.enableAutoRange(x=True)
        color = pg.intColor(self._overlay_n, hues=12)
        pen = pg.mkPen(color=color, width=1.)
        self._overlay_n += 1
        self.data_plot_0.plot(x, y, pen=pen)

    def stream_trace(self, x: np.ndarray, y_0: np.ndarray, y_1: np.ndarray):
        """Initialize the 1D trace plots."""
        if y_0 is not None:
            self.data_plot_0.clear()
            self.data_curve_0 = self.data_plot_0.plot()
            self.data_curve_0.setDownsampling(auto=True, method="peak")
            self.data_curve_0.setSkipFiniteCheck(True)
            self.data_plot_0.enableAutoRange(x=True)
            self.data_curve_0.setData(x, y_0)
        if y_1 is not None:
            self.data_plot_1.clear()
            self.data_curve_1 = self.data_plot_1.plot()
            self.data_curve_1.setDownsampling(auto=True, method="peak")
            self.data_curve_1.setSkipFiniteCheck(True)
            self.data_plot_1.enableAutoRange(x=True)
            self.data_curve_1.setData(x, y_1)

    def stream_trace_update(self, xt: np.ndarray, counts_0: np.ndarray, counts_1: np.ndarray):
        if not self.btn_profile_0.isChecked():
            self.data_curve_0.setData(xt, counts_0)
        if not self.btn_profile_1.isChecked():
            self.data_curve_1.setData(xt, counts_1)

    def set_graph_with_axes(self, img_0: np.ndarray, img_1: np.ndarray,
                            x_axis=None, y_axis=None, levels=None):
        self.set_graph_image(img_0, img_1, levels)
        if self.psr_fn % 8 == 0 and self.btn_metrics_enable.isChecked():
            self.image_metrics(img_0, img_1)

        if x_axis is not None and y_axis is not None:
            self.x_min, self.x_max = x_axis[0], x_axis[-1]
            self.y_min, self.y_max = y_axis[0], y_axis[-1]

            h, w = img_0.shape
            pixel_width = (self.x_max - self.x_min) / w
            pixel_height = (self.y_max - self.y_min) / h
            self.graph_img_item_0.setRect(pg.QtCore.QRectF(self.x_min, self.y_min,
                                                           self.x_max - self.x_min,
                                                           self.y_max - self.y_min))
            self.graph_img_item_1.setRect(pg.QtCore.QRectF(self.x_min, self.y_min,
                                                           self.x_max - self.x_min,
                                                           self.y_max - self.y_min))
            self.graph_plot_0.setAspectLocked(True, ratio=pixel_height / pixel_width)
            self.graph_plot_1.setAspectLocked(True, ratio=pixel_height / pixel_width)
            self.graph_plot_0.setRange(xRange=[self.x_min, self.x_max],
                                       yRange=[self.y_min, self.y_max], padding=0)
            self.graph_plot_1.setRange(xRange=[self.x_min, self.x_max],
                                       yRange=[self.y_min, self.y_max], padding=0)

    def set_graph_image(self, img_0: np.ndarray, img_1: np.ndarray, levels=None):
        self.graph_img_item_0.setImage(img_0, autoLevels=(levels is None))
        if levels is not None:
            self.graph_img_item_0.setLevels(levels)
            self.color_bar_0.setLevels(low=levels[0], high=levels[1])
        else:
            lo, hi = float(img_0.min()), float(img_0.max())
            self.color_bar_0.setLevels(low=lo, high=hi)

        self.graph_img_item_1.setImage(img_1, autoLevels=(levels is None))
        if levels is not None:
            self.graph_img_item_1.setLevels(levels)
            self.color_bar_1.setLevels(low=levels[0], high=levels[1])
        else:
            lo, hi = float(img_1.min()), float(img_1.max())
            self.color_bar_1.setLevels(low=lo, high=hi)

    def on_psr_frame(self):
        counts_0 = np.array(self.photon_pool.buf_0)
        counts_1 = np.array(self.photon_pool.buf_1)
        self.stream_trace_update(self.photon_pool.xt, counts_0, counts_1)

        if self.btn_profile_0.isChecked():
            self._draw_profile(self.photon_pool.img_0, self.data_plot_0,
                               self.rb_axis_x_0.isChecked(),
                               self.spin_center_0.value(),
                               self.spin_width_0.value())
        if self.btn_profile_1.isChecked():
            self._draw_profile(self.photon_pool.img_1, self.data_plot_1,
                               self.rb_axis_x_1.isChecked(),
                               self.spin_center_1.value(),
                               self.spin_width_1.value())
        if self.psr_mode:
            self.psr_fn += 1
            self.set_graph_with_axes(self.photon_pool.img_0, self.photon_pool.img_1)

    def image_metrics(self, img_0, img_1):
        img_0_sobel = ipr.calculate_focus_measure_with_sobel(img_0)
        img_0_lap = ipr.calculate_focus_measure_with_laplacian(img_0)
        img_1_sobel = ipr.calculate_focus_measure_with_sobel(img_1)
        img_1_lap = ipr.calculate_focus_measure_with_laplacian(img_1)
        self.display_metrics(img_0_sobel, img_0_lap, img_1_sobel, img_1_lap)

    def display_metrics(self, img_0_sobel: float | None, img_0_lap: float | None,
                        img_1_sobel: float | None, img_1_lap: float | None):
        if img_0_sobel is not None:
            self.lcdNumber_img_0_sobel.display(float(img_0_sobel))
        if img_0_lap is not None:
            self.lcdNumber_img_0_laplacian.display(float(img_0_lap))
        if img_1_sobel is not None:
            self.lcdNumber_img_1_sobel.display(float(img_1_sobel))
        if img_1_lap is not None:
            self.lcdNumber_img_1_laplacian.display(float(img_1_lap))
