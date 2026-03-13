# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


from collections import deque

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, pyqtSignal, Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSplitter, QGridLayout

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
        pass

    def _create_plot_0_widget(self):
        layout_plot_0 = QGridLayout()

        self.graph_plot_0 = pg.PlotWidget()
        self.graph_plot_0.setAspectLocked(True)
        self.graph_plot_0.setLabel('left', 'Y Position', units='v')
        self.graph_plot_0.setLabel('bottom', 'X Position', units='v')
        self.graph_plot_0.getAxis('left').enableAutoSIPrefix(False)
        self.graph_plot_0.getAxis('bottom').enableAutoSIPrefix(False)

        self.graph_img_item_0 = pg.ImageItem(axisOrder="row-major")  # numpy (H,W)
        self.graph_plot_0.addItem(self.graph_img_item_0)
        self.graph_plot_0.invertY(True)

        self.data_plot_0 = pg.PlotWidget()
        self.data_plot_0.showGrid(x=True, y=True)

        self.data_curve_0 = self.data_plot_0.plot()

        self.data_curve_0.setDownsampling(auto=True, method="peak")
        self.data_curve_0.setSkipFiniteCheck(True)

        pi_0 = self.data_plot_0.getPlotItem()
        pi_0.setClipToView(True)
        pi_0.enableAutoRange(x=False)

        layout_plot_0.addWidget(cw.LabelWidget(str('MPD #0')), 0, 0)
        layout_plot_0.addWidget(self.graph_plot_0, 1, 0)
        layout_plot_0.addWidget(self.data_plot_0, 1, 1)

        plot_0_widget = QWidget()
        plot_0_widget.setLayout(layout_plot_0)

        return plot_0_widget

    def _create_plot_1_widget(self):
        layout_plot_1 = QGridLayout()

        self.QComboBox_plot_selection = cw.ComboBoxWidget(list_items=["MPD #1", "PMT"], length=80)

        self.graph_plot_1 = pg.PlotWidget()
        self.graph_plot_1.setAspectLocked(True)
        self.graph_plot_1.setLabel('left', 'Y Position', units='v')
        self.graph_plot_1.setLabel('bottom', 'X Position', units='v')
        self.graph_plot_1.getAxis('left').enableAutoSIPrefix(False)
        self.graph_plot_1.getAxis('bottom').enableAutoSIPrefix(False)

        self.graph_img_item_1 = pg.ImageItem(axisOrder="row-major")  # numpy (H,W)
        self.graph_plot_1.addItem(self.graph_img_item_1)
        self.graph_plot_1.invertY(True)

        self.data_plot_1 = pg.PlotWidget()
        self.data_plot_1.showGrid(x=True, y=True)

        self.data_curve_1 = self.data_plot_1.plot()

        self.data_curve_1.setDownsampling(auto=True, method="peak")
        self.data_curve_1.setSkipFiniteCheck(True)

        pi_1 = self.data_plot_1.getPlotItem()
        pi_1.setClipToView(True)
        pi_1.enableAutoRange(x=False)

        layout_plot_1.addWidget(self.QComboBox_plot_selection, 0, 0)
        layout_plot_1.addWidget(self.graph_plot_1, 1, 0)
        layout_plot_1.addWidget(self.data_plot_1, 1, 1)

        plot_1_widget = QWidget()
        plot_1_widget.setLayout(layout_plot_1)

        return plot_1_widget

    def _create_metric_widget(self):
        layout_metric = QGridLayout()

        self.lcdNumber_img_0_max = cw.LCDNumberWidget()
        self.lcdNumber_img_0_min = cw.LCDNumberWidget()
        self.lcdNumber_img_0_avg = cw.LCDNumberWidget()
        self.lcdNumber_img_0_sobel = cw.LCDNumberWidget()
        self.lcdNumber_img_0_laplacian = cw.LCDNumberWidget()
        self.lcdNumber_img_1_max = cw.LCDNumberWidget()
        self.lcdNumber_img_1_min = cw.LCDNumberWidget()
        self.lcdNumber_img_1_avg = cw.LCDNumberWidget()
        self.lcdNumber_img_1_sobel = cw.LCDNumberWidget()
        self.lcdNumber_img_1_laplacian = cw.LCDNumberWidget()

        layout_metric.addWidget(cw.LabelWidget(str('Max'), align=2), 0, 0)
        layout_metric.addWidget(self.lcdNumber_img_0_max, 0, 1)
        layout_metric.addWidget(self.lcdNumber_img_1_max, 1, 1)
        layout_metric.addWidget(cw.LabelWidget(str('Min'), align=2), 0, 2)
        layout_metric.addWidget(self.lcdNumber_img_0_min, 0, 3)
        layout_metric.addWidget(self.lcdNumber_img_1_min, 1, 3)
        layout_metric.addWidget(cw.LabelWidget(str('Avg'), align=2), 0, 4)
        layout_metric.addWidget(self.lcdNumber_img_0_avg, 0, 5)
        layout_metric.addWidget(self.lcdNumber_img_1_avg, 1, 5)
        layout_metric.addWidget(cw.LabelWidget(str('Sobel'), align=2), 0, 6)
        layout_metric.addWidget(self.lcdNumber_img_0_sobel, 0, 7)
        layout_metric.addWidget(self.lcdNumber_img_1_sobel, 1, 7)
        layout_metric.addWidget(cw.LabelWidget(str('Laplacian'), align=2), 0, 8)
        layout_metric.addWidget(self.lcdNumber_img_0_laplacian, 0, 9)
        layout_metric.addWidget(self.lcdNumber_img_1_laplacian, 1, 9)

        metric_widget = QWidget()
        metric_widget.setLayout(layout_metric)

        return metric_widget

    def set_plot_1(self, n):
        self.QComboBox_plot_selection.setCurrentIndex(n)

    def plot_trace(self,
                   y,
                   x=None,
                   overlay=False):
        y = np.asarray(y)
        if y.size == 0:
            return
        if not overlay:
            self.data_plot_0.clear()
            self._overlay_n = 0
        if x is None:
            x = np.arange(y.size)
        self.data_plot_0.enableAutoRange(x=True)
        color = pg.intColor(self._overlay_n, hues=12)  # 12 distinct-ish hues, repeats after 12
        pen = pg.mkPen(color=color, width=1.)
        self._overlay_n += 1
        self.data_plot_0.plot(x, y, pen=pen)

    def stream_trace(self,
                     x: np.ndarray,
                     y_0: np.ndarray, y_1: np.ndarray):
        """
        Update the 1D trace plot lively.
        """
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

    def stream_trace_update(self,
                            xt: np.ndarray,
                            counts_0: np.ndarray, counts_1: np.ndarray):
        self.data_curve_0.setData(xt, counts_0)
        self.data_curve_1.setData(xt, counts_1)

    def set_graph_with_axes(self,
                            img_0: np.ndarray, img_1: np.ndarray,
                            x_axis=None, y_axis=None,
                            levels=None):
        self.set_graph_image(img_0, img_1, levels)
        if self.psr_fn % 8 == 0:
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
        # else:
        #     self.graph_img_item_0.setRect(pg.QtCore.QRectF(self.x_min, self.y_min,
        #                                                    self.x_max - self.x_min,
        #                                                    self.y_max - self.y_min))
        #     self.graph_img_item_1.setRect(pg.QtCore.QRectF(self.x_min, self.y_min,
        #                                                    self.x_max - self.x_min,
        #                                                    self.y_max - self.y_min))
        #
        #     self.graph_plot_0.setRange(xRange=[self.x_min, self.x_max],
        #                                yRange=[self.y_min, self.y_max], padding=0)
        #     self.graph_plot_1.setRange(xRange=[self.x_min, self.x_max],
        #                                yRange=[self.y_min, self.y_max], padding=0)

    def set_graph_image(self,
                        img_0: np.ndarray, img_1: np.ndarray,
                        levels=None):
        self.graph_img_item_0.setImage(img_0, autoLevels=(levels is None))
        if levels is not None:
            self.graph_img_item_0.setLevels(levels)
        self.graph_img_item_1.setImage(img_1, autoLevels=(levels is None))
        if levels is not None:
            self.graph_img_item_1.setLevels(levels)

    def on_psr_frame(self):
        self.stream_trace_update(self.photon_pool.xt, np.array(self.photon_pool.buf_0),
                                 np.array(self.photon_pool.buf_1))
        if self.psr_mode:
            self.psr_fn += 1
            self.set_graph_with_axes(self.photon_pool.img_0, self.photon_pool.img_1)

    def image_metrics(self, img_0, img_1):
        img_0_max, img_0_min, img_0_avg = ipr.img_statistics(img_0)
        img_1_max, img_1_min, img_1_avg = ipr.img_statistics(img_1)
        img_0_sobel = ipr.calculate_focus_measure_with_sobel(img_0)
        img_0_lap = ipr.calculate_focus_measure_with_laplacian(img_0)
        img_1_sobel = ipr.calculate_focus_measure_with_sobel(img_1)
        img_1_lap = ipr.calculate_focus_measure_with_laplacian(img_1)
        self.display_metrics(img_0_max, img_0_min, img_0_avg, img_0_sobel, img_0_lap, img_1_max, img_1_min, img_1_avg,
                             img_1_sobel, img_1_lap)

    def display_metrics(self,
                        img_0_max: int | None, img_0_min: int | None, img_0_avg: float | None,
                        img_0_sobel: float | None, img_0_lap: float | None,
                        img_1_max: int | None, img_1_min: int | None, img_1_avg: float | None,
                        img_1_sobel: float | None, img_1_lap: float | None):
        if img_0_max is not None:
            self.lcdNumber_img_0_max.display(img_0_max)
        if img_0_min is not None:
            self.lcdNumber_img_0_min.display(img_0_min)
        if img_0_avg is not None:
            self.lcdNumber_img_0_avg.display(img_0_avg)
        if img_0_sobel is not None:
            self.lcdNumber_img_0_sobel.display(img_0_sobel)
        if img_0_lap is not None:
            self.lcdNumber_img_0_laplacian.display(img_0_lap)
        if img_1_max is not None:
            self.lcdNumber_img_1_max.display(img_1_max)
        if img_1_min is not None:
            self.lcdNumber_img_1_min.display(img_1_min)
        if img_1_avg is not None:
            self.lcdNumber_img_1_avg.display(img_1_avg)
        if img_1_sobel is not None:
            self.lcdNumber_img_1_sobel.display(img_1_sobel)
        if img_1_lap is not None:
            self.lcdNumber_img_1_laplacian.display(img_1_lap)
