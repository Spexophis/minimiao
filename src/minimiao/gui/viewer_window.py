# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import matplotlib
import numpy as np
from PyQt6.QtCore import QObject, QMutex, QMutexLocker
from PyQt6.QtCore import pyqtSlot, pyqtSignal, Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QSplitter, QHBoxLayout, QStackedWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from . import gl_viewer
from . import custom_widgets as cw

matplotlib.rcParams.update({
    'axes.facecolor': '#232629',
    'figure.facecolor': '#232629',
    'axes.edgecolor': '#EEEEEE',
    'axes.labelcolor': '#EEEEEE',
    'xtick.color': '#EEEEEE',
    'ytick.color': '#EEEEEE',
    'text.color': '#EEEEEE',
    'grid.color': '#555555',
    'axes.grid': True,
    'savefig.facecolor': '#232629'
})


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, dpi=100):
        fig = Figure(dpi=dpi, facecolor="#232629")
        self.axes = fig.add_subplot(111, facecolor="#232629")
        super().__init__(fig)
        self.setStyleSheet("background-color: #232629;")  # Panel background


class FramePool(QObject):
    def __init__(self, shape=(2048, 2048), dtype=np.uint16, n_buffers=4):
        super().__init__()
        self._buffers = [np.empty(shape, dtype=dtype) for _ in range(n_buffers)]
        self._free = list(range(n_buffers))
        self._in_use = set()
        self._m = QMutex()

    def acquire(self):
        """Reserve a buffer index for writing. Returns idx or None if none free."""
        with QMutexLocker(self._m):
            if not self._free:
                return None
            idx = self._free.pop()
            self._in_use.add(idx)
            return idx

    def buffer(self, idx: int) -> np.ndarray:
        return self._buffers[idx]

    @pyqtSlot(object)
    def release(self, token):
        """Return a buffer to the free list after viewer consumed/discarded it."""
        idx = int(token)
        with QMutexLocker(self._m):
            if idx in self._in_use:
                self._in_use.remove(idx)
                self._free.append(idx)


class LiveViewer(QWidget):
    frame_idx_signal = pyqtSignal(int)

    def __init__(self, config, logg, parent=None):
        super().__init__(parent)
        self.config = config
        self.logg = logg
        self._setup_ui()
        self.h = 1024
        self.w = 1024
        self.pool = FramePool(shape=(self.h, self.w), dtype=np.uint16, n_buffers=4)
        self.fft_mode = False
        self.fft_worker = None
        self.view_stack.setCurrentIndex(0)
        self._setup_signal_connections()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Vertical)

        image_widget = QWidget()
        image_layout = self._create_image_widgets()
        image_widget.setLayout(image_layout)
        splitter.addWidget(image_widget)

        plot_widget = QWidget()
        plot_layout = self._create_plot_widgets()
        plot_widget.setLayout(plot_layout)
        splitter.addWidget(plot_widget)

        layout.addWidget(splitter)
        self.setLayout(layout)

    def _setup_signal_connections(self):
        self.QSlider_black.valueChanged.connect(self.on_black_change)
        self.QSlider_white.valueChanged.connect(self.on_white_change)
        self.QPushButton_contrast_auto.clicked.connect(self.auto_contrast)
        self.QPushButton_contrast_manual.clicked.connect(self.manual_contrast)
        self.image_viewer.mousePixelChanged.connect(self.on_mouse)
        self.image_viewer.frameConsumed.connect(self.pool.release, Qt.ConnectionType.QueuedConnection)
        self.image_viewer.frameDiscarded.connect(self.pool.release, Qt.ConnectionType.QueuedConnection)
        self.frame_idx_signal.connect(self.on_frame_idx, Qt.ConnectionType.QueuedConnection)
        self.QComboBox_viewer_selection.currentIndexChanged.connect(self.switch_viewer)

    def _create_image_widgets(self):
        layout_view = QVBoxLayout()
        layout_view.setContentsMargins(4, 4, 4, 4)

        self.image_viewer = gl_viewer.GLGray16Viewer(use_pbo=True)  # camera frames (big, fast)
        self.image_viewer.set_levels(0, 65535, 1.0)

        self.fft_viewer = gl_viewer.GLGray16Viewer(use_pbo=False)  # FFT frames (smaller; PBO not needed)

        self.view_stack = QStackedWidget()
        self.view_stack.addWidget(self.image_viewer)  # index 0
        self.view_stack.addWidget(self.fft_viewer)  # index 1

        controls = QWidget()
        row = QHBoxLayout(controls)
        self.QComboBox_viewer_selection = cw.ComboBoxWidget(list_items=["Image", "FFT"])
        self.QSlider_black = cw.SliderWidget(0, 65535, 0)
        self.QSpinBox_black = cw.SpinBoxWidget(0, 65535, 1, 0)
        self.QSlider_white = cw.SliderWidget(0, 65535, 65535)
        self.QSpinBox_white = cw.SpinBoxWidget(0, 65535, 1, 65535)
        self.QPushButton_contrast_manual = cw.PushButtonWidget("Set")
        self.QPushButton_contrast_auto = cw.PushButtonWidget("Auto Set")
        row.addWidget(self.QComboBox_viewer_selection)
        row.addWidget(cw.LabelWidget("Min"))
        row.addWidget(cw.LabelWidget("0"))
        row.addWidget(self.QSlider_black)
        row.addWidget(self.QSpinBox_black)
        row.addWidget(cw.LabelWidget("Max"))
        row.addWidget(self.QSlider_white)
        row.addWidget(self.QSpinBox_white)
        row.addWidget(cw.LabelWidget("65535"))
        row.addWidget(self.QPushButton_contrast_manual)
        row.addWidget(self.QPushButton_contrast_auto)

        self.QLabel_cursor = cw.LabelWidget("x:-  y:-  v:-")

        layout_view.addWidget(controls)
        layout_view.addWidget(self.view_stack, stretch=1)
        layout_view.addWidget(self.QLabel_cursor)
        return layout_view

    def on_mouse(self, ix, iy, val):
        if ix < 0:
            self.QLabel_cursor.setText("x:-  y:-  v:-")
        else:
            self.QLabel_cursor.setText(f"x:{ix}  y:{iy}  v:{val}")

    def switch_camera(self, h, w):
        self.h, self.w = h, w
        self.pool = FramePool(shape=(self.h, self.w), dtype=np.uint16, n_buffers=4)
        self.image_viewer.frameConsumed.connect(self.pool.release, Qt.ConnectionType.QueuedConnection)
        self.image_viewer.frameDiscarded.connect(self.pool.release, Qt.ConnectionType.QueuedConnection)

    @pyqtSlot(int)
    def switch_viewer(self, ind: int):
        self.view_stack.setCurrentIndex(ind)

    def on_camera_update_from_thread(self, frame: np.ndarray):
        """Runs in camera thread. Do NOT touch Qt widgets here."""
        if frame is None:
            return

        # normalize shape/dtype
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = frame[..., 0]
        if frame.dtype != np.uint16:
            frame = frame.astype(np.uint16, copy=False)

        idx = self.pool.acquire()
        if idx is None:
            return  # drop if GUI behind

        dst = self.pool.buffer(idx)
        np.copyto(dst, frame, casting="no")

        # send only index to GUI thread
        self.frame_idx_signal.emit(idx)

    @pyqtSlot(int)
    def on_black_change(self, value: int):
        self.QSpinBox_black.setValue(value)

    @pyqtSlot(int)
    def on_white_change(self, value: int):
        self.QSpinBox_white.setValue(value)

    @pyqtSlot()
    def manual_contrast(self):
        self.image_viewer.set_levels(self.QSpinBox_black.value(), self.QSpinBox_white.value())

    @pyqtSlot()
    def auto_contrast(self):
        b, w = self.image_viewer.auto_levels()
        self.QSlider_black.setValue(b)
        self.QSlider_white.setValue(w)

    @pyqtSlot(int)
    def on_frame_idx(self, idx: int):
        self.image_viewer.set_frame(self.pool.buffer(idx), token=idx)
        if self.fft_mode:
            self.fft_worker.push_frame(self.pool.buffer(idx))

    def on_fft_frame(self, frame_u16):
        self.fft_viewer.set_frame(frame_u16)

    def _create_plot_widgets(self):
        layout_plot = QGridLayout()
        self.canvas_plot = MplCanvas(self, dpi=64)
        self.canvas_show = MplCanvas(self, dpi=64)
        toolbar = NavigationToolbar(self.canvas_plot, self)
        layout_plot.addWidget(toolbar, 0, 0, 1, 2)
        layout_plot.addWidget(self.canvas_plot, 1, 0, 1, 1)
        layout_plot.addWidget(self.canvas_show, 1, 1, 1, 1)
        return layout_plot

    def plot_profile(self, data, x=None, sp=None):
        if x is not None:
            self.canvas_plot.axes.plot(x, data)
        else:
            self.canvas_plot.axes.plot(data)
        if sp is not None:
            self.canvas_plot.axes.axhline(y=sp, color='r', linestyle='--')
        self.canvas_plot.axes.grid(True)
        self.canvas_plot.draw()

    def update_plot(self, data, x=None, sp=None):
        self.canvas_plot.axes.cla()
        self.plot_profile(data, x, sp)
