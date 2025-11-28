import matplotlib
import numpy as np
from PyQt6.QtCore import QObject, QMutex, QMutexLocker, pyqtSlot
from PyQt6.QtCore import pyqtSlot, pyqtSignal, Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QSplitter, QSizePolicy
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from gui import gl_viewer

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
        self.image_viewer.set_frame(np.random.randint(0, 2**14, size=(2048, 2048), dtype=np.uint16))

        self.pool = FramePool(shape=(2048, 2048), dtype=np.uint16, n_buffers=4)

        self.image_viewer.frameConsumed.connect(self.pool.release, Qt.ConnectionType.QueuedConnection)
        self.image_viewer.frameDiscarded.connect(self.pool.release, Qt.ConnectionType.QueuedConnection)

        self.frame_idx_signal.connect(self.on_frame_idx, Qt.ConnectionType.QueuedConnection)

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

    def _create_image_widgets(self):
        layout_view = QVBoxLayout()
        layout_view.setContentsMargins(4, 4, 4, 4)

        self.image_viewer = gl_viewer.GLGray16Viewer(use_pbo=True)
        self.image_viewer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        black = getattr(self.config, "black_u16", 0)
        white = getattr(self.config, "white_u16", 65535)
        gamma = getattr(self.config, "gamma", 1.0)
        self.image_viewer.set_levels(black, white, gamma)

        layout_view.addWidget(self.image_viewer, stretch=1)
        return layout_view

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
    def on_frame_idx(self, idx: int):
        # Pass the *pre-allocated* array directly to GL viewer with token=idx
        frame = self.pool.buffer(idx)
        self.image_viewer.set_frame(frame, token=idx)

    def _create_plot_widgets(self):
        layout_plot = QGridLayout()
        self.canvas_show = MplCanvas(self, dpi=64)
        self.canvas_plot = MplCanvas(self, dpi=64)
        toolbar = NavigationToolbar(self.canvas_plot, self)
        layout_plot.addWidget(toolbar, 0, 0, 1, 2)
        layout_plot.addWidget(self.canvas_show, 1, 0, 1, 1)
        layout_plot.addWidget(self.canvas_plot, 1, 1, 1, 1)
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
