# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QEvent, QObject, QMutex, QMutexLocker, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSplitter

try:
    from . import custom_widgets as cw
except ImportError:
    from minimiao.gui import custom_widgets as cw

try:
    from . import vispy_viewer
except ImportError:
    from minimiao.gui import vispy_viewer


class FramePool(QObject):
    """Fixed set of preallocated frame buffers shared between the camera thread and the GUI thread."""

    def __init__(self, shape=(2400, 2400), dtype=np.uint16, n_buffers=4):
        super().__init__()
        self.shape = tuple(shape)
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

    def release(self, idx: int):
        """Return a buffer to the free list after the viewer consumed it."""
        with QMutexLocker(self._m):
            if idx in self._in_use:
                self._in_use.remove(idx)
                self._free.append(idx)


class LiveViewer(QWidget):
    """
    Top: live camera display (vispy, GPU; contrast min/max, fit, auto, cursor readout).
    Bottom: target image for spot picking / calibration results (pyqtgraph). Picked spots
    are drawn as numbered markers; the index matches the hologram labels ("spot i").
    """
    SPOT_DEFAULT_Z_UM = 0.0
    SPOT_DEFAULT_INTENSITY = 1.0
    frame_token_signal = pyqtSignal(object)

    def __init__(self, logg, parent=None):
        super().__init__(parent)
        self.logg = logg
        self.pool = FramePool(shape=(2400, 2400), dtype=np.uint16, n_buffers=4)
        self._setup_ui()

        self.live_worker = None

        self._target_img = None
        self._picking_enabled = False
        self._picking_n = None
        self.target_points = []

        self.image_plot.scene().sigMouseClicked.connect(self._on_target_mouse_clicked)
        self.image_plot.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.image_plot.installEventFilter(self)

        # picked spots: ring markers + index labels; calibration detections: separate cross markers
        self.target_spots_item = pg.ScatterPlotItem(size=12, pen=pg.mkPen((255, 220, 0), width=2),
                                                    brush=None, symbol="o")
        self.image_plot.addItem(self.target_spots_item)
        self._spot_labels = []
        self.calib_spots_item = pg.ScatterPlotItem(size=10, pen=pg.mkPen((0, 220, 255), width=2), symbol="+")
        self.image_plot.addItem(self.calib_spots_item)

        self._setup_signal_connections()

    # ---- layout ----------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Vertical)

        live_widget = QWidget()
        live_widget.setLayout(self._create_live_widgets())
        splitter.addWidget(live_widget)

        image_widget = self._create_image_widgets()
        splitter.addWidget(image_widget)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)
        self.setLayout(layout)

    def _create_live_widgets(self):
        layout_view = QVBoxLayout()
        layout_view.setContentsMargins(4, 4, 4, 4)

        self.live_viewer = vispy_viewer.VispyViewer()
        self.live_viewer.set_levels(0, 65535, 1.0)

        controls = QWidget()
        row = QHBoxLayout(controls)
        self.QSlider_black = cw.SliderWidget(0, 65535, 0)
        self.QSpinBox_black = cw.SpinBoxWidget(0, 65535, 1, 0)
        self.QSlider_white = cw.SliderWidget(0, 65535, 65535)
        self.QSpinBox_white = cw.SpinBoxWidget(0, 65535, 1, 65535)
        self.QPushButton_fit_view = cw.PushButtonWidget("Fit to Window")
        self.QPushButton_contrast_manual = cw.PushButtonWidget("Set")
        self.QPushButton_contrast_auto = cw.PushButtonWidget("Auto Set")
        row.addWidget(cw.LabelWidget("Min"))
        row.addWidget(cw.LabelWidget("0"))
        row.addWidget(self.QSlider_black)
        row.addWidget(self.QSpinBox_black)
        row.addWidget(cw.LabelWidget("Max"))
        row.addWidget(self.QSlider_white)
        row.addWidget(self.QSpinBox_white)
        row.addWidget(cw.LabelWidget("65535"))
        row.addWidget(self.QPushButton_fit_view)
        row.addWidget(self.QPushButton_contrast_manual)
        row.addWidget(self.QPushButton_contrast_auto)

        self.QLabel_cursor = cw.LabelWidget("x:-  y:-  v:-")

        layout_view.addWidget(controls)
        layout_view.addWidget(self.live_viewer, stretch=1)
        layout_view.addWidget(self.QLabel_cursor)
        return layout_view

    def _create_image_widgets(self):
        layout_image = QVBoxLayout()

        self.image_plot = pg.PlotWidget()
        self.image_plot.setAspectLocked(True)
        self.image_plot.getPlotItem().hideAxis("left")
        self.image_plot.getPlotItem().hideAxis("bottom")

        self.graph_img_item = pg.ImageItem(axisOrder="row-major")
        self.image_plot.addItem(self.graph_img_item)
        self.image_plot.invertY(True)

        layout_image.addWidget(cw.LabelWidget("Target Image (spot picking / calibration)"))
        layout_image.addWidget(self.image_plot, stretch=1)

        image_widget = QWidget()
        image_widget.setLayout(layout_image)

        return image_widget

    def _setup_signal_connections(self):
        self.QSlider_black.valueChanged.connect(self.on_black_change)
        self.QSlider_white.valueChanged.connect(self.on_white_change)
        self.QPushButton_fit_view.clicked.connect(self.fit_view)
        self.QPushButton_contrast_auto.clicked.connect(self.auto_contrast)
        self.QPushButton_contrast_manual.clicked.connect(self.manual_contrast)
        self.live_viewer.mousePixelChanged.connect(self.on_mouse)
        self.live_viewer.frameConsumed.connect(self._release_token, Qt.ConnectionType.QueuedConnection)
        self.live_viewer.frameDiscarded.connect(self._release_token, Qt.ConnectionType.QueuedConnection)
        self.frame_token_signal.connect(self.on_frame_token, Qt.ConnectionType.QueuedConnection)

    # ---- live camera display ---------------------------------------------------

    def on_camera_update_from_thread(self, frame: np.ndarray):
        """Safe from any thread: copies the frame into a pool buffer and hands its token to the GUI thread.
        Frames are dropped (not queued) when the GUI is behind."""
        if frame is None:
            return
        frame = np.asarray(frame)
        if frame.ndim == 3 and frame.shape[-1] == 1:
            frame = frame[..., 0]
        if frame.ndim != 2:
            return
        if frame.dtype != np.uint16:
            frame = frame.astype(np.uint16, copy=False)

        pool = self.pool
        if pool.shape != frame.shape:
            # ROI / binning changed: new pool; buffers still in flight are released to their own pool
            pool = FramePool(shape=frame.shape, dtype=np.uint16, n_buffers=4)
            self.pool = pool
        idx = pool.acquire()
        if idx is None:
            return
        np.copyto(pool.buffer(idx), frame, casting="no")
        self.frame_token_signal.emit((pool, idx))

    @pyqtSlot(object)
    def on_frame_token(self, token):
        pool, idx = token
        self.live_viewer.set_frame(pool.buffer(idx), token=token)

    @pyqtSlot(object)
    def _release_token(self, token):
        pool, idx = token
        pool.release(idx)

    def on_mouse(self, ix, iy, val):
        if ix < 0:
            self.QLabel_cursor.setText("x:-  y:-  v:-")
        else:
            self.QLabel_cursor.setText(f"x:{ix}  y:{iy}  v:{val}")

    @pyqtSlot()
    def fit_view(self):
        self.live_viewer.fit_to_window()

    @pyqtSlot(int)
    def on_black_change(self, value: int):
        self.QSpinBox_black.setValue(value)

    @pyqtSlot(int)
    def on_white_change(self, value: int):
        self.QSpinBox_white.setValue(value)

    @pyqtSlot()
    def manual_contrast(self):
        self.live_viewer.set_levels(self.QSpinBox_black.value(), self.QSpinBox_white.value())

    @pyqtSlot()
    def auto_contrast(self):
        b, w = self.live_viewer.auto_levels()
        self.QSlider_black.setValue(int(b))
        self.QSlider_white.setValue(int(w))

    # ---- target image (spot picking / calibration) -----------------------------

    def set_graph_image(self, img: np.ndarray, levels=None):
        old_shape = None if self._target_img is None else self._target_img.shape
        self._target_img = img
        self.graph_img_item.setImage(img, autoLevels=(levels is None))

        if old_shape != img.shape:
            self.image_plot.getViewBox().autoRange()

    def start_target_picking(self):
        self.logg.info("Start picking...")
        self._picking_enabled = True

        self.target_points = []
        self._update_target_spots_overlay()

        self.image_plot.setFocus()
        self.image_plot.setCursor(Qt.CursorShape.CrossCursor)

        self._update_pick_status()

    def finish_target_picking(self):
        self._picking_enabled = False
        self.image_plot.unsetCursor()
        self._update_pick_status(done=True)
        spots = self.get_target_spots()
        self.logg.info(f"Picked {len(spots)} spot(s) [index: x, y (cam px), z (um), intensity]: "
                       + "; ".join(f"{i}: {x:.0f}, {y:.0f}, {z:g}, {a:g}" for i, (x, y, z, a) in enumerate(spots)))

    def cancel_target_picking(self):
        self._picking_enabled = False
        self.image_plot.unsetCursor()
        self._update_pick_status(cancelled=True)

    def get_target_spots(self):
        """Picked spots as [x, y, z_um, intensity] rows (camera pixels), index = hologram spot index."""
        return [[float(u), float(v), self.SPOT_DEFAULT_Z_UM, self.SPOT_DEFAULT_INTENSITY]
                for (u, v) in self.target_points]

    def show_calibration_points(self, xs, ys):
        """Overlay calibration detections (no index labels; picked spots stay visible)."""
        self.calib_spots_item.setData(list(xs), list(ys))

    def clear_calibration_points(self):
        self.calib_spots_item.setData([], [])

    def _update_pick_status(self, done=False, cancelled=False):
        n = len(getattr(self, "target_points", []))
        if cancelled:
            title = "Picking cancelled"
        elif done:
            title = f"Picked {n} spot(s)"
        elif self._picking_enabled:
            title = f"Picking: {n} spot(s)  |  Left-click add, Backspace undo, Enter finish, Esc cancel"
        else:
            title = f"Spots: {n}"
        self.image_plot.getPlotItem().setTitle(title)

    def _on_target_mouse_clicked(self, ev):
        if not self._picking_enabled or self._target_img is None:
            return

        # right-click = finish (optional)
        if ev.button() == Qt.MouseButton.RightButton:
            self.finish_target_picking()
            return

        if ev.button() != Qt.MouseButton.LeftButton:
            return

        vb = self.image_plot.getPlotItem().vb
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return

        p = vb.mapSceneToView(ev.scenePos())
        x, y = float(p.x()), float(p.y())

        h, w = self._target_img.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return

        u = int(np.clip(np.floor(x), 0, w - 1))
        v = int(np.clip(np.floor(y), 0, h - 1))
        self.target_points.append((u, v))
        self._update_target_spots_overlay()
        self._update_pick_status()

    def _update_target_spots_overlay(self):
        for t in self._spot_labels:
            self.image_plot.removeItem(t)
        self._spot_labels = []
        if not self.target_points:
            self.target_spots_item.setData([], [])
            return
        xs = [p[0] for p in self.target_points]
        ys = [p[1] for p in self.target_points]
        self.target_spots_item.setData(xs, ys)
        for i, (u, v) in enumerate(self.target_points):
            label = pg.TextItem(str(i), color=(255, 220, 0), anchor=(-0.3, 1.1))
            label.setPos(u, v)
            self.image_plot.addItem(label)
            self._spot_labels.append(label)

    def eventFilter(self, obj, event):
        if obj is self.image_plot and self._picking_enabled:
            if event.type() == QEvent.Type.KeyPress:
                key = event.key()

                if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                    self.finish_target_picking()
                    return True

                if key == Qt.Key.Key_Backspace:
                    if self.target_points:
                        self.target_points.pop()
                        self._update_target_spots_overlay()
                    return True

                if key == Qt.Key.Key_Escape:
                    self.cancel_target_picking()
                    return True

        return super().eventFilter(obj, event)
