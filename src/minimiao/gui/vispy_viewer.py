# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.

"""
High-throughput uint16 grayscale image viewer.

Rendering is built directly on vispy's scene graph -- the same GPU
rendering core (SceneCanvas, PanZoomCamera, the Image visual with its
clim/gamma color transform) that napari uses under its own layer system.
"""

import numpy as np
import vispy

vispy.use(app="pyqt6")

from vispy import scene
from PyQt6.QtCore import QSize, pyqtSignal
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QSizePolicy


class VispyViewer(QWidget):
    """
    Call .set_frame(frame_u16) from the GUI thread (or via Qt signal with QueuedConnection).
    """
    frameConsumed = pyqtSignal(object)   # token
    frameDiscarded = pyqtSignal(object)  # token when replaced before upload
    mousePixelChanged = pyqtSignal(int, int, int)  # x, y, value (uint16), -1 if invalid

    def __init__(self, parent=None):
        super().__init__(parent)

        self._display_frame = None
        self._display_token = None
        self._img_w = 0
        self._img_h = 0

        self.black_u16 = 0
        self.white_u16 = 65535
        self.gamma = 1.0

        self.canvas = scene.SceneCanvas(keys=None, show=False, bgcolor="black")
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.PanZoomCamera(aspect=1)
        self.view.camera.flip = (False, True, False)  # row 0 at top, like a numpy image

        self._image = scene.visuals.Image(
            np.zeros((1, 1), dtype=np.uint16),
            parent=self.view.scene,
            cmap="grays",
            clim=(0, 65535),
            interpolation="linear",
            texture_format="auto",
        )
        self.view.camera.rect = (0, 0, 1, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas.native)
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.canvas.events.mouse_move.connect(self._on_mouse_move)

    def sizeHint(self) -> QSize:
        return QSize(700, 700)

    def minimumSizeHint(self) -> QSize:
        return QSize(300, 300)

    def _on_mouse_move(self, event):
        if self._img_w <= 0 or self._img_h <= 0:
            self.mousePixelChanged.emit(-1, -1, -1)
            return

        x, y = self._image.get_transform("canvas", "visual").map(event.pos)[:2]
        ix, iy = int(np.floor(x)), int(np.floor(y))

        if ix < 0 or ix >= self._img_w or iy < 0 or iy >= self._img_h or self._display_frame is None:
            self.mousePixelChanged.emit(-1, -1, -1)
            return

        val = int(self._display_frame[iy, ix])
        self.mousePixelChanged.emit(ix, iy, val)

    def set_levels(self, black_u16: int, white_u16: int, gamma: float = 1.0):
        self.black_u16 = int(np.clip(black_u16, 0, 65535))
        self.white_u16 = int(np.clip(white_u16, 0, 65535))
        if self.white_u16 <= self.black_u16:
            self.white_u16 = min(65535, self.black_u16 + 1)
        self.gamma = float(max(gamma, 1e-6))

        self._image.clim = (self.black_u16, self.white_u16)
        # vispy applies pow(v, gamma); invert so gamma > 1 brightens, matching the old shader.
        self._image.gamma = 1.0 / self.gamma
        self.canvas.update()

    def fit_to_window(self):
        if self._img_w > 0 and self._img_h > 0:
            self.view.camera.rect = (0, 0, self._img_w, self._img_h)

    def auto_levels(self):
        black = self._display_frame.min()
        white = self._display_frame.max()
        self.set_levels(black, white)
        return black, white

    def set_frame(self, frame_u16: np.ndarray, token=None):
        frame = np.ascontiguousarray(frame_u16)
        h, w = frame.shape

        # vispy copies the array into its own buffer synchronously (copy=True),
        # so the caller's buffer is safe to reuse the moment this call returns.
        self._image.set_data(frame, copy=True)

        if (w, h) != (self._img_w, self._img_h):
            self._img_w, self._img_h = w, h
            self.view.camera.rect = (0, 0, w, h)

        self._display_frame = frame
        self._display_token = token
        self.canvas.update()

        if token is not None:
            self.frameConsumed.emit(token)
