import numpy as np
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class LiveViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        self.status_label = QLabel("No image")
        layout.addWidget(self.status_label)
        self.fig = Figure(figsize=(6, 6), facecolor="#232629")
        self.ax = self.fig.add_subplot(111, facecolor="#232629")
        self.ax.axis('off')
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas, 1)
        self._im = None

    @pyqtSlot(np.ndarray)
    def update_image(self, img: np.ndarray):
        factor = max(img.shape[0] // 512, 1)
        if factor > 1:
            img_disp = img[::factor, ::factor]
        else:
            img_disp = img

        self.status_label.setText(f"Image shape: {img.shape} (displayed {img_disp.shape})")

        if self._im is None:
            self.ax.clear()
            self.ax.axis('off')
            self._im = self.ax.imshow(img_disp, cmap='gray', vmin=0, vmax=255, animated=True)
        else:
            self._im.set_data(img_disp)
        self.canvas.draw_idle()
