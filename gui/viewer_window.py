import matplotlib
import numpy as np
from PyQt6.QtCore import pyqtSlot, Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QHBoxLayout, QSlider, QLabel
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from gui import custom_widgets as cw

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


class LiveViewer(QWidget):
    def __init__(self, config, logg, parent=None):
        super().__init__(parent)
        self.config = config
        self.logg = logg
        self._setup_ui()
        self._im = None
        self.update_image(np.random.rand(2048, 2048) * 2048)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        self.status_label = cw.LabelWidget("No image")
        layout.addWidget(self.status_label)
        self.ax, self.canvas = self._create_image_window()
        layout.addWidget(self.canvas)

        self._contrast_layout = QHBoxLayout()
        self._contrast_label = QLabel("Contrast:")
        self._slider = QSlider()
        self._slider.setOrientation(Qt.Orientation.Horizontal)
        self._slider.setMinimum(10)
        self._slider.setMaximum(255)
        self._slider.setValue(255)
        self._slider.valueChanged.connect(self._on_contrast_changed)
        self._contrast_value_label = QLabel("255")
        self._contrast_layout.addWidget(self._contrast_label)
        self._contrast_layout.addWidget(self._slider)
        self._contrast_layout.addWidget(self._contrast_value_label)
        layout.addLayout(self._contrast_layout)

        self._plot_layout = self._create_plot_widgets()
        layout.addLayout(self._plot_layout)
        layout.addStretch(1)
        self.setLayout(layout)

    @staticmethod
    def _create_image_window():
        fig = Figure(figsize=(8, 8), facecolor="#232629")
        ax = fig.add_subplot(111, facecolor="#232629")
        ax.axis('off')
        canvas = FigureCanvas(fig)
        return ax, canvas

    def _create_plot_widgets(self):
        layout_plot = QGridLayout()
        self.canvas_show = MplCanvas(self, dpi=64)
        self.canvas_plot = MplCanvas(self, dpi=64)
        toolbar = NavigationToolbar(self.canvas_plot, self)
        layout_plot.addWidget(toolbar, 0, 0, 1, 2)
        layout_plot.addWidget(self.canvas_show, 1, 0, 1, 1)
        layout_plot.addWidget(self.canvas_plot, 1, 1, 1, 1)
        return layout_plot

    @pyqtSlot(np.ndarray)
    def update_image(self, img: np.ndarray):
        factor = max(img.shape[0] // 512, 1)
        img_disp = img[::factor, ::factor] if factor > 1 else img
        self.status_label.setText(f"Image shape: {img.shape} (displayed {img_disp.shape})")

        if self._im is None:
            self.ax.clear()
            self.ax.axis('off')
            self._im = self.ax.imshow(img_disp, cmap='gray', vmin=0, vmax=255, animated=True)
        else:
            self._im.set_data(img_disp)
        self.canvas.draw_idle()

    def _on_contrast_changed(self, val):
        self._contrast_value_label.setText(str(val))
        if self._im is not None:
            self._im.set_clim(0, val)
            self.canvas.draw_idle()

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
