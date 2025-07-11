import matplotlib
import numpy as np
from PyQt6.QtCore import pyqtSlot
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QGridLayout
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
        self.update_image(np.zeros((2048, 2048), dtype=np.uint8))

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        self.status_label = cw.LabelWidget("No image")
        layout.addWidget(self.status_label)
        self.ax, self.canvas = self._create_image_window()
        layout.addWidget(self.canvas)
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
