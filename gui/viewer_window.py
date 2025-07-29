import matplotlib
import numpy as np
from PyQt6.QtCore import pyqtSlot, pyqtSignal, Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QGridLayout, QSplitter
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from gui import napari_tool

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
    update_image_signal = pyqtSignal(np.ndarray)

    def __init__(self, config, logg, parent=None):
        super().__init__(parent)
        self.config = config
        self.logg = logg
        self._setup_ui()
        self._set_napari_layers()
        self.update_image_signal.connect(self.update_image)

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
        napari_tool.addNapariGrayclipColormap()
        self.napari_viewer = napari_tool.EmbeddedNapari()
        layout_view.addWidget(self.napari_viewer.get_widget())
        return layout_view

    def _set_napari_layers(self):
        self.napari_layers = {}
        self.img_layers = {0: "Kinetix Camera", 1: "Hamamatsu Camera"}
        for name in reversed(list(self.img_layers.values())):
            self.napari_layers[name] = self.add_napari_layer(name)

    def add_napari_layer(self, name):
        return self.napari_viewer.add_image(np.zeros((1024, 1024)), rgb=False, name=name, blending='additive',
                                           colormap=None, protected=True)

    @pyqtSlot(np.ndarray)
    def update_image(self, img: np.ndarray):
        name = "Kinetix Camera"
        if isinstance(img, np.ndarray):
            self.napari_layers[name].data = img

    def get_image(self, name):
        return self.napari_layers[name].data

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
