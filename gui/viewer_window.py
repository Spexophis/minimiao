import matplotlib
import numpy as np
from PyQt6.QtCore import pyqtSlot, pyqtSignal, Qt
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
        super().__init__(parent)
        fig = Figure(dpi=dpi, facecolor="#232629")
        self.axes = fig.add_subplot(111, facecolor="#232629")
        super().__init__(fig)
        self.setStyleSheet("background-color: #232629;")


class LiveViewer(QWidget):
    update_image_signal = pyqtSignal(np.ndarray)

    def __init__(self, config, logg, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.config = config
        self.logg = logg
        self._setup_ui()
        self._im = None
        self._orig_img = np.random.rand(2048, 2048) * 4096
        self._orig_img_shape = (2048, 2048)
        self._downsample_factor = 4
        self.update_image(self._orig_img)
        self.update_image_signal.connect(self.update_image)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        self.status_label = cw.LabelWidget("No image")
        layout.addWidget(self.status_label)
        self.ax, self.canvas = self._create_image_window()

        layout.addWidget(self.canvas)

        self.pixel_pos_label = cw.LabelWidget("Pixel: - , -")
        layout.addWidget(self.pixel_pos_label)

        self.contrast_slide = self._create_slide()
        layout.addLayout(self.contrast_slide)

        self.plot_layout = self._create_plot_widgets()
        layout.addLayout(self.plot_layout)
        layout.addStretch(1)
        self.setLayout(layout)

    @staticmethod
    def _create_image_window():
        fig = Figure(figsize=(8, 8), facecolor="#232629")
        ax = fig.add_subplot(111, facecolor="#232629")
        ax.axis('off')
        canvas = FigureCanvas(fig)
        return ax, canvas

    def _create_slide(self):
        contrast_layout = QHBoxLayout()
        contrast_label = cw.LabelWidget("Contrast:")
        slider = cw.SliderWidget(mi=0, ma=65535, value=65535)
        slider.setOrientation(Qt.Orientation.Horizontal)
        slider.valueChanged.connect(self._on_contrast_changed)
        self.contrast_value_label = cw.LabelWidget("65536")
        contrast_layout.addWidget(contrast_label)
        contrast_layout.addWidget(slider)
        contrast_layout.addWidget(self.contrast_value_label)
        return contrast_layout

    def _create_plot_widgets(self):
        layout_plot = QGridLayout()
        self.canvas_show = MplCanvas(self, dpi=80)
        self.canvas_plot = MplCanvas(self, dpi=80)
        toolbar = NavigationToolbar(self.canvas_plot, self)
        layout_plot.addWidget(toolbar, 0, 0, 1, 2)
        layout_plot.addWidget(self.canvas_show, 1, 0, 1, 1)
        layout_plot.addWidget(self.canvas_plot, 1, 1, 1, 1)
        return layout_plot

    @pyqtSlot(np.ndarray)
    def update_image(self, img: np.ndarray):
        self._orig_img = img
        self._orig_img_shape = img.shape
        factor = max(img.shape[0] // 512, 1)
        self._downsample_factor = factor
        img_disp = img[::factor, ::factor] if factor > 1 else img
        self.status_label.setText(f"Image shape: {img.shape} (displayed {img_disp.shape})")

        if self._im is None:
            self.ax.clear()
            self.ax.axis('off')
            self._im = self.ax.imshow(img_disp, cmap='gray', vmin=0, vmax=255, animated=True)
        else:
            self._im.set_data(img_disp)
        self.canvas.draw_idle()

    @pyqtSlot(int)
    def _on_contrast_changed(self, val: int):
        self.contrast_value_label.setText(str(val))
        if self._im is not None:
            self._im.set_clim(0, val)
            self.canvas.draw_idle()

    def _on_mouse_move(self, event):
        if event.inaxes == self.ax and self._im is not None:
            try:
                x_disp, y_disp = int(event.xdata + 0.5), int(event.ydata + 0.5)
                factor = getattr(self, "_downsample_factor", 1)
                orig_x = x_disp * factor
                orig_y = y_disp * factor
                orig_shape = getattr(self, "_orig_img_shape", None)
                if orig_shape and 0 <= orig_x < orig_shape[1] and 0 <= orig_y < orig_shape[0]:
                    val = self._orig_img[orig_y, orig_x]
                    self.pixel_pos_label.setText(f"Pixel: {orig_x}, {orig_y}   Value: {val:.1f}")
                else:
                    self.pixel_pos_label.setText("Pixel: - , -")
            except Exception:
                self.pixel_pos_label.setText("Pixel: - , -")
        else:
            self.pixel_pos_label.setText("Pixel: - , -")

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
