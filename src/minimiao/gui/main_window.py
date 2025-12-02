import sys

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QSpinBox, QDoubleSpinBox
from . import custom_widgets as cw
from . import controller_panel, ao_panel, viewer_window


class MainWindow(QMainWindow):
    aboutToClose = pyqtSignal()

    def __init__(self, config=None, logg=None, path=None):
        super().__init__()
        self.config = config
        self.logg = logg or self.setup_logging()
        self.data_folder = path
        self._set_dark_theme()
        self._setup_ui()

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    def closeEvent(self, event, **kwargs):
        self.aboutToClose.emit()
        self.ctrl_panel.save_spinbox_values()
        self.ao_panel.save_spinbox_values()
        super().closeEvent(event)

    def _setup_ui(self):
        self.ctrl_panel = controller_panel.ControlPanel(self.config, self.logg)
        self.ctrl_dock = cw.DockWidget("Control Panel")
        self.ctrl_dock.setWidget(self.ctrl_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.ctrl_dock)

        self.viewer = viewer_window.LiveViewer(self.config, self.logg)
        self.setCentralWidget(self.viewer)

        self.ao_panel = ao_panel.AOPanel(self.config, self.logg)
        self.ao_dock = cw.DockWidget("AO Panel")
        self.ao_dock.setWidget(self.ao_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.ao_dock)

    def _set_dark_theme(self):
        dark_stylesheet = """
        QWidget {
            background-color: #232629;
            color: #f0f0f0;
            font-size: 13px;
        }
        QPushButton {
            background-color: #444;
            border: 1px solid #555;
            color: #f0f0f0;
            padding: 4px;
            border-radius: 2px;
        }
        QPushButton:hover {
            background-color: #666;
        }
        QLabel {
            color: #e0e0e0;
        }
        QSpinBox {
            background-color: #222;
            color: #f0f0f0;
            border: 1px solid #333;
        }
        QGroupBox {
            border: 1px solid #555;
            margin-top: 10px;
        }
        """
        self.setStyleSheet(dark_stylesheet)


if __name__ == '__main__':
    import json
    app = QApplication(sys.argv)
    with open(r"C:\Users\ruizhe.lin\Documents\data\config_files\microscope_configurations_20240426.json", 'r') as f:
        cfg = json.load(f)
    win = MainWindow(config=cfg)
    win.show()
    sys.exit(app.exec())
