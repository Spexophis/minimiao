import sys

from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QDialog, QVBoxLayout, QLabel

import controller
import viewer


class MainWindow(QMainWindow):
    aboutToClose = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.resize(1000, 800)
        self._set_dark_theme()
        self._setup_ui()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        self.ctrl_panel = controller.ControlPanel()
        layout.addWidget(self.ctrl_panel, 1)

        self.viewer = viewer.LiveViewer()
        layout.addWidget(self.viewer, 4)

    def _set_dark_theme(self):
        dark_stylesheet = """
        QWidget {
            background-color: #232629;
            color: #f0f0f0;
            font-size: 14px;
        }
        QPushButton {
            background-color: #444;
            border: 1px solid #555;
            color: #f0f0f0;
            padding: 6px;
            border-radius: 4px;
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

    def closeEvent(self, event):
        self.aboutToClose.emit()
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
