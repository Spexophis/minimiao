# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import os
import sys

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog

try:
    from . import controller_panel, hg_panel, viewer_window
except ImportError:
    from minimiao.gui import controller_panel, hg_panel, viewer_window

try:
    from . import custom_widgets as cw
except ImportError:
    from minimiao.gui import custom_widgets as cw

from minimiao import logger


class MainWindow(QMainWindow):
    aboutToClose = pyqtSignal()

    def __init__(self, logg=None, path=None):
        super().__init__()
        self.logg = logg or logger.setup_logging()
        self.data_folder = path
        self._set_dark_theme()
        self._setup_ui()
        self.dialog, self.dialog_text = None, None

    def closeEvent(self, event, **kwargs):
        self.aboutToClose.emit()
        self.ctrl_panel.save_spinbox_values()
        self.hg_panel.save_spinbox_values()
        super().closeEvent(event)

    def _setup_ui(self):
        self.ctrl_panel = controller_panel.ControlPanel(self.logg)
        self.ctrl_dock = cw.DockWidget("Ctrl Panel")
        self.ctrl_dock.setWidget(self.ctrl_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.ctrl_dock)

        self.viewer = viewer_window.LiveViewer(self.logg)
        self.setCentralWidget(self.viewer)

        self.hg_panel = hg_panel.HGPanel(self.logg)
        self.hg_dock = cw.DockWidget("HG Panel")
        self.hg_dock.setWidget(self.hg_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.hg_dock)

    def _set_dark_theme(self):
        dark_stylesheet = """
        QWidget {
            background-color: #232629;
            color: #f0f0f0;
            font-size: 12px;
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

    def get_file_dialog(self, sw="Save File"):
        file_dialog = cw.FileDialogWidget(name=sw, file_filter="All Files (*)", default_dir=self.data_folder)
        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            selected_file = file_dialog.selectedFiles()
            if selected_file:
                return os.path.basename(selected_file[0])
            else:
                return None
        return None

    def get_dialog(self, txt, interrupt=False):
        self.dialog, self.dialog_text = cw.create_dialog(labtex=True, interrupt=interrupt)
        self.dialog.setModal(True)
        self.dialog.show()
        self.dialog_text.setText(f"Task {txt} is running, please wait...")
        self.refresh_gui()

    def select_file_from_folder(self, data_folder):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select a File", data_folder, "All Files (*)")
        return file_path if file_path else None

    def get_file_name(self):
        selected_file = self.select_file_from_folder(self.data_folder)
        if not selected_file:
            self.logg.error("No file selected.")
            return None
        else:
            self.logg.info(f"Selected file: {selected_file}")
            return selected_file

    @staticmethod
    def refresh_gui():
        QApplication.processEvents()


if __name__ == '__main__':
    import sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
