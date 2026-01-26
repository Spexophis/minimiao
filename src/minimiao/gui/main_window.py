# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import sys
import os
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog
from . import custom_widgets as cw
from . import controller_panel, ao_panel, viewer_window


class MainWindow(QMainWindow):
    aboutToClose = pyqtSignal()

    def __init__(self, config=None, logg=None, path=None):
        super().__init__()
        self.config = config
        self.logg = logg or logger.setup_logging()
        self.data_folder = path
        self._set_dark_theme()
        self._setup_ui()
        self.dialog, self.dialog_text = None, None

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

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

    @staticmethod
    def refresh_gui():
        QApplication.processEvents()

    def closeEvent(self, event, **kwargs):
        self._cleanup_workers()
        self.aboutToClose.emit()
        self.ctrl_panel.save_spinbox_values()
        self.ao_panel.save_spinbox_values()
        super().closeEvent(event)

    def _cleanup_workers(self):
        """Stop and cleanup all active workers"""
        # Cleanup FFT worker
        if hasattr(self.viewer, 'fft_worker') and self.viewer.fft_worker is not None:
            try:
                self.viewer.fft_worker.stop()
                self.viewer.fft_worker.clear_data()
                self.viewer.fft_worker.deleteLater()
            except Exception as e:
                print(f"Error cleaning up FFT worker: {e}")
            self.viewer.fft_worker = None

        # Cleanup PSR worker
        if hasattr(self.viewer, 'psr_worker') and self.viewer.psr_worker is not None:
            try:
                self.viewer.psr_worker.stop()
                self.viewer.psr_worker.clear_data()
                self.viewer.psr_worker.deleteLater()
            except Exception as e:
                print(f"Error cleaning up PSR worker: {e}")
            self.viewer.psr_worker = None


if __name__ == '__main__':
    import json
    app = QApplication(sys.argv)
    with open(r"C:\Users\ruizhe.lin\Documents\data\config_files\microscope_configurations_20240426.json", 'r') as f:
        cfg = json.load(f)
    win = MainWindow(config=cfg)
    win.show()
    sys.exit(app.exec())
