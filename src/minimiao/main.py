# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import datetime
import getpass
import os
import sys

from PyQt6.QtWidgets import QApplication

try:
    from . import executor, logger
except ImportError:
    from minimiao import executor, logger

try:
    from .computations import computator
except ImportError:
    from minimiao.computations import computator

try:
    from .devices import device
except ImportError:
    from minimiao.devices import device

try:
    from .gui import main_window
except ImportError:
    from minimiao.gui import main_window


def setup_folder():
    documents_dir = os.path.join(os.path.expanduser('~'), 'Documents', 'data')
    today_str = datetime.datetime.now().strftime("%Y%m%d")
    username = getpass.getuser()
    folder_name = f"{today_str}_{username}"
    full_path = os.path.join(documents_dir, folder_name)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        print(f"Created folder: {full_path}")
    else:
        print(f"Folder already exists: {full_path}")
    return full_path


class AppWrapper:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setStyleSheet("""
        QWidget { background-color: #232629; color: #f0f0f0; font-size: 14px; }
        QPushButton { background-color: #444; border: 1px solid #555; color: #f0f0f0; }
        QPushButton:hover { background-color: #666; }
        QLabel { color: #e0e0e0; }
        QSpinBox { background-color: #222; color: #f0f0f0; border: 1px solid #333; }
        """)
        self.data_folder = setup_folder()
        self.error_logger = logger.setup_logger(self.data_folder)
        self.mwd = main_window.MainWindow(logg=self.error_logger, path=self.data_folder)
        self.cmp = computator.ComputationManager(logg=self.error_logger, path=self.data_folder)
        self.devices = device.DeviceManager(logg=self.error_logger)
        # self.cmd_exc = executor.CommandExecutor(self.devices, self.mwd, self.cmp, self.data_folder, self.error_logger)
        self.mwd.aboutToClose.connect(self.close)

    def run(self):
        try:
            self.mwd.show()
            sys.exit(self.app.exec())
        except Exception as e:
            print(f"Fatal error: {e}")
            sys.exit(1)

    def close(self):
        if self.devices is not None:
            self.devices.close()
        self.app.exit()


def main():
    app_wrapper = AppWrapper()
    app_wrapper.run()


if __name__ == '__main__':
    main()
