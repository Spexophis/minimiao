# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import datetime
import getpass
import logging
import os
import sys
import json
from PyQt6.QtWidgets import QApplication, QFileDialog

from .devices import device
from . import executor
from .gui import main_window
from .computations import computator


def setup_folder():
    documents_dir = r'C:\\Users\\Public\\Data'
    today_str = datetime.datetime.now().strftime("%Y%m%d")
    folder_name = f"{today_str}"
    full_path = os.path.join(documents_dir, folder_name)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        print(f"Created folder: {full_path}")
    else:
        print(f"Folder already exists: {full_path}")
    return full_path


def setup_logger(log_path):
    today = datetime.datetime.now()
    log_name = today.strftime("%Y%m%d_%H%M.log")
    log_file = os.path.join(log_path, log_name)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=log_file,
                        filemode='w')
    logger = logging.getLogger('shared_logger')
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info("Logger initialized.")
    return logger


def select_file_from_folder(parent, data_folder):
    file_path, _ = QFileDialog.getOpenFileName(
        parent,
        "Select a File",
        data_folder,
        "All Files (*)"
    )
    return file_path if file_path else None


def load_config(dfd):
    with open(dfd, 'r') as f:
        cfg = json.load(f)
    return cfg


def change_config(values, dfd):
    with open(dfd, 'w') as f:
        json.dump(values, f)


class AppWrapper:
    def __init__(self):
        self.app = QApplication(sys.argv)  # Create an instance of QApplication
        self.app.setStyleSheet("""
        QWidget { background-color: #232629; color: #f0f0f0; font-size: 14pt; }
        QPushButton { background-color: #444; border: 1px solid #555; color: #f0f0f0; }
        QPushButton:hover { background-color: #666; }
        QLabel { color: #e0e0e0; }
        QSpinBox { background-color: #222; color: #f0f0f0; border: 1px solid #333; }
        """)
        self.data_folder = setup_folder()
        self.error_logger = setup_logger(self.data_folder)

        selected_file = select_file_from_folder(None, self.data_folder)
        if not selected_file:
            print("No file selected. Exiting.")
            sys.exit(0)

        print(f"Selected file: {selected_file}")
        self.configs = load_config(selected_file)
        self.devices = device.DeviceManager(config=self.configs, logg=self.error_logger, path=self.data_folder)
        self.cmp = computator.ComputationManager(config=self.configs, logg=self.error_logger, path=self.data_folder)
        self.mwd = main_window.MainWindow(config=self.configs, logg=self.error_logger, path=self.data_folder)
        self.cmd_exc = executor.CommandExecutor(self.devices, self.mwd, self.cmp, self.data_folder, self.error_logger)
        self.mwd.aboutToClose.connect(self.close)

    def run(self):
        try:
            self.mwd.show()
            sys.exit(self.app.exec())
        except Exception as e:
            print(f"Fatal error: {e}")
            sys.exit(1)

    def close(self):
        self.devices.close()
        self.app.exit()


def main():
    app_wrapper = AppWrapper()
    app_wrapper.run()


if __name__ == '__main__':
    main()
