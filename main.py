import datetime
import getpass
import logging
import os
import sys
import json
from PyQt6.QtWidgets import QApplication, QFileDialog

from devices import device
import executor
from gui import main_window
from processors import processor


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
        "All Files (*)"  # Or specify e.g. "Images (*.png *.jpg);;All Files (*)"
    )
    return file_path if file_path else None


def load_config(dfd):
    with open(dfd, 'r') as f:
        cfg = json.load(f)
    return cfg


def change_config(values, dfd):
    with open(dfd, 'w') as f:
        json.dump(values, f)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet("""
    QWidget { background-color: #232629; color: #f0f0f0; font-size: 14px; }
    QPushButton { background-color: #444; border: 1px solid #555; color: #f0f0f0; }
    QPushButton:hover { background-color: #666; }
    QLabel { color: #e0e0e0; }
    QSpinBox { background-color: #222; color: #f0f0f0; border: 1px solid #333; }
    """)

    data_folder = setup_folder()
    error_logger = setup_logger(data_folder)

    selected_file = select_file_from_folder(None, data_folder)
    if not selected_file:
        print("No file selected. Exiting.")
        sys.exit(0)

    print(f"Selected file: {selected_file}")
    configs = load_config(selected_file)
    devices = device.DeviceManager(config=configs, logg=error_logger, path=data_folder)
    proc = processor.ProcessorManager(config=configs, logg=error_logger, path=data_folder)
    mwd = main_window.MainWindow(config=configs, logg=error_logger, path=data_folder)
    cmd_exc = executor.CommandExecutor(devices, mwd, proc, data_folder, error_logger)
    mwd.aboutToClose.connect(devices.close)

    mwd.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
