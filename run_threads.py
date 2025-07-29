import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


class LiveViewThread(QThread):
    new_frame = pyqtSignal(np.ndarray)

    def __init__(self, camera, interval_ms=50):
        super().__init__()
        self.camera = camera
        self.interval_ms = interval_ms
        self._running = False

    def run(self):
        self._running = True
        while self._running:
            frame = self.camera.get_last_image()
            self.new_frame.emit(frame)
            self.msleep(self.interval_ms)

    def stop(self):
        self._running = False
        self.wait()


class AcquisitionThread(QThread):
    data_stack = pyqtSignal(np.ndarray)

    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self._running = False

    def run(self):
        self._running = True
        while self._running:
            frames = self.camera.get_data()
            self.data_stack.emit(frames)

    def stop(self):
        self._running = False
        self.wait()
