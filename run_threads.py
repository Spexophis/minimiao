import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


class AcquisitionThread(QThread):
    new_frame = pyqtSignal(np.ndarray)

    def __init__(self, camera, interval_ms=50):
        super().__init__()
        self.camera = camera
        self.interval_ms = interval_ms
        self._running = False

    def run(self):
        self._running = True
        while self._running:
            frame = self.camera.get_frame()
            self.new_frame.emit(frame)
            self.msleep(self.interval_ms)

    def stop(self):
        self._running = False
        self.wait()
