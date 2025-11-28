import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


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
