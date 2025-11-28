import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtCore import QObject, QMutex, QMutexLocker, pyqtSlot

class FramePool(QObject):
    def __init__(self, shape=(1024, 1024), dtype=np.uint16, n_buffers=4):
        super().__init__()
        self._buffers = [np.empty(shape, dtype=dtype) for _ in range(n_buffers)]
        self._free = list(range(n_buffers))
        self._in_use = set()
        self._m = QMutex()

    def acquire(self):
        """Reserve a buffer index for writing. Returns idx or None if none free."""
        with QMutexLocker(self._m):
            if not self._free:
                return None
            idx = self._free.pop()
            self._in_use.add(idx)
            return idx

    def buffer(self, idx: int) -> np.ndarray:
        return self._buffers[idx]

    @pyqtSlot(object)
    def release(self, token):
        """Return a buffer to the free list after viewer consumed/discarded it."""
        idx = int(token)
        with QMutexLocker(self._m):
            if idx in self._in_use:
                self._in_use.remove(idx)
                self._free.append(idx)


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
