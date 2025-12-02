import time
import threading
from collections import deque
import numpy as np


class MockCamera:
    def __init__(self, shape=(2048, 2048)):
        self.shape = shape
        self.buffer_size = 4
        self.data = None
        self.acq_thread = None

    def start_live(self):
        self.data = DataList(self.buffer_size)
        self.acq_thread = AcquisitionThread(self)
        self.acq_thread.start()

    def stop_live(self):
        self.acq_thread.stop()
        self.acq_thread = None
        self.data = None

    def get_images(self):
        self.data.add_element([np.random.randint(0, 2**14, size=self.shape, dtype=np.uint16)])

    def get_last_image(self):
        return np.random.randint(0, 2**14, size=self.shape, dtype=np.uint16)


class AcquisitionThread(threading.Thread):
    def __init__(self, cam):
        super().__init__()
        self.cam = cam
        self.running = False
        self.lock = threading.Lock()   # instance lock, not class-level

    def run(self):
        self.running = True
        while self.running:
            with self.lock:
                self.cam.get_images()
            time.sleep(0.001)  # 1 ms yield (tune)

    def stop(self):
        self.running = False
        self.join()


class DataList:
    def __init__(self, max_length):
        self.data_list = deque(maxlen=max_length)
        self.callback = None
        self._lock = threading.Lock()

    def add_element(self, elements):
        with self._lock:
            self.data_list.extend(elements)
            last = self.data_list[-1] if self.data_list else None
        if self.callback is not None and last is not None:
            self.callback(last)  # passes ndarray

    def get_last_element(self, copy=False):
        with self._lock:
            if not self.data_list:
                return None
            arr = self.data_list[-1]
        return arr.copy() if copy else arr  # no copy for display

    def on_update(self, callback):
        self.callback = callback
