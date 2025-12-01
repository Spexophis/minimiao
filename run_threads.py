import threading
import time
from collections import deque

import numpy as np


class CameraAcquisitionThread(threading.Thread):
    def __init__(self, cam):
        super().__init__()
        self.cam = cam
        self.running = False
        self.lock = threading.Lock()  # instance lock, not class-level

    def run(self):
        self.running = True
        while self.running:
            with self.lock:
                self.cam.get_images()
            time.sleep(0.001)  # 1 ms yield (tune)

    def stop(self):
        self.running = False
        self.join()


class CameraDataList:
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


class PhotonCountThread(threading.Thread):
    running = False
    lock = threading.Lock()

    def __init__(self, daq):
        threading.Thread.__init__(self)
        self.daq = daq

    def run(self):
        self.running = True
        while self.running:
            with self.lock:
                self.daq.get_photon_count()

    def stop(self):
        self.running = False
        self.join()


class PhotonCountList:

    def __init__(self, max_length):
        self.data_list = deque(maxlen=max_length)
        self.count_list = deque(maxlen=max_length)
        self.data_list.extend([0])
        self.count_list.extend([0])

    def add_element(self, elements):
        d = np.array(elements, dtype=int)
        counts = np.diff(np.insert(d, 0, self.data_list[-1]))
        self.count_list.extend(list(counts))
        self.data_list.extend(elements)

    def get_elements(self):
        return np.array(self.data_list) if self.data_list else None, np.array(
            self.count_list) if self.data_list else None

    def get_last_element(self):
        return self.data_list[-1].copy() if self.data_list else None

    def is_empty(self):
        return len(self.data_list) == 0
