# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import threading
import time
import traceback
from collections import deque

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot


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
        self.ind_list = deque(maxlen=max_length)
        self.callback = None
        self._lock = threading.Lock()

    def add_element(self, elements, ids=None):
        with self._lock:
            self.data_list.extend(elements)
            if ids is not None:
                self.ind_list.extend(list(range(ids[0], ids[1] + 1)))
            last = self.data_list[-1] if self.data_list else None
        if self.callback is not None and last is not None:
            self.callback(last)  # passes ndarray

    def get_elements(self):
        return np.array(self.data_list) if self.data_list else None

    def get_last_element(self, copy=False):
        with self._lock:
            if not self.data_list:
                return None
            arr = self.data_list[-1]
        return arr.copy() if copy else arr  # no copy for display

    def on_update(self, callback):
        self.callback = callback


class MPDCountThread(threading.Thread):

    def __init__(self, daq, ind):
        threading.Thread.__init__(self)
        self.daq = daq
        self.ind = ind
        self.running = False
        self.lock = threading.Lock()

    def run(self):
        self.running = True
        while self.running:
            with self.lock:
                self.daq.get_photon_counts(self.ind)
            time.sleep(0.008)  # 8 ms yield (tune)

    def stop(self):
        self.running = False
        self.join()


class MPDCountList:

    def __init__(self, max_length):
        self.data_lists = [deque(maxlen=max_length), deque(maxlen=max_length)]
        for data_list in self.data_lists:
            data_list.extend([0])
        self.count_lists = [deque(maxlen=max_length), deque(maxlen=max_length)]
        self.ind_lists = [deque(maxlen=max_length), deque(maxlen=max_length)]
        self.count_starts = [1, 1]
        self.count_ends = [1, 1]
        self.count_lens = [max_length, max_length]
        self.callback = None
        self.request = None
        self.lock = threading.Lock()

    def add_element(self, elements: list, num: int, ind: int):
        with self.lock:
            d = np.asarray(elements, dtype=np.int64)
            self.count_starts[ind] = self.count_ends[ind] % self.count_lens[ind]
            self.count_ends[ind] = (self.count_ends[ind] + num) % self.count_lens[ind]
            counts = np.diff(np.insert(d, 0, self.data_lists[ind][-1]))
            self.count_lists[ind].extend(counts.tolist())
            self.data_lists[ind].extend(elements)
            if self.count_starts[ind] <= self.count_ends[ind]:
                indices = np.arange(self.count_starts[ind], self.count_ends[ind])
            else:
                indices = np.concatenate(
                    (np.arange(self.count_starts[ind], self.count_lens[ind]), np.arange(self.count_ends[ind])))
            self.ind_lists[ind].extend(indices.tolist())
            if self.callback is not None:
                self.callback(counts, list(indices), ind)

    def get_elements(self, ind):
        return (np.array(self.data_lists[ind]) if self.data_lists[ind] else None,
                self.count_lists[ind] if self.count_lists[ind] else None)

    def on_update(self, callback):
        self.callback = callback

class PMTAmpThread(threading.Thread):

    def __init__(self, daq):
        threading.Thread.__init__(self)
        self.daq = daq
        self.running = False
        self.lock = threading.Lock()

    def run(self):
        self.running = True
        while self.running:
            with self.lock:
                self.daq.get_pmt_amps()
            time.sleep(0.008)  # 8 ms yield (tune)

    def stop(self):
        self.running = False
        self.join()


class PMTAmpList:

    def __init__(self, max_length):
        self.data_list = deque(maxlen=max_length)
        self.ind_list = deque(maxlen=max_length)
        self.count_len = max_length
        self.count_starts = 1
        self.count_ends = 1
        self.callback = None
        self.request = None
        self.lock = threading.Lock()

    def add_element(self, elements: list, num: int):
        with self.lock:
            self.count_starts = self.count_ends % self.count_len
            self.count_ends = (self.count_ends + num) % self.count_len
            self.data_list.extend(elements)
            if self.count_starts <= self.count_ends:
                indices = list(range(self.count_starts, self.count_ends))
            else:
                indices = list(range(self.count_starts, self.count_len))
                indices.extend(list(range(self.count_ends)))
            self.ind_list.extend(indices)
            if self.callback is not None:
                self.callback(np.array(elements), indices, 1)

    def get_elements(self):
        return np.array(self.data_list) if self.data_list else None

    def on_update(self, callback):
        self.callback = callback


class PSLiveWorker(QThread):
    psr_ready = pyqtSignal(object, object, object)
    psr_new = pyqtSignal()

    def __init__(self, reco, mpd_dat=None, pmt_dat=None, fps=10, parent=None):
        super().__init__(parent)
        self.mpd_dat = mpd_dat
        self.pmt_dat = pmt_dat
        self.reco = reco
        self.period_ms = max(1, int(1000 / max(float(fps), 0.1)))
        self._running = True

    def stop(self):
        self._running = False
        self.wait()

    def run(self):
        while self._running:
            self.msleep(self.period_ms)
            with self.reco.lock:
                img_copy = self.reco.live_rec.copy()
            counts_copy = self.mpd_dat.count_lists.copy()
            if self.pmt_dat is not None:
                amp_copy = self.pmt_dat.data_list.copy()
            else:
                amp_copy = None
            self.psr_ready.emit(img_copy, counts_copy, amp_copy)
            self.psr_new.emit()


class FFTWorker(QThread):
    fft_ready = pyqtSignal(object)

    def __init__(self, fps=10, parent=None):
        super().__init__(parent)
        self.fps = float(fps)
        self._running = True
        self._latest = None
        self._win = None  # cached window for ROI

    def stop(self):
        self._running = False
        self.wait(2)

    def push_frame(self, frame_u16: np.ndarray):
        if frame_u16 is None or frame_u16.ndim != 2:
            return
        f = frame_u16
        self._latest = np.array(f, copy=True)

    def _ensure_window(self, n: int):
        if self._win is None or self._win.shape[0] != n:
            w1 = np.hanning(n).astype(np.float32)
            self._win = np.outer(w1, w1)

    def run(self):
        period = 1.0 / max(self.fps, 0.1)
        next_t = time.perf_counter()

        while self._running:
            now = time.perf_counter()
            if now < next_t:
                self.msleep(int((next_t - now) * 1000))
                continue
            next_t = now + period

            if self._latest is None:
                continue

            img = self._latest
            n = img.shape[0]
            self._ensure_window(n)

            ft = np.fft.fftshift(np.fft.fft2(img * self._win))
            mag = np.log1p(np.abs(ft)).astype(np.float32)

            mn = float(mag.min())
            mx = float(mag.max())
            if mx <= mn:
                out = np.zeros_like(mag, dtype=np.uint16)
            else:
                out = ((mag - mn) * (65535.0 / (mx - mn))).astype(np.uint16)

            self.fft_ready.emit(out)


class TaskWorker(QThread):
    error = pyqtSignal(tuple)

    def __init__(self, task=None, n=1, parent=None):
        super().__init__(parent)
        self.task = task if task is not None else self._do_nothing
        self.n = n

    def run(self):
        try:
            for i in range(self.n):
                self.task()
        except Exception as e:
            self.error.emit((e, traceback.format_exc()))

    @pyqtSlot()
    def _do(self):
        self.task()

    @staticmethod
    def _do_nothing():
        pass
