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


class PhotonCountThread(threading.Thread):

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
                self.daq.get_photon_count(self.ind)
            time.sleep(0.001)  # 1 ms yield (tune)

    def stop(self):
        self.running = False
        self.join()


class PhotonCountList:

    def __init__(self, max_length):
        self.data_lists = [deque(maxlen=max_length), deque(maxlen=max_length)]
        for data_list in self.data_lists:
            data_list.extend([0])
        self.count_arrays = [np.zeros(max_length, dtype=np.int64), np.zeros(max_length, dtype=np.int64)]
        self.count_indices = [1, 1]
        self.count_lens = [max_length, max_length]
        self.callback = None
        self.request = None
        self.lock = threading.Lock()

    def add_element(self, elements: list, num: int, ind: int):
        with self.lock:
            d = np.asarray(elements, dtype=np.int64)
            counts = np.diff(np.insert(d, 0, self.data_lists[ind][-1]))
            # Circular buffer
            start_idx = self.count_indices[ind]
            end_idx = (self.count_indices[ind] + num) % self.count_lens[ind]
            if end_idx > start_idx:
                self.count_arrays[ind][start_idx:end_idx] = counts
            else:
                # Wrap around
                remaining = self.count_lens[ind] - start_idx
                self.count_arrays[ind][start_idx:] = counts[:remaining]
                self.count_arrays[ind][:end_idx] = counts[remaining:]

            self.count_indices[ind] = end_idx
            self.data_lists[ind].extend(elements)

            if self.callback is not None:
                self.callback(counts.tolist(), list(range(start_idx, end_idx)))

    def get_recent_elements(self, ind, n_samples=None):
        if n_samples is None:
            return self.count_arrays[ind].copy()
        idx = self.count_indices[ind]
        if n_samples <= idx:
            return self.count_arrays[ind][idx-n_samples:idx].copy()
        else:
            # Wrap around
            return np.concatenate([
                self.count_arrays[ind][self.count_lens[ind]-(n_samples-idx):],
                self.count_arrays[ind][:idx]
            ])

    def get_elements(self, ind):
        return (np.array(self.data_lists[ind]) if self.data_lists[ind] else None,
                self.count_arrays[ind] if self.count_arrays[ind] else None)

    def on_update(self, callback):
        self.callback = callback


class PSLiveWorker(QThread):
    psr_ready = pyqtSignal(object, object)
    psr_new = pyqtSignal()

    def __init__(self, dat, reco, fps=10, parent=None):
        super().__init__(parent)
        self.dat = dat
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
                img_0_copy = self.reco.live_rec_0.copy()
                img_1_copy = self.reco.live_rec_1.copy()
            counts_0_copy = self.dat.count_arrays[0].copy()
            counts_1_copy = self.dat.count_arrays[1].copy()
            self.psr_ready.emit(counts_0_copy, img_0_copy, counts_1_copy, img_1_copy)
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
