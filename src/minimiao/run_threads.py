# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import threading
import traceback
from collections import deque

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot


class PhotonCountThread(threading.Thread):

    def __init__(self, daq, ind, interval=0.004):
        threading.Thread.__init__(self)
        self.daq = daq
        self.ind = ind
        self.interval = interval
        self.started = False
        self.running = False
        self.paused = False
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

    def run(self):
        self.started = True
        self.running = True
        while self.running:
            with self.condition:

                while self.paused and self.running:
                    self.condition.wait()

                if not self.running:
                    break

                self.condition.wait(timeout=self.interval)

                if not self.running:
                    break

                if not self.paused and self.running:
                    self.daq.get_photon_counts(self.ind)

    def pause(self):
        with self.condition:
            self.paused = True

    def resume(self):
        with self.condition:
            self.paused = False
            self.condition.notify()

    def stop(self):
        with self.condition:
            self.running = False
            self.paused = False
            self.condition.notify()
        self.join()

    def trigger(self):
        """Manually trigger an immediate count"""
        with self.condition:
            self.condition.notify()


class PhotonCountList:

    def __init__(self, n, max_length):
        self.data_lists = [deque(maxlen=max_length) for _ in range(n)]
        for data_list in self.data_lists:
            data_list.extend([0])
        self.count_lists = [deque(maxlen=max_length) for _ in range(n)]
        self.ind_lists = [deque(maxlen=max_length) for _ in range(n)]
        self.count_starts = [1 for _ in range(n)]
        self.count_ends = [1 for _ in range(n)]
        self.count_lens = [max_length for _ in range(n)]
        self.callback = None
        self.request = None
        self.lock = threading.Lock()

    def add_element(self, elements: list, num: int, ind: int):
        with self.lock:
            d = np.asarray(elements, dtype=np.int64)
            self.count_starts[ind] = self.count_ends[ind] % self.count_lens[ind]
            self.count_ends[ind] = (self.count_ends[ind] + num) % self.count_lens[ind]
            counts = np.diff(np.insert(d, 0, self.data_lists[ind][-1])).astype(np.uint32)
            self.count_lists[ind].extend(counts.tolist())
            self.data_lists[ind].extend(elements)
            if self.count_starts[ind] < self.count_ends[ind]:
                indices = np.arange(self.count_starts[ind], self.count_ends[ind])
            else:
                indices = np.concatenate(
                    (np.arange(self.count_starts[ind], self.count_lens[ind]), np.arange(self.count_ends[ind])))
            self.ind_lists[ind].extend(indices.tolist())
            if self.callback is not None:
                self.callback(counts, list(indices), ind)

    def get_last_element(self, ind):
        c = self.data_lists[ind][-1] if self.data_lists[ind][-1] else None
        return c

    def get_elements(self, ind):
        r = np.array(self.data_lists[ind]) if self.data_lists[ind] else None
        d = self.count_lists[ind] if self.count_lists[ind] else None
        return r, d

    def on_update(self, callback):
        self.callback = callback


class PMTThread(threading.Thread):

    def __init__(self, daq, interval=0.004):
        threading.Thread.__init__(self)
        self.daq = daq
        self.interval = interval
        self.started = False
        self.running = False
        self.paused = False
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

    def run(self):
        self.started = True
        self.running = True
        while self.running:
            with self.condition:

                while self.paused and self.running:
                    self.condition.wait()

                if not self.running:
                    break

                self.condition.wait(timeout=self.interval)

                if not self.running:
                    break

                if not self.paused and self.running:
                    self.daq.get_pmt_voltages()

    def pause(self):
        with self.condition:
            self.paused = True

    def resume(self):
        with self.condition:
            self.paused = False
            self.condition.notify()

    def stop(self):
        with self.condition:
            self.running = False
            self.paused = False
            self.condition.notify()
        self.join()

    def trigger(self):
        """Manually trigger an immediate count"""
        with self.condition:
            self.condition.notify()


class PMTList:

    def __init__(self, max_length):
        self.data_list = deque(maxlen=max_length)
        self.ind_list = deque(maxlen=max_length)
        self.read_start = 0
        self.read_end = 0
        self.read_len = max_length
        self.callback = None
        self.request = None
        self.lock = threading.Lock()

    def add_element(self, elements: list, num: int):
        with self.lock:
            d = np.asarray(elements, dtype=np.float64)
            self.read_start = self.read_end % self.read_len
            self.read_end = (self.read_end + num) % self.read_len
            self.data_list.extend(elements)
            if self.read_start < self.read_end:
                indices = np.arange(self.read_start, self.read_end)
            else:
                indices = np.concatenate((np.arange(self.read_start, self.read_len), np.arange(self.read_end)))
            self.ind_list.extend(indices.tolist())
            if self.callback is not None:
                self.callback(d, list(indices), 0)

    def get_elements(self):
        return np.array(self.data_list) if self.data_list else None

    def on_update(self, callback):
        self.callback = callback


class PSLiveWorker(QThread):
    psr_ready = pyqtSignal(object, object)
    psr_new = pyqtSignal()
    pmt_ready = pyqtSignal(object)

    def __init__(self, reco, mpd_dat=None, pmt_dat=None, dn=None, fps=10, parent=None):
        super().__init__(parent)
        self.mpd_dat = mpd_dat
        self.pmt_dat = pmt_dat
        self.reco = reco
        self.dn = dn or [2, 2]  # default: both empty
        self.period_ms = max(1, int(1000 / max(float(fps), 0.1)))
        self._running = True
        self._lock = threading.Lock()

    def stop(self):
        """Stop worker thread gracefully"""
        self._running = False

        if not self.wait(2000):
            self.terminate()
            self.wait(1000)

        self.clear_data()

    def clear_data(self):
        """Release references to large data objects"""
        with self._lock:
            self.mpd_dat = None
            self.pmt_dat = None
            self.reco = None

    def run(self):
        try:
            while self._running:
                self.msleep(self.period_ms)
                with self._lock:
                    if self.reco is None:
                        break
                    img_copy = self.reco.live_rec   # [slot0_img, slot1_img]

                    # Build [slot0_counts, slot1_counts]
                    _empty = deque([0])
                    # Slot 0
                    if self.dn[0] == 0 and self.mpd_dat is not None:
                        slot0 = self.mpd_dat.count_lists[0]
                    elif self.dn[0] == 1 and self.pmt_dat is not None:
                        slot0 = self.pmt_dat.data_list
                    else:
                        slot0 = _empty
                    # Slot 1 — hardware counter index is 1 if spot0 is MPD, else 0
                    mpd_idx1 = 1 if self.dn[0] == 0 else 0
                    if self.dn[1] in (0, 1) and self.mpd_dat is not None:
                        slot1 = self.mpd_dat.count_lists[mpd_idx1]
                    else:
                        slot1 = _empty

                self.psr_ready.emit(img_copy, [slot0, slot1])
                self.psr_new.emit()
        except Exception as e:
            import logging
            logging.error(f"PSLiveWorker error: {e}")
        finally:
            self.clear_data()


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
