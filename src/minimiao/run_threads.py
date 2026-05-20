# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import threading
import time
import traceback
import tifffile as tf
from collections import deque
from queue import Queue, Empty
from pathlib import Path
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
            time.sleep(0.02)  # 1 ms yield (tune)

    def stop(self):
        self.running = False
        self.join()


class CameraDataList:

    def __init__(
        self,
        max_length,
        save_to_disk=False,
        save_dir=None,
        file_prefix="stack",
    ):
        self.max_length = max_length
        self.save_to_disk = save_to_disk

        # In normal mode: rolling display buffer
        # In saving mode: batch buffer that resets after each full stack
        if self.save_to_disk:
            self.data_list = []
            self.ind_list = []
        else:
            self.data_list = deque(maxlen=max_length)
            self.ind_list = deque(maxlen=max_length)

        self.callback = None
        self._lock = threading.Lock()

        # ---------- disk saving ----------
        self.save_dir = Path(save_dir) if save_dir is not None else None
        self.file_prefix = file_prefix
        self._stack_save_count = 0

        self._save_queue = None
        self._save_thread = None
        self._stop_saver = threading.Event()

        if self.save_to_disk:
            if self.save_dir is None:
                raise ValueError("save_dir must be provided when save_to_disk=True")

            self.save_dir.mkdir(parents=True, exist_ok=True)

            self._save_queue = Queue()
            self._save_thread = threading.Thread(
                target=self._save_worker,
                daemon=True
            )
            self._save_thread.start()

    def add_element(self, elements, ids=None):
        """
        elements: iterable of ndarray images
        ids: optional tuple like (start_id, end_id)
        """
        elements = list(elements)
        if len(elements) == 0:
            return

        if ids is not None:
            frame_ids = list(range(ids[0], ids[1] + 1))
            if len(frame_ids) != len(elements):
                raise ValueError("Number of ids does not match number of elements")
        else:
            frame_ids = [None] * len(elements)

        last = elements[-1]

        if self.save_to_disk:
            self._add_element_saving_mode(elements, frame_ids)
        else:
            self._add_element_normal_mode(elements, frame_ids)

        # Live display/update callback
        if self.callback is not None and last is not None:
            self.callback(last)

    def _add_element_normal_mode(self, elements, frame_ids):
        """
        Original rolling-buffer behavior when save_to_disk=False.
        """
        with self._lock:
            self.data_list.extend(elements)

            valid_ids = [i for i in frame_ids if i is not None]
            if valid_ids:
                self.ind_list.extend(valid_ids)

    def _add_element_saving_mode(self, elements, frame_ids):
        """
        Batch-saving behavior:
        fill data_list until max_length,
        save the full stack,
        then start a new empty list.
        """
        current_pos = 0
        n_new = len(elements)

        while current_pos < n_new:
            stack_to_save = None
            stack_ids_to_save = None

            with self._lock:
                remaining_capacity = self.max_length - len(self.data_list)
                n_take = min(remaining_capacity, n_new - current_pos)

                # Add only what fits into the current stack
                self.data_list.extend(
                    elements[current_pos:current_pos + n_take]
                )
                self.ind_list.extend(
                    frame_ids[current_pos:current_pos + n_take]
                )

                current_pos += n_take

                # If full, package this stack for saving
                if len(self.data_list) == self.max_length:
                    stack_to_save = np.stack(
                        [np.array(img, copy=True) for img in self.data_list],
                        axis=0
                    )
                    stack_ids_to_save = list(self.ind_list)

                    # Start a brand-new list for the next stack
                    self.data_list = []
                    self.ind_list = []

            # Queue full stack outside the lock
            if stack_to_save is not None:
                self._save_queue.put(
                    (
                        self._stack_save_count,
                        stack_ids_to_save,
                        stack_to_save,
                    )
                )
                self._stack_save_count += 1

    def _save_worker(self):
        """
        Background worker that saves one full image stack per file.
        """
        while not self._stop_saver.is_set() or not self._save_queue.empty():
            try:
                stack_index, stack_ids, stack = self._save_queue.get(timeout=0.1)
            except Empty:
                continue

            valid_ids = [i for i in stack_ids if i is not None]

            if valid_ids:
                first_id = valid_ids[0]
                last_id = valid_ids[-1]
                file_name = (
                    f"{self.file_prefix}_{stack_index:06d}"
                    f"_ids_{first_id}_{last_id}.tiff"
                )
            else:
                file_name = f"{self.file_prefix}_{stack_index:06d}.tiff"

            file_path = self.save_dir / file_name
            tf.imwrite(file_path, stack)

            self._save_queue.task_done()

    def wait_until_saved(self):
        """
        Wait until all queued full stacks have been written to disk.
        """
        if self.save_to_disk and self._save_queue is not None:
            self._save_queue.join()

    def close(self):
        """
        Finish saving queued full stacks cleanly.
        Note: an unfinished partial stack is not saved here.
        """
        if self.save_to_disk and self._save_thread is not None:
            self._save_queue.join()
            self._stop_saver.set()
            self._save_thread.join()

    def get_elements(self):
        with self._lock:
            return np.array(self.data_list) if self.data_list else None

    def get_last_element(self, copy=False):
        with self._lock:
            if not self.data_list:
                return None
            arr = self.data_list[-1]

        return arr.copy() if copy else arr

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
            self.psr_ready.emit(list(self.dat.count_list).copy(), self.reco.live_rec.copy())
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
