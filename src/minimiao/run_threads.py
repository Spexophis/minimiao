# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import threading
import time
import traceback
from collections import deque
from pathlib import Path
from queue import Queue, Empty

import numpy as np
import tifffile as tf
from PyQt6.QtCore import QThread, pyqtSignal, pyqtSlot


class CameraAcquisitionThread(threading.Thread):

    def __init__(self, cam, interval=0.05):
        super().__init__(daemon=True)
        self.cam = cam
        self._running = False
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.interval = interval

    def run(self):
        self._running = True
        while self._running:
            with self.condition:
                self.condition.wait(timeout=self.interval)

                if not self._running:
                    break

                self.cam.get_images()

    def stop(self, timeout=5.0):
        with self.condition:
            self._running = False
            self.condition.notify()  # Wake up thread immediately
        self.join(timeout=timeout)
        if self.is_alive():
            self.cam.logg.error("CameraAcquisitionThread did not exit within %.1fs; ", timeout)

    def is_running(self):
        return self._running


class CameraDataList:

    def __init__(self, max_length, save_to_disk=False, save_dir=None, file_prefix="stack"):
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
            self._save_thread = threading.Thread(target=self._save_worker, daemon=True)
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
                self.data_list.extend(elements[current_pos:current_pos + n_take])
                self.ind_list.extend(frame_ids[current_pos:current_pos + n_take])

                current_pos += n_take

                # If full, hand off the lists and reset — heavy numpy work done outside lock
                if len(self.data_list) == self.max_length:
                    imgs_to_stack = self.data_list
                    stack_ids_to_save = list(self.ind_list)
                    self.data_list = []
                    self.ind_list = []
                else:
                    imgs_to_stack = None
                    stack_ids_to_save = None

            # Build and queue the array outside the lock to avoid blocking get_images()
            if imgs_to_stack is not None:
                stack_to_save = np.stack([np.array(img, copy=True) for img in imgs_to_stack], axis=0)
                self._save_queue.put((self._stack_save_count, stack_ids_to_save, stack_to_save))
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
                file_name = (f"{self.file_prefix}_{stack_index:06d}"
                             f"_ids_{first_id}_{last_id}.tiff")
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
            self._save_thread.join(timeout=5.0)

    def get_elements(self):
        with self._lock:
            return np.array(self.data_list) if self.data_list else None

    def get_last_element(self):
        with self._lock:
            return np.array(self.data_list[-1]) if self.data_list else None

    def on_update(self, callback):
        self.callback = callback


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


class DPCWorker(QThread):
    dpc_ready = pyqtSignal(object)

    def __init__(self, fps=10, parent=None):
        super().__init__(parent)
        self.fps = float(fps)
        self._running = True
        self._latest = None

    def stop(self):
        self._running = False
        self.wait(2)

    def push_frame(self, dpc_image: np.ndarray):
        if dpc_image is None or dpc_image.ndim != 2:
            return
        self._latest = np.array(dpc_image, copy=True)

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
            mn = float(img.min())
            mx = float(img.max())
            if mx <= mn:
                out = np.zeros_like(img, dtype=np.uint16)
            else:
                out = ((img - mn) * (65535.0 / (mx - mn))).astype(np.uint16)

            self.dpc_ready.emit(out)


def dpc_reconstruct(frames):
    """
    Reconstruct a differential phase contrast (DPC) image from 4 raw frames
    acquired under complementary half-circle illumination patterns, in
    acquisition order (top, bottom, left, right).

    Returns (dpc_x, dpc_y, dpc_combined) as float32 arrays:
      dpc_x        = (right - left) / (right + left)  -- horizontal phase gradient
      dpc_y        = (bottom - top) / (bottom + top)   -- vertical phase gradient
      dpc_combined = 0.5 * (dpc_x + dpc_y)             -- single-image live preview

    This is the standard raw normalized-difference DPC contrast used for live
    display; it is not a deconvolved quantitative phase (that requires the
    optical system's weak-object transfer functions and is better done as an
    offline post-processing step).
    """
    if len(frames) != 4:
        raise ValueError(f"DPC reconstruction requires exactly 4 frames, got {len(frames)}")

    top, bottom, left, right = (np.asarray(f, dtype=np.float32) for f in frames)

    sum_h = left + right
    sum_v = top + bottom
    dpc_x = np.divide(right - left, sum_h, out=np.zeros_like(sum_h), where=sum_h > 0)
    dpc_y = np.divide(bottom - top, sum_v, out=np.zeros_like(sum_v), where=sum_v > 0)
    dpc_combined = 0.5 * (dpc_x + dpc_y)
    return dpc_x, dpc_y, dpc_combined


class DPCCameraDataList:
    """
    Rolling buffer for a DPC camera. Raw frames arrive one at a time from
    the acquisition thread (same as CameraDataList) but must always be
    consumed in complete groups of 4 (top/bottom/left/right illumination),
    since a DPC image only makes sense once all 4 are available. Frames are
    staged until a full group accumulates, then immediately reconstructed
    via dpc_reconstruct(); only the reconstructed DPC images are exposed.
    """

    def __init__(self, max_length, group_size=4, save_to_disk=False, save_dir=None, file_prefix="dpc"):
        self.group_size = group_size
        self.max_length = max_length
        self.save_to_disk = save_to_disk
        if self.save_to_disk:
            self.data_list = []  # reconstructed dpc_combined images
            self.raw_group_list = []  # matching raw 4-frame windows
            self.ind_list = []
        else:
            self.data_list = deque(maxlen=max_length)
            self.raw_group_list = deque(maxlen=max_length)
            self.ind_list = deque(maxlen=max_length)
        self._window = deque(maxlen=group_size)
        self._id_window = deque(maxlen=group_size)
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
            self._save_thread = threading.Thread(target=self._save_worker, daemon=True)
            self._save_thread.start()

    def add_element(self, elements, ids=None):
        elements = list(elements)
        if not elements:
            return

        if ids is not None:
            frame_ids = list(range(ids[0], ids[1] + 1))
            if len(frame_ids) != len(elements):
                raise ValueError("Number of ids does not match number of elements")
        else:
            frame_ids = [None] * len(elements)

        ready_groups = []
        with self._lock:
            for element, frame_id in zip(elements, frame_ids):
                self._window.append(element)
                self._id_window.append(frame_id)
                if len(self._window) == self.group_size:
                    ready_groups.append((list(self._window), list(self._id_window)))
                    self._window.clear()
                    self._id_window.clear()

        for group, group_ids in ready_groups:
            _, _, dpc_combined = dpc_reconstruct(group)
            if self.save_to_disk:
                self._add_group_saving_mode(dpc_combined, group, group_ids)
            else:
                self._add_group_normal_mode(dpc_combined, group, group_ids)
            if self.callback is not None:
                self.callback(dpc_combined)

    def _add_group_normal_mode(self, dpc_combined, group, group_ids):
        """
        Rolling-buffer behavior when save_to_disk=False.
        """
        with self._lock:
            self.data_list.append(dpc_combined)
            self.raw_group_list.append(group)

            valid_ids = [i for i in group_ids if i is not None]
            if valid_ids:
                self.ind_list.extend(valid_ids)

    def _add_group_saving_mode(self, dpc_combined, group, group_ids):
        """
        Batch-saving behavior: collect DPC images until max_length of them are
        held, hand the full stack off to the writer, then start a new stack.
        """
        with self._lock:
            self.data_list.append(dpc_combined)
            self.raw_group_list.append(group)
            self.ind_list.extend(group_ids)

            if len(self.data_list) < self.max_length:
                return

            stack_index, stack_ids, dpc_group, raw_group = self._take_stack()

        self._queue_stack(stack_index, stack_ids, dpc_group, raw_group)

    def _take_stack(self):
        """
        Hand off the buffered stack and reset. Caller must hold the lock.
        """
        dpc_group = self.data_list
        raw_group = self.raw_group_list
        stack_ids = list(self.ind_list)
        self.data_list = []
        self.raw_group_list = []
        self.ind_list = []
        stack_index = self._stack_save_count
        self._stack_save_count += 1
        return stack_index, stack_ids, dpc_group, raw_group

    def _queue_stack(self, stack_index, stack_ids, dpc_group, raw_group):
        """
        Build the arrays outside the lock, so get_images() is never blocked by
        the numpy work, and queue them for the writer.
        """
        dpc_stack = np.stack([np.asarray(img, dtype=np.float32) for img in dpc_group], axis=0)
        raw_stack = np.stack([np.array(frame, copy=True) for grp in raw_group for frame in grp], axis=0)
        self._save_queue.put((stack_index, stack_ids, dpc_stack, raw_stack))

    def _save_worker(self):
        """
        Background worker that saves one full DPC stack per file, plus the raw
        frames that stack was reconstructed from.
        """
        while not self._stop_saver.is_set() or not self._save_queue.empty():
            try:
                stack_index, stack_ids, dpc_stack, raw_stack = self._save_queue.get(timeout=0.1)
            except Empty:
                continue

            valid_ids = [i for i in stack_ids if i is not None]

            if valid_ids:
                suffix = (f"_{stack_index:06d}"
                          f"_ids_{valid_ids[0]}_{valid_ids[-1]}")
            else:
                suffix = f"_{stack_index:06d}"

            tf.imwrite(self.save_dir / f"{self.file_prefix}_dpc{suffix}.tiff", dpc_stack)
            tf.imwrite(self.save_dir / f"{self.file_prefix}_raw{suffix}.tiff", raw_stack)

            self._save_queue.task_done()

    def flush(self):
        """
        Queue whatever partial stack is buffered. A DPC acquisition is stopped
        by hand, so the last stack is normally incomplete and would otherwise
        be lost.
        """
        if not self.save_to_disk:
            return

        with self._lock:
            if not self.data_list:
                return
            stack_index, stack_ids, dpc_group, raw_group = self._take_stack()

        self._queue_stack(stack_index, stack_ids, dpc_group, raw_group)

    def wait_until_saved(self):
        """
        Wait until all queued stacks have been written to disk.
        """
        if self.save_to_disk and self._save_queue is not None:
            self._save_queue.join()

    def close(self):
        """
        Save the partial stack still buffered, then finish the queued stacks
        cleanly.
        """
        if self.save_to_disk and self._save_thread is not None:
            self.flush()
            self._save_queue.join()
            self._stop_saver.set()
            self._save_thread.join(timeout=5.0)

    def get_last_element(self):
        with self._lock:
            return self.data_list[-1] if self.data_list else None

    def get_elements(self):
        with self._lock:
            return np.array(self.data_list) if self.data_list else None

    def on_update(self, callback):
        self.callback = callback


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
