# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import threading
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
            self.cam.logg.error(
                "CameraAcquisitionThread did not exit within %.1fs; ",
                timeout,
            )

    def is_running(self):
        return self._running


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
                stack_to_save = np.stack(
                    [np.array(img, copy=True) for img in imgs_to_stack],
                    axis=0
                )
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
            self._save_thread.join(timeout=5.0)

    def get_elements(self):
        with self._lock:
            return np.array(self.data_list) if self.data_list else None

    def get_last_element(self):
        with self._lock:
            return np.array(self.data_list[-1]) if self.data_list else None

    def on_update(self, callback):
        self.callback = callback


class ImgLiveWorker(QThread):
    img_ready = pyqtSignal(object)

    def __init__(self, data=None, fps=12, parent=None):
        super().__init__(parent)
        self.data = data
        self.period_ms = max(1, int(1000 / max(float(fps), 0.1)))
        self._running = True
        self._lock = threading.Lock()

    def stop(self):
        """Stop worker thread gracefully"""
        self._running = False

        if not self.wait(1000):
            self.terminate()
            self.wait(1000)

    def run(self):
        try:
            while self._running:
                self.msleep(self.period_ms)
                with self._lock:
                    if self.data is None:
                        break
                    img_copy = self.data.get_last_element()
                self.img_ready.emit(img_copy)
        except Exception as e:
            import logging
            logging.error(f"PSLiveWorker error: {e}")


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


class StackRecorder:
    """
    Lossless recording of a camera stream into ONE multi-page (3D) TIFF.

    poll() must return (frame ndarray, frame_nr) for the next unread frame,
    or None if no frame arrived within its own timeout. A poller thread
    drains the SDK buffer as fast as frames arrive; a writer thread appends
    frames to the TIFF so slow disks do not stall polling. Gaps in frame_nr
    are recorded as dropped frames.

    stop(): keeps draining until the stream is idle (external trigger) or
    `drain_s` has passed (free-running), then returns a summary dict.
    """

    def __init__(self, poll, path, bigtiff=False, preview=None, preview_hz=10.0, drain_s=0.3):
        self.poll = poll
        self.path = str(path)
        self.bigtiff = bigtiff
        self.preview = preview
        self.preview_period = 1.0 / preview_hz if preview_hz else None
        self.drain_s = drain_s
        self._queue = Queue()
        self._stop = threading.Event()
        self._t_stop = None
        self.frame_nrs = []
        self.timestamps = []  # time.perf_counter() when each frame was retrieved
        self.max_backlog = 0
        self.error = None
        self.shape = None
        self.dtype = None
        self._poller = threading.Thread(target=self._poll_loop, daemon=True)
        self._writer = threading.Thread(target=self._write_loop, daemon=True)

    def start(self):
        self._writer.start()
        self._poller.start()

    def _poll_loop(self):
        import time
        last_preview = 0.0
        try:
            while True:
                if self._stop.is_set() and time.perf_counter() - self._t_stop > self.drain_s:
                    break
                got = self.poll()
                if got is None:
                    if self._stop.is_set():
                        break  # idle after stop request: stream finished
                    continue
                frame, frame_nr = got
                t = time.perf_counter()
                self.frame_nrs.append(frame_nr)
                self.timestamps.append(t)
                self._queue.put(frame)
                self.max_backlog = max(self.max_backlog, self._queue.qsize())
                if self.preview is not None and self.preview_period and t - last_preview >= self.preview_period:
                    last_preview = t
                    try:
                        self.preview(frame)
                    except Exception:
                        pass
        except Exception as e:
            self.error = f"poll: {e}"
        finally:
            self._queue.put(None)

    def _write_loop(self):
        try:
            with tf.TiffWriter(self.path, bigtiff=self.bigtiff) as tw:
                while True:
                    frame = self._queue.get()
                    if frame is None:
                        break
                    if self.shape is None:
                        self.shape, self.dtype = frame.shape, frame.dtype
                    tw.write(frame, contiguous=True, photometric="minisblack")
        except Exception as e:
            self.error = f"write: {e}"
            while self._queue.get() is not None:  # keep draining so the poller never blocks
                pass

    def stop(self, timeout=60.0):
        import time
        self._t_stop = time.perf_counter()
        self._stop.set()
        self._poller.join(timeout=timeout)
        self._writer.join(timeout=timeout)
        nrs = [n for n in self.frame_nrs if n is not None]
        dropped = []
        for a, b in zip(nrs[:-1], nrs[1:]):
            if b - a > 1:
                dropped.extend(range(a + 1, b))
        dt = np.diff(self.timestamps) if len(self.timestamps) > 1 else np.array([])
        return {
            "path": self.path,
            "n_frames": len(self.frame_nrs),
            "frame_shape": list(self.shape) if self.shape is not None else None,
            "dtype": str(self.dtype) if self.dtype is not None else None,
            "frame_nrs": self.frame_nrs,
            "timestamps_perf_s": self.timestamps,
            "dropped_frame_nrs": dropped,
            "n_dropped": len(dropped),
            "measured_fps": float(1.0 / np.median(dt)) if dt.size else None,
            "max_write_backlog": self.max_backlog,
            "error": self.error,
        }
