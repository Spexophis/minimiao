# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import threading
import traceback
from collections import deque
from queue import Queue

import numpy as np
import tifffile as tf
from PyQt6.QtCore import QThread, pyqtSignal


class CameraAcquisitionThread(threading.Thread):

    def __init__(self, cam, interval=0.005):
        super().__init__(daemon=True)
        self.cam = cam
        self._running = False
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.interval = interval

    def run(self):
        """Drain every frame the SDK has buffered; only pause briefly when no new frame
        arrived (cam.get_images() returns False). Waiting a fixed interval per frame
        capped readout at ~20 fps and let the SDK ring buffer overwrite unread frames."""
        self._running = True
        while self._running:
            got = self.cam.get_images()
            if got:
                continue
            with self.condition:
                if not self._running:
                    break
                self.condition.wait(timeout=self.interval)

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


class CameraDataList:
    """Rolling buffer of the newest frames (live video); an optional callback gets each new last frame."""

    def __init__(self, max_length):
        self.max_length = max_length
        self.data_list = deque(maxlen=max_length)
        self.ind_list = deque(maxlen=max_length)
        self.callback = None
        self._lock = threading.Lock()

    def add_element(self, elements, ids=None):
        """elements: iterable of frames; ids: optional (first_id, last_id) frame numbers."""
        elements = list(elements)
        if not elements:
            return
        with self._lock:
            self.data_list.extend(elements)
            if ids is not None:
                self.ind_list.extend(range(ids[0], ids[1] + 1))
        if self.callback is not None and elements[-1] is not None:
            self.callback(elements[-1])

    def get_last_element(self):
        with self._lock:
            return np.array(self.data_list[-1]) if self.data_list else None

    def on_update(self, callback):
        self.callback = callback

    def close(self):
        """Nothing to flush (frames are only buffered for display)."""


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

    max_frames: fixed-length recording (camera sequence mode). The poller stops by
    itself after max_frames frames; wait_done() blocks until then.
    """

    def __init__(self, poll, path, bigtiff=False, preview=None, preview_hz=10.0, drain_s=0.3, max_frames=None):
        self.poll = poll
        self.path = str(path)
        self.bigtiff = bigtiff
        self.preview = preview
        self.preview_period = 1.0 / preview_hz if preview_hz else None
        self.drain_s = drain_s
        self.max_frames = max_frames
        self._done = threading.Event()
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
                if self.max_frames and len(self.frame_nrs) >= self.max_frames:
                    break  # fixed-length recording complete
                if self.preview is not None and self.preview_period and t - last_preview >= self.preview_period:
                    last_preview = t
                    try:
                        self.preview(frame)
                    except Exception:
                        pass
        except Exception as e:
            self.error = f"poll: {e}"
        finally:
            self._done.set()
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

    @property
    def n_frames(self):
        return len(self.frame_nrs)

    def is_done(self):
        return self._done.is_set()

    def wait_done(self, timeout=None) -> bool:
        """Block until the poller has finished (max_frames reached or stopped)."""
        return self._done.wait(timeout)

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
            "n_frames_requested": self.max_frames,
            "complete": None if not self.max_frames else len(self.frame_nrs) >= self.max_frames,
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
