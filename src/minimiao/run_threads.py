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


def available_ram_bytes():
    """Physical RAM currently available to new allocations (bytes), or None if unknown."""
    try:
        import sys
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return int(st.ullAvailPhys)
            return None
        import os
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except Exception:
        return None


def auto_queue_mb(reserve_gb=4.0, fraction=0.7, floor_mb=2048):
    """RAM for frames waiting for the disk: 70% of available RAM, leaving >= 4 GB free (at least 2 GB)."""
    avail = available_ram_bytes()
    if not avail:
        return floor_mb
    mb = min(fraction * avail, avail - reserve_gb * 2 ** 30) / 2 ** 20
    return int(max(floor_mb, mb))


class StackRecorder:
    """
    Lossless recording of a camera stream into one multi-page (3D) TIFF, or into a series of
    TIFFs of frames_per_file frames each (<path stem>_00000.tif, _00001.tif, ...; the last one
    holds the remainder).

    poll() must return (frame ndarray, frame_nr) for the next unread frame,
    or None if no frame arrived within its own timeout. A poller thread
    drains the SDK buffer as fast as frames arrive; a writer thread appends
    frames to the TIFF so slow disks do not stall polling. Gaps in frame_nr
    are recorded as dropped frames.

    stop(): keeps draining until the stream is idle (external trigger) or
    `drain_s` has passed (free-running), then returns a summary dict.

    max_frames: fixed-length recording (camera sequence mode). The poller stops by
    itself after max_frames frames; wait_done() blocks until then.

    max_queue_mb: RAM for frames waiting to be written (None: auto_queue_mb(), i.e. most of the
    free RAM, so a short fast recording can sit in RAM and be written out afterwards). If the
    disk is slower than the camera, the poller waits once this is full (the camera's own buffer
    then fills; frames lost after that show up as frame_nr gaps) instead of growing memory until
    the PC runs out.
    """

    def __init__(self, poll, path, bigtiff=False, preview=None, preview_hz=10.0, drain_s=0.3, max_frames=None,
                 max_queue_mb=None, frames_per_file=None, on_file_saved=None):
        self.poll = poll
        self.path = str(path)
        self.frames_per_file = int(frames_per_file) if frames_per_file else None
        self.files = []  # [{"file", "first_frame", "n_frames"}] in recording order
        self.on_file_saved = on_file_saved  # called (writer thread) with the file's entry when a stack is full
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
        self.max_queue_bytes = int((auto_queue_mb() if max_queue_mb is None else max_queue_mb) * 2 ** 20)
        self._queued_bytes = 0
        self._space = threading.Condition()
        self.max_backlog_bytes = 0
        self.blocked_s = 0.0  # time the poller waited for queue space (disk too slow)
        self._n_before_block = None  # frames read before the first wait (camera rate is clean there)
        self.bytes_written = 0
        self.write_busy_s = 0.0
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
                nb = frame.nbytes
                with self._space:
                    if self._queued_bytes + nb > self.max_queue_bytes and self._queued_bytes > 0:
                        t_wait = time.perf_counter()
                        if self._n_before_block is None:
                            self._n_before_block = len(self.frame_nrs)
                        while (self._queued_bytes + nb > self.max_queue_bytes and self._queued_bytes > 0
                               and self.error is None):
                            self._space.wait(0.05)
                        self.blocked_s += time.perf_counter() - t_wait
                    self._queued_bytes += nb
                    self.max_backlog_bytes = max(self.max_backlog_bytes, self._queued_bytes)
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
        import os
        import time
        tw = None
        n_written = 0
        try:
            while True:
                frame = self._queue.get()
                if frame is None:
                    break
                if self.shape is None:
                    self.shape, self.dtype = frame.shape, frame.dtype
                t0 = time.perf_counter()
                if tw is None:
                    path = (self.path if not self.frames_per_file else
                            f"{os.path.splitext(self.path)[0]}_{len(self.files):05d}.tif")
                    tw = tf.TiffWriter(path, bigtiff=self.bigtiff)
                    self.files.append({"file": os.path.basename(path), "first_frame": n_written, "n_frames": 0})
                tw.write(frame, contiguous=True, photometric="minisblack")
                self.files[-1]["n_frames"] += 1
                n_written += 1
                if self.frames_per_file and self.files[-1]["n_frames"] >= self.frames_per_file:
                    tw.close()  # this stack is complete on disk now, not when the next frame arrives
                    tw = None
                    if self.on_file_saved is not None:
                        try:
                            self.on_file_saved(dict(self.files[-1]))
                        except Exception:
                            pass
                self.write_busy_s += time.perf_counter() - t0
                self.bytes_written += frame.nbytes
                self._release(frame.nbytes)
            if tw is not None:
                tw.close()
        except Exception as e:
            try:
                if tw is not None:
                    tw.close()
            except Exception:
                pass
            self.error = f"write: {e}"
            with self._space:
                self._space.notify_all()
            while True:  # keep draining so the poller never blocks
                frame = self._queue.get()
                if frame is None:
                    break
                self._release(frame.nbytes)

    def _release(self, nbytes):
        with self._space:
            self._queued_bytes -= nbytes
            self._space.notify_all()

    def _camera_fps(self, nrs):
        """Rate the camera took frames: frame-number span / time span (counts frames lost on the way)."""
        ts = [t for t, n in zip(self.timestamps, self.frame_nrs) if n is not None]
        if len(nrs) < 2 or len(ts) != len(nrs):
            return None
        k = self._n_before_block
        if k is not None and k >= 50 and len(ts) == len(self.frame_nrs):
            nrs, ts = nrs[:k], ts[:k]  # after the disk fell behind, read times no longer follow the camera
        if ts[-1] <= ts[0]:
            return None
        return float((nrs[-1] - nrs[0]) / (ts[-1] - ts[0]))

    @property
    def bytes_per_frame(self):
        return int(np.prod(self.shape)) * np.dtype(self.dtype).itemsize if self.shape is not None else 0

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
        t_flush, last = time.perf_counter(), self.bytes_written
        while self._writer.is_alive():  # flush the RAM backlog; give up only if writing stalls
            self._writer.join(timeout=1.0)
            if self.bytes_written != last:
                t_flush, last = time.perf_counter(), self.bytes_written
            elif time.perf_counter() - t_flush > timeout:
                self.error = (self.error or "") + f" writer stalled for {timeout:.0f} s; file may be incomplete"
                break
        self.flush_s = time.perf_counter() - self._t_stop
        nrs = [n for n in self.frame_nrs if n is not None]
        dropped = []
        for a, b in zip(nrs[:-1], nrs[1:]):
            if b - a > 1:
                dropped.extend(range(a + 1, b))
        dt = np.diff(self.timestamps) if len(self.timestamps) > 1 else np.array([])
        return {
            "path": self.path,
            "frames_per_file": self.frames_per_file,
            "files": self.files,
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
            "max_write_backlog_mb": self.max_backlog_bytes / 2 ** 20,
            "write_queue_limit_mb": self.max_queue_bytes / 2 ** 20,
            "poller_blocked_s": self.blocked_s,
            "stop_to_saved_s": getattr(self, "flush_s", None),
            "camera_fps": self._camera_fps(nrs),
            "data_mb_s": (self._camera_fps(nrs) * self.bytes_per_frame / 2 ** 20
                          if self._camera_fps(nrs) and self.shape is not None else None),
            "disk_write_mb_s": self.bytes_written / 2 ** 20 / self.write_busy_s if self.write_busy_s > 0 else None,
            "error": self.error,
        }
