# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.

"""
Subprocess proxy for KinetixCamera.

PVCAM SDK conflicts with PyQt6 when both run in the same Windows process.
This proxy runs KinetixCamera in a dedicated child process
so the PVCAM DLL never loads in the main process.
"""

import multiprocessing
import threading
from collections import deque

from minimiao import logger


# ---------------------------------------------------------------------------
# Worker — runs inside the child process
# ---------------------------------------------------------------------------

def _camera_worker(cmd_queue, result_queue, frame_queue):
    """Entry point for the camera subprocess."""

    class _Logger:
        def __init__(self, q):
            self._q = q
        def _put(self, level, msg):
            self._q.put(('log', level, str(msg)))
        def info(self, msg):    self._put('info', msg)
        def error(self, msg):   self._put('error', msg)
        def warning(self, msg): self._put('warning', msg)
        def debug(self, msg):   self._put('debug', msg)

    proc_log = _Logger(result_queue)

    try:
        from minimiao.devices.teledyne_kinetix import KinetixCamera
        cam = KinetixCamera(logg=proc_log)
        result_queue.put(('ready', None, _snap(cam)))
    except Exception as e:
        result_queue.put(('error', str(e), {}))
        return

    while True:
        try:
            msg = cmd_queue.get()
        except Exception:
            break

        cmd = msg[0]
        args = msg[1] if len(msg) > 1 else ()
        kwargs = msg[2] if len(msg) > 2 else {}

        if cmd == 'close':
            try:
                cam.close()
                result_queue.put(('ok', None, {}))
            except Exception as e:
                result_queue.put(('error', str(e), {}))
            break

        elif cmd == 'set_attr':
            attr, value = args
            try:
                if hasattr(cam._settings, attr):
                    setattr(cam._settings, attr, value)
                else:
                    setattr(cam, attr, value)
                result_queue.put(('ok', None, _snap(cam)))
            except Exception as e:
                result_queue.put(('error', str(e), _snap(cam)))

        elif cmd == 'start_live':
            try:
                cam.start_live(*args, **kwargs)
                if cam.data is not None:
                    def _fwd(frame, q=frame_queue):
                        try:
                            q.put_nowait(frame)
                        except Exception:
                            pass
                    cam.data.on_update(_fwd)
                result_queue.put(('ok', None, _snap(cam)))
            except Exception as e:
                result_queue.put(('error', str(e), _snap(cam)))

        elif cmd == 'start_data_acquisition':
            n, fd, fn = args
            try:
                cam.start_data_acquisition(n, fd, fn)
                if cam.data is not None:
                    def _fwd(frame, q=frame_queue):
                        try:
                            q.put_nowait(frame)
                        except Exception:
                            pass
                    cam.data.on_update(_fwd)
                result_queue.put(('ok', None, _snap(cam)))
            except Exception as e:
                result_queue.put(('error', str(e), _snap(cam)))

        else:
            try:
                method = getattr(cam, cmd)
                result = method(*args, **kwargs)
                result_queue.put(('ok', result, _snap(cam)))
            except Exception as e:
                result_queue.put(('error', str(e), _snap(cam)))


def _snap(cam):
    """Return a picklable snapshot of the settings the proxy needs to cache."""
    s = cam._settings
    return {
        'pixels_x':  s.pixels_x,
        'pixels_y':  s.pixels_y,
        't_exposure': s.t_exposure,
        's_bin':     s.s_bin,
        'p_bin':     s.p_bin,
        't_clean':   getattr(cam, 't_clean', 0),
        't_readout': getattr(cam, 't_readout', 0),
        'fps':       getattr(cam, 'fps', 0),
    }


# ---------------------------------------------------------------------------
# Proxy helpers
# ---------------------------------------------------------------------------

class _ProxyData:
    """
    Mirrors the CameraDataList interface in the main process.

    Frames received from the subprocess are stored in a local rolling buffer
    so that get_last_element() and get_elements() work without an extra
    round-trip to the subprocess.  Disk saving still happens inside the
    subprocess (via the real CameraDataList) — this buffer is for display
    and in-process analysis only.
    """

    def __init__(self, max_length=16):
        self.callback = None
        self._frames = deque(maxlen=max_length)
        self._lock = threading.Lock()

    def on_update(self, callback):
        self.callback = callback

    def _add_frame(self, frame):
        with self._lock:
            self._frames.append(frame)

    def get_last_element(self):
        import numpy as np
        with self._lock:
            return np.array(self._frames[-1]) if self._frames else None

    def get_elements(self):
        import numpy as np
        with self._lock:
            return np.array(list(self._frames)) if self._frames else None


# ---------------------------------------------------------------------------
# Proxy — lives in the main process, mirrors the KinetixCamera public API
# ---------------------------------------------------------------------------

class KinetixCameraProxy:
    """Drop-in replacement for KinetixCamera running PVCAM in a subprocess."""

    def __init__(self, logg=None):
        self.logg = logg or logger.setup_logging()
        self.data = None

        self._cache = {
            'pixels_x': 2400, 'pixels_y': 2400,
            't_exposure': 10,
            't_clean': 0, 't_readout': 0, 'fps': 0,
            's_bin': 1, 'p_bin': 1,
        }

        self._cmd_q   = multiprocessing.Queue()
        self._res_q   = multiprocessing.Queue()
        self._frame_q = multiprocessing.Queue(maxsize=4)  # drop oldest if viewer lags

        self._proc = multiprocessing.Process(
            target=_camera_worker,
            args=(self._cmd_q, self._res_q, self._frame_q),
            daemon=True,
        )
        self._proc.start()

        status, val, settings = self._recv(timeout=60)
        if status != 'ready':
            self._proc.terminate()
            raise RuntimeError(f"Camera subprocess failed: {val}")
        self._apply(settings)

        self._frame_thread  = None
        self._frame_running = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recv(self, timeout=30):
        import queue as _q
        while True:
            try:
                msg = self._res_q.get(timeout=timeout)
            except _q.Empty:
                return ('error', 'timeout waiting for camera subprocess', {})
            if msg[0] == 'log':
                _, level, text = msg
                getattr(self.logg, level, self.logg.info)(f"[Camera] {text}")
            else:
                return msg

    def _call(self, cmd, *args, **kwargs):
        self._cmd_q.put((cmd, args, kwargs))
        status, val, settings = self._recv()
        self._apply(settings)
        if status == 'error':
            raise RuntimeError(f"Camera.{cmd} failed: {val}")
        return val

    def _apply(self, settings):
        if settings:
            self._cache.update(settings)

    def _start_frames(self):
        self._frame_running = True
        self._frame_thread = threading.Thread(
            target=self._frame_loop, daemon=True
        )
        self._frame_thread.start()

    def _stop_frames(self):
        self._frame_running = False
        if self._frame_thread is not None:
            self._frame_thread.join(timeout=2.0)
            self._frame_thread = None

    def _frame_loop(self):
        import queue as _q
        while self._frame_running:
            try:
                frame = self._frame_q.get(timeout=0.1)
                if self.data is not None:
                    self.data._add_frame(frame)
                    if self.data.callback is not None:
                        self.data.callback(frame)
            except _q.Empty:
                continue
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Attribute access — cached settings
    # ------------------------------------------------------------------

    @property
    def pixels_x(self):  return self._cache['pixels_x']
    @property
    def pixels_y(self):  return self._cache['pixels_y']
    @property
    def t_exposure(self): return self._cache['t_exposure']
    @property
    def t_clean(self):   return self._cache['t_clean']
    @property
    def t_readout(self): return self._cache['t_readout']
    @property
    def fps(self):       return self._cache['fps']

    _ATTR_ALIASES = {
        'bin_h':   's_bin',
        'bin_v':   'p_bin',
        'start_h': 's1',
        'end_h':   's2',
        'start_v': 'p1',
        'end_v':   'p2',
        'gain':    'gain_index',
    }

    _SETTINGS_ATTRS = {
        'pixels_x', 'pixels_y',
        't_exposure', 's_bin', 'p_bin',
        's1', 's2', 'p1', 'p2',
        'temp_setpoint', 'port_index', 'speed_index', 'gain_index',
        'buffer_frame_count',
    }

    def __setattr__(self, name, value):
        if name.startswith('_') or name in ('logg', 'data', 'camera_name'):
            super().__setattr__(name, value)
            return

        settings_name = self._ATTR_ALIASES.get(name, name)

        if settings_name in self._SETTINGS_ATTRS:
            self._cache[settings_name] = value
            self._cmd_q.put(('set_attr', (settings_name, value), {}))
            self._recv()
        else:
            super().__setattr__(name, value)

    # ------------------------------------------------------------------
    # Public API — mirrors KinetixCamera
    # ------------------------------------------------------------------

    def close(self):
        self._stop_frames()
        if self._proc.is_alive():
            self._cmd_q.put(('close', (), {}))
            self._recv(timeout=15)
            self._proc.join(timeout=15)
            if self._proc.is_alive():
                self._proc.terminate()

    def prepare_live(self, port=1, speed=0, gain=1, exp_mode=1792, expose_out=0):
        self._call('prepare_live', port, speed, gain, exp_mode, expose_out)

    def start_live(self):
        self.data = _ProxyData()
        self._call('start_live')
        self._start_frames()

    def stop_live(self):
        self._stop_frames()
        self.data = None
        try:
            self._call('stop_live')
        except Exception as e:
            self.logg.error(f"Camera stop_live: {e}")

    def finish(self):
        try:
            self._call('finish')
        except Exception:
            pass

    def start_data_acquisition(self, n, fd, fn):
        self.data = _ProxyData()
        self._cmd_q.put(('start_data_acquisition', (n, fd, fn), {}))
        status, val, settings = self._recv()
        self._apply(settings)
        if status == 'error':
            raise RuntimeError(f"Camera.start_data_acquisition failed: {val}")
        self._start_frames()

    def stop_data_acquisition(self):
        self._stop_frames()
        self.data = None
        try:
            self._call('stop_data_acquisition')
        except Exception as e:
            self.logg.error(f"Camera stop_data_acquisition: {e}")

    def get_last_image(self):
        if self.data is not None:
            return self.data.get_last_element()
        return None

    def get_data(self):
        if self.data is not None:
            return self.data.get_elements()
        return None

    def set_roi(self):
        self._call('set_roi')

    def prepare_data_acquisition(self, port=0, speed=0, gain=1,
                                  exp_mode=None, expose_out=None):
        # Pass only the args that are explicitly provided; let the subprocess
        # method use its own pvc_consts defaults for exp_mode / expose_out.
        args = (port, speed, gain)
        kwargs = {}
        if exp_mode is not None:
            kwargs['exp_mode'] = exp_mode
        if expose_out is not None:
            kwargs['expose_out'] = expose_out
        self._cmd_q.put(('prepare_data_acquisition', args, kwargs))
        status, val, settings = self._recv()
        self._apply(settings)
        if status == 'error':
            raise RuntimeError(f"Camera.prepare_data_acquisition failed: {val}")
