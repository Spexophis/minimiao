# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.

"""
Subprocess proxy for HamamatsuSLM.
"""

import multiprocessing
import numpy as np

from minimiao import logger


def _slm_worker(cmd_queue, result_queue, serial_number):
    """Entry point for the SLM subprocess."""

    try:
        from minimiao.devices.hamamatsu_slm import HamamatsuSLM

        class _ForwardingHandler:
            """Forward log records to the main process via result_queue."""
            import logging as _logging

            def __init__(self, q):
                self._q = q

            def emit(self, record):
                self._q.put(('log', record.levelname.lower(), record.getMessage()))

        import logging
        _handler = _ForwardingHandler(result_queue)

        class _Logger:
            def __init__(self, q):
                self._q = q

            def _put(self, level, msg):
                self._q.put(('log', level, msg))

            def info(self, msg): self._put('info', msg)
            def error(self, msg): self._put('error', msg)
            def warning(self, msg): self._put('warning', msg)
            def debug(self, msg): self._put('debug', msg)

        proxy_log = _Logger(result_queue)
        slm = HamamatsuSLM(serial_number=serial_number, logg=proxy_log)
        result_queue.put(('ready', None))

    except Exception as e:
        result_queue.put(('error', str(e)))
        return

    while True:
        try:
            cmd, args, kwargs = cmd_queue.get()
        except Exception:
            break

        if cmd == 'close':
            try:
                slm.close()
                result_queue.put(('ok', None))
            except Exception as e:
                result_queue.put(('error', str(e)))
            break

        try:
            method = getattr(slm, cmd)
            result = method(*args, **kwargs)
            # Return pattern array alongside result so proxy can cache it
            extra = slm.pattern if hasattr(slm, 'pattern') else None
            result_queue.put(('ok', result, extra))
        except Exception as e:
            result_queue.put(('error', str(e), None))


# ---------------------------------------------------------------------------
# Proxy — lives in the main process, mirrors the HamamatsuSLM public API
# ---------------------------------------------------------------------------

class HamamatsuSLMProxy:
    """Drop-in replacement for HamamatsuSLM that runs the SLM in a subprocess."""

    def __init__(self, serial_number="LSH0805629", logg=None):
        self.logg = logg or logger.setup_logging()
        self.pattern = None
        self.nx = 1272
        self.ny = 1024
        self.array_size = self.nx * self.ny

        self._cmd_q = multiprocessing.Queue()
        self._res_q = multiprocessing.Queue()
        self._proc = multiprocessing.Process(
            target=_slm_worker,
            args=(self._cmd_q, self._res_q, serial_number),
            daemon=True,
        )
        self._proc.start()

        status, val = self._recv(timeout=30)
        if status != 'ready':
            self._proc.terminate()
            raise RuntimeError(f"SLM subprocess failed to start: {val}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recv(self, timeout=30):
        """Read from result queue, forwarding log messages to self.logg."""
        import queue as _queue
        while True:
            try:
                msg = self._res_q.get(timeout=timeout)
            except _queue.Empty:
                return ('error', 'timeout waiting for SLM subprocess')

            if msg[0] == 'log':
                _, level, text = msg
                getattr(self.logg, level, self.logg.info)(f"[SLM] {text}")
            else:
                return msg

    def _call(self, method_name, *args, **kwargs):
        """Send a command and return its result, raising on error."""
        self._cmd_q.put((method_name, args, kwargs))
        msg = self._recv()
        status = msg[0]
        val = msg[1]
        extra = msg[2] if len(msg) > 2 else None
        if status == 'error':
            raise RuntimeError(f"SLM.{method_name} failed: {val}")
        if extra is not None:
            self.pattern = extra
        return val

    # ------------------------------------------------------------------
    # Public API — matches HamamatsuSLM exactly
    # ------------------------------------------------------------------

    def close(self):
        if self._proc.is_alive():
            self._cmd_q.put(('close', (), {}))
            self._recv(timeout=10)
            self._proc.join(timeout=10)
            if self._proc.is_alive():
                self._proc.terminate()
        if hasattr(self, 'logg'):
            self.logg.info("SLM Closed")

    def load_pattern(self, pfd, slot_no=0):
        return self._call('load_pattern', pfd, slot_no)

    def display_pattern(self, slot_no=0):
        return self._call('display_pattern', slot_no)

    def check_temp(self):
        return self._call('check_temp')

    def mode_select(self, mode):
        return self._call('mode_select', mode)

    def mode_check(self, bid=None):
        return self._call('mode_check', bid)
