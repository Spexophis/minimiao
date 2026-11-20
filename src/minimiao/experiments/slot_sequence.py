# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.

"""
Timed SLM slot sequence: one continuous camera recording over a list of SLM slots, each followed
by one TriggerBox sequence (the Ctrl Panel DO#1..8 and AO#1..4 rows).

    camera start (free-running)
    [pre delay]
    upload + ARM the TriggerBox sequence once (the box replays it on every RUN)
    for each slot:
        SLM: display slot                    -> t_slm (driver call returned)
        wait until t_slm + slm_to_trigger    (sleep, then busy-wait the last ms)
        TriggerBox RUN                       -> t_run (command written)
        wait until t_run + sequence length   (sleep, then busy-wait)
        STATUS until the box reports it stopped
    [post delay]
    camera stop -> <stem>.tif + <stem>.json

Host jitter is kept small by: no logging / preview frames / GUI work in the timed loop, garbage
collection paused, a short GIL switch interval, high process/thread priority and 1 ms timer
resolution on Windows, sleep + busy-wait for every deadline, and RUN/STATUS read without the
normal 50 ms reply gap. Every step's measured times are saved, so the achieved delay and jitter
can be checked after each run. What the host cannot remove: USB transfer of RUN to the TriggerBox
(typically up to ~1 ms) and the SLM's own frame update / liquid-crystal response after the slot change.
"""

import ctypes
import datetime
import gc
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import numpy as np

from minimiao.devices import trigger_box as tbx
from minimiao.experiments import acquisition as acq

SCHEMA = "minimiao.slot_sequence/1"
SPIN_S = 0.002  # busy-wait this much before every deadline


@dataclass
class SlotSequenceSettings:
    slots: list = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    slm_to_trigger_ms: float = 20.0  # SLM display command returned -> TriggerBox RUN
    pre_delay_s: float = 0.5  # camera running before the first slot is displayed
    post_delay_s: float = 0.5  # after the last sequence finished, before the camera stops
    laser: str = "473"
    laser_power_mw: float = 1.0
    end_margin_ms: float = 1.0  # extra wait after the sequence's nominal end before checking STATUS
    name: str = ""

    def validate(self):
        errs = []
        if not self.slots:
            errs.append("no slots")
        elif any(int(s) < 0 for s in self.slots):
            errs.append("slots must be >= 0")
        if self.slm_to_trigger_ms < 0:
            errs.append("SLM -> trigger delay must be >= 0")
        if self.pre_delay_s < 0 or self.post_delay_s < 0:
            errs.append("delays must be >= 0")
        if self.laser_power_mw < 0:
            errs.append("laser power must be >= 0")
        if errs:
            raise ValueError("; ".join(errs))


def build_trigger(outputs):
    """TriggerBox Sequence from the DO / AO rows; returns (seq, summary, duration_ms, last_pulse_end_ms)."""
    seq, summary = tbx.sequence_from_outputs(outputs)
    return seq, summary, float(seq.duration_ms), float(max(s["end_ms"] for s in summary))


# ---- precise waiting ----------------------------------------------------------------

def wait_until(t_target, abort: threading.Event, spin_s: float = SPIN_S):
    """Sleep until spin_s before t_target (perf_counter), then busy-wait. Raises on abort."""
    while True:
        rem = t_target - time.perf_counter()
        if rem <= spin_s:
            break
        if abort.is_set():
            raise acq.AcquisitionAborted("aborted by user")
        time.sleep(min(rem - spin_s, 0.05))
    while time.perf_counter() < t_target:
        pass


@contextmanager
def realtime():
    """Reduce host jitter for the timed loop; everything is restored afterwards."""
    old_switch = sys.getswitchinterval()
    gc_was_on = gc.isenabled()
    win = sys.platform == "win32"
    restore = []
    try:
        gc.collect()
        gc.disable()
        sys.setswitchinterval(1e-4)  # another Python thread hands the GIL back within ~0.1 ms
        if win:
            try:
                winmm, k32 = ctypes.windll.winmm, ctypes.windll.kernel32
                if winmm.timeBeginPeriod(1) == 0:
                    restore.append(lambda: winmm.timeEndPeriod(1))
                proc, thread = k32.GetCurrentProcess(), k32.GetCurrentThread()
                old_cls, old_prio = k32.GetPriorityClass(proc), k32.GetThreadPriority(thread)
                if k32.SetPriorityClass(proc, 0x00000080):  # HIGH_PRIORITY_CLASS
                    restore.append(lambda: k32.SetPriorityClass(proc, old_cls))
                if k32.SetThreadPriority(thread, 2):  # THREAD_PRIORITY_HIGHEST
                    restore.append(lambda: k32.SetThreadPriority(thread, old_prio))
            except Exception:
                pass
        yield
    finally:
        for f in reversed(restore):
            try:
                f()
            except Exception:
                pass
        sys.setswitchinterval(old_switch)
        if gc_was_on:
            gc.enable()


# ---- runner -----------------------------------------------------------------------

class SlotSequenceRunner:
    """Runs one slot sequence on the real devices. Call from a worker thread."""

    def __init__(self, devs, logg, slot_info: Optional[Callable[[int], dict]] = None,
                 context: Optional[dict] = None, progress: Optional[Callable[[str], None]] = None):
        self.devs = devs
        self.logg = logg
        self.slot_info = slot_info or (lambda s: {})
        self.context = context or {}
        self.progress = progress or (lambda msg: None)
        self.abort = threading.Event()
        self.camera_info = {}

    def prepare_camera(self, readout=(1, 0, 1)):
        port, speed, gain = readout
        self.devs.camera.prepare_recording("internal", port, speed, gain)
        self.camera_info = self.devs.camera.get_recording_info() or {}

    @staticmethod
    def _run(tb):
        """RUN the armed sequence (timestamped). If the box answers 'not armed' (ERR 13), ARM and RUN
        again; returns (t_write_start, t_write_end, rearmed)."""
        with tb._lock:
            t0, t1 = tb.write_timed("RUN")
            try:
                tb.read_reply_fast("RUN", timeout=0.5, quiet=0.0)
                return t0, t1, False
            except tbx.TriggerBoxError as e:
                if e.code != 13:
                    raise
            tb.command_fast("ARM", timeout=2.0, quiet=0.003)
            t0, t1 = tb.write_timed("RUN")
            tb.read_reply_fast("RUN", timeout=0.5, quiet=0.0)
            return t0, t1, True

    def _wait_stopped(self, tb, timeout_s):
        """Poll STATUS (fast) until the box is no longer running; returns (state, tick, n_polls)."""
        t_end = time.perf_counter() + timeout_s
        n = 0
        while True:
            state, tick = tb.status_fast()
            n += 1
            if "RUN" not in state.upper():
                return state, tick, n
            if self.abort.is_set():
                raise acq.AcquisitionAborted("aborted by user")
            if time.perf_counter() > t_end:
                raise TimeoutError(f"TriggerBox still running {timeout_s:.1f} s after the expected end")

    def run(self, ss: SlotSequenceSettings, outputs, stem: str) -> dict:
        ss.validate()
        seq, summary, dur_ms, last_end_ms = build_trigger(outputs)
        cam, slm, laser, tb = self.devs.camera, self.devs.slm, self.devs.laser, self.devs.trg
        slots = [int(s) for s in ss.slots]
        tif_path, json_path = stem + ".tif", stem + ".json"
        est_s = ss.pre_delay_s + ss.post_delay_s + len(slots) * (ss.slm_to_trigger_ms + dur_ms + 30) / 1000
        exposure_ms = float(self.camera_info.get("exposure_ms") or getattr(cam, "t_exposure", 10))
        frame_bytes = 2 * int(self.camera_info.get("pixels_x") or 2400) * int(self.camera_info.get("pixels_y") or 2400)
        bigtiff = (est_s * 1000.0 / max(exposure_ms, 1e-3)) * frame_bytes > acq.BIGTIFF_THRESHOLD_BYTES
        meta = {
            "schema": SCHEMA, "status": "running", "error": None,
            "created": datetime.datetime.now().isoformat(timespec="milliseconds"),
            "software": {"package": "minimiao", "git_commit": acq._git_commit()},
            "files": {"stack": os.path.basename(tif_path), "bigtiff": bigtiff},
            "settings": asdict(ss),
            "slots": [{"slot": s, **self.slot_info(s)} for s in slots],
            "laser": {"name": ss.laser, "mode": "digital modulation (on while TTL high)",
                      "power_mw": ss.laser_power_mw},
            "trigger_box": {"sysclk_hz": seq.sysclk_hz, "commands": seq.commands(), "outputs": summary,
                            "sequence_ms": dur_ms, "last_pulse_end_ms": last_end_ms},
            "camera": dict(self.camera_info, trigger="internal (free-running)"),
            "steps": [], "frames": {}, "context": self.context, "warnings": [],
            "timing_note": ("host perf_counter seconds relative to camera_start. t_slm = SLM driver call "
                            "returned; t_run = RUN written to the TriggerBox; delay_ms = t_run - t_slm; "
                            "t_stopped = STATUS first reported the box stopped."),
        }
        rate_text, rate_warn = acq.data_rate_message(self.camera_info)
        self.logg.info(f"Slot sequence: {rate_text}")
        if rate_warn:
            meta["warnings"].append(rate_warn)
        t = {}
        recording = False
        try:
            self.progress("preparing")
            laser.set_modulation_mode([ss.laser], [ss.laser_power_mw])
            meta["trigger_box"]["arm"] = (tb.prepare(seq, validate=False) or {}).get("raw", "reused armed sequence")

            t["camera_start"] = time.perf_counter()
            cam.start_recording(tif_path, n_frames=None, bigtiff=bigtiff, preview=False)
            recording = True
            self.progress(f"recording: {len(slots)} slots")
            wait_until(time.perf_counter() + ss.pre_delay_s, self.abort)

            delay_s = ss.slm_to_trigger_ms / 1000.0
            with realtime():
                for i, slot in enumerate(slots):
                    if self.abort.is_set():
                        raise acq.AcquisitionAborted("aborted by user")
                    t_cmd = time.perf_counter()
                    t_slm0, t_slm = slm.display_timed(slot)
                    t_ret = time.perf_counter()
                    if not (t_cmd - 0.5 <= t_slm <= t_ret + 1e-3):  # clocks not comparable: use our own
                        t_slm = t_ret
                    wait_until(t_slm + delay_s, self.abort)
                    t_run0, t_run1, rearmed = self._run(tb)
                    wait_until(t_run0 + (dur_ms + ss.end_margin_ms) / 1000.0, self.abort)
                    state, tick, n_polls = self._wait_stopped(tb, timeout_s=2.0)
                    t_stop = time.perf_counter()
                    meta["steps"].append({
                        "index": i, "slot": slot, "t_slm_call": t_slm0, "t_slm": t_slm,
                        "t_slm_returned_here": t_ret, "t_run": t_run0, "t_run_write_end": t_run1,
                        "t_stopped": t_stop, "delay_ms": (t_run0 - t_slm) * 1e3,
                        "slm_call_ms": (t_slm - t_slm0) * 1e3, "run_write_ms": (t_run1 - t_run0) * 1e3,
                        "sequence_host_ms": (t_stop - t_run0) * 1e3, "status_polls": n_polls,
                        "final_state": state, "final_tick": tick, "rearmed": rearmed})
                    self.progress(f"slot {slot} done ({i + 1}/{len(slots)})")
            t["last_done"] = time.perf_counter()
            wait_until(t["last_done"] + ss.post_delay_s, self.abort)
            meta["status"] = "complete"
        except acq.AcquisitionAborted as e:
            meta["status"], meta["error"] = "aborted", str(e)
            self._stop_box(tb)
        except Exception as e:
            meta["status"], meta["error"] = "error", f"{type(e).__name__}: {e}"
            self._stop_box(tb)
        finally:
            if recording:
                t["camera_stop"] = time.perf_counter()
                try:
                    cam_summary = cam.stop_recording(wait_s=0.0) or {}
                except Exception as e:
                    cam_summary = {}
                    meta["warnings"].append(f"camera stop failed: {e}")
                self._add_frames(meta, cam_summary, t)
            self._finish_timing(meta, t, ss)
            acq.write_json(json_path, meta)
        if meta["status"] == "aborted":
            raise acq.AcquisitionAborted(meta["error"])
        if meta["status"] == "error":
            raise RuntimeError(meta["error"])
        return meta

    @staticmethod
    def _stop_box(tb):
        try:
            tb.stop()
        except Exception:
            pass

    @staticmethod
    def _finish_timing(meta, t, ss):
        t0 = t.get("camera_start", 0.0)
        for st in meta["steps"]:
            for k in ("t_slm_call", "t_slm", "t_slm_returned_here", "t_run", "t_run_write_end", "t_stopped"):
                st[k] = st[k] - t0
        meta["host_times_s"] = {k: v - t0 for k, v in t.items()}
        d = np.array([s["delay_ms"] for s in meta["steps"]], dtype=float)
        if d.size:
            meta["delay_stats_ms"] = {"target": ss.slm_to_trigger_ms, "mean": float(d.mean()),
                                      "min": float(d.min()), "max": float(d.max()),
                                      "max_error": float(np.abs(d - ss.slm_to_trigger_ms).max())}
            gaps = [meta["steps"][k + 1]["t_slm"] - meta["steps"][k]["t_stopped"]
                    for k in range(len(meta["steps"]) - 1)]
            meta["stopped_to_next_slm_ms"] = [g * 1e3 for g in gaps]
            if meta["delay_stats_ms"]["max_error"] > 0.5:
                meta["warnings"].append(f"SLM -> RUN delay off by up to "
                                        f"{meta['delay_stats_ms']['max_error']:.3f} ms on the host")
            if any(s["rearmed"] for s in meta["steps"]):
                meta["warnings"].append("TriggerBox answered 'not armed' at least once; it was re-armed and RUN "
                                        "sent again (that step's SLM -> RUN delay includes the ARM)")

    @staticmethod
    def _add_frames(meta, summary, t):
        t0 = t.get("camera_start", 0.0)
        ts = summary.pop("timestamps_perf_s", []) or []
        nrs = summary.pop("frame_nrs", []) or []
        rel = [x - t0 for x in ts]
        meta["frames"] = {"frame_nrs": nrs, "retrieved_s": rel,
                          "note": "retrieved_s = host time the frame was read from the SDK buffer, "
                                  "relative to camera_start (exposure ended shortly before)"}
        meta["camera"].update({k: v for k, v in summary.items() if k != "path"})
        if summary.get("error"):
            meta["warnings"].append(f"recorder: {summary['error']}")
        if summary.get("n_dropped"):
            meta["warnings"].append(f"{summary['n_dropped']} dropped frame(s)")
        meta["warnings"].extend(acq.recorder_warnings(summary))
        if rel:
            ends = np.asarray(ts) - t0
            for st in meta["steps"]:
                for key, src in (("frame_at_slm", "t_slm"), ("frame_at_run", "t_run")):
                    idx = int(np.searchsorted(ends, st[src] - t0))  # steps still hold absolute times
                    st[key] = idx if idx < len(ends) else None
            meta["frames"]["step_frames_method"] = "approximate (host timestamps; first frame read after the event)"
