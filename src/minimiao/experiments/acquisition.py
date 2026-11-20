# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.

"""
Single acquisition block and loops over SLM slots / laser power.

One block:
  1. display an SLM slot
  2. set the laser power (digital modulation: light only while the TriggerBox TTL is high)
  3. upload + ARM the TriggerBox sequence (laser pulse train, optional camera frame clock),
     unless the same sequence is already armed (the box replays it on RUN)
  4. start camera recording (every frame streamed into one 3D TIFF)
  5. RUN: baseline (pre delay) -> pulse train -> post delay
  6. stop the camera, write <stem>.tif + <stem>.json (all parameters and timing)

Camera trigger modes:
  internal : camera free-runs (frame rate set by exposure/readout). Pulse onset
             frames are estimated from host timestamps (approximate).
  external : the TriggerBox also outputs the camera frame clock, so every
             frame and every pulse share one clock and onset frames are exact.
"""

import datetime
import hashlib
import json
import math
import os
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable, Optional

import numpy as np

from minimiao.devices import trigger_box as tbx

SCHEMA = "minimiao.acquisition_block/1"
BIGTIFF_THRESHOLD_BYTES = 3.5e9


class AcquisitionAborted(RuntimeError):
    pass


@dataclass
class BlockSettings:
    slot: int = 1
    laser: str = "473"
    laser_power_mw: float = 1.0
    n_pulses: int = 5
    pulse_width_ms: float = 10.0
    pulse_period_ms: float = 100.0
    laser_channel: int = 1  # TriggerBox DIG channel (output = same number)
    pre_delay_s: float = 1.0  # baseline before the first pulse
    post_delay_s: float = 1.0  # after the pulse train ends
    slm_settle_ms: float = 100.0
    camera_trigger: str = "internal"  # "internal" | "external"
    frame_rate_hz: int = 100  # external mode only
    camera_channel: int = 2  # TriggerBox DIG channel for the camera clock (external mode)
    camera_trigger_width_us: int = 1000
    name: str = ""

    def validate(self):
        """Block-level checks; the pulse train itself is validated by trigger_box.PulseTrain."""
        errs = []
        if self.pre_delay_s < 0 or self.post_delay_s < 0:
            errs.append("delays must be >= 0")
        if self.laser_power_mw < 0:
            errs.append("laser power must be >= 0")
        if self.camera_trigger not in ("internal", "external"):
            errs.append("camera_trigger must be 'internal' or 'external'")
        if self.camera_trigger == "external":
            if self.camera_channel == self.laser_channel:
                errs.append("camera and laser must use different TriggerBox channels")
            if self.frame_rate_hz < 1:
                errs.append("frame rate must be >= 1 Hz")
            elif self.camera_trigger_width_us * 1e-6 >= 1.0 / self.frame_rate_hz:
                errs.append("camera trigger width must be shorter than the frame period")
        if errs:
            raise ValueError("; ".join(errs))


def build_sequence(bs: BlockSettings):
    """
    TriggerBox sequence + timing for one block (pure; no hardware).
    Returns (Sequence, timing dict). All times are ms from RUN.
    """
    bs.validate()
    pre_ms = int(round(bs.pre_delay_s * 1000))
    train = tbx.PulseTrain(bs.n_pulses, bs.pulse_width_ms, bs.pulse_period_ms, pre_ms,
                           bs.laser_channel, bs.laser_channel)
    train_stop = train.stop_ms
    onsets_ms = train.onsets_ms
    last_end = train.last_pulse_end_ms

    channels = [train.channel_def()]
    freqs, widths = [train.freq_hz], [train.width_us]
    timing = {"pre_ms": pre_ms, "train_start_ms": pre_ms, "train_stop_ms": train_stop,
              "last_pulse_end_ms": last_end, "pulse_onsets_ms": onsets_ms, "pulse_freq_hz": train.freq_hz}

    if bs.camera_trigger == "external":
        total = int(math.ceil(last_end + bs.post_delay_s * 1000))
        fps = int(bs.frame_rate_hz)
        channels.append(tbx.DigitalPulse(bs.camera_channel, start_ms=0, stop_ms=total, freq_hz=fps,
                                         width_us=int(bs.camera_trigger_width_us), output=bs.camera_channel))
        freqs.append(fps)
        widths.append(int(bs.camera_trigger_width_us))
        n_frames = int(math.ceil(total * fps / 1000.0 - 1e-9))
        timing.update({"camera_clock_stop_ms": total, "frames_expected": n_frames,
                       # frame i is exposed from i/fps; the frame that contains each pulse onset
                       "pulse_onset_frames": [int(math.floor(t * fps / 1000.0 + 1e-9)) for t in onsets_ms],
                       "pulse_onset_frames_method": "exact (camera frame clock from TriggerBox)"})
        timing["sequence_stop_ms"] = total
    else:
        timing["sequence_stop_ms"] = train_stop

    sysclk = tbx.choose_sysclk(freqs, min_width_us=min(widths), widths_us=widths)
    if sysclk is None:
        # laser train far from the camera clock in frequency: send its pulses as level pulses instead
        try:
            channels, sysclk, converted = tbx._split_trains_into_levels(channels)
            timing["laser_as_level_pulses_on_dig"] = converted[0][1] if converted else None
        except ValueError:
            sysclk = None
    if sysclk is None:
        hint = ""
        if bs.camera_trigger == "external":
            # highest feasible clock ignoring widths, and the shortest pulse it allows (2 ticks)
            f_max = tbx.choose_sysclk(freqs)
            if f_max:
                hint = (f" The highest TriggerBox clock for these frequencies is {f_max} Hz, so every pulse must be "
                        f">= {2e6 / f_max:.0f} us (2 ticks); the camera trigger ({bs.camera_trigger_width_us} us) "
                        f"must fit that and be shorter than the frame period ({1e6 / bs.frame_rate_hz:.0f} us).")
        raise ValueError(f"No TriggerBox system clock fits pulse frequencies {freqs} Hz with widths {widths} us "
                         f"(spread too wide for the shared prescaler, or a pulse shorter than 2 ticks).{hint}")
    seq = tbx.Sequence(sysclk_hz=sysclk, external_trigger=False)
    seq.add(*channels)
    timing["sysclk_hz"] = sysclk
    return seq, timing


def camera_timing_issues(bs: BlockSettings, exposure_ms: float, readout_ms: float = 0.0):
    """
    (errors, warnings) for the camera with this block in TriggerBox (external) camera mode:
    exposure must be shorter than the frame period; a readout time that long risks missed triggers.
    Internal mode has no constraint here.
    """
    errors, warnings = [], []
    if bs.camera_trigger == "external":
        period_ms = 1000.0 / bs.frame_rate_hz
        if exposure_ms >= period_ms:
            errors.append(f"exposure {exposure_ms:g} ms must be shorter than the frame period {period_ms:.3f} ms")
        if readout_ms and readout_ms >= period_ms:
            warnings.append(f"camera readout time {readout_ms:.3f} ms >= frame period {period_ms:.3f} ms; "
                            f"triggers may be missed (smaller ROI / faster readout port)")
    return errors, warnings


def sequence_frames(bs: BlockSettings, timing: dict, exposure_ms: float, readout_ms: float = 0.0) -> int:
    """
    Frames for the block's fixed-length camera sequence.
    external: exactly one frame per TriggerBox camera-clock edge.
    internal: block duration / max(exposure, readout). The camera cannot run faster than
              that, so the sequence always covers the whole block (it may run a little longer).
    """
    if bs.camera_trigger == "external":
        return int(timing["frames_expected"])
    total_ms = max(timing["sequence_stop_ms"], timing["last_pulse_end_ms"] + bs.post_delay_s * 1000.0) + 50.0
    period_ms = max(float(exposure_ms), float(readout_ms or 0.0), 1e-3)
    return int(math.ceil(total_ms / period_ms)) + 1


HIGH_DATA_RATE_MB_S = 400.0  # above this, sustained recording needs a fast local NVMe SSD


def expected_data_rate(info: dict, fps: Optional[float] = None):
    """
    (frames/s, MB/s) the camera will deliver with these recording settings. Free-running:
    1000 / max(exposure, readout) frames/s; pass fps for TriggerBox-clocked recordings.
    """
    px = int(info.get("pixels_x") or 0) // max(1, int(info.get("s_bin") or 1))
    py = int(info.get("pixels_y") or 0) // max(1, int(info.get("p_bin") or 1))
    if fps is None:
        period = max(float(info.get("exposure_ms") or 0.0), float(info.get("readout_time_ms") or 0.0))
        fps = 1000.0 / period if period > 0 else 0.0
    return fps, fps * px * py * 2 / 2 ** 20


def data_rate_message(info: dict, fps: Optional[float] = None) -> tuple:
    """(log text, warning or None) about the expected data rate."""
    fps, mb_s = expected_data_rate(info, fps)
    text = f"expected ~{fps:.0f} frames/s, {mb_s:.0f} MB/s to disk"
    warn = None
    if mb_s > HIGH_DATA_RATE_MB_S:
        warn = (f"data rate {mb_s:.0f} MB/s: sustained recording needs a fast local NVMe SSD (not a network / "
                f"cloud-synced folder); otherwise frames wait in RAM and the camera buffer may overflow")
    return text, warn


def recorder_warnings(summary: dict) -> list:
    """Warnings about the disk keeping up, from a StackRecorder summary."""
    out = []
    blocked = summary.get("poller_blocked_s") or 0.0
    disk, data = summary.get("disk_write_mb_s"), summary.get("data_mb_s")
    rates = (f" (disk {disk:.0f} MB/s vs camera {data:.0f} MB/s)" if disk and data else "")
    if blocked > 0:
        out.append(f"disk could not keep up: frame RAM buffer full, reading waited {blocked:.2f} s{rates}; "
                   f"check dropped frames")
    else:
        lim, peak = summary.get("write_queue_limit_mb"), summary.get("max_write_backlog_mb") or 0.0
        if lim and peak > 0.5 * lim:
            out.append(f"disk fell behind: up to {peak:.0f} MB of frames waited in RAM (limit {lim:.0f} MB){rates}")
    return out


def _git_commit():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except Exception:
        return None


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=_jsonable)
    os.replace(tmp, path)


class AcquisitionRunner:
    """
    Runs blocks on the real devices. Call from a worker thread.
    devs: DeviceManager-like (camera proxy, slm proxy, laser, trg TriggerBox).
    slot_info(slot) -> dict with label / pattern hash, if known.
    context: static metadata (CGH / calibration / correction) stored in every JSON.
    """

    def __init__(self, devs, logg, slot_info: Optional[Callable[[int], dict]] = None,
                 context: Optional[dict] = None, progress: Optional[Callable[[str], None]] = None,
                 preview: Optional[Callable] = None):
        self.devs = devs
        self.logg = logg
        self.slot_info = slot_info or (lambda s: {})
        self.context = context or {}
        self.progress = progress or (lambda msg: None)
        self.preview = preview  # thread-safe frame sink for the ~10 Hz recording preview
        self.abort = threading.Event()
        self.camera_info = {}

    # ---- helpers ---------------------------------------------------------------
    def _sleep(self, seconds):
        if seconds > 0 and self.abort.wait(seconds):
            raise AcquisitionAborted("aborted by user")

    def _check_abort(self):
        if self.abort.is_set():
            raise AcquisitionAborted("aborted by user")

    def prepare_camera(self, trigger, readout=(1, 0, 1)):
        """Apply ROI / exposure / readout (port, speed, gain) / trigger mode once before one or many blocks."""
        port, speed, gain = readout
        self.devs.camera.prepare_recording(trigger, port, speed, gain)
        self.camera_info = self.devs.camera.get_recording_info() or {}

    def _wait_sequence(self, expected_ms):
        tb = self.devs.trg
        t0 = time.monotonic()
        seen_running = False
        while True:
            if self.abort.is_set():
                tb.stop()
                raise AcquisitionAborted("aborted by user")
            state, tick = tb.status()
            running = "RUN" in state.upper()
            seen_running |= running
            elapsed = time.monotonic() - t0
            if not running and (seen_running or elapsed > expected_ms / 1000.0):
                return state, tick
            if elapsed > expected_ms / 1000.0 + 5.0:
                tb.stop()
                raise TimeoutError(f"TriggerBox sequence not finished after {elapsed:.1f} s (state {state})")
            time.sleep(0.05)

    # ---- one block -------------------------------------------------------------
    def run_block(self, bs: BlockSettings, stem: str, loop: Optional[dict] = None) -> dict:
        """Run one block; writes <stem>.tif and <stem>.json. Returns the metadata dict."""
        seq, timing = build_sequence(bs)
        seq.validate()
        cam, slm, laser, tb = self.devs.camera, self.devs.slm, self.devs.laser, self.devs.trg
        exposure_ms = float(self.camera_info.get("exposure_ms") or getattr(cam, "t_exposure", 10))
        readout_ms = float(self.camera_info.get("readout_time_ms") or 0.0)
        errors, early_warnings = camera_timing_issues(bs, exposure_ms, readout_ms)
        if errors:
            raise ValueError("; ".join(errors))
        rate_text, rate_warn = data_rate_message(self.camera_info,
                                                 bs.frame_rate_hz if bs.camera_trigger == "external" else None)
        self.logg.info(f"Block: {rate_text}")
        if rate_warn:
            early_warnings = early_warnings + [rate_warn]
        n_est = sequence_frames(bs, timing, exposure_ms, readout_ms)
        frame_bytes = 2 * int(self.camera_info.get("pixels_x") or 2400) * int(self.camera_info.get("pixels_y") or 2400)
        bigtiff = n_est * frame_bytes > BIGTIFF_THRESHOLD_BYTES
        tif_path, json_path = stem + ".tif", stem + ".json"
        meta = self._block_metadata(bs, seq, timing, loop, n_est, bigtiff, tif_path, early_warnings)
        recording = False
        t = {}
        try:
            self._check_abort()
            self.progress(f"slot {bs.slot}, {bs.laser_power_mw:g} mW: preparing")
            t_slm = time.perf_counter()
            slm.display_pattern(bs.slot)
            laser.set_modulation_mode([bs.laser], [bs.laser_power_mw])
            arm = tb.prepare(seq, validate=False)  # upload + ARM only if this sequence is not armed already
            meta["trigger_box"]["arm"] = arm.get("raw") if arm else "reused armed sequence"
            self._sleep(bs.slm_settle_ms / 1000.0 - (time.perf_counter() - t_slm))

            t["camera_start"] = time.perf_counter()
            cam.start_recording(tif_path, n_frames=n_est, bigtiff=bigtiff)
            recording = True
            if self.preview is not None and getattr(cam, "data", None) is not None:
                cam.data.on_update(self.preview)
            self.progress(f"slot {bs.slot}, {bs.laser_power_mw:g} mW: recording")

            t["run"] = time.perf_counter()
            tb.run_armed()
            self._wait_sequence(timing["sequence_stop_ms"])
            t["sequence_done"] = time.perf_counter()
            if bs.camera_trigger == "internal":
                # post delay counts from the end of the last pulse, not the window stop
                done_ms = timing["sequence_stop_ms"] - timing["last_pulse_end_ms"]
                self._sleep(bs.post_delay_s - done_ms / 1000.0)
            else:  # post delay is inside the camera clock; let the last frames arrive
                self._sleep(2.0 / bs.frame_rate_hz + 0.05)
            meta["status"] = "complete"
        except AcquisitionAborted as e:
            meta["status"], meta["error"] = "aborted", str(e)
        except Exception as e:
            meta["status"], meta["error"] = "error", f"{type(e).__name__}: {e}"
            try:
                tb.stop()
            except Exception:
                pass
        finally:
            if recording:
                # complete block: wait for the rest of the fixed-length sequence; abort/error: stop now
                wait_s = (self._sequence_wait_s(bs, n_est, exposure_ms, readout_ms, t["camera_start"])
                          if meta["status"] == "complete" else 0.0)
                t["camera_stop"] = time.perf_counter()
                try:
                    summary = cam.stop_recording(wait_s=wait_s) or {}
                except Exception as e:
                    summary = {}
                    meta["warnings"].append(f"camera stop failed: {e}")
                self._add_frames(meta, summary, t, bs, timing, exposure_ms)
            meta["timing"]["host_perf_s"] = t
            write_json(json_path, meta)
        if meta["status"] == "aborted":
            raise AcquisitionAborted(meta["error"])
        if meta["status"] == "error":
            raise RuntimeError(meta["error"])
        return meta

    def _block_metadata(self, bs, seq, timing, loop, n_frames, bigtiff, tif_path, warnings):
        """Metadata written next to a block's TIFF (status / frames / timing are filled in while running)."""
        external = bs.camera_trigger == "external"
        return {
            "schema": SCHEMA, "status": "running", "error": None,
            "created": datetime.datetime.now().isoformat(timespec="milliseconds"),
            "software": {"package": "minimiao", "git_commit": _git_commit()},
            "files": {"stack": os.path.basename(tif_path), "bigtiff": bigtiff},
            "block": asdict(bs), "loop": loop,
            "slm": {"slot": bs.slot, "settle_ms": bs.slm_settle_ms, **self.slot_info(bs.slot)},
            "laser": {"name": bs.laser, "mode": "digital modulation (on while TTL high)",
                      "power_mw": bs.laser_power_mw},
            "trigger_box": {"sysclk_hz": seq.sysclk_hz, "commands": seq.commands(),
                            "laser_channel": bs.laser_channel,
                            "camera_channel": bs.camera_channel if external else None},
            "camera": dict(self.camera_info, frame_rate_hz_requested=bs.frame_rate_hz if external else None,
                           frames_sequence=n_frames,
                           frames_sequence_rule=("one per camera-clock edge" if external
                                                 else "block duration / max(exposure, readout); may exceed the block")),
            "timing": dict(timing), "frames": {}, "context": self.context, "warnings": list(warnings),
        }

    @staticmethod
    def _sequence_wait_s(bs, n_frames, exposure_ms, readout_ms, t_camera_start):
        """How long to wait for the rest of a block's fixed-length camera sequence (generous)."""
        period_s = (1.0 / bs.frame_rate_hz if bs.camera_trigger == "external"
                    else max(exposure_ms, readout_ms, 1e-3) / 1000.0)
        remaining = n_frames * period_s - (time.perf_counter() - t_camera_start)
        return max(0.5, 2.0 * remaining + 1.0)

    def _add_frames(self, meta, summary, t, bs, timing, exposure_ms):
        t_run = t.get("run")
        ts = summary.pop("timestamps_perf_s", []) or []
        nrs = summary.pop("frame_nrs", []) or []
        meta["frames"] = {
            "frame_nrs": nrs,
            "retrieved_s_from_run": [x - t_run for x in ts] if t_run is not None else [],
            "note": "retrieved_s_from_run = host time the frame was read from the SDK buffer, relative to RUN",
        }
        meta["camera"].update({k: v for k, v in summary.items() if k != "path"})
        if summary.get("error"):
            meta["warnings"].append(f"recorder: {summary['error']}")
        if summary.get("complete") is False and meta["status"] == "complete":
            meta["warnings"].append(f"sequence incomplete: {summary.get('n_frames')} of "
                                    f"{summary.get('n_frames_requested')} frames received")
        if summary.get("n_dropped"):
            meta["warnings"].append(f"{summary['n_dropped']} dropped frame(s)")
        meta["warnings"].extend(recorder_warnings(summary))
        if bs.camera_trigger == "external":
            exp = timing.get("frames_expected")
            if summary.get("n_frames") is not None and summary["n_frames"] != exp:
                meta["warnings"].append(f"received {summary['n_frames']} frames, expected {exp}; "
                                        f"exact pulse-onset frames assume no missing frames")
        elif t_run is not None and ts:
            # approximate: frame exposure end ~ retrieval time; take the first frame
            # whose estimated exposure window [ret - exposure, ret] ends after the onset
            ends = np.asarray(ts) - t_run
            frames = []
            for on in timing["pulse_onsets_ms"]:
                idx = int(np.searchsorted(ends, on / 1000.0))
                frames.append(idx if idx < len(ends) else None)
            meta["timing"]["pulse_onset_frames"] = frames
            meta["timing"]["pulse_onset_frames_method"] = (
                "approximate (host timestamps; free-running camera; typically within 1-2 frames)")

    # ---- loops -------------------------------------------------------------------
    def _run_loop(self, kind, blocks, folder, interval_s, values_key):
        os.makedirs(folder, exist_ok=True)
        summary = {"schema": "minimiao.acquisition_loop/1", "kind": kind,
                   "created": datetime.datetime.now().isoformat(timespec="milliseconds"),
                   "interval_s": interval_s, "n_blocks": len(blocks), "blocks": [], "status": "running"}
        loop_json = os.path.join(folder, "loop.json")
        try:
            for i, (bs, stem_name) in enumerate(blocks):
                self.progress(f"{kind} block {i + 1}/{len(blocks)}")
                stem = os.path.join(folder, stem_name)
                entry = {"index": i, "stack": stem_name + ".tif", "metadata": stem_name + ".json",
                         values_key: getattr(bs, values_key)}
                try:
                    meta = self.run_block(bs, stem, loop={"kind": kind, "index": i, "n": len(blocks),
                                                          values_key: getattr(bs, values_key)})
                    entry["status"] = meta["status"]
                    entry["warnings"] = meta["warnings"]
                finally:
                    entry.setdefault("status", "aborted" if self.abort.is_set() else "error")
                    summary["blocks"].append(entry)
                    write_json(loop_json, summary)
                if i < len(blocks) - 1:
                    self._sleep(interval_s)
            summary["status"] = "complete"
        except AcquisitionAborted:
            summary["status"] = "aborted"
            raise
        except Exception as e:
            summary["status"] = f"error: {e}"
            raise
        finally:
            write_json(loop_json, summary)
        return summary

    @staticmethod
    def _vary(base: BlockSettings, key, values, name):
        """Blocks equal to `base` except for `key`; name(i, block) gives each block's file stem."""
        blocks = [BlockSettings(**{**asdict(base), key: v}) for v in values]
        return [(bs, name(i, bs)) for i, bs in enumerate(blocks)]

    def run_slot_loop(self, base: BlockSettings, slots, folder, interval_s=2.0):
        blocks = self._vary(base, "slot", [int(s) for s in slots],
                            lambda i, bs: f"block_{i:03d}_slot{bs.slot:02d}")
        return self._run_loop("slot_loop", blocks, folder, interval_s, "slot")

    def run_power_loop(self, base: BlockSettings, powers, folder, interval_s=2.0):
        blocks = self._vary(base, "laser_power_mw", [float(p) for p in powers],
                            lambda i, bs: f"block_{i:03d}_slot{bs.slot:02d}_{bs.laser_power_mw:.2f}mW")
        return self._run_loop("power_loop", blocks, folder, interval_s, "laser_power_mw")


def parse_slots(text: str):
    """'1-5, 8, 10' -> [1, 2, 3, 4, 5, 8, 10] (order kept, duplicates allowed)."""
    out = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = (int(x) for x in part.split("-", 1))
            step = 1 if b >= a else -1
            out.extend(range(a, b + step, step))
        else:
            out.append(int(part))
    if not out:
        raise ValueError("No slots given")
    if any(s < 0 for s in out):
        raise ValueError("Slots must be >= 0")
    return out


def power_steps(start, step, n):
    if n < 1:
        raise ValueError("Number of power steps must be >= 1")
    powers = [round(start + i * step, 4) for i in range(n)]
    if any(p < 0 for p in powers):
        raise ValueError("Laser power must be >= 0")
    return powers


def pattern_sha1(pattern) -> Optional[str]:
    if pattern is None:
        return None
    return hashlib.sha1(np.ascontiguousarray(pattern).tobytes()).hexdigest()
