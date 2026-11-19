# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import os
import threading
import time

import numpy as np
import tifffile as tf
from PyQt6.QtCore import QObject, pyqtSlot, pyqtSignal, QTimer

from . import run_threads
from .computations import slm_calibration as slm_cal
from .experiments import acquisition as acq
from .experiments import slot_sequence as slseq
from .devices import trigger_box as tbx


class CommandExecutor(QObject):
    sig_progress = pyqtSignal(str)
    sig_trigger_status = pyqtSignal(str)

    def __init__(self, dev, cwd, cmp, path, logger=None):
        super().__init__()
        self.devs = dev
        self.vw = cwd
        self.ctrl_panel = self.vw.ctrl_panel
        self.viewer = self.vw.viewer
        self.hg_panel = self.vw.hg_panel
        self.acq_panel = self.vw.acq_panel
        self.runner = None
        self._fire_thread = None
        self._recording = None  # Acquire-button recording in progress
        self._fps_timer = QTimer(self)  # measured camera frame rate -> FPS display, twice a second
        self._fps_timer.timeout.connect(self._update_fps_display)
        self._fps_timer.start(500)
        self._fire_stopped = threading.Event()
        self.cgh = cmp.cgh
        self.path = path
        self.logg = logger or self.setup_logging()
        self._set_signal_executions()
        self._initial_setup()
        self.task_worker = None
        self._after_task = None
        self.slot_patterns = {}  # SLM memory slot -> (label, uint8 pattern) written this session
        self.calib_holo_pts = None  # hologram-plane points currently in slots 1..K
        self.calib_meta = {}  # metadata of the active affine calibration
        self.calib_file = None  # file the active calibration came from
        self._calib_result = None
        self._calib_abort = threading.Event()  # Esc in a calibration dialog

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    def _set_signal_executions(self):
        # SLM
        self.ctrl_panel.Signal_slm_correction.connect(self.load_slm_correction)
        self.ctrl_panel.Signal_slm_load.connect(self.load_slm_pattern)
        self.ctrl_panel.Signal_slm_base.connect(self.apply_slm_base)
        self.ctrl_panel.Signal_slm_display.connect(self.display_slm_slot)
        # Lasers
        self.ctrl_panel.Signal_set_laser.connect(self.set_laser)
        # Trigger
        self.ctrl_panel.Signal_trigger_fire.connect(self.fire_trigger)
        self.ctrl_panel.Signal_trigger_reset.connect(self.stop_trigger)
        self.sig_trigger_status.connect(self.ctrl_panel.display_trigger_status)
        # Acquisition
        self.ctrl_panel.Signal_video.connect(self.video)
        self.ctrl_panel.Signal_data_acquire.connect(self.acquisition)
        # Hologram
        self.hg_panel.Signal_load_target.connect(self.load_cgh_target)
        self.hg_panel.Signal_pick_spot.connect(self.pick_cgh_spots)
        self.hg_panel.Signal_compute_cgh.connect(self.compute_cgh_pattern)
        self.hg_panel.Signal_save_pattern.connect(self.save_cgh_pattern)
        self.hg_panel.Signal_calib_generate.connect(self.generate_calibration_holograms)
        self.hg_panel.Signal_calib_measure.connect(self.measure_calibration)
        self.hg_panel.Signal_calib_load.connect(self.load_calibration)
        self.hg_panel.Signal_calib_auto.connect(self.auto_calibration)
        self.hg_panel.Signal_calib_check.connect(self.check_spots)
        # Acquisition blocks
        self.acq_panel.Signal_run_block.connect(self.run_acquisition_block)
        self.acq_panel.Signal_run_slot_loop.connect(self.run_slot_loop)
        self.acq_panel.Signal_run_power_loop.connect(self.run_power_loop)
        self.acq_panel.Signal_run_slot_sequence.connect(self.run_slot_sequence)
        self.sig_progress.connect(self._show_progress)

    def _initial_setup(self):
        self.logg.info("Finish setting up controllers")
        try:
            if self.devs.camera is not None:
                options = self.devs.camera.get_readout_options() or []
                self.ctrl_panel.set_readout_options(options)
                self.logg.info(f"Camera readout modes: {len(options)} found")
        except Exception as e:
            self.logg.error(f"Could not read camera readout modes (default port 1 / speed 0 / gain 1 kept): {e}")
        try:
            self._load_latest_calibration()
        except Exception as e:
            self.logg.error(f"Could not auto-load the camera-SLM calibration: {e}")

    def set_camera_roi(self):
        try:
            x, y, nx, ny, bn = self.ctrl_panel.get_scmos_roi()
            self.devs.camera.s_bin, self.devs.camera.p_bin = bn, bn
            self.devs.camera.s1, self.devs.camera.pixels_x = x, nx
            self.devs.camera.p1, self.devs.camera.pixels_y = y, ny
            self.devs.camera.t_exposure = self.ctrl_panel.get_scmos_exposure()
        except Exception as e:
            self.logg.error(f"Camera Error: {e}")

    # ---- TriggerBox: manual fire -------------------------------------------------

    def _trigger_firing(self):
        return self._fire_thread is not None and self._fire_thread.is_alive()

    @pyqtSlot(list)
    def fire_trigger(self, outputs: list):
        """Build one sequence from the DO#1..8 and AO#1..4 rows and fire it."""
        tb = getattr(self.devs, "trg", None)
        if tb is None:
            self.logg.error("TriggerBox not connected")
            return
        if self._trigger_firing():
            self.logg.error("TriggerBox is already firing; press Stop / Reset to abort it")
            return
        if getattr(self, "task_worker", None) is not None and self.task_worker.isRunning():
            self.logg.error("A task is running (it may be using the TriggerBox); fire refused")
            return
        if self._recording is not None:
            self.logg.info("Firing during an Acquire recording")
        try:
            seq, summary = tbx.sequence_from_outputs(outputs)
        except Exception as e:
            self.logg.error(f"Trigger settings: {e}")
            self.ctrl_panel.display_trigger_status(f"Error: {e}")
            return
        desc = tbx.describe_outputs(summary)
        total = max(s["end_ms"] for s in summary)
        self.logg.info(f"Firing TriggerBox ({seq.sysclk_hz} Hz clock, {total:g} ms): {desc}")
        self.ctrl_panel.display_trigger_status(f"Firing... {desc}")

        self._fire_stopped.clear()

        def job():
            try:
                tb.execute(seq, wait=True)
                if self._fire_stopped.is_set():
                    self.sig_trigger_status.emit(f"Stopped early: {desc}")
                else:
                    self.sig_trigger_status.emit(f"Done ({total:g} ms): {desc}")
            except Exception as e:
                self.logg.error(f"TriggerBox fire failed: {e}")
                self.sig_trigger_status.emit(f"Error: {e}")

        self._fire_thread = threading.Thread(target=job, daemon=True)
        self._fire_thread.start()

    @pyqtSlot()
    def stop_trigger(self):
        """Stop any running sequence and reset the TriggerBox."""
        tb = getattr(self.devs, "trg", None)
        if tb is None:
            self.logg.error("TriggerBox not connected")
            return
        if self._trigger_firing():
            self._fire_stopped.set()
        try:
            tb.stop()
            tb.reset()
            self.ctrl_panel.display_trigger_status("Stopped / reset")
            self.logg.info("TriggerBox stopped and reset")
        except Exception as e:
            self.logg.error(f"TriggerBox stop/reset failed: {e}")

    # ---- acquisition blocks / loops --------------------------------------------

    def _show_progress(self, msg: str):
        if getattr(self.vw, "dialog_text", None) is not None:
            self.vw.dialog_text.setText(f"{msg}  (Esc to abort)")

    def _slot_info(self, slot):
        entry = self.slot_patterns.get(slot)
        if entry is None:
            return {"label": None, "pattern_sha1": None, "note": "slot not written in this session"}
        return {"label": entry[0], "pattern_sha1": acq.pattern_sha1(entry[1])}

    def _acq_context(self):
        hs = self.cgh.holo_sys
        return {
            "cgh_at_last_hologram_computation": {
                    "f_reference_m": hs.f_fresnel, "holograms_lens": self.cgh.holograms_lens,
                    "slot0": {"lens_enabled": hs.lens_enabled, "cross_enabled": hs.cross_enabled,
                              "f_fresnel_m": hs.f_fresnel if hs.lens_enabled else None},
                    "hologram_composition": "correction + focal-array hologram + Fresnel lens (f_fresnel_m)",
                    "slm_offset": [hs.slm_offset_x, hs.slm_offset_y],
                    "correction_value": hs.correction_value, "mode_last_computed": self.hg_panel.get_cgh_mode(),
                    "spots_xy_um_z": [list(map(float, sp)) for sp in (self.cgh.spots_xy or [])]},
            "affine_calibration": {"in_use": self.cgh.affine is not None, "file": self.calib_file,
                                   "M": None if self.cgh.affine is None else self.cgh.affine.tolist(),
                                   "meta": self.calib_meta},
            "camera_roi_gui": list(self.ctrl_panel.get_scmos_roi()),
            "exposure_ms_gui": self.ctrl_panel.get_scmos_exposure(),
        }

    def _start_acquisition_task(self, title, settings: dict, job):
        """Common checks + camera setup, then run job(runner, block_settings) in the worker thread."""
        if self._slm_busy():
            return
        if self._trigger_firing():
            self.logg.error("TriggerBox is busy firing; wait or press Stop / Reset")
            return
        if self._recording is not None:
            self.logg.error("Acquire is recording or still saving; release it / wait for 'Acquisition saved'")
            return
        missing = [n for n in ("camera", "slm", "laser", "trg") if getattr(self.devs, n, None) is None]
        if missing:
            self.logg.error(f"Acquisition needs: {', '.join(missing)} (not connected)")
            return
        try:
            bs = acq.BlockSettings(**settings)
            acq.build_sequence(bs)  # validates timing / TriggerBox clock before touching hardware
            errors, _ = acq.camera_timing_issues(bs, self.ctrl_panel.get_scmos_exposure())
            if errors:
                raise ValueError("; ".join(errors))
        except Exception as e:
            self.logg.error(f"Invalid acquisition settings: {e}")
            return
        if getattr(self.viewer, "live_worker", None) is not None:
            self.logg.info("Stopping live video for the acquisition")
            self.stop_video()
            self.ctrl_panel.QPushButton_video.setChecked(False)
        self.set_camera_roi()
        readout = self.ctrl_panel.get_scmos_readout()
        runner = acq.AcquisitionRunner(self.devs, self.logg, slot_info=self._slot_info,
                                       context=self._acq_context(), progress=self.sig_progress.emit,
                                       preview=self.viewer.on_camera_update_from_thread)
        self.runner = runner

        def task():
            try:
                runner.prepare_camera(bs.camera_trigger, readout)
                job(runner, bs)
            except acq.AcquisitionAborted:
                self.logg.info(f"{title} aborted; partial data and metadata saved")
            except Exception as e:
                self.logg.error(f"{title} failed: {e}")

        self._after_task = lambda: setattr(self, "runner", None)
        self.vw.get_dialog(title, interrupt=True)
        self.vw.dialog.dialog_closed.connect(runner.abort.set)
        self.run_task(task=task)

    def _stem(self, bs, kind):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name = f"_{bs.name}" if bs.name else ""
        return f"{stamp}{name}_{kind}"

    @pyqtSlot(dict)
    def run_acquisition_block(self, settings: dict):
        def job(runner, bs):
            stem = os.path.join(self.path, self._stem(bs, f"block_slot{bs.slot:02d}_{bs.laser_power_mw:.2f}mW"))
            meta = runner.run_block(bs, stem)
            self.logg.info(f"Block saved: {stem}.tif ({meta['camera'].get('n_frames')} frames); "
                           f"warnings: {meta['warnings'] or 'none'}")
        self._start_acquisition_task("Acquisition Block", settings, job)

    @pyqtSlot(dict, str, float)
    def run_slot_loop(self, settings: dict, slots_text: str, interval_s: float):
        try:
            slots = acq.parse_slots(slots_text)
        except Exception as e:
            self.logg.error(f"Invalid slot list {slots_text!r}: {e}")
            return

        def job(runner, bs):
            folder = os.path.join(self.path, self._stem(bs, "slotloop"))
            s = runner.run_slot_loop(bs, slots, folder, interval_s)
            self.logg.info(f"Slot loop saved: {folder} ({s['n_blocks']} blocks)")
        self._start_acquisition_task("Slot Loop", settings, job)

    @pyqtSlot(dict, float, float, int, float)
    def run_power_loop(self, settings: dict, start: float, step: float, n: int, interval_s: float):
        try:
            powers = acq.power_steps(start, step, n)
        except Exception as e:
            self.logg.error(f"Invalid power steps: {e}")
            return
        max_p = self.ctrl_panel.QDoubleSpinBox_laserpower_473.maximum()
        if max(powers) > max_p:
            self.logg.error(f"Power loop exceeds {max_p} mW: {powers}")
            return

        def job(runner, bs):
            folder = os.path.join(self.path, self._stem(bs, "powerloop"))
            s = runner.run_power_loop(bs, powers, folder, interval_s)
            self.logg.info(f"Power loop saved: {folder} ({s['n_blocks']} blocks)")
        self._start_acquisition_task("Power Loop", settings, job)

    @pyqtSlot(dict)
    def run_slot_sequence(self, settings: dict):
        """One recording over several SLM slots; per slot: display, wait, fire the DO / AO rows (timed)."""
        if self._slm_busy():
            return
        if self._trigger_firing():
            self.logg.error("TriggerBox is busy firing; wait or press Stop / Reset")
            return
        if self._recording is not None:
            self.logg.error("Acquire is recording or still saving; release it / wait for 'Acquisition saved'")
            return
        missing = [n for n in ("camera", "slm", "laser", "trg") if getattr(self.devs, n, None) is None]
        if missing:
            self.logg.error(f"Slot sequence needs: {', '.join(missing)} (not connected)")
            return
        outputs = self.ctrl_panel.get_trigger_outputs()
        try:
            ss = slseq.SlotSequenceSettings(**{**settings, "slots": acq.parse_slots(settings["slots"])})
            ss.validate()
            seq, summary, dur_ms, _ = slseq.build_trigger(outputs)
        except Exception as e:
            self.logg.error(f"Invalid slot sequence settings: {e}")
            return
        max_p = self.ctrl_panel.QDoubleSpinBox_laserpower_473.maximum()
        if ss.laser_power_mw > max_p:
            self.logg.error(f"Laser power {ss.laser_power_mw} mW exceeds {max_p} mW")
            return
        unwritten = [s for s in ss.slots if s not in self.slot_patterns]
        if unwritten:
            self.logg.warning(f"Slot sequence: slot(s) {unwritten} not written in this session")
        if getattr(self.viewer, "live_worker", None) is not None:
            self.logg.info("Stopping live video for the slot sequence")
            self.stop_video()
            self.ctrl_panel.QPushButton_video.setChecked(False)
        self.set_camera_roi()
        readout = self.ctrl_panel.get_scmos_readout()
        runner = slseq.SlotSequenceRunner(self.devs, self.logg, slot_info=self._slot_info,
                                          context=self._acq_context(), progress=self.sig_progress.emit)
        self.runner = runner
        stem = os.path.join(self.path, f"{time.strftime('%Y%m%d_%H%M%S')}"
                                       f"{'_' + ss.name if ss.name else ''}_slotseq")
        self.logg.info(f"Slot sequence: slots {ss.slots}, {ss.slm_to_trigger_ms:g} ms SLM -> trigger, "
                       f"{dur_ms:g} ms TriggerBox sequence per slot ({seq.sysclk_hz} Hz clock)")

        def task():
            try:
                runner.prepare_camera(readout)
                meta = runner.run(ss, outputs, stem)
                st = meta.get("delay_stats_ms", {})
                self.logg.info(f"Slot sequence saved: {stem}.tif ({meta['camera'].get('n_frames')} frames); "
                               f"SLM -> RUN {st.get('mean', float('nan')):.3f} ms "
                               f"(min {st.get('min', float('nan')):.3f}, max {st.get('max', float('nan')):.3f}); "
                               f"warnings: {meta['warnings'] or 'none'}")
            except acq.AcquisitionAborted:
                self.logg.info("Slot sequence aborted; partial data and metadata saved")
            except Exception as e:
                self.logg.error(f"Slot sequence failed: {e}")

        self._after_task = lambda: setattr(self, "runner", None)
        self.vw.get_dialog("Slot Sequence", interrupt=True)
        self.vw.dialog.dialog_closed.connect(runner.abort.set)
        self.run_task(task=task)

    CORRECTION_SLOT = 0  # holds the correction pattern; never overwritten by holograms
    FIRST_HOLOGRAM_SLOT = 1

    def _slm_busy(self):
        if getattr(self, "task_worker", None) is not None and self.task_worker.isRunning():
            self.logg.error("SLM is busy (hologram computation/loading running)")
            return True
        return False

    @pyqtSlot()
    def load_slm_correction(self):
        """Load the correction pattern into SLM slot 0 and display it."""
        if self._slm_busy():
            return
        file_name = self.vw.get_file_name()
        if file_name is None:
            return
        try:
            self.cgh.load_correction_pattern(file_name)
            self.devs.slm.load_pattern(file_name, self.CORRECTION_SLOT)
        except Exception as e:
            self.logg.error(f"Error loading SLM correction: {e}")
            return
        self.slot_patterns[self.CORRECTION_SLOT] = ("correction", self.devs.slm.pattern)
        self.ctrl_panel.display_loaded_pattern(file_name)
        self.hg_panel.set_pattern_image(self.devs.slm.pattern)
        hs = self.cgh.holo_sys
        if hs.lens_enabled or hs.cross_enabled:  # keep slot 0 = new correction + active lens / cross
            self.apply_slm_base(hs.lens_enabled, hs.cross_enabled)

    @pyqtSlot()
    def load_slm_pattern(self):
        """Load a pattern file into the slot chosen in the SLM panel (slot 0 is reserved)."""
        if self._slm_busy():
            return
        file_name = self.vw.get_file_name()
        if file_name is None:
            return
        _, slot = self.ctrl_panel.get_slm_parameters()
        if slot == self.CORRECTION_SLOT:
            self.logg.error("Slot 0 is reserved for the correction pattern; use 'Load Correction'")
            return
        try:
            self.devs.slm.load_pattern(file_name, slot)
        except Exception as e:
            self.logg.error(f"Error loading SLM pattern: {e}")
            return
        self.slot_patterns[slot] = (file_name, self.devs.slm.pattern)
        self.ctrl_panel.display_loaded_pattern(file_name)
        self.hg_panel.set_pattern_image(self.devs.slm.pattern)

    @pyqtSlot(bool, bool)
    def apply_slm_base(self, lens: bool, cross: bool):
        """
        Rewrite slot 0 = correction + Fresnel lens (if lens) + 2x2 0/pi cross (if cross).
        Holograms computed afterwards include the lens when it is on (never the cross).
        """
        hs = self.cgh.holo_sys
        applied = (hs.lens_enabled, hs.cross_enabled)
        if self._slm_busy() or self.devs.slm is None:
            if self.devs.slm is None:
                self.logg.error("No SLM connected")
            self.ctrl_panel.set_slm_base_buttons(*applied)
            return
        f_mm = self.ctrl_panel.get_slm_focal_mm()
        if lens and f_mm <= 0:
            self.logg.error("Set a Fresnel focal length > 0 to add the lens")
            lens = False
        (ox, oy), _ = self.ctrl_panel.get_slm_parameters()
        hs.slm_offset_x, hs.slm_offset_y = ox, oy
        try:
            pattern, ap = self.cgh.compute_base_pattern(f_mm * 1e-3 if lens else None, cross)
            self.devs.slm.write_array(pattern, self.CORRECTION_SLOT)
            self.devs.slm.display_pattern(self.CORRECTION_SLOT)
        except Exception as e:
            self.logg.error(f"Error writing slot 0: {e}")
            self.ctrl_panel.set_slm_base_buttons(*applied)
            return
        hs.lens_enabled, hs.cross_enabled = lens, cross
        hs.f_fresnel = max(f_mm, 0.0) * 1e-3  # same focal length as the holograms
        self.ctrl_panel.set_slm_base_buttons(lens, cross)
        parts = ["correction"]
        if lens:
            parts.append(f"Fresnel lens f={f_mm:g} mm, D={ap['diameter_px']:.0f} px ({ap['diameter_mm']:.2f} mm) "
                         f"at ({ap['center_x']:.1f}, {ap['center_y']:.1f})")
        if cross:
            parts.append("cross (2x2 0/pi)")
        label = " + ".join(parts)
        self.ctrl_panel.display_lens_aperture(ap["diameter_px"] if ap else 0, ap["diameter_mm"] if ap else 0)
        self.slot_patterns[self.CORRECTION_SLOT] = (label, pattern)
        self.hg_panel.set_pattern_image(pattern)
        self.ctrl_panel.QSpinBox_SLM_Slot.setValue(self.CORRECTION_SLOT)
        self.logg.info(f"SLM slot 0: {label}, offset ({ox}, {oy})")
        # holograms already in slots 1..N were built with the lens state at their computation
        if any(k >= self.FIRST_HOLOGRAM_SLOT for k in self.slot_patterns):
            built = self.cgh.holograms_lens
            now = {"lens_in_holograms": self.cgh.hologram_lens_f() is not None,
                   "f_fresnel_m": self.cgh.hologram_lens_f()}
            if self.cgh.holograms and built != now:
                self.logg.warning(f"Holograms in slots >= 1 were computed with {built}; the focal length is now "
                                  f"{now}. Recompute holograms (and recalibrate) so they match.")

    @pyqtSlot(int)
    def display_slm_slot(self, slot: int):
        if self._slm_busy():
            return
        if self.devs.slm is None:
            self.logg.error("No SLM connected")
            return
        try:
            self.devs.slm.display_pattern(slot)
        except Exception as e:
            self.logg.error(f"Error displaying SLM slot {slot}: {e}")
            return
        entry = self.slot_patterns.get(slot)
        if entry is not None:
            label, pattern = entry
            self.hg_panel.set_pattern_image(pattern)
            self.logg.info(f"SLM displaying slot {slot}: {label}")
        else:
            self.logg.info(f"SLM displaying slot {slot} (not written in this session)")

    @pyqtSlot(list, bool, float)
    def set_laser(self, laser: list, sw: bool, pw: float):
        if sw:
            try:
                self.devs.laser.set_constant_power(laser, [pw])
            except Exception as e:
                self.logg.error(f"Cobolt Laser Error: {e}")
        else:
            try:
                self.devs.laser.set_modulation_mode(laser, [pw])
            except Exception as e:
                self.logg.error(f"Cobolt Laser Error: {e}")

    def prepare_video(self):
        self.set_camera_roi()
        port, speed, gain = self.ctrl_panel.get_scmos_readout()
        self.devs.camera.prepare_live(port, speed, gain)

    @pyqtSlot(bool)
    def video(self, sw: bool):
        if sw:
            try:
                self.prepare_video()
            except Exception as e:
                self.logg.error(f"Error preparing imaging video: {e}")
                return
            self.start_video()
        else:
            self.stop_video()

    def start_video(self):
        try:
            self.devs.camera.start_live()
            self.viewer.live_worker = run_threads.ImgLiveWorker(self.devs.camera.data, fps=10, parent=self.viewer)
            self.viewer.live_worker.img_ready.connect(self.viewer.on_camera_update_from_thread)
            self.viewer.live_worker.start()
            self.logg.info("Live Video Started")
        except Exception as e:
            self.logg.error(f"Error starting imaging video: {e}")
            self.stop_video()
            return

    def stop_video(self):
        try:
            if getattr(self.viewer, "live_worker", None) is not None:
                self.viewer.live_worker.stop()
                self.viewer.live_worker = None
            self.devs.camera.stop_live()
            self.logg.info(r"Live Video Stopped")
        except Exception as e:
            self.logg.error(f"Error stopping imaging video: {e}")

    def run_task(self, task, iteration=1, parent=None):
        if getattr(self, "task_worker", None) is not None and self.task_worker.isRunning():
            return
        self.task_worker = run_threads.TaskWorker(task=task, n=iteration, parent=parent)
        self.task_worker.finished.connect(self.task_finish)
        self.task_worker.start()

    @pyqtSlot()
    def task_finish(self):
        w = self.task_worker
        self.task_worker = None
        w.deleteLater()
        self.vw.dialog.close()
        callback, self._after_task = self._after_task, None
        if callback is not None:
            callback()

    @pyqtSlot(bool, int)
    def acquisition(self, sw: bool, n_frames: int):
        """Acquire button: record continuously until the button is released (+ one JSON).
        n_frames > 0: a new TIFF every n frames (<name>_00000.tif, _00001.tif, ...);
        0: everything in one TIFF."""
        if sw:
            self.start_acquisition(n_frames)
        else:
            self.stop_acquisition()

    def _update_fps_display(self):
        fps = getattr(self.devs.camera, "measured_fps", None) if self.devs.camera is not None else None
        self.ctrl_panel.display_frame_rate(float(fps) if fps else 0.0)

    def _release_acquire_button(self):
        btn = self.ctrl_panel.QPushButton_acquire
        btn.blockSignals(True)
        btn.setChecked(False)
        btn.blockSignals(False)

    def start_acquisition(self, n_frames: int):
        cam = self.devs.camera
        if cam is None:
            self.logg.error("No camera connected")
            self._release_acquire_button()
            return
        if self._recording is not None:
            if self._recording.get("saving"):
                self.logg.error("The previous acquisition is still being written to disk; wait for "
                                "'Acquisition saved' in the log")
                self._release_acquire_button()
            return
        if (getattr(self, "task_worker", None) is not None and self.task_worker.isRunning()) or self._trigger_firing():
            self.logg.error("Another task is using the camera / TriggerBox; acquisition refused")
            self._release_acquire_button()
            return
        fn = self.vw.get_file_dialog()
        stamp = time.strftime("%Y%m%d%H%M%S")
        name = f"{stamp}_{fn}" if fn else stamp
        name = os.path.splitext(name)[0]
        if getattr(self.viewer, "live_worker", None) is not None:
            self.logg.info("Stopping live video for the acquisition")
            self.stop_video()
            self.ctrl_panel.QPushButton_video.setChecked(False)
        port, speed, gain = self.ctrl_panel.get_scmos_readout()
        try:
            self.set_camera_roi()
            cam.prepare_recording("internal", port, speed, gain)
            info = cam.get_recording_info() or {}
            frame_bytes = 2 * int(info.get("pixels_x") or 2400) * int(info.get("pixels_y") or 2400)
            # one open-ended file: BigTIFF so it can exceed 4 GB; split files: only if one file can
            bigtiff = (not n_frames) or n_frames * frame_bytes > acq.BIGTIFF_THRESHOLD_BYTES
            path = os.path.join(self.path, name + ".tif")
            rate_text, rate_warn = acq.data_rate_message(info)
            self.logg.info(f"Acquisition: {rate_text}")
            if rate_warn:
                self.logg.warning(rate_warn)
            t0 = time.perf_counter()
            cam.start_recording(path, n_frames=None, bigtiff=bigtiff, frames_per_file=n_frames or None)
            if getattr(cam, "data", None) is not None:
                cam.data.on_update(self.viewer.on_camera_update_from_thread)
        except Exception as e:
            self.logg.error(f"Error starting acquisition: {e}")
            self._release_acquire_button()
            return
        self._recording = {"path": path, "n_frames": n_frames, "t0": t0, "info": info, "bigtiff": bigtiff,
                           "rate_warning": rate_warn,
                           "readout": {"port": port, "speed": speed, "gain": gain},
                           "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.logg.info(f"Acquisition started: {name}{'_NNNNN' if n_frames else ''}.tif "
                       f"({f'a new file every {n_frames} frames, ' if n_frames else ''}until Acquire is released)")

    def stop_acquisition(self):
        """Stop the camera and save in the background: at kHz rates gigabytes may still be waiting
        in RAM for the disk, and the GUI must not freeze meanwhile. self._recording stays set
        (refusing new acquisitions / tasks) until the file is complete."""
        self._release_acquire_button()
        rec = self._recording
        if rec is None or rec.get("saving"):
            return
        rec["saving"] = True
        rec["context"] = self._acq_context()  # read the GUI here, not in the saving thread
        target = os.path.splitext(rec["path"])[0] + ("_NNNNN.tif" if rec["n_frames"] else ".tif")
        self.logg.info(f"Acquisition stopping; writing the remaining frames to {target} ...")
        threading.Thread(target=self._finish_acquisition, args=(rec,), daemon=True).start()

    def _finish_acquisition(self, rec):
        try:
            self._save_acquisition(rec)
        except Exception as e:
            self.logg.error(f"Error saving acquisition: {e}")
        finally:
            self._recording = None

    def _save_acquisition(self, rec):
        try:
            summary = self.devs.camera.stop_recording() or {}
        except Exception as e:
            self.logg.error(f"Error stopping acquisition: {e}")
            summary = {"error": str(e)}
        ts = summary.pop("timestamps_perf_s", []) or []
        nrs = summary.pop("frame_nrs", []) or []
        meta = {"schema": "minimiao.acquisition/1", "created": rec["created"],
                "software": {"package": "minimiao", "git_commit": acq._git_commit()},
                "files": {"stacks": summary.pop("files", None) or [{"file": os.path.basename(rec["path"])}],
                          "frames_per_file": rec["n_frames"] or None, "bigtiff": rec["bigtiff"]},
                "camera": dict(rec["info"], readout=rec["readout"],
                               **{k: v for k, v in summary.items() if k != "path"}),
                "frames": {"frame_nrs": nrs, "retrieved_s_from_start": [x - rec["t0"] for x in ts],
                           "note": "host time each frame was read from the SDK buffer, relative to start"},
                "context": rec["context"], "warnings": []}
        if summary.get("n_dropped"):
            meta["warnings"].append(f"{summary['n_dropped']} dropped frame(s)")
        if summary.get("error"):
            meta["warnings"].append(f"recorder: {summary['error']}")
        meta["warnings"].extend(acq.recorder_warnings(summary))
        if rec.get("rate_warning"):
            meta["warnings"].append(rec["rate_warning"])
        for w in meta["warnings"]:
            self.logg.warning(f"Acquisition: {w}")
        json_path = os.path.splitext(rec["path"])[0] + ".json"
        try:
            acq.write_json(json_path, meta)
        except Exception as e:
            self.logg.error(f"Error writing acquisition metadata: {e}")
        n_files = len(meta["files"]["stacks"])
        self.logg.info(f"Acquisition saved: {json_path} ({summary.get('n_frames')} frames"
                       f"{f' in {n_files} files' if n_files > 1 else ''}"
                       f"{', measured %.1f fps' % summary['measured_fps'] if summary.get('measured_fps') else ''}"
                       f", {summary.get('n_dropped', 0)} dropped)")

    @pyqtSlot()
    def load_cgh_target(self):
        file_name = self.vw.get_file_name()
        self.viewer.clear_calibration_points()
        if file_name is not None:
            self.cgh.load_mask(file_name)
            if self.cgh.cell_mask is not None:
                self.viewer.set_graph_image(self.cgh.cell_mask)
        else:
            frame = self.devs.camera.get_last_image() if self.devs.camera is not None else None
            if frame is None:
                self.logg.error("No camera frame to use as target: start Video, or choose an image file")
                return
            self.cgh.cell_mask = frame
            self.viewer.set_graph_image(self.cgh.cell_mask)

    @pyqtSlot()
    def pick_cgh_spots(self):
        self.viewer.start_target_picking()

    def update_cgh_parameters(self):
        n = self.hg_panel.get_cgh_iterations()
        o, idx = self.ctrl_panel.get_slm_parameters()
        f_mm = self.ctrl_panel.get_slm_focal_mm()
        # holograms always include the Fresnel lens with this focal length (0 = no lens)
        self.cgh.update_parameters(n, max(f_mm, 0.0) * 1e-3, o)
        if f_mm <= 0:
            self.logg.warning("Fresnel focal length is 0: holograms carry no lens and will not focus "
                              "at the sample through the relay; set the focal length in the SLM panel")
        if self.cgh.affine is not None:
            for msg in self._calibration_mismatches():
                self.logg.warning(f"Calibration mismatch: {msg}; spot positions will be wrong "
                                  f"(restore the setting or run Auto Calibrate again)")
        rows = self.viewer.get_target_spots()  # picked spots, index = hologram "spot i"
        pts = [tuple(float(v) for v in r) for r in rows
               if r is not None and all(isinstance(v, (int, float)) for v in r)]
        self.cgh.load_spots_picked(pts)

    def cgh_computation(self, mode):
        """Worker thread: compute holograms, then write them to SLM slots 1..N."""
        try:
            holograms = self.cgh.compute_holograms(mode)
        except Exception as e:
            self.logg.error(f"Error Running CGH Computation: {e}")
            return
        self.logg.info(f"CGH ({mode}): {len(holograms)} hologram(s) computed")
        if self.devs.slm is None:
            self.logg.error("No SLM connected; holograms computed but not loaded")
            return
        for slot, h in enumerate(holograms, start=self.FIRST_HOLOGRAM_SLOT):
            try:
                self.devs.slm.write_array(h["pattern"], slot)
            except Exception as e:
                self.logg.error(f"Error loading hologram '{h['label']}' to SLM slot {slot}: {e}")
                return
            self.slot_patterns[slot] = (h["label"], h["pattern"])
            coords = ", ".join(f"({self.cgh.spots_xy[i][0]:.0f}, {self.cgh.spots_xy[i][1]:.0f})" for i in h["spots"])
            self.logg.info(f"SLM slot {slot}: {h['label']} at cam px {coords}")

    def _show_first_hologram(self):
        entry = self.slot_patterns.get(self.FIRST_HOLOGRAM_SLOT)
        if self.cgh.holograms and entry is not None:
            self.hg_panel.set_pattern_image(entry[1])
            self.ctrl_panel.QSpinBox_SLM_Slot.setValue(self.FIRST_HOLOGRAM_SLOT)

    def _clear_hologram_slots(self):
        """Forget hologram slots >= 1 so stale patterns are not mistaken for the current set."""
        for slot in [k for k in self.slot_patterns if k >= self.FIRST_HOLOGRAM_SLOT]:
            del self.slot_patterns[slot]
        self.cgh.holograms = []
        self.calib_holo_pts = None

    # ---- camera -> SLM affine calibration ------------------------------------

    @pyqtSlot(int, float)
    def generate_calibration_holograms(self, n: int, span: float):
        """Write one single-spot hologram per grid point to SLM slots 1..n*n."""
        if self._slm_busy():
            return
        if self.devs.slm is None:
            self.logg.error("No SLM connected")
            return
        try:
            self.update_cgh_parameters()
        except Exception as e:
            self.logg.error(f"Error reading CGH parameters: {e}")
            return
        grid = slm_cal.make_grid(self.cgh.holo_sys.phase_nx, self.cgh.holo_sys.phase_ny, n, span)
        self._clear_hologram_slots()

        def task():
            try:
                self._write_point_holograms(grid)
            except Exception as e:
                self.logg.error(f"Error loading calibration holograms: {e}")
                self.calib_holo_pts = None
                return
            self.logg.info(f"Calibration: {len(grid)} holograms in slots "
                           f"{self.FIRST_HOLOGRAM_SLOT}..{self.FIRST_HOLOGRAM_SLOT + len(grid) - 1}")

        self._after_task = lambda: self.hg_panel.set_calibration_status(
            f"Calibration holograms: {len(grid)} in slots 1..{len(grid)}" if self.calib_holo_pts is not None
            else "Calibration holograms: failed (see log)")
        self.vw.get_dialog(r"Calibration Holograms")
        self.run_task(task=task)

    def _grab_frame(self, n_avg: int, exposure_s: float, not_before: float = None, timeout_s: float = 3.0,
                    wait_hint: float = 0.0):
        """
        Average n_avg distinct camera frames that were read off the camera at or after `not_before`
        (perf_counter time; e.g. SLM switch + settle + exposure). Preview frames reach the GUI
        process late (throttled to ~20/s plus transfer), so neither "newest frame" nor "arrived
        after the wait" guarantees a frame taken after the SLM switched; the read time does.
        """
        data = getattr(self.devs.camera, "data", None)
        if data is not None and hasattr(data, "get_last_with_count"):
            _, last, _ = data.get_last_with_count()
            frames = []
            # deadline on this process's clock (not_before may be on the camera process's clock)
            deadline = time.perf_counter() + wait_hint + timeout_s + n_avg * max(exposure_s, 0.05)
            while len(frames) < n_avg:
                if self._calib_abort.is_set():
                    raise RuntimeError("calibration aborted")
                f, c, t_read = data.get_last_with_count()
                if f is not None and c > last and (not_before is None or t_read is None or t_read >= not_before):
                    frames.append(np.asarray(f, dtype=np.float64))
                    last = c
                elif f is not None and c > last:
                    last = c  # new frame, but taken before the SLM settled: skip it
                elif time.monotonic() > deadline:
                    break
                else:
                    time.sleep(0.002)
            if not frames:
                raise RuntimeError("No new camera frames; is live video running?")
            return np.mean(frames, axis=0)
        # camera without a frame counter: wait, then newest frame, spaced by the exposure
        if not_before is not None:
            time.sleep(max(0.0, not_before - time.perf_counter()))
        frames = []
        for _ in range(n_avg):
            f = self.devs.camera.get_last_image()
            if f is not None:
                frames.append(np.asarray(f, dtype=np.float64))
            time.sleep(exposure_s)
        if not frames:
            raise RuntimeError("No camera frames; is live video running?")
        return np.mean(frames, axis=0)

    def _show_calib_outcome(self, title, text):
        """Non-blocking message box with a calibration's outcome (also logged)."""
        from PyQt6.QtWidgets import QMessageBox
        self.logg.info(f"{title}: {text}")
        box = QMessageBox(self.vw)
        box.setWindowTitle(title)
        box.setText(text)
        box.setIcon(QMessageBox.Icon.Information)
        box.setModal(False)
        box.show()
        self._calib_msg = box  # keep a reference while it is open

    def _calib_dialog(self, title):
        """Calibration progress dialog; Esc aborts (checked between SLM slots / frames)."""
        self._calib_abort.clear()
        self.vw.get_dialog(title, interrupt=True)
        self.vw.dialog.dialog_closed.connect(self._calib_abort.set)

    def _measure_slots(self, n_slots, wait_s, n_avg, exposure_s, slots=None):
        """Worker thread: display slots 1..n_slots (or the given slot list) in turn and record one
        averaged frame each."""
        frames = []
        slots = list(slots) if slots is not None else [self.FIRST_HOLOGRAM_SLOT + k for k in range(n_slots)]
        try:
            for slot in slots:
                if self._calib_abort.is_set():
                    raise RuntimeError("calibration aborted")
                self.devs.slm.display_pattern(slot)
                # only frames taken after settle + exposure (wait_s) from now show this slot
                # read the camera process's own clock right after the switch, so "frame taken after
                # settle" compares times on one clock (no cross-process clock assumption)
                cam = self.devs.camera
                t_ref = cam.clock() if hasattr(cam, "clock") else time.perf_counter()
                frames.append(self._grab_frame(n_avg, exposure_s, not_before=t_ref + wait_s, wait_hint=wait_s))
        finally:
            self.devs.slm.display_pattern(self.CORRECTION_SLOT)
        return np.stack(frames)

    def _write_point_holograms(self, holo_pts, tag="calib"):
        """Worker thread: write single-spot holograms for hologram-plane points to slots 1..K.
        Points are rounded to whole hologram pixels, which is where the hologram actually
        puts the spot (build_target_from_spots rounds), so the fit uses the true positions."""
        holo_pts = np.round(np.asarray(holo_pts, float))
        self._clear_hologram_slots()
        for k, (x1, y1) in enumerate(holo_pts):
            slot = self.FIRST_HOLOGRAM_SLOT + k
            pattern = self.cgh.compute_hologram_point(x1, y1)
            if pattern is None:
                raise ValueError(f"Calibration point {k} ({x1:.1f}, {y1:.1f}) is outside the field")
            self.devs.slm.write_array(pattern, slot)
            self.slot_patterns[slot] = (f"{tag} {k} ({x1:.1f}, {y1:.1f})", pattern)
        self.calib_holo_pts = np.asarray(holo_pts, float)

    def _project_and_detect(self, holo_pts, wait_s, n_avg, exposure_s, tag, min_frac=0.75):
        """Worker thread: write, measure, detect. Retries once with 2x averaging if too few spots are found."""
        self._write_point_holograms(holo_pts, tag)
        holo_pts = self.calib_holo_pts  # rounded positions actually projected
        for avg in (n_avg, 2 * n_avg):
            frames = self._measure_slots(len(holo_pts), wait_s, avg, exposure_s)
            cam_pts, snr, max_proj = slm_cal.detect_spots(frames)
            found = np.isfinite(cam_pts[:, 0]).mean()
            self.logg.info(f"Auto calibration [{tag}]: {found:.0%} of {len(holo_pts)} spots found "
                           f"(avg {avg} frames, min SNR {np.min(snr):.1f})")
            if found >= min_frac:
                break
        return holo_pts, cam_pts, snr, max_proj

    @pyqtSlot(dict)
    def auto_calibration(self, settings: dict):
        """
        Full automatic calibration:
          1. coarse pass: hologram-space grid -> affine M1
          2. refine pass: grid covering the camera FOV (via M1), avoiding the zero order;
             M1's prediction error on these spots is reported as validation
          3. final affine fitted on both passes; saved and applied (all picked spots
             are mapped through it).
        Starts live video and switches the laser to constant power if needed,
        and restores the laser mode afterwards.
        """
        prep = self._calib_prepare()
        if prep is None:
            return
        power, laser_was_on = prep

        exposure_s = max(float(getattr(self.devs.camera, "t_exposure", 20) or 20), 1.0) * 1e-3
        wait_s = settings["settle_ms"] * 1e-3 + 2 * exposure_s
        n_avg = settings["n_avg"]
        roi = list(self.ctrl_panel.get_scmos_roi())
        phase_n = (self.cgh.holo_sys.phase_nx, self.cgh.holo_sys.phase_ny)
        self._calib_result = None

        def task():
            try:
                t0 = time.monotonic()
                while self.devs.camera.get_last_image() is None:
                    if time.monotonic() - t0 > 10:
                        raise RuntimeError("No camera frames after 10 s")
                    time.sleep(0.1)
                # 1. coarse
                holo1 = slm_cal.make_grid(*phase_n, settings["grid"], settings["span"])
                holo1, cam1, _, mp1 = self._project_and_detect(holo1, wait_s, n_avg, exposure_s, "coarse")
                try:
                    n1 = slm_cal.check_point_set(cam1)
                except ValueError as e:
                    raise RuntimeError(
                        f"coarse pass: {e}. Many grid spots may fall outside the camera image: reduce Span "
                        f"(now {settings['span']:g}) or enlarge the ROI; otherwise check focus on the layer, "
                        f"Fresnel focal length, laser power and Avg frames") from None
                detected = {"coarse": [n1, len(holo1)]}
                # robust: a few wrong detections must not drag the fit (they did: rms 3605 px)
                M1, res1, inl1 = slm_cal.fit_affine_robust(cam1, holo1)
                try:
                    slm_cal.check_point_set(np.where(inl1[:, None], cam1, np.nan))
                except ValueError as e:
                    raise RuntimeError(f"coarse pass: detections are not consistent with one geometry ({e}); "
                                       f"check focus on the layer, stray light and laser power") from None
                self.logg.info(f"Auto calibration coarse: {int(inl1.sum())} consistent of {n1} detected, rms "
                               f"{np.sqrt(np.mean(res1[inl1] ** 2)):.2f} px, {slm_cal.describe(M1)}")
                holo_all, cam_all, mp = [holo1], [cam1], mp1
                validation = None
                # 2. refine over the camera FOV
                if settings["refine"] >= 2:
                    # grid over the part of the hologram field the camera sees (the camera image can be
                    # much larger than the hologram field, so a camera-wide grid lands mostly outside it)
                    n_ref = settings["refine"]
                    holo2, _ = slm_cal.holo_grid_in_camera(M1, mp1.shape, n_ref, phase_n=phase_n)
                    if len(holo2) < max(6, n_ref * n_ref // 2):   # camera image smaller than the field
                        holo2, _ = slm_cal.camera_grid_to_holo(M1, mp1.shape, n_ref, phase_n=phase_n)
                    if len(holo2) >= 3:
                        holo2, cam2, _, mp2 = self._project_and_detect(holo2, wait_s, n_avg, exposure_s,
                                                                        "refine")
                        cam2_exp = slm_cal.holo_to_cam(M1, holo2)  # coarse-model prediction
                        ok = np.isfinite(cam2[:, 0])
                        detected["refine"] = [int(ok.sum()), len(holo2)]
                        err = np.linalg.norm(cam2[ok] - cam2_exp[ok], axis=1)
                        good = err < 5.0
                        validation = {"n": int(ok.sum()), "n_consistent": int(good.sum()),
                                      "rms_cam_px": float(np.sqrt(np.mean(err[good] ** 2))) if good.any() else None,
                                      "max_cam_px": float(err.max())} if ok.any() else None
                        self.logg.info(f"Auto calibration validation (coarse model on refine spots): {validation}")
                        holo_all.append(holo2)
                        cam_all.append(cam2)
                        mp = np.maximum(mp1, mp2)
                    else:
                        self.logg.warning("Refine grid: too few points inside the hologram field; skipped")
                # 3. final fit
                holo_all, cam_all = np.vstack(holo_all), np.vstack(cam_all)
                M, res, inl = slm_cal.fit_affine_robust(cam_all, holo_all)
                ok_q, _, why_q = slm_cal.calibration_quality(res, inl)
                stamp = time.strftime("%Y%m%d%H%M%S")
                base = os.path.join(self.path, f"{stamp}_slm_affine_calibration" + ("" if ok_q else "_rejected"))
                meta = {"camera_roi": roi, "settle_ms": settings["settle_ms"], "n_avg": n_avg,
                        "auto": True, "settings": settings, "validation": validation,
                        "phase_n": list(phase_n), "detected": detected,
                        "slm_offset": [self.cgh.holo_sys.slm_offset_x, self.cgh.holo_sys.slm_offset_y],
                        "f_fresnel": self.cgh.hologram_lens_f(),
                        "lens_in_holograms": self.cgh.hologram_lens_f() is not None,
                        "applied": ok_q, "rejected_because": why_q}
                data = slm_cal.save_calibration(base + ".json", M, cam_all, holo_all, res, meta, inliers=inl)
                tf.imwrite(base + "_maxproj.tif", mp.astype(np.float32))
                if not ok_q:
                    raise RuntimeError("calibration rejected: " + "; ".join(why_q)
                                       + f". Saved for inspection: {os.path.basename(base)}.json")
                self.cgh.set_affine(M)
                self.calib_meta = meta
                self.calib_file = base + ".json"
                self._calib_result = (data, cam_all, mp, base + ".json")
                self.logg.info(f"Auto calibration saved: {base}.json; rms {data['rms_cam_px']:.2f} px over "
                               f"{data['n_inliers']} consistent spots; outliers {data['outliers']}")
            except Exception as e:
                error[0] = str(e)
                if not self._calib_abort.is_set():
                    self.logg.error(f"Auto calibration failed: {e}")

        error = [None]

        def finish():
            if not laser_was_on:
                self.set_laser(["473"], False, power)
            self._show_calibration_result()
            if self._calib_result is not None:
                data, _, _, path = self._calib_result
                sm = data["summary"]
                val = data["meta"].get("validation") or {}
                det = data["meta"].get("detected", {})
                warn = ""
                if val.get("rms_cam_px") is not None and val["rms_cam_px"] > 3:
                    warn = ("\n\nWARNING: the coarse model predicts the refine spots poorly "
                            f"(rms {val['rms_cam_px']:.1f} px): detections may be wrong (stray light, wrong focus, "
                            "frames from the previous slot). Run Check Spots before using it.")
                self._show_calib_outcome(
                    "Auto Calibrate: calibration applied",
                    f"{os.path.basename(path)}\nfit error rms {data['rms_cam_px']:.2f} camera px"
                    + (f", independent check rms {val['rms_cam_px']:.2f} px" if val.get("rms_cam_px") is not None else "")
                    + "\nspots detected: " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in det.items())
                    + f"; used {data.get('n_inliers')} consistent"
                    + (f", ignored {len(data['outliers'])} inconsistent detection(s)" if data.get("outliers") else "")
                    + f"\nrotation {sm['rotation_deg']:.1f} deg, scale {sm['scale_x']:.4f} / {sm['scale_y']:.4f}"
                    + (", mirrored" if sm["mirrored"] else "") + f", shear {sm['shear']:.3f}" + warn)
            else:
                why = "aborted" if self._calib_abort.is_set() else f"failed: {error[0] or 'see log'}"
                self.hg_panel.set_calibration_status(f"Calibration: {why}; previous calibration kept"
                                                     if self.cgh.affine is not None else f"Calibration: {why}")
                self._show_calib_outcome("Auto Calibrate: calibration NOT changed", why)

        self._after_task = finish
        self._calib_dialog(r"Auto Calibration")
        self.run_task(task=task)

    def _calib_prepare(self):
        """Common set-up for automatic calibrations: checks, live video on, laser to constant power.
        Returns (power, laser_was_on) or None if a check failed."""
        if self._slm_busy():
            return None
        if self.devs.slm is None or self.devs.camera is None:
            self.logg.error("Calibration needs both the SLM and the camera")
            return None
        if self._recording is not None:
            self.logg.error("Acquire is recording or still saving; release it / wait for 'Acquisition saved'")
            return None
        power = self.ctrl_panel.QDoubleSpinBox_laserpower_473.value()
        laser_was_on = self.ctrl_panel.QPushButton_laser_473.isChecked()
        if power <= 0:
            self.logg.error("Set a laser power > 0 for calibration")
            return None
        try:
            self.update_cgh_parameters()
        except Exception as e:
            self.logg.error(f"Error reading CGH parameters: {e}")
            return None
        if getattr(self.devs.camera, "data", None) is None:
            self.video(True)
            self.ctrl_panel.QPushButton_video.setChecked(True)
        if not laser_was_on:
            self.set_laser(["473"], True, power)
        return power, laser_was_on

    @pyqtSlot(dict)
    def check_spots(self, settings: dict):
        """
        Verify targeting on the fluorescent layer with the current calibration and settings:
        computes Hybrid holograms for the picked spots (slot 1 = all spots, slots 2.. = one spot
        each), projects slot 0 (background) and each hologram, and measures where every spot lands.
        Reports per-spot errors for single spots and for the all-at-once hologram.
        """
        if self.cgh.affine is None:
            self._show_calib_outcome("Check Spots: no calibration", "Run Auto Calibrate (or Load Calib) first.")
            return
        if not self.viewer.get_target_spots():
            self._show_calib_outcome("Check Spots: no spots", "Pick spots on the target image first.")
            return
        prep = self._calib_prepare()
        if prep is None:
            return
        power, laser_was_on = prep
        picked = np.array([[x, y] for x, y, _, _ in self.cgh.spots_xy], dtype=float)
        exposure_s = max(float(getattr(self.devs.camera, "t_exposure", 20) or 20), 1.0) * 1e-3
        wait_s = settings["settle_ms"] * 1e-3 + 2 * exposure_s
        n_avg = settings["n_avg"]
        mism = self._calibration_mismatches()
        result = {}

        def task():
            try:
                holograms = self.cgh.compute_holograms("Hybrid")
                self._clear_hologram_slots_keep()
                for slot, h in enumerate(holograms, start=self.FIRST_HOLOGRAM_SLOT):
                    self.devs.slm.write_array(h["pattern"], slot)
                    self.slot_patterns[slot] = (h["label"], h["pattern"])
                slots = [self.CORRECTION_SLOT] + list(range(self.FIRST_HOLOGRAM_SLOT,
                                                           self.FIRST_HOLOGRAM_SLOT + len(holograms)))
                frames = self._measure_slots(0, wait_s, n_avg, exposure_s, slots=slots)
                # background: median of slot 0 and the single-spot frames. Each spot is in only one of
                # them, while anything common to all holograms (e.g. light from the trigger stripe,
                # weakened zero order) is in most of them and cancels; a slot-0 frame alone keeps it.
                single_frames = [fr for h, fr in zip(holograms, frames[1:]) if h["label"] != "all"]
                bg = (np.median(np.stack([frames[0]] + single_frames), axis=0) if len(single_frames) >= 2
                      else frames[0])
                singles = np.full_like(picked, np.nan)
                multi_pts = np.zeros((0, 2))
                for h, fr in zip(holograms, frames[1:]):
                    pts, _ = slm_cal.find_peaks(fr - bg, len(h["spots"]))
                    if h["label"] == "all":
                        multi_pts = pts
                    elif len(pts):
                        singles[h["spots"][0]] = pts[0]
                err_single = np.hypot(*(singles - picked).T)
                # how the landed single spots relate to the picked positions (diagnosis of a miss)
                ok = np.isfinite(singles[:, 0])
                relation = None
                if ok.sum() >= 3:
                    try:
                        Md, _ = slm_cal.fit_affine(picked[ok], singles[ok])
                        relation = dict(slm_cal.describe(Md), mean_shift_px=(singles[ok] - picked[ok]).mean(0).tolist())
                    except ValueError:
                        relation = None
                elif ok.any():
                    relation = {"mean_shift_px": (singles[ok] - picked[ok]).mean(0).tolist()}
                err_multi = np.array([np.min(np.hypot(*(multi_pts - q).T)) if len(multi_pts) else np.nan
                                      for q in picked])
                stamp = time.strftime("%Y%m%d%H%M%S")
                base = os.path.join(self.path, f"{stamp}_check_spots")
                record = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "calibration_file": self.calib_file,
                          "settings_mismatches": mism, "picked_cam": picked.tolist(),
                          "single_landed_cam": singles.tolist(), "single_error_px": err_single.tolist(),
                          "all_at_once_landed_cam": multi_pts.tolist(), "all_at_once_error_px": err_multi.tolist(),
                          "landed_vs_picked": relation,
                          "slm_offset": [self.cgh.holo_sys.slm_offset_x, self.cgh.holo_sys.slm_offset_y],
                          "f_fresnel_m": self.cgh.hologram_lens_f(), "camera_roi": list(self.ctrl_panel.get_scmos_roi())}
                acq.write_json(base + ".json", record)
                tf.imwrite(base + "_frames.tif", np.stack(frames).astype(np.float32))
                result.update(singles=singles, multi=multi_pts, es=err_single, em=err_multi, path=base + ".json",
                              relation=relation,
                              img=np.clip(np.max(frames[1:] - bg, axis=0), 0, None))
            except Exception as e:
                result["error"] = str(e)
                if not self._calib_abort.is_set():
                    self.logg.error(f"Check Spots failed: {e}")

        def finish():
            if not laser_was_on:
                self.set_laser(["473"], False, power)
            if "es" not in result:
                why = "aborted" if self._calib_abort.is_set() else f"failed: {result.get('error', 'see log')}"
                self._show_calib_outcome("Check Spots: " + why.split(":")[0], why)
                return
            es, em = result["es"], result["em"]
            self.viewer.set_graph_image(result["img"])
            pts = np.vstack([result["singles"][np.isfinite(result["singles"][:, 0])], result["multi"]])
            self.viewer.show_calibration_points(pts[:, 0], pts[:, 1])
            fmt = lambda e: ("not detected" if not np.isfinite(e).any() else
                             f"rms {np.sqrt(np.nanmean(e ** 2)):.1f} px, max {np.nanmax(e):.1f} px"
                             + (f", {int(np.isnan(e).sum())} not detected" if np.isnan(e).any() else ""))
            per = "\n".join(f"  spot {i}: single {a:.1f} px, all-at-once {b:.1f} px" for i, (a, b) in enumerate(zip(es, em)))
            single_ok = np.isfinite(es).all() and np.nanmax(es) <= 3
            multi_ok = np.isfinite(em).all() and np.nanmax(em) <= 3
            if single_ok and multi_ok:
                verdict = "Spots land where picked."
            elif single_ok:
                verdict = "Single spots land correctly but the all-at-once hologram does not: multi-spot hologram problem."
            else:
                verdict = ("Single spots miss: the calibration does not match this setup or this target image "
                           "(check camera ROI / binning, SLM offset, focal length, and that the target image is a "
                           "camera frame) - run Auto Calibrate again.")
            rel = result.get("relation")
            if rel and not single_ok:
                dx, dy = rel["mean_shift_px"]
                if "rotation_deg" in rel:
                    rot = rel["rotation_deg"]
                    rot_off = min(abs(rot), abs(abs(rot) - 360))
                    same_geom = rot_off < 2 and abs(rel["scale_x"] - 1) < 0.03 and abs(rel["scale_y"] - 1) < 0.03 \
                        and not rel["mirrored"]
                    if same_geom:
                        verdict += (f"\nLanded spots are the picked pattern shifted by ({dx:.0f}, {dy:.0f}) px: "
                                    f"something moved or changed since calibration (sample/focus, SLM offset, lens, "
                                    f"camera ROI).")
                    else:
                        verdict += (f"\nLanded spots = picked pattern rotated {rot:.1f} deg, scaled "
                                    f"{rel['scale_x']:.3f} / {rel['scale_y']:.3f}"
                                    + (", MIRRORED" if rel["mirrored"] else "")
                                    + f", shifted ({dx:.0f}, {dy:.0f}) px: the calibration geometry itself is wrong.")
                else:
                    verdict += f"\nMean shift of landed spots: ({dx:.0f}, {dy:.0f}) px."
            if mism:
                verdict += "\nSettings changed since calibration: " + "; ".join(mism)
            self.hg_panel.set_calibration_status(f"Check Spots: single {fmt(es)}; all-at-once {fmt(em)}")
            self._show_calib_outcome("Check Spots", f"{verdict}\n\nSingle spots: {fmt(es)}\nAll at once: {fmt(em)}\n"
                                                    f"{per}\n\nSaved: {os.path.basename(result['path'])}")

        self._after_task = finish
        self._calib_dialog(r"Check Spots")
        self.run_task(task=task)

    def _clear_hologram_slots_keep(self):
        """Forget hologram slot labels >= 1 without discarding self.cgh.holograms (just computed)."""
        for slot in [k for k in self.slot_patterns if k >= self.FIRST_HOLOGRAM_SLOT]:
            del self.slot_patterns[slot]

    @pyqtSlot(int, int)
    def measure_calibration(self, settle_ms: int, n_avg: int):
        """Step through the calibration slots, locate each spot on the camera, fit the affine."""
        if self._slm_busy():
            return
        holo_pts = self.calib_holo_pts
        if holo_pts is None:
            self.logg.error("Generate calibration holograms first")
            return
        if self.devs.camera is None or getattr(self.devs.camera, "data", None) is None:
            self.logg.error("Start live video (laser on) before measuring the calibration")
            return
        exposure_s = max(float(getattr(self.devs.camera, "t_exposure", 20) or 20), 1.0) * 1e-3
        wait_s = settle_ms * 1e-3 + 2 * exposure_s  # SLM settle + flush frames exposed during the switch
        roi = list(self.ctrl_panel.get_scmos_roi())
        self._calib_result = None

        def task():
            try:
                frames = self._measure_slots(len(holo_pts), wait_s, n_avg, exposure_s)
                cam_pts, snr, max_proj = slm_cal.detect_spots(frames)
                n_bad = int(np.sum(~np.isfinite(cam_pts[:, 0])))
                if n_bad:
                    self.logg.warning(f"Calibration: {n_bad} spot(s) not detected (SNR {np.round(snr, 1).tolist()})")
                M, res, inl = slm_cal.fit_affine_robust(cam_pts, holo_pts)
                ok_q, _, why_q = slm_cal.calibration_quality(res, inl)
                stamp = time.strftime("%Y%m%d%H%M%S")
                base = os.path.join(self.path, f"{stamp}_slm_affine_calibration" + ("" if ok_q else "_rejected"))
                meta = {"camera_roi": roi, "settle_ms": settle_ms, "n_avg": n_avg,
                        "applied": ok_q, "rejected_because": why_q,
                        "snr": snr.tolist(),
                        "phase_n": [self.cgh.holo_sys.phase_nx, self.cgh.holo_sys.phase_ny],
                        "slm_offset": [self.cgh.holo_sys.slm_offset_x, self.cgh.holo_sys.slm_offset_y],
                        "f_fresnel": self.cgh.hologram_lens_f(),
                        "lens_in_holograms": self.cgh.hologram_lens_f() is not None}
                data = slm_cal.save_calibration(base + ".json", M, cam_pts, holo_pts, res, meta, inliers=inl)
                tf.imwrite(base + "_maxproj.tif", max_proj.astype(np.float32))
                if not ok_q:
                    raise RuntimeError("calibration rejected: " + "; ".join(why_q)
                                       + f". Saved for inspection: {os.path.basename(base)}.json")
                self.cgh.set_affine(M)
                self.calib_meta = meta
                self.calib_file = base + ".json"
                self._calib_result = (data, cam_pts, max_proj, base + ".json")
                self.logg.info(f"Calibration saved: {base}.json; rms {data['rms_cam_px']:.2f} cam px over "
                               f"{data['n_inliers']} consistent spots; outliers {data['outliers']}; {data['summary']}")
            except Exception as e:
                self.logg.error(f"Calibration measurement failed: {e}")

        self._after_task = self._show_calibration_result
        self._calib_dialog(r"Calibration Measurement")
        self.run_task(task=task)

    def _show_calibration_result(self):
        if self._calib_result is None:
            self.hg_panel.set_calibration_status("Calibration: measurement failed (see log)")
            return
        data, cam_pts, max_proj, path = self._calib_result
        self.viewer.set_graph_image(max_proj)
        ok = np.all(np.isfinite(cam_pts), axis=1)
        self.viewer.show_calibration_points(cam_pts[ok, 0], cam_pts[ok, 1])
        self._set_calib_status(data, os.path.basename(path))

    def _set_calib_status(self, data, name):
        sm = data.get("summary", {})
        self.hg_panel.set_calibration_status(
            f"{name}: rms {data.get('rms_cam_px', float('nan')):.2f} px, "
            f"rot {sm.get('rotation_deg', 0):.1f} deg, scale {sm.get('scale_x', 0):.3f}/{sm.get('scale_y', 0):.3f}"
            f"{', mirrored' if sm.get('mirrored') else ''}")

    @pyqtSlot()
    def load_calibration(self):
        file_name = self.vw.get_file_name()
        if file_name is None:
            return
        try:
            M, data = slm_cal.load_calibration(file_name)
        except Exception as e:
            self.logg.error(f"Error loading calibration: {e}")
            return
        self._apply_calibration_file(file_name, M, data)

    def _apply_calibration_file(self, file_name, M, data, how="loaded"):
        self.cgh.set_affine(M)
        self.calib_meta = data.get("meta", {})
        self.calib_file = file_name
        self.logg.info(f"Calibration {how}: {file_name} (created {data.get('created', '?')})")
        self._set_calib_status(data, f"{os.path.basename(file_name)} ({how})")
        mism = self._calibration_mismatches()
        for msg in mism:
            self.logg.warning(f"Calibration mismatch: {msg}")
        if mism:
            self.hg_panel.set_calibration_status(self.hg_panel.QLabel_Calib_Status.text()
                                                 + " | MISMATCH: " + "; ".join(mism))

    def _calibration_mismatches(self):
        """Settings the calibration depends on that differ from the current ones (list of messages)."""
        meta, out = self.calib_meta or {}, []
        roi = list(self.ctrl_panel.get_scmos_roi())
        if meta.get("camera_roi") not in (None, roi):
            out.append(f"camera ROI/binning {roi} vs calibration {meta.get('camera_roi')}")
        (ox, oy), _ = self.ctrl_panel.get_slm_parameters()
        if meta.get("slm_offset") not in (None, [ox, oy]):
            out.append(f"SLM offset ({ox}, {oy}) vs calibration {tuple(meta.get('slm_offset'))}")
        f_now = self.ctrl_panel.get_slm_focal_mm() * 1e-3
        f_cal = meta.get("f_fresnel")
        if f_cal is not None and abs(float(f_cal) - f_now) > 1e-6:
            out.append(f"Fresnel focal length {f_now * 1e3:g} mm vs calibration {float(f_cal) * 1e3:g} mm")
        if meta and meta.get("lens_in_holograms") is False and f_now > 0:
            out.append("calibration was measured with holograms WITHOUT the Fresnel lens")
        return out

    def _load_latest_calibration(self):
        """At start-up: load the newest *_slm_affine_calibration.json from the data folders."""
        import glob
        root = os.path.dirname(os.path.abspath(self.path))
        files = glob.glob(os.path.join(root, "*", "*_slm_affine_calibration.json")) + \
            glob.glob(os.path.join(self.path, "*_slm_affine_calibration.json"))
        for f in sorted(set(files), key=os.path.getmtime, reverse=True):
            try:
                M, data = slm_cal.load_calibration(f)
            except Exception as e:
                self.logg.warning(f"Skipping unreadable calibration {f}: {e}")
                continue
            self._apply_calibration_file(f, M, data, how="auto-loaded, newest")
            return
        self.logg.info("No camera-SLM calibration found; run Auto Calibrate before computing holograms")

    def compute_cgh_pattern(self):
        if self._slm_busy():
            return
        if self.cgh.affine is None:
            self.logg.error("No camera-SLM calibration: run Auto Calibrate (or Load Calib) before computing "
                            "holograms for picked spots")
            self._show_calib_outcome("Compute CGH: no calibration",
                                     "Picked spots are mapped to the SLM through the camera-SLM calibration.\n"
                                     "Run Auto Calibrate (or Load Calib) first.")
            return
        try:
            self.update_cgh_parameters()
        except Exception as e:
            self.logg.error(f"Error reading CGH parameters: {e}")
            return
        mode = self.hg_panel.get_cgh_mode()
        self._clear_hologram_slots()
        self._after_task = self._show_first_hologram
        self.vw.get_dialog(r"CGH Computation")
        self.run_task(task=lambda: self.cgh_computation(mode))

    @pyqtSlot()
    def save_cgh_pattern(self):
        file_name = self.vw.get_file_dialog()
        if file_name is not None:
            self.cgh.save_cgh(os.path.join(self.path, file_name))
