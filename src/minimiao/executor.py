# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import os
import threading
import time

import numpy as np
import pandas as pd
import tifffile as tf
from PyQt6.QtCore import QObject, pyqtSlot, Qt, pyqtSignal, QTimer

from . import run_threads
from .computations import slm_calibration as slm_cal
from .experiments import acquisition as acq
from .devices import trigger_box as tbx


class CommandExecutor(QObject):
    svd = pyqtSignal(str)
    sig_progress = pyqtSignal(str)
    sig_trigger_status = pyqtSignal(str)
    sig_plt = pyqtSignal(list, list)
    sig_auto_focus = pyqtSignal(float)

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
        self._fire_stopped = threading.Event()
        self.trg = cmp.trg
        self.cgh = cmp.cgh
        self.path = path
        self.logg = logger or self.setup_logging()
        self._set_signal_executions()
        self._initial_setup()
        self.lasers = []
        self.task_worker = None
        self._after_task = None
        self.slot_patterns = {}  # SLM memory slot -> (label, uint8 pattern) written this session
        self.calib_holo_pts = None  # hologram-plane points currently in slots 1..K
        self.calib_meta = {}  # metadata of the active affine calibration
        self._calib_result = None

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    def _set_signal_executions(self):
        # SLM
        self.ctrl_panel.Signal_slm_correction.connect(self.load_slm_correction)
        self.ctrl_panel.Signal_slm_load.connect(self.load_slm_pattern)
        self.ctrl_panel.Signal_slm_lens.connect(self.apply_slm_lens)
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
        self.svd.connect(self.save_data)
        # Hologram
        self.hg_panel.Signal_load_target.connect(self.load_cgh_target)
        self.hg_panel.Signal_pick_spot.connect(self.pick_cgh_spots)
        self.hg_panel.Signal_compute_cgh.connect(self.compute_cgh_pattern)
        self.hg_panel.Signal_save_pattern.connect(self.save_cgh_pattern)
        self.hg_panel.Signal_display_slot.connect(self.display_slm_slot)
        self.hg_panel.Signal_calib_generate.connect(self.generate_calibration_holograms)
        self.hg_panel.Signal_calib_measure.connect(self.measure_calibration)
        self.hg_panel.Signal_calib_load.connect(self.load_calibration)
        self.hg_panel.Signal_calib_use.connect(self.use_calibration)
        self.hg_panel.Signal_calib_auto.connect(self.auto_calibration)
        # Acquisition blocks
        self.acq_panel.Signal_run_block.connect(self.run_acquisition_block)
        self.acq_panel.Signal_run_slot_loop.connect(self.run_slot_loop)
        self.acq_panel.Signal_run_power_loop.connect(self.run_power_loop)
        self.sig_progress.connect(self._show_progress)

    def _initial_setup(self):
        try:
            self.laser_lists = list(self.devs.laser.lasers.keys())
            self.logg.info("Finish setting up controllers")
        except Exception as e:
            self.logg.error(f"Initial setup Error: {e}")

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
        """Build one sequence from the DO#1..8 rows (start / pulse / interval / N) and fire it."""
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
        try:
            seq, summary = tbx.sequence_from_outputs(outputs)
        except Exception as e:
            self.logg.error(f"Trigger settings: {e}")
            self.ctrl_panel.display_trigger_status(f"Error: {e}")
            return
        desc = "; ".join(f"DO#{s['channel']}: {len(s['onsets_ms'])} x {s['width_ms']:g} ms"
                         + (f" every {s['onsets_ms'][1] - s['onsets_ms'][0]:g} ms" if len(s['onsets_ms']) > 1 else "")
                         + f" from {s['onsets_ms'][0]:g} ms" for s in summary)
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
            "cgh_at_last_hologram_computation": {"magnification": hs.magnification, "center": [hs.image_center_x, hs.image_center_y],
                    "f_reference_m": hs.f_fresnel, "holograms_include_lens": False, "slm_offset": [hs.slm_offset_x, hs.slm_offset_y],
                    "correction_value": hs.correction_value, "mode_last_computed": self.hg_panel.get_cgh_mode(),
                    "spots_xy_um_z": [list(map(float, sp)) for sp in (self.cgh.spots_xy or [])]},
            "affine_calibration": {"in_use": bool(self.cgh.use_affine and self.cgh.affine is not None),
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
        missing = [n for n in ("camera", "slm", "laser", "trg") if getattr(self.devs, n, None) is None]
        if missing:
            self.logg.error(f"Acquisition needs: {', '.join(missing)} (not connected)")
            return
        try:
            bs = acq.BlockSettings(**settings)
            acq.build_sequence(bs)  # validates timing / TriggerBox clock before touching hardware
            exposure = self.ctrl_panel.get_scmos_exposure()
            if bs.camera_trigger == "external" and round(exposure) >= 1000.0 / bs.frame_rate_hz:
                raise ValueError(f"exposure {round(exposure)} ms must be shorter than the frame period "
                                 f"{1000.0 / bs.frame_rate_hz:.2f} ms (exposure is whole ms)")
        except Exception as e:
            self.logg.error(f"Invalid acquisition settings: {e}")
            return
        if getattr(self.viewer, "live_worker", None) is not None:
            self.logg.info("Stopping live video for the acquisition")
            self.stop_video()
            self.ctrl_panel.QPushButton_video.setChecked(False)
        self.set_camera_roi()
        runner = acq.AcquisitionRunner(self.devs, self.logg, slot_info=self._slot_info,
                                       context=self._acq_context(), progress=self.sig_progress.emit,
                                       preview=self.viewer.on_camera_update_from_thread)
        self.runner = runner

        def task():
            try:
                runner.prepare_camera(bs.camera_trigger)
                job(runner, bs)
            except acq.AcquisitionAborted:
                self.logg.info(f"{title} aborted; partial data and metadata saved")
            except Exception as e:
                self.logg.error(f"{title} failed: {e}")

        def finish():
            self.runner = None
        self._after_task = finish
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

    @pyqtSlot()
    def apply_slm_lens(self):
        """Replace slot 0 with correction + Fresnel lens (f from the SLM panel; 0 = correction only)."""
        if self._slm_busy():
            return
        if self.devs.slm is None:
            self.logg.error("No SLM connected")
            return
        f_mm = self.ctrl_panel.get_slm_focal_mm()
        (ox, oy), _ = self.ctrl_panel.get_slm_parameters()
        self.cgh.holo_sys.slm_offset_x, self.cgh.holo_sys.slm_offset_y = ox, oy
        try:
            pattern, ap = self.cgh.compute_lens_pattern(f_mm * 1e-3)
            self.devs.slm.write_array(pattern, self.CORRECTION_SLOT)
            self.devs.slm.display_pattern(self.CORRECTION_SLOT)
        except Exception as e:
            self.logg.error(f"Error applying Fresnel lens to slot 0: {e}")
            return
        label = (f"correction + Fresnel lens f={f_mm:g} mm, D={ap['diameter_px']:.0f} px "
                 f"({ap['diameter_mm']:.2f} mm) at ({ap['center_x']:.1f}, {ap['center_y']:.1f})"
                 if f_mm > 0 else "correction (no lens)")
        self.ctrl_panel.display_lens_aperture(ap["diameter_px"], ap["diameter_mm"])
        self.slot_patterns[self.CORRECTION_SLOT] = (label, pattern)
        self.hg_panel.set_pattern_image(pattern)
        self.hg_panel.QSpinBox_SLM_Display_Slot.setValue(self.CORRECTION_SLOT)
        self.logg.info(f"SLM slot 0: {label}, offset ({ox}, {oy})")

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

    # def reset_stage_positions(self):
    #     pos_x, pos_y, pos_z = self.ctrl_panel.get_stage_positions()
    #     self.set_stage_position_x(pos_x[0], port="software")
    #     self.set_stage_position_y(pos_y[0], port="software")
    #     self.set_stage_position_z(pos_z[0], port="software")
    #     self.set_stage_position_x(pos_x[1], port="analog")
    #     self.set_stage_position_y(pos_y[1], port="analog")
    #     self.set_stage_position_z(pos_z[1], port="analog")
    #     self.ctrl_panel.display_stage_position_x(self.devs.stage.read_position(0))
    #     self.ctrl_panel.display_stage_position_y(self.devs.stage.read_position(1))
    #     self.ctrl_panel.display_stage_position_z(self.devs.stage.read_position(2))
    #
    # @pyqtSlot(str, float, float, float)
    # def set_stage_positions_usb(self, axis: str, value_x: float, value_y: float, value_z: float):
    #     if axis == "x":
    #         self.set_stage_position_x(value_x, port="software")
    #     if axis == "y":
    #         self.set_stage_position_y(value_y, port="software")
    #     if axis == "z":
    #         self.set_stage_position_z(value_z, port="software")
    #
    # @pyqtSlot(str, float, float, float)
    # def set_stage_positions(self, axis: str, value_x: float, value_y: float, value_z: float):
    #     if axis == "x":
    #         self.set_stage_position_x(value_x, port="analog")
    #     if axis == "y":
    #         self.set_stage_position_y(value_y, port="analog")
    #     if axis == "z":
    #         self.set_stage_position_z(value_z, port="analog")
    #
    # def set_stage_position_x(self, pos_x, port="analog"):
    #     try:
    #         if port == "software":
    #             self.devs.stage.move_position(0, pos_x)
    #             QTimer.singleShot(100, lambda: self._update_stage_display_x())
    #         else:
    #             self.devs.trigger.set_stage_position([pos_x / 10.], [0])
    #             QTimer.singleShot(100, lambda: self._update_stage_display_x())
    #     except Exception as e:
    #         self.logg.error(f" stage Error: {e}")
    #
    # def _update_stage_display_x(self):
    #     try:
    #         position = self.devs.stage.read_position(0)
    #         self.ctrl_panel.display_stage_position_x(position)
    #     except Exception as e:
    #         self.logg.error(f" stage Read Error: {e}")
    #
    # def set_stage_position_y(self, pos_y, port="analog"):
    #     try:
    #         if port == "software":
    #             self.devs.stage.move_position(1, pos_y)
    #             QTimer.singleShot(100, lambda: self._update_stage_display_y())
    #         else:
    #             self.devs.trigger.set_stage_position([pos_y / 10.], [1])
    #             QTimer.singleShot(100, lambda: self._update_stage_display_y())
    #     except Exception as e:
    #         self.logg.error(f" stage Error: {e}")
    #
    # def _update_stage_display_y(self):
    #     try:
    #         position = self.devs.stage.read_position(1)
    #         self.ctrl_panel.display_stage_position_y(position)
    #     except Exception as e:
    #         self.logg.error(f" stage Read Error: {e}")
    #
    # def set_stage_position_z(self, pos_z, port="analog"):
    #     try:
    #         if port == "software":
    #             self.devs.stage.move_position(2, pos_z)
    #             QTimer.singleShot(100, lambda: self._update_stage_display_z())
    #         else:
    #             self.devs.trigger.set_stage_position([pos_z / 10.], [2])
    #             QTimer.singleShot(100, lambda: self._update_stage_display_z())
    #     except Exception as e:
    #         self.logg.error(f" stage Error: {e}")
    #
    # def _update_stage_display_z(self):
    #     try:
    #         position = self.devs.stage.read_position(2)
    #         self.ctrl_panel.display_stage_position_z(position)
    #     except Exception as e:
    #         self.logg.error(f" stage Read Error: {e}")
    #
    # def update_stage_scanning(self):
    #     axis_lengths, step_sizes = self.ctrl_panel.get_stage_scan_parameters()
    #     pos_x, pos_y, pos_z = self.ctrl_panel.get_stage_positions()
    #     positions = [pos_x[1], pos_y[1], pos_z[1]]
    #     return_time, line_time = self.ctrl_panel.get_stage_scan_time()
    #     self.trg.update_stage_scan_parameters(axis_lengths, step_sizes, positions, return_time, line_time)

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

    # @pyqtSlot(int)
    # def update_trigger_sample_rate(self, sr: int):
    #     self.trg.update_sampling_rate(sr * 1000)
    #     self.devs.trigger.sample_rate = sr * 1000
    #
    # @pyqtSlot()
    # def reset_trigger_channels(self):
    #     self.devs.trigger.stop_triggers()
    #
    # def update_digital_triggers(self):
    #     digital_starts, digital_ends = self.ctrl_panel.get_digital_parameters()
    #     self.trg.update_digital_parameters(digital_starts, digital_ends)
    #
    # def update_trigger_parameters(self):
    #     """Ensure that the camera acquisition is fully set up before executing this function."""
    #     try:
    #         self.update_digital_triggers()
    #         self.update_stage_scanning()
    #         self.logg.info(f"Trigger Updated")
    #     except Exception as e:
    #         self.logg.error(f"Trigger Error: {e}")

    def prepare_video(self, vd_mod):
        # self.update_trigger_parameters()
        self.set_camera_roi()
        self.devs.camera.prepare_live()
        # self.trg.update_camera_parameters(initial_time=self.devs.camera.t_clean,
        #                                   exposure_time=self.devs.camera.t_exposure,
        #                                   standby_time=self.devs.camera.t_readout,
        #                                   frame_rate=self.devs.camera.fps)
        # dtr, chs = self.trg.generate_digital_triggers()
        # self.ctrl_panel.display_scmos_timings(exposure_time=self.trg.exposure_time, kinetic_time=self.trg.cycle_time)
        # self.devs.trigger.write_triggers(digital_sequences=dtr, digital_channels=chs, finite=False, trg=False)

    @pyqtSlot(bool, str)
    def video(self, sw: bool, md: str):
        if sw:
            try:
                self.prepare_video(md)
            except Exception as e:
                self.logg.error(f"Error preparing imaging video: {e}")
                # self.devs.trigger.stop_triggers()
                return
            self.start_video()
        else:
            self.stop_video()

    def start_video(self):
        try:
            self.devs.camera.start_live()
            self.viewer.live_worker = run_threads.ImgLiveWorker(self.devs.camera.data, fps=10, parent=self.viewer)
            self.viewer.live_worker.img_ready.connect(self.viewer.on_camera_update_from_thread)
            # self.devs.trigger.run_triggers()
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
            # self.devs.trigger.stop_triggers()
            # time.sleep(0.04)
            self.devs.camera.stop_live()
            self.logg.info(r"Live Video Stopped")
            # self.reset_stage_positions()
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

    @pyqtSlot(bool, str, int)
    def acquisition(self, sw: bool, acq_mod: str, acq_num: int):
        if sw:
            fn = self.vw.get_file_dialog()
            tim = time.strftime("%Y%m%d%H%M%S")
            if fn is not None:
                file_name = tim + "_" + acq_mod + "_" + fn
            else:
                file_name = tim + "_" + acq_mod
            try:
                self.prepare_acquisition()
            except Exception as e:
                self.logg.error(f"Error preparing widefield: {e}")
                # self.devs.trigger.stop_triggers()
                return
            self.start_acquisition(file_name, acq_num)
        else:
            self.stop_acquisition()

    def prepare_acquisition(self):
        # self.update_trigger_parameters()
        self.set_camera_roi()
        self.devs.camera.prepare_live()
        # self.trg.update_camera_parameters(initial_time=self.devs.camera.t_clean,
        #                                   exposure_time=self.devs.camera.t_exposure,
        #                                   standby_time=self.devs.camera.t_readout,
        #                                   frame_rate=self.devs.camera.fps)
        # dtr, chs = self.trg.generate_digital_triggers(self.lasers, self.cameras["imaging"])
        # self.ctrl_panel.display_scmos_timings(exposure_time=self.trg.exposure_time, kinetic_time=self.trg.cycle_time)
        # self.devs.trigger.write_triggers(digital_sequences=dtr, digital_channels=chs, finite=False, trg=False)

    def start_acquisition(self, labl: str, acq_num: int):
        try:
            self.devs.camera.start_data_acquisition(n=acq_num, fd=self.path, fn=labl)
            self.devs.camera.data.on_update(self.viewer.on_camera_update_from_thread)
            # self.devs.trigger.run_triggers()
        except Exception as e:
            self.stop_acquisition()
            self.logg.error(f"Error start acquisition: {e}")
            return

    def stop_acquisition(self):
        try:
            # self.devs.trigger.stop_triggers()
            # time.sleep(0.04)
            self.devs.camera.stop_data_acquisition()
            # self.reset_stage_positions()
        except Exception as e:
            self.logg.error(f"Error stop acquisition: {e}")

    @pyqtSlot(str)
    def save_data(self, fd: str):
        pass

    @pyqtSlot()
    def load_cgh_target(self):
        file_name = self.vw.get_file_name()
        if file_name is not None:
            self.cgh.load_mask(file_name)
            self.viewer.set_graph_image(self.cgh.cell_mask)
        else:
            self.cgh.cell_mask = self.devs.camera.data.get_last_element()
            self.viewer.set_graph_image(self.cgh.cell_mask)

    @pyqtSlot()
    def pick_cgh_spots(self):
        self.viewer.start_target_picking()

    def update_cgh_parameters(self):
        n, m, c = self.hg_panel.get_cgh_parameters()
        o, idx = self.ctrl_panel.get_slm_parameters()
        f_mm = self.ctrl_panel.get_slm_focal_mm()
        # holograms carry no lens; f is only the reference distance for z-offset spots
        self.cgh.update_parameters(n, m, f_mm * 1e-3 if f_mm > 0 else self.cgh.holo_sys.f_fresnel, c, o)
        if self.cgh.use_affine and self.cgh.affine is not None:
            self.logg.info("Using affine calibration; Magnification / Center-X / Center-Y are ignored")
            roi = list(self.ctrl_panel.get_scmos_roi())
            if self.calib_meta.get("camera_roi") not in (None, roi):
                self.logg.warning(f"Camera ROI/binning {roi} differs from calibration "
                                  f"{self.calib_meta.get('camera_roi')}; spot positions will be wrong")
        rows = self.viewer.get_target_spots()
        # skip empty / incomplete table rows
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
            self.logg.info(f"SLM slot {slot}: {h['label']}")

    def _show_first_hologram(self):
        entry = self.slot_patterns.get(self.FIRST_HOLOGRAM_SLOT)
        if self.cgh.holograms and entry is not None:
            self.hg_panel.set_pattern_image(entry[1])
            self.hg_panel.QSpinBox_SLM_Display_Slot.setValue(self.FIRST_HOLOGRAM_SLOT)

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

    def _grab_frame(self, n_avg: int, exposure_s: float):
        frames = []
        for _ in range(n_avg):
            f = self.devs.camera.get_last_image()
            if f is not None:
                frames.append(np.asarray(f, dtype=np.float64))
            time.sleep(exposure_s)
        if not frames:
            raise RuntimeError("No camera frames; is live video running?")
        return np.mean(frames, axis=0)

    def _measure_slots(self, n_slots, wait_s, n_avg, exposure_s):
        """Worker thread: display slots 1..n_slots in turn and record one averaged frame each."""
        frames = []
        try:
            for k in range(n_slots):
                self.devs.slm.display_pattern(self.FIRST_HOLOGRAM_SLOT + k)
                time.sleep(wait_s)
                frames.append(self._grab_frame(n_avg, exposure_s))
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
        for attempt, avg in enumerate((n_avg, 2 * n_avg)):
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
          3. final affine fitted on both passes; saved, applied, and the
             magnification / center fields are updated to the closest equivalent.
        Starts live video and switches the laser to constant power if needed,
        and restores the laser mode afterwards.
        """
        if self._slm_busy():
            return
        if self.devs.slm is None or self.devs.camera is None:
            self.logg.error("Auto calibration needs both the SLM and the camera")
            return
        power = self.ctrl_panel.QDoubleSpinBox_laserpower_473.value()
        laser_was_on = self.ctrl_panel.QPushButton_laser_473.isChecked()
        if power <= 0:
            self.logg.error("Set a laser power > 0 for calibration")
            return
        try:
            self.update_cgh_parameters()
        except Exception as e:
            self.logg.error(f"Error reading CGH parameters: {e}")
            return
        if getattr(self.devs.camera, "data", None) is None:
            self.video(True, self.ctrl_panel.QComboBox_live_modes.currentText())
            self.ctrl_panel.QPushButton_video.setChecked(True)
        if not laser_was_on:
            self.set_laser(["473"], True, power)

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
                holo1, cam1, snr1, mp1 = self._project_and_detect(holo1, wait_s, n_avg, exposure_s, "coarse")
                M1, res1 = slm_cal.fit_affine(cam1, holo1)
                self.logg.info(f"Auto calibration coarse: rms {np.sqrt(np.nanmean(res1 ** 2)):.2f} px, "
                               f"{slm_cal.describe(M1)}")
                holo_all, cam_all, mp = [holo1], [cam1], mp1
                validation = None
                # 2. refine over the camera FOV
                if settings["refine"] >= 2:
                    holo2, _ = slm_cal.camera_grid_to_holo(M1, mp1.shape, settings["refine"],
                                                           phase_n=phase_n)
                    if len(holo2) >= 3:
                        holo2, cam2, snr2, mp2 = self._project_and_detect(holo2, wait_s, n_avg, exposure_s,
                                                                        "refine")
                        cam2_exp = slm_cal.holo_to_cam(M1, holo2)  # coarse-model prediction
                        ok = np.isfinite(cam2[:, 0])
                        err = np.linalg.norm(cam2[ok] - cam2_exp[ok], axis=1)
                        validation = {"n": int(ok.sum()), "rms_cam_px": float(np.sqrt(np.mean(err ** 2))),
                                      "max_cam_px": float(err.max())} if ok.any() else None
                        self.logg.info(f"Auto calibration validation (coarse model on refine spots): {validation}")
                        holo_all.append(holo2)
                        cam_all.append(cam2)
                        mp = np.maximum(mp1, mp2)
                    else:
                        self.logg.warning("Refine grid: too few points inside the hologram field; skipped")
                # 3. final fit
                holo_all, cam_all = np.vstack(holo_all), np.vstack(cam_all)
                M, res = slm_cal.fit_affine(cam_all, holo_all)
                simple = slm_cal.equivalent_simple_params(M, phase_n)
                stamp = time.strftime("%Y%m%d%H%M%S")
                base = os.path.join(self.path, f"{stamp}_slm_affine_calibration")
                meta = {"camera_roi": roi, "settle_ms": settings["settle_ms"], "n_avg": n_avg,
                        "auto": True, "settings": settings, "validation": validation,
                        "equivalent_simple": simple, "phase_n": list(phase_n),
                        "slm_offset": [self.cgh.holo_sys.slm_offset_x, self.cgh.holo_sys.slm_offset_y],
                        "f_fresnel": self.cgh.holo_sys.f_fresnel}
                data = slm_cal.save_calibration(base + ".json", M, cam_all, holo_all, res, meta)
                tf.imwrite(base + "_maxproj.tif", mp.astype(np.float32))
                self.cgh.set_affine(M)
                self.calib_meta = meta
                self._calib_result = (data, cam_all, mp, base + ".json")
                self.logg.info(f"Auto calibration saved: {base}.json; final rms {data['rms_cam_px']:.2f} px")
                if np.nanmax(res) > 3.0:
                    self.logg.warning(f"Auto calibration: max residual {np.nanmax(res):.1f} cam px")
            except Exception as e:
                self.logg.error(f"Auto calibration failed: {e}")

        def finish():
            if not laser_was_on:
                self.set_laser(["473"], False, power)
            self._show_calibration_result()
            if self._calib_result is not None:
                simple = self._calib_result[0]["meta"]["equivalent_simple"]
                self.hg_panel.set_simple_mapping(simple["magnification"], simple["center_x"], simple["center_y"])
                if not simple["exact"]:
                    self.logg.info("Magnification/center updated to the closest equivalent; the setup has "
                                   "rotation/mirror/shear, so keep 'Use Calib' on for accurate targeting")

        self._after_task = finish
        self.vw.get_dialog(r"Auto Calibration")
        self.run_task(task=task)

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
                M, res = slm_cal.fit_affine(cam_pts, holo_pts)
                stamp = time.strftime("%Y%m%d%H%M%S")
                base = os.path.join(self.path, f"{stamp}_slm_affine_calibration")
                meta = {"camera_roi": roi, "settle_ms": settle_ms, "n_avg": n_avg,
                        "snr": snr.tolist(),
                        "phase_n": [self.cgh.holo_sys.phase_nx, self.cgh.holo_sys.phase_ny],
                        "slm_offset": [self.cgh.holo_sys.slm_offset_x, self.cgh.holo_sys.slm_offset_y],
                        "f_fresnel": self.cgh.holo_sys.f_fresnel}
                data = slm_cal.save_calibration(base + ".json", M, cam_pts, holo_pts, res, meta)
                tf.imwrite(base + "_maxproj.tif", max_proj.astype(np.float32))
                self.cgh.set_affine(M)
                self.calib_meta = meta
                self._calib_result = (data, cam_pts, max_proj, base + ".json")
                self.logg.info(f"Calibration saved: {base}.json; rms {data['rms_cam_px']:.2f} cam px; "
                               f"{data['summary']}")
                if np.nanmax(res) > 3.0:
                    self.logg.warning(f"Calibration: max residual {np.nanmax(res):.1f} cam px; "
                                      f"check per-point residuals in the JSON")
            except Exception as e:
                self.logg.error(f"Calibration measurement failed: {e}")

        self._after_task = self._show_calibration_result
        self.vw.get_dialog(r"Calibration Measurement")
        self.run_task(task=task)

    def _show_calibration_result(self):
        if self._calib_result is None:
            self.hg_panel.set_calibration_status("Calibration: measurement failed (see log)")
            return
        data, cam_pts, max_proj, path = self._calib_result
        self.viewer.set_graph_image(max_proj)
        ok = np.all(np.isfinite(cam_pts), axis=1)
        self.viewer.target_spots_item.setData(cam_pts[ok, 0], cam_pts[ok, 1])
        self._set_calib_status(data, os.path.basename(path))

    def _set_calib_status(self, data, name):
        sm = data.get("summary", {})
        self.hg_panel.set_calibration_status(
            f"{name}: rms {data.get('rms_cam_px', float('nan')):.2f} px, "
            f"rot {sm.get('rotation_deg', 0):.1f} deg, scale {sm.get('scale_x', 0):.3f}/{sm.get('scale_y', 0):.3f}"
            f"{', mirrored' if sm.get('mirrored') else ''}", use=True)

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
        self.cgh.set_affine(M)
        self.calib_meta = data.get("meta", {})
        self.logg.info(f"Calibration loaded: {file_name}")
        self._set_calib_status(data, os.path.basename(file_name))

    @pyqtSlot(bool)
    def use_calibration(self, use: bool):
        if use and self.cgh.affine is None:
            self.logg.error("No affine calibration loaded or measured")
            self.hg_panel.set_calibration_status("Calibration: none", use=False)
            return
        self.cgh.use_affine = use
        self.logg.info(f"Affine calibration {'enabled' if use else 'disabled'}")

    def compute_cgh_pattern(self):
        if self._slm_busy():
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

    # def prepare_task(self, single=True):
    #     self.update_trigger_parameters()
    #     self.set_camera_roi()
    #     self.devs.camera.t_exposure = slm_on
    #     self.devs.camera.prepare_live()
    #     self.trg.update_camera_parameters(initial_time=self.devs.camera.t_clean,
    #                                       exposure_time=self.devs.camera.t_exposure,
    #                                       standby_time=self.devs.camera.t_readout,
    #                                       frame_rate=self.devs.camera.fps)
        # dtr, chs = self.trg.generate_digital_triggers()
        # self.devs.trigger.write_triggers(digital_sequences=dtr, digital_channels=chs, finite=single, trg=False)

    # def finish_task(self):
    #     try:
    #         self.devs.trigger.stop_triggers()
    #         time.sleep(0.04)
    #         self.devs.camera.stop_snap()
    #         self.logg.info("Focus Finding Finish")
    #     except Exception as e:
    #         self.logg.error(f"Error Stopping Focus Finding: {e}")
