# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import os
import time

import numpy as np
import pandas as pd
import tifffile as tf
from PyQt6.QtCore import QObject, pyqtSlot, Qt, pyqtSignal, QTimer

from . import run_threads


class CommandExecutor(QObject):
    svd = pyqtSignal(str)
    sig_plt = pyqtSignal(list, list)
    sig_auto_focus = pyqtSignal(float)

    def __init__(self, dev, cwd, cmp, path, logger=None):
        super().__init__()
        self.devs = dev
        self.vw = cwd
        self.ctrl_panel = self.vw.ctrl_panel
        self.viewer = self.vw.viewer
        self.hg_panel = self.vw.hg_panel
        self.trg = cmp.trg
        self.cgh = cmp.cgh
        self.path = path
        self.logg = logger or self.setup_logging()
        self._set_signal_executions()
        self._initial_setup()
        self.lasers = []
        self.task_worker = None

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    def _set_signal_executions(self):
        # SLM
        self.ctrl_panel.Signal_slm_correction.connect(self.load_slm_correction)
        self.ctrl_panel.Signal_slm_load.connect(self.load_slm_pattern)
        # Lasers
        self.ctrl_panel.Signal_set_laser.connect(self.set_laser)
        # Trigger
        self.ctrl_panel.Signal_trigger_reset.connect(self.reset_trigger_channels)
        self.ctrl_panel.Signal_trigger_update.connect(self.update_trigger_sample_rate)
        # Acquisition
        self.ctrl_panel.Signal_video.connect(self.video)
        self.ctrl_panel.Signal_data_acquire.connect(self.acquisition)
        self.svd.connect(self.save_data)
        # Hologram
        self.hg_panel.Signal_load_target.connect(self.load_cgh_target)
        # self.hg_panel.Signal_pick_spot.connect(self.pick_focal_spots)
        # self.hg_panel.Signal_compute_cgh.connect(self.compute_cgh_pattern)
        # self.hg_panel.Signal_save_pattern.connect(self.save_cgh_pattern)

    def _initial_setup(self):
        try:
            self.laser_lists = list(self.devs.laser.lasers.keys())
            self.logg.info("Finish setting up controllers")
        except Exception as e:
            self.logg.error(f"Initial setup Error: {e}")

    def set_camera_roi(self):
        try:
            x, y, nx, ny, bn = self.ctrl_panel.get_scmos_roi()
            self.devs.camera.bin_h, self.devs.camera.bin_v = bn, bn
            self.devs.camera.start_h, self.devs.camera.end_h = x, x + nx - 1
            self.devs.camera.start_v, self.devs.camera.end_v = y, y + ny - 1
            self.devs.camera.gain = self.ctrl_panel.get_scmos_gain()
            self.devs.camera.t_exposure = self.ctrl_panel.get_scmos_exposure()
        except Exception as e:
            self.logg.error(f"Camera Error: {e}")

    @pyqtSlot()
    def load_slm_correction(self):
        file_name = self.vw.get_file_name()
        if file_name is not None:
            self.cgh.load_correction_pattern(file_name)
            self.load_slm_pattern(file_name)

    @pyqtSlot()
    def load_slm_pattern(self, fn=None):
        if fn is not None:
            file_name = fn
        else:
            file_name = self.vw.get_file_name()
        if file_name is not None:
            self.devs.slm.load_pattern(file_name)
            self.hg_panel.set_pattern_image(self.devs.slm.pattern)

    def reset_stage_positions(self):
        pos_x, pos_y, pos_z = self.ctrl_panel.get_stage_positions()
        self.set_stage_position_x(pos_x[0], port="software")
        self.set_stage_position_y(pos_y[0], port="software")
        self.set_stage_position_z(pos_z[0], port="software")
        self.set_stage_position_x(pos_x[1], port="analog")
        self.set_stage_position_y(pos_y[1], port="analog")
        self.set_stage_position_z(pos_z[1], port="analog")
        self.ctrl_panel.display_stage_position_x(self.devs.stage.read_position(0))
        self.ctrl_panel.display_stage_position_y(self.devs.stage.read_position(1))
        self.ctrl_panel.display_stage_position_z(self.devs.stage.read_position(2))

    @pyqtSlot(str, float, float, float)
    def set_stage_positions_usb(self, axis: str, value_x: float, value_y: float, value_z: float):
        if axis == "x":
            self.set_stage_position_x(value_x, port="software")
        if axis == "y":
            self.set_stage_position_y(value_y, port="software")
        if axis == "z":
            self.set_stage_position_z(value_z, port="software")

    @pyqtSlot(str, float, float, float)
    def set_stage_positions(self, axis: str, value_x: float, value_y: float, value_z: float):
        if axis == "x":
            self.set_stage_position_x(value_x, port="analog")
        if axis == "y":
            self.set_stage_position_y(value_y, port="analog")
        if axis == "z":
            self.set_stage_position_z(value_z, port="analog")

    def set_stage_position_x(self, pos_x, port="analog"):
        try:
            if port == "software":
                self.devs.stage.move_position(0, pos_x)
                QTimer.singleShot(100, lambda: self._update_stage_display_x())
            else:
                self.devs.trigger.set_stage_position([pos_x / 10.], [0])
                QTimer.singleShot(100, lambda: self._update_stage_display_x())
        except Exception as e:
            self.logg.error(f" stage Error: {e}")

    def _update_stage_display_x(self):
        try:
            position = self.devs.stage.read_position(0)
            self.ctrl_panel.display_stage_position_x(position)
        except Exception as e:
            self.logg.error(f" stage Read Error: {e}")

    def set_stage_position_y(self, pos_y, port="analog"):
        try:
            if port == "software":
                self.devs.stage.move_position(1, pos_y)
                QTimer.singleShot(100, lambda: self._update_stage_display_y())
            else:
                self.devs.trigger.set_stage_position([pos_y / 10.], [1])
                QTimer.singleShot(100, lambda: self._update_stage_display_y())
        except Exception as e:
            self.logg.error(f" stage Error: {e}")

    def _update_stage_display_y(self):
        try:
            position = self.devs.stage.read_position(1)
            self.ctrl_panel.display_stage_position_y(position)
        except Exception as e:
            self.logg.error(f" stage Read Error: {e}")

    def set_stage_position_z(self, pos_z, port="analog"):
        try:
            if port == "software":
                self.devs.stage.move_position(2, pos_z)
                QTimer.singleShot(100, lambda: self._update_stage_display_z())
            else:
                self.devs.trigger.set_stage_position([pos_z / 10.], [2])
                QTimer.singleShot(100, lambda: self._update_stage_display_z())
        except Exception as e:
            self.logg.error(f" stage Error: {e}")

    def _update_stage_display_z(self):
        try:
            position = self.devs.stage.read_position(2)
            self.ctrl_panel.display_stage_position_z(position)
        except Exception as e:
            self.logg.error(f" stage Read Error: {e}")

    def update_stage_scanning(self):
        axis_lengths, step_sizes = self.ctrl_panel.get_stage_scan_parameters()
        pos_x, pos_y, pos_z = self.ctrl_panel.get_stage_positions()
        positions = [pos_x[1], pos_y[1], pos_z[1]]
        return_time, line_time = self.ctrl_panel.get_stage_scan_time()
        self.trg.update_stage_scan_parameters(axis_lengths, step_sizes, positions, return_time, line_time)

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

    @pyqtSlot(int)
    def update_trigger_sample_rate(self, sr: int):
        self.trg.update_sampling_rate(sr * 1000)
        self.devs.trigger.sample_rate = sr * 1000

    @pyqtSlot()
    def reset_trigger_channels(self):
        self.devs.trigger.stop_triggers()

    def update_digital_triggers(self):
        digital_starts, digital_ends = self.ctrl_panel.get_digital_parameters()
        self.trg.update_digital_parameters(digital_starts, digital_ends)

    def update_trigger_parameters(self):
        """Ensure that the camera acquisition is fully set up before executing this function."""
        try:
            self.update_digital_triggers()
            self.update_stage_scanning()
            self.logg.info(f"Trigger Updated")
        except Exception as e:
            self.logg.error(f"Trigger Error: {e}")

    def prepare_video(self, vd_mod):
        self.update_trigger_parameters()
        self.set_camera_roi()
        self.devs.camera.prepare_live()
        self.trg.update_camera_parameters(initial_time=self.devs.camera.t_clean,
                                          exposure_time=self.devs.camera.t_exposure,
                                          standby_time=self.devs.camera.t_readout,
                                          frame_rate=self.devs.camera.fps)
        # dtr, chs = self.trg.generate_digital_triggers()
        self.viewer.switch_camera(self.devs.camera.pixels_x,
                                  self.devs.camera.pixels_y)
        self.ctrl_panel.display_scmos_timings(exposure_time=self.trg.exposure_time, kinetic_time=self.trg.cycle_time)
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
            self.devs.camera.data.on_update(self.viewer.on_camera_update_from_thread)
            # self.devs.trigger.run_triggers()
            self.logg.info("Live Video Started")
        except Exception as e:
            self.logg.error(f"Error starting imaging video: {e}")
            self.stop_video()
            return

    def stop_video(self):
        try:
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
        self.update_trigger_parameters()
        self.set_camera_roi()
        self.devs.camera.prepare_live()
        self.trg.update_camera_parameters(initial_time=self.devs.camera.t_clean,
                                          exposure_time=self.devs.camera.t_exposure,
                                          standby_time=self.devs.camera.t_readout,
                                          frame_rate=self.devs.camera.fps)
        # dtr, chs = self.trg.generate_digital_triggers(self.lasers, self.cameras["imaging"])
        self.viewer.switch_camera(self.devs.camera.pixels_x,
                                  self.devs.camera.pixels_y)
        self.ctrl_panel.display_scmos_timings(exposure_time=self.trg.exposure_time, kinetic_time=self.trg.cycle_time)
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

    def load_cgh_target(self):
        file_name = self.vw.get_file_name()
        if file_name is not None:
            self.cgh.load_mask(file_name)
            self.hg_panel.set_target_image(self.cgh.cell_mask)

    # def pick_focal_spots
    # def compute_cgh_pattern
    # def save_cgh_pattern

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
