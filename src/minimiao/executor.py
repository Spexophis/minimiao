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
        self.ao_panel = self.vw.hg_panel
        self.trg = cmp.trg
        self.path = path
        self.logg = logger or self.setup_logging()
        self._set_signal_executions()
        self._initial_setup()
        self.lasers = []
        self.slm_seq = ""
        self.cameras = {"imaging": 0, "wfs": 1, "focus_lock": 2}
        self.task_worker = None

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    def _set_signal_executions(self):
        # Camera
        self.ctrl_panel.Signal_check_emccd_temperature.connect(self.check_camera_temperature)
        self.ctrl_panel.Signal_switch_emccd_cooler.connect(self.switch_camera_cooler)
        # Cobolt Lasers
        self.ctrl_panel.Signal_set_laser.connect(self.set_laser)
        # DAQ
        self.ctrl_panel.Signal_daq_reset.connect(self.reset_daq_channels)
        self.ctrl_panel.Signal_daq_update.connect(self.update_daq_sample_rate)
        # SLM
        self.ao_panel.Signal_set_zernike.connect(self.set_zernike)
        self.ao_panel.Signal_set_dm.connect(self.set_dm_current)
        self.ao_panel.Signal_set_dm_flat.connect(self.set_dm_flat)
        self.ao_panel.Signal_update_cmd.connect(self.update_dm)
        self.ao_panel.Signal_save_dm.connect(self.save_dm)
        # Acquisition
        self.ctrl_panel.Signal_video.connect(self.video)
        self.ctrl_panel.Signal_fft.connect(self.fft)
        self.ctrl_panel.Signal_plot_profile.connect(self.profile_plot)
        self.ctrl_panel.Signal_add_profile.connect(self.plot_add)
        self.ctrl_panel.Signal_data_acquire.connect(self.acquisition)
        self.svd.connect(self.save_data)
        self.sig_plt.connect(self.viewer.plot_trace)

    def _initial_setup(self):
        try:

            p = self.devs.deck.get_position_steps_taken(3)
            self.ctrl_panel.display_deck_position(p)

            self.reset_piezo_positions()

            self.laser_lists = list(self.devs.laser.lasers.keys())

            for key in self.devs.slm.ord_dict.keys():
                self.ctrl_panel.QComboBox_slm_sequence.addItem(key)

            self.ao_panel.update_dm_display(self.devs.dfm.dpp_cmd[self.devs.dfm.current_cmd])

            self.logg.info("Finish setting up controllers")
        except Exception as e:
            self.logg.error(f"Initial setup Error: {e}")

    @pyqtSlot()
    def check_camera_temperature(self):
        try:
            self.devs.camera.get_temperature()
            self.ctrl_panel.display_emccd_temperature(self.devs.camera.temperature)
        except Exception as e:
            self.logg.error(f"CCD Camera Error: {e}")

    def reset_piezo_positions(self):
        pos_x, pos_y, pos_z = self.ctrl_panel.get_piezo_positions()
        self.set_piezo_position_x(pos_x[0], port="software")
        self.set_piezo_position_y(pos_y[0], port="software")
        self.set_piezo_position_z(pos_z[0], port="software")
        self.set_piezo_position_x(pos_x[1], port="analog")
        self.set_piezo_position_y(pos_y[1], port="analog")
        self.set_piezo_position_z(pos_z[1], port="analog")
        self.ctrl_panel.display_piezo_position_x(self.devs.piezo.read_position(0))
        self.ctrl_panel.display_piezo_position_y(self.devs.piezo.read_position(1))
        self.ctrl_panel.display_piezo_position_z(self.devs.piezo.read_position(2))

    @pyqtSlot(str, float, float, float)
    def set_piezo_positions_usb(self, axis: str, value_x: float, value_y: float, value_z: float):
        if axis == "x":
            self.set_piezo_position_x(value_x, port="software")
        if axis == "y":
            self.set_piezo_position_y(value_y, port="software")
        if axis == "z":
            self.set_piezo_position_z(value_z, port="software")

    @pyqtSlot(str, float, float, float)
    def set_piezo_positions(self, axis: str, value_x: float, value_y: float, value_z: float):
        if axis == "x":
            self.set_piezo_position_x(value_x, port="analog")
        if axis == "y":
            self.set_piezo_position_y(value_y, port="analog")
        if axis == "z":
            self.set_piezo_position_z(value_z, port="analog")

    def set_piezo_position_x(self, pos_x, port="analog"):
        try:
            if port == "software":
                self.devs.piezo.move_position(0, pos_x)
                QTimer.singleShot(100, lambda: self._update_piezo_display_x())
            else:
                self.devs.daq.set_piezo_position([pos_x / 10.], [0])
                QTimer.singleShot(100, lambda: self._update_piezo_display_x())
        except Exception as e:
            self.logg.error(f"MCL Piezo Error: {e}")

    def _update_piezo_display_x(self):
        try:
            position = self.devs.piezo.read_position(0)
            self.ctrl_panel.display_piezo_position_x(position)
        except Exception as e:
            self.logg.error(f"MCL Piezo Read Error: {e}")

    def set_piezo_position_y(self, pos_y, port="analog"):
        try:
            if port == "software":
                self.devs.piezo.move_position(1, pos_y)
                QTimer.singleShot(100, lambda: self._update_piezo_display_y())
            else:
                self.devs.daq.set_piezo_position([pos_y / 10.], [1])
                QTimer.singleShot(100, lambda: self._update_piezo_display_y())
        except Exception as e:
            self.logg.error(f"MCL Piezo Error: {e}")

    def _update_piezo_display_y(self):
        try:
            position = self.devs.piezo.read_position(1)
            self.ctrl_panel.display_piezo_position_y(position)
        except Exception as e:
            self.logg.error(f"MCL Piezo Read Error: {e}")

    def set_piezo_position_z(self, pos_z, port="analog"):
        try:
            if port == "software":
                self.devs.piezo.move_position(2, pos_z)
                QTimer.singleShot(100, lambda: self._update_piezo_display_z())
            else:
                self.devs.daq.set_piezo_position([pos_z / 10.], [2])
                QTimer.singleShot(100, lambda: self._update_piezo_display_z())
        except Exception as e:
            self.logg.error(f"MCL Piezo Error: {e}")

    def _update_piezo_display_z(self):
        try:
            position = self.devs.piezo.read_position(2)
            self.ctrl_panel.display_piezo_position_z(position)
        except Exception as e:
            self.logg.error(f"MCL Piezo Read Error: {e}")

    def update_piezo_scanning(self):
        axis_lengths, step_sizes = self.ctrl_panel.get_piezo_scan_parameters()
        pos_x, pos_y, pos_z = self.ctrl_panel.get_piezo_positions()
        positions = [pos_x[1], pos_y[1], pos_z[1]]
        return_time, line_time = self.ctrl_panel.get_piezo_scan_time()
        self.trg.update_piezo_scan_parameters(axis_lengths, step_sizes, positions, return_time, line_time)

    @pyqtSlot(list, bool, float)
    def set_laser(self, laser: list, sw: bool, pw: float):
        if sw:
            try:
                self.devs.laser.set_constant_power(laser, [pw])
                self.devs.laser.laser_on(laser)
            except Exception as e:
                self.logg.error(f"Cobolt Laser Error: {e}")
        else:
            try:
                self.devs.laser.laser_off(laser)
            except Exception as e:
                self.logg.error(f"Cobolt Laser Error: {e}")

    def set_lasers(self, lasers):
        pw = self.ctrl_panel.get_cobolt_laser_power("488_1")
        try:
            self.devs.laser.set_modulation_mode(["488_1"], [pw])
            self.devs.laser.laser_on(["488_1"])
        except Exception as e:
            self.logg.error(f"Cobolt Laser Error: {e}")

    def lasers_off(self):
        try:
            self.devs.laser.laser_off("all")
        except Exception as e:
            self.logg.error(f"Cobolt Laser Error: {e}")

    def set_camera_roi(self, key="imaging"):
        try:
            if self.cameras[key] == 0:
                x, y, nx, ny, bn = self.ctrl_panel.get_emccd_roi()
                self.devs.cam_set[0].bin_h, self.devs.cam_set[0].bin_v = bn, bn
                self.devs.cam_set[0].start_h, self.devs.cam_set[0].end_h = x, x + nx - 1
                self.devs.cam_set[0].start_v, self.devs.cam_set[0].end_v = y, y + ny - 1
                self.devs.cam_set[0].gain = self.ctrl_panel.get_emccd_gain()
                self.devs.cam_set[0].t_exposure = self.ctrl_panel.get_emccd_exposure()
            elif self.cameras[key] == 1:
                x, y, nx, ny, bn = self.ctrl_panel.get_scmos_roi()
                self.devs.cam_set[1].set_roi(bn, bn, x, nx, y, ny)
            else:
                self.logg.error(f"Camera Error: Invalid camera")
        except Exception as e:
            self.logg.error(f"Camera Error: {e}")

    @pyqtSlot(int)
    def update_daq_sample_rate(self, sr: int):
        self.trg.update_sampling_rate(sr * 1000)
        self.devs.daq.sample_rate = sr * 1000

    @pyqtSlot()
    def reset_daq_channels(self):
        self.devs.daq.stop_triggers()

    def update_digital_triggers(self):
        digital_starts, digital_ends = self.ctrl_panel.get_digital_parameters()
        self.trg.update_digital_parameters(digital_starts, digital_ends)

    def update_trigger_parameters(self, cam_key):
        """Ensure that the camera acquisition is fully set up before executing this function."""
        try:
            self.update_digital_triggers()
            self.update_piezo_scanning()
            self.logg.info(f"Trigger Updated")
        except Exception as e:
            self.logg.error(f"Trigger Error: {e}")

    def prepare_video(self, vd_mod):
        self.update_trigger_parameters("imaging")
        self.lasers = self.ctrl_panel.get_lasers()
        self.set_lasers(self.lasers)
        self.slm_seq = self.ctrl_panel.get_slm_sequence()
        slm_total, slm_end, slm_on = self.devs.slm.select_order(self.devs.slm.ord_dict[self.slm_seq])
        self.trg.update_slm_parameters(total_time=slm_total, on_time=slm_on, end_time=slm_end)
        self.cameras["imaging"] = self.ctrl_panel.get_imaging_camera()
        self.set_camera_roi("imaging")
        self.devs.cam_set[self.cameras["imaging"]].t_exposure = slm_on
        self.devs.cam_set[self.cameras["imaging"]].prepare_live()
        self.trg.update_camera_parameters(initial_time=self.devs.cam_set[self.cameras["imaging"]].t_clean,
                                          exposure_time=self.devs.cam_set[self.cameras["imaging"]].t_exposure,
                                          standby_time=self.devs.cam_set[self.cameras["imaging"]].t_readout,
                                          frame_rate=self.devs.cam_set[self.cameras["imaging"]].fps)
        dtr, chs = self.trg.generate_digital_triggers(self.lasers, self.cameras["imaging"])
        self.viewer.switch_camera(self.devs.cam_set[self.cameras["imaging"]].pixels_x,
                                  self.devs.cam_set[self.cameras["imaging"]].pixels_y)
        self.ctrl_panel.display_emccd_timings(exposure_time=self.trg.exposure_time, kinetic_time=self.trg.cycle_time)
        self.devs.daq.write_triggers(digital_sequences=dtr, digital_channels=chs, finite=False, trg=False)

    @pyqtSlot(bool, str)
    def video(self, sw: bool, md: str):
        if sw:
            try:
                self.prepare_video(md)
            except Exception as e:
                self.logg.error(f"Error preparing imaging video: {e}")
                self.devs.daq.stop_triggers()
                self.lasers_off()
                return
            self.start_video()
        else:
            self.stop_video()

    def start_video(self):
        try:
            self.devs.slm.activate()
            self.devs.cam_set[self.cameras["imaging"]].start_live()
            self.devs.cam_set[self.cameras["imaging"]].data.on_update(self.viewer.on_camera_update_from_thread)
            self.devs.daq.run_triggers()
            self.logg.info("Live Video Started")
        except Exception as e:
            self.logg.error(f"Error starting imaging video: {e}")
            self.stop_video()
            return

    def stop_video(self):
        try:
            self.devs.daq.stop_triggers()
            time.sleep(0.04)
            self.devs.cam_set[self.cameras["imaging"]].stop_live()
            self.logg.info(r"Live Video Stopped")
            self.devs.slm.deactivate()
            self.lasers_off()
            self.reset_piezo_positions()
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
                self.devs.daq.stop_triggers()
                self.lasers_off()
                return
            self.start_acquisition(file_name, acq_num)
        else:
            self.stop_acquisition()

    def prepare_acquisition(self):
        self.update_trigger_parameters("imaging")
        self.lasers = self.ctrl_panel.get_lasers()
        self.set_lasers(self.lasers)
        self.slm_seq = self.ctrl_panel.get_slm_sequence()
        slm_total, slm_end, slm_on = self.devs.slm.select_order(self.devs.slm.ord_dict[self.slm_seq])
        self.trg.update_slm_parameters(total_time=slm_total, on_time=slm_on, end_time=slm_end)
        self.cameras["imaging"] = self.ctrl_panel.get_imaging_camera()
        self.set_camera_roi("imaging")
        self.devs.cam_set[self.cameras["imaging"]].t_exposure = slm_on
        self.devs.cam_set[self.cameras["imaging"]].prepare_live()
        self.trg.update_camera_parameters(initial_time=self.devs.cam_set[self.cameras["imaging"]].t_clean,
                                          exposure_time=self.devs.cam_set[self.cameras["imaging"]].t_exposure,
                                          standby_time=self.devs.cam_set[self.cameras["imaging"]].t_readout,
                                          frame_rate=self.devs.cam_set[self.cameras["imaging"]].fps)
        dtr, chs = self.trg.generate_digital_triggers(self.lasers, self.cameras["imaging"])
        self.viewer.switch_camera(self.devs.cam_set[self.cameras["imaging"]].pixels_x,
                                  self.devs.cam_set[self.cameras["imaging"]].pixels_y)
        self.ctrl_panel.display_emccd_timings(exposure_time=self.trg.exposure_time, kinetic_time=self.trg.cycle_time)
        self.devs.daq.write_triggers(digital_sequences=dtr, digital_channels=chs, finite=False, trg=False)

    def start_acquisition(self, labl: str, acq_num: int):
        try:
            self.devs.slm.activate()
            self.devs.cam_set[self.cameras["imaging"]].start_data_acquisition(n=acq_num, fd=self.path, fn=labl)
            self.devs.cam_set[self.cameras["imaging"]].data.on_update(self.viewer.on_camera_update_from_thread)
            self.devs.daq.run_triggers()
        except Exception as e:
            self.stop_acquisition()
            self.logg.error(f"Error start acquisition: {e}")
            return

    def stop_acquisition(self):
        try:
            self.devs.daq.stop_triggers()
            time.sleep(0.04)
            self.devs.cam_set[self.cameras["imaging"]].stop_data_acquisition()
            self.lasers_off()
            self.devs.slm.deactivate()
            self.reset_piezo_positions()
        except Exception as e:
            self.logg.error(f"Error stop acquisition: {e}")

    @pyqtSlot(str)
    def save_data(self, fd: str):
        with pd.ExcelWriter(str(fd + r"_metadata.xlsx"), engine="openpyxl") as writer:
            for i, arr in enumerate(self.trg.piezo_scan_positions):
                df_pos = pd.DataFrame(arr, columns=[f"axis_{i}"])
                df_pos.to_excel(writer, sheet_name=f"axis_{i}", index=False)

    def prepare_task(self, single=True):
        self.update_trigger_parameters("imaging")
        self.lasers = self.ctrl_panel.get_lasers()
        self.set_lasers(self.lasers)
        self.slm_seq = self.ctrl_panel.get_slm_sequence()
        slm_total, slm_end, slm_on = self.devs.slm.select_order(self.devs.slm.ord_dict[self.slm_seq])
        self.trg.update_slm_parameters(total_time=slm_total, on_time=slm_on, end_time=slm_end)
        self.cameras["imaging"] = self.ctrl_panel.get_imaging_camera()
        self.set_camera_roi("imaging")
        self.devs.cam_set[self.cameras["imaging"]].t_exposure = slm_on
        self.devs.cam_set[self.cameras["imaging"]].prepare_live()
        self.trg.update_camera_parameters(initial_time=self.devs.cam_set[self.cameras["imaging"]].t_clean,
                                          exposure_time=self.devs.cam_set[self.cameras["imaging"]].t_exposure,
                                          standby_time=self.devs.cam_set[self.cameras["imaging"]].t_readout,
                                          frame_rate=self.devs.cam_set[self.cameras["imaging"]].fps)
        dtr, chs = self.trg.generate_digital_triggers(self.lasers, self.cameras["imaging"])
        self.devs.daq.write_triggers(digital_sequences=dtr, digital_channels=chs, finite=single, trg=False)

    def finish_task(self):
        try:
            self.devs.daq.stop_triggers()
            time.sleep(0.04)
            self.devs.cam_set[self.cameras["imaging"]].stop_snap()
            self.devs.slm.deactivate()
            self.lasers_off()
            self.logg.info("Focus Finding Finish")
        except Exception as e:
            self.logg.error(f"Error Stopping Focus Finding: {e}")
