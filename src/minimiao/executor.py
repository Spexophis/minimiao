# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import os
import time

import numpy as np
import pandas as pd
import tifffile as tf
from PyQt6.QtCore import QObject, pyqtSlot, Qt, pyqtSignal

from . import run_threads
from .utilities import image_processor as ipr


class CommandExecutor(QObject):
    svd = pyqtSignal(str)
    psv = pyqtSignal(str)

    def __init__(self, dev, cwd, cmp, path, logger=None):
        super().__init__()
        self.devs = dev
        self.vw = cwd
        self.ctrl_panel = self.vw.ctrl_panel
        self.viewer = self.vw.viewer
        self.ao_panel = self.vw.ao_panel
        self.rec = cmp.rec
        self.trg = cmp.trg
        self.wfr = cmp.wfp
        self.flk = cmp.flp
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
        # EMCCD
        self.ctrl_panel.Signal_check_emccd_temperature.connect(self.check_emdccd_temperature)
        self.ctrl_panel.Signal_switch_emccd_cooler.connect(self.switch_emdccd_cooler)
        # MCL Piezo
        self.ctrl_panel.Signal_piezo_move_usb.connect(self.set_piezo_positions_usb)
        self.ctrl_panel.Signal_piezo_move.connect(self.set_piezo_positions)
        self.ctrl_panel.Signal_focus_finding.connect(self.run_focus_finding)
        # self.ctrl_panel.Signal_focus_locking.connect(self.run_focus_locking)
        # MCL Mad Deck
        self.ctrl_panel.Signal_deck_read_position.connect(self.deck_read_position)
        self.ctrl_panel.Signal_deck_zero_position.connect(self.deck_zero_position)
        self.ctrl_panel.Signal_deck_move_single_step.connect(self.move_deck_single_step)
        self.ctrl_panel.Signal_deck_move_continuous.connect(self.move_deck_continuous)
        # Cobolt Lasers
        self.ctrl_panel.Signal_set_laser.connect(self.set_laser)
        # DAQ
        self.ctrl_panel.Signal_daq_reset.connect(self.reset_daq_channels)
        self.ctrl_panel.Signal_daq_update.connect(self.update_daq_sample_rate)
        # Acquisition
        self.ctrl_panel.Signal_video.connect(self.video)
        self.ctrl_panel.Signal_fft.connect(self.fft)
        self.ctrl_panel.Signal_plot_profile.connect(self.profile_plot)
        self.ctrl_panel.Signal_add_profile.connect(self.plot_add)
        self.ctrl_panel.Signal_data_acquire.connect(self.data_acquisition)
        self.svd.connect(self.save_data)
        self.psv.connect(self.save_scan)

    def _initial_setup(self):
        try:

            p = self.devs.deck.get_position_steps_taken(3)
            self.ctrl_panel.display_deck_position(p)

            self.reset_piezo_positions()

            self.laser_lists = list(self.devs.laser.lasers.keys())

            for key in self.devs.slm.ord_dict.keys():
                self.ctrl_panel.QComboBox_slm_sequence.addItem(key)

            self.logg.info("Finish setting up controllers")
        except Exception as e:
            self.logg.error(f"Initial setup Error: {e}")

    @pyqtSlot()
    def check_emdccd_temperature(self):
        try:
            self.devs.emccd.get_temperature()
            self.ctrl_panel.display_emccd_temperature(self.devs.emccd.temperature)
        except Exception as e:
            self.logg.error(f"CCD Camera Error: {e}")

    @pyqtSlot(bool)
    def switch_emdccd_cooler(self, sw: bool):
        if sw:
            self.switch_emdccd_cooler_on()
        else:
            self.switch_emdccd_cooler_off()

    def switch_emdccd_cooler_on(self):
        try:
            self.devs.emccd.cooler_on()
        except Exception as e:
            self.logg.error(f"CCD Camera Error: {e}")

    def switch_emdccd_cooler_off(self):
        try:
            self.devs.emccd.cooler_off()
        except Exception as e:
            self.logg.error(f"CCD Camera Error: {e}")

    @pyqtSlot()
    def deck_read_position(self):
        self.ctrl_panel.display_deck_position(self.devs.deck.position)

    @pyqtSlot()
    def deck_zero_position(self):
        self.devs.deck.position = 0
        self.ctrl_panel.display_deck_position(self.devs.deck.position)

    @pyqtSlot(bool)
    def move_deck_single_step(self, direction: bool):
        if direction:
            self.move_deck_up()
        else:
            self.move_deck_down()

    def move_deck_up(self):
        try:
            _moving = self.devs.deck.is_moving()
            if _moving:
                self.logg.info("MadDeck is moving")
            else:
                self.devs.deck.move_relative(3, 0.000762, velocity=0.8)
                self.ctrl_panel.display_deck_position(self.devs.deck.position)
        except Exception as e:
            self.logg.error(f"MadDeck Error: {e}")

    def move_deck_down(self):
        try:
            _moving = self.devs.deck.is_moving()
            if _moving:
                self.logg.info("MadDeck is moving")
            else:
                self.devs.deck.move_relative(3, -0.000762, velocity=0.8)
                self.ctrl_panel.display_deck_position(self.devs.deck.position)
        except Exception as e:
            self.logg.error(f"MadDeck Error: {e}")

    @pyqtSlot(bool, int, float)
    def move_deck_continuous(self, moving: bool, direction: int, velocity: float):
        if moving:
            self.devs.deck.move_deck(direction, velocity)
        else:
            self.devs.deck.stop_deck()

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
                time.sleep(0.1)
                self.ctrl_panel.display_piezo_position_x(self.devs.piezo.read_position(0))
            else:
                self.devs.daq.set_piezo_position([pos_x / 10.], [0])
                time.sleep(0.1)
                self.ctrl_panel.display_piezo_position_x(self.devs.piezo.read_position(0))
        except Exception as e:
            self.logg.error(f"MCL Piezo Error: {e}")

    def set_piezo_position_y(self, pos_y, port="analog"):
        try:
            if port == "software":
                self.devs.piezo.move_position(1, pos_y)
                time.sleep(0.1)
                self.ctrl_panel.display_piezo_position_y(self.devs.piezo.read_position(1))
            else:
                self.devs.daq.set_piezo_position([pos_y / 10.], [1])
                time.sleep(0.1)
                self.ctrl_panel.display_piezo_position_y(self.devs.piezo.read_position(1))
        except Exception as e:
            self.logg.error(f"MCL Piezo Error: {e}")

    def set_piezo_position_z(self, pos_z, port="analog"):
        try:
            if port == "software":
                self.devs.piezo.move_position(2, pos_z)
                time.sleep(0.1)
                self.ctrl_panel.display_piezo_position_z(self.devs.piezo.read_position(2))
            else:
                self.devs.daq.set_piezo_position([pos_z / 10.], [2])
                time.sleep(0.1)
                self.ctrl_panel.display_piezo_position_z(self.devs.piezo.read_position(2))
        except Exception as e:
            self.logg.error(f"MCL Piezo Error: {e}")

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
        pws = self.ctrl_panel.get_cobolt_laser_power("all")
        ln = []
        pw = []
        for ls in lasers:
            ln.append(self.laser_lists[ls])
            pw.append(pws[ls])
        try:
            self.devs.laser.set_modulation_mode(ln, pw)
            self.devs.laser.laser_on(ln)
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
            elif self.cameras[key] == 1:
                x, y, nx, ny, bn = self.ctrl_panel.get_scmos_roi()
                self.devs.cam_set[1].set_roi(bn, bn, x, nx, y, ny)
            elif self.cameras[key] == 2:
                expo = self.ctrl_panel.get_cmos_exposure()
                self.devs.cam_set[2].set_exposure(expo)
                x, y, nx, ny, bx, by = self.ctrl_panel.get_cmos_roi()
                self.devs.cam_set[2].set_roi(x, y, nx, ny)
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
            if self.cameras[cam_key] <= 2:
                self.trg.update_camera_parameters(initial_time=self.devs.cam_set[self.cameras[cam_key]].t_clean,
                                                  standby_time=self.devs.cam_set[self.cameras[cam_key]].t_readout)
            self.logg.info(f"Trigger Updated")
        except Exception as e:
            self.logg.error(f"Trigger Error: {e}")

    def prepare_video(self, vd_mod):
        self.lasers = self.ctrl_panel.get_lasers()
        self.set_lasers(self.lasers)
        self.cameras["imaging"] = self.ctrl_panel.get_imaging_camera()
        if self.cameras["imaging"] <= 2:
            self.set_camera_roi("imaging")
            self.devs.cam_set[self.cameras["imaging"]].prepare_live()
            self.update_trigger_parameters("imaging")
            self.viewer.switch_camera(self.devs.cam_set[self.cameras["imaging"]].pixels_x,
                                      self.devs.cam_set[self.cameras["imaging"]].pixels_y)
            self.slm_seq = self.ctrl_panel.get_slm_sequence()
            if self.slm_seq != "None":
                self.devs.slm.select_order(self.devs.slm.ord_dict[self.slm_seq])
            if vd_mod == "Wide Field":
                dtr, chs = self.trg.generate_digital_triggers(self.lasers, self.cameras["imaging"], self.slm_seq)
                self.devs.daq.write_triggers(digital_sequences=dtr, digital_channels=chs, finite=False)
                if self.cameras["imaging"] == 0:
                    self.ctrl_panel.display_emccd_timings(clean=self.trg.initial_time,
                                                          exposure=self.trg.exposure_time,
                                                          standby=self.trg.standby_time)
                if self.cameras["imaging"] == 1:
                    self.ctrl_panel.display_scmos_timings(clean=self.trg.initial_time,
                                                          exposure=self.trg.exposure_time,
                                                          standby=self.trg.standby_time)
            if vd_mod == "SIM":
                dtr, chs = self.trg.generate_sim_triggers(self.lasers, self.cameras["imaging"], self.slm_seq, 2)
                self.devs.daq.write_triggers(digital_sequences=dtr, digital_channels=chs, finite=False)
                if self.cameras["imaging"] == 0:
                    self.ctrl_panel.display_emccd_timings(clean=self.trg.initial_time,
                                                          exposure=self.trg.exposure_time,
                                                          standby=self.trg.standby_time)
                if self.cameras["imaging"] == 1:
                    self.ctrl_panel.display_scmos_timings(clean=self.trg.initial_time,
                                                          exposure=self.trg.exposure_time,
                                                          standby=self.trg.standby_time)
            if vd_mod == "Focus Lock":
                self.logg.info(f"Focus Lock live")
        elif self.cameras["imaging"] == 3:
            self.slm_seq = self.ctrl_panel.get_slm_sequence()
            if self.slm_seq != "None":
                self.devs.slm.select_order(self.devs.slm.ord_dict[self.slm_seq])
            self.update_trigger_parameters("imaging")
            if vd_mod == "Wide Field":
                dtr, dch, dwl = self.trg.generate_live_point_scan_2d(self.lasers, "None")
                self.devs.daq.write_triggers(digital_sequences=dtr, digital_channels=dch, finite=False)
                self.devs.daq.photon_counter_mode = 0
                self.viewer.psr_mode = False
            elif vd_mod == "Point Scan":
                ptr, pch, dtr, dch, dwl = self.trg.generate_piezo_point_scan_2d(self.lasers)
                self.devs.daq.write_triggers(piezo_sequences=ptr, piezo_channels=pch,
                                             digital_sequences=dtr, digital_channels=dch, finite=False)
                self.devs.daq.photon_counter_mode = 1
                self.devs.daq.psr = self.rec
                self.viewer.psr_mode = True
            else:
                raise Exception(f"Invalid video mode {vd_mod} for MPD")
            self.rec.point_scan_gate_mask = dtr[-1]
            self.rec.set_point_scan_params(n_lines=self.trg.piezo_scan_pos[1],
                                           n_pixels=self.trg.piezo_scan_pos[0],
                                           dwell_samples=dwl)
            self.rec.prepare_point_scan_live_recon()
            self.devs.daq.photon_counter_length = dtr.shape[1]
            self.devs.daq.prepare_photon_counter()
            self.viewer.photon_pool.reset_buffer(max_len=self.devs.daq.photon_counter_length,
                                                 dt_s=1/self.devs.daq.sample_rate,
                                                 px=(self.trg.piezo_scan_pos[1], self.trg.piezo_scan_pos[0]))
            if getattr(self.viewer, "psr_worker", None) is None:
                self.viewer.psr_worker = run_threads.PSLiveWorker(self.devs.daq.data, self.rec, fps=10)
                self.viewer.psr_worker.psr_ready.connect(self.viewer.photon_pool.new_acquire)
                self.viewer.psr_worker.psr_new.connect(self.viewer.on_psr_frame)
                self.viewer.psr_worker.start()
        else:
            raise Exception(f"Invalid camera selection")

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
            if self.slm_seq != "None":
                self.devs.slm.activate()
            if self.cameras["imaging"] <= 2:
                self.devs.cam_set[self.cameras["imaging"]].start_live()
                self.devs.cam_set[self.cameras["imaging"]].data.on_update(self.viewer.on_camera_update_from_thread)
                self.devs.daq.run_triggers()
            else:
                self.devs.daq.start_triggers()
                self.devs.daq.start_photon_count()
                self.devs.daq.run_triggers()
                self.viewer.stream_trace(self.viewer.photon_pool.xt, self.viewer.photon_pool.buf)
            self.logg.info("Live Video Started")
        except Exception as e:
            self.logg.error(f"Error starting imaging video: {e}")
            self.stop_video()
            return

    def stop_video(self):
        try:
            if self.cameras["imaging"] <= 2:
                self.devs.cam_set[self.cameras["imaging"]].stop_live()
                self.devs.daq.stop_triggers()
            else:
                if getattr(self.viewer, "psr_worker", None) is not None:
                    self.viewer.psr_worker.stop()
                    self.viewer.psr_worker = None
                self.devs.daq.stop_photon_count()
                self.devs.daq.stop_triggers()
            self.logg.info(r"Live Video Stopped")
            if self.slm_seq != "None":
                self.devs.slm.deactivate()
            self.lasers_off()
            self.reset_piezo_positions()
        except Exception as e:
            self.logg.error(f"Error stopping imaging video: {e}")

    @pyqtSlot(bool)
    def fft(self, on: bool):
        if on:
            if getattr(self.viewer, "fft_worker", None) is None:
                self.viewer.fft_worker = run_threads.FFTWorker(fps=10)
                self.viewer.fft_worker.fft_ready.connect(self.viewer.on_fft_frame, Qt.ConnectionType.QueuedConnection)
                self.viewer.fft_worker.start()
            self.viewer.fft_mode = True
        else:
            self.viewer.fft_mode = False
            if getattr(self.viewer, "fft_worker", None) is not None:
                self.viewer.fft_worker.stop()
                self.viewer.fft_worker = None

    @pyqtSlot()
    def profile_plot(self):
        try:
            ax = self.ctrl_panel.get_profile_axis()
            self.viewer.plot_trace(ipr.get_profile(self.viewer.image_viewer._display_frame, ax, norm=True))
        except Exception as e:
            self.logg.error(f"Error plotting profile: {e}")

    @pyqtSlot()
    def plot_add(self):
        try:
            ax = self.ctrl_panel.get_profile_axis()
            self.viewer.plot_trace(ipr.get_profile(self.viewer.image_viewer._display_frame, ax, norm=True), overlay=True)
        except Exception as e:
            self.logg.error(f"Error plotting profile: {e}")

    @pyqtSlot()
    def plot_trigger(self):
        try:
            self.slm_seq = self.ctrl_panel.get_slm_sequence()
            self.devs.slm.select_order(self.devs.slm.ord_dict[self.slm_seq])
            self.update_trigger_parameters("imaging")
            dtr, dch = self.trg.generate_digital_triggers(self.lasers, self.cameras["imaging"], self.slm_seq)
            self.viewer.plot_trace(dtr[0])
            for i in range(dtr.shape[0] - 1):
                self.viewer.plot(dtr[i + 1] + i + 1)
        except Exception as e:
            self.logg.error(f"Error plotting digital triggers: {e}")

    @pyqtSlot(list, list)
    def plot_(self, x, d):
        try:
            self.viewer.plot_update(data=d, x=x)
        except Exception as e:
            self.logg.error(f"Error plotting profile: {e}")

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

    @pyqtSlot(str, int)
    def data_acquisition(self, acq_mod: str, acq_num: int):
        if acq_mod == "Point Scan 2D":
            self.run_point_scan(acq_num)
        elif acq_mod == "Wide Field":
            self.run_widefield(acq_num)
        elif acq_mod == "SIM 2D":
            self.run_sim_2d(acq_num)
        elif acq_mod == "SIM 3D":
            self.run_sim_3d(acq_num)
        else:
            self.logg.error(f"Invalid video mode")

    @pyqtSlot(str)
    def save_data(self, tm: str):
        fn = self.vw.get_file_dialog()
        if fn is not None:
            fd = os.path.join(self.path, tm + '_' + fn)
        else:
            fd = os.path.join(self.path, tm)
        tf.imwrite(str(fd + r".tif"), data=self.devs.cam_set[self.cameras["imaging"]].get_data())
        with pd.ExcelWriter(str(fd + r"_metadata.xlsx"), engine="openpyxl") as writer:
            df_idx = pd.DataFrame(list(self.devs.cam_set[self.cameras["imaging"]].data.ind_list),
                                  columns=["acquisition_sequence"])
            df_idx.to_excel(writer, sheet_name="acquisition_sequence", index=False)
            for i, arr in enumerate(self.trg.piezo_scan_positions):
                df_pos = pd.DataFrame(arr, columns=[f"axis_{i}"])
                df_pos.to_excel(writer, sheet_name=f"axis_{i}", index=False)
        self.devs.cam_set[self.cameras["imaging"]].data = None

    @pyqtSlot(str)
    def save_scan(self, tm: str):
        fn = self.vw.get_file_dialog()
        if fn is not None:
            fd = os.path.join(self.path, tm + '_' + fn)
        else:
            fd = os.path.join(self.path, tm)
        tf.imwrite(str(fd + r"_recon_image.tif"), self.rec.live_rec.astype(np.float16))
        try:
            res = np.zeros((3, self.rec.gate_len))
            res[0] = np.arange(self.rec.gate_len) / self.trg.sample_rate
            res[1] = self.rec.point_scan_gate_mask * 1
            res[2] = np.array(self.devs.daq.data.count_list)
            np.save(str(fd + r"_photon_counts.npy"), res)
        except Exception as e:
            self.logg.error(f"Error writing photon counting data: {e}")
        try:
            with pd.ExcelWriter(str(fd + r"_scan_positions.xlsx"), engine="openpyxl") as writer:
                for i, arr in enumerate(self.trg.piezo_scan_positions):
                    df_pos = pd.DataFrame(arr, columns=[f"axis_{i}"])
                    df_pos.to_excel(writer, sheet_name=f"axis_{i}", index=False)
        except Exception as e:
            self.logg.error(f"Error writing piezo scanning data: {e}")
        self.devs.daq.data = None

    def prepare_focus_finding(self):
        self.lasers = self.ctrl_panel.get_lasers()
        self.set_lasers(self.lasers)
        self.cameras["imaging"] = self.ctrl_panel.get_imaging_camera()
        self.set_camera_roi("imaging")
        self.slm_seq = self.ctrl_panel.get_slm_sequence()
        if self.slm_seq != "None":
            self.devs.slm.select_order(self.devs.slm.ord_dict[self.slm_seq])
        self.devs.cam_set[self.cameras["imaging"]].prepare_live()
        self.update_trigger_parameters("imaging")
        dtr, chs = self.trg.generate_digital_triggers(self.lasers, self.cameras["imaging"], self.slm_seq)
        self.devs.daq.write_triggers(digital_sequences=dtr, digital_channels=chs, finite=True)
        if self.cameras["imaging"] == 0:
            self.ctrl_panel.display_emccd_timings(clean=self.trg.initial_time,
                                                  exposure=self.trg.exposure_time,
                                                  standby=self.trg.standby_time)
        if self.cameras["imaging"] == 1:
            self.ctrl_panel.display_scmos_timings(clean=self.trg.initial_time,
                                                  exposure=self.trg.exposure_time,
                                                  standby=self.trg.standby_time)
        # self.devs.cam_set[self.cameras["focus_lock"]].set_exposure(self.ctrl_panel.get_tis_expo())
        # self.devs.cam_set[self.cameras["focus_lock"]].prepare_live()

    def focus_finding(self):
        try:
            self.prepare_focus_finding()
        except Exception as e:
            self.logg.error(f"Error starting focus finding: {e}")
            self.devs.daq.stop_triggers()
            self.lasers_off()
            return
        try:
            pos_x, pos_y, pos_z = self.ctrl_panel.get_piezo_positions()
            center_pos, axis_length, step_size = pos_z[0], 0.96, 0.06
            start = center_pos - axis_length
            end = center_pos + axis_length
            zps = np.arange(start, end + step_size, step_size)
            data = []
            # data_calib = []
            pzs = []
            if self.slm_seq != "None":
                self.devs.slm.activate()
            self.devs.cam_set[self.cameras["imaging"]].start_live()
            # self.devs.cam_set[self.cameras["focus_lock"]].start_live()
            for i, z in enumerate(zps):
                self.set_piezo_position_z(z, port="software")
                time.sleep(0.1)
                self.devs.daq.run_triggers()
                time.sleep(0.04)
                self.devs.daq.stop_triggers(_close=False)
                temp = self.devs.cam_set[self.cameras["imaging"]].get_last_image()
                data.append(temp)
                # data_calib.append(self.devs.cam_set[self.cameras["focus_lock"]].get_last_image())
                pzs.append(ipr.calculate_focus_measure_with_sobel(temp - temp.min()))
            fd = os.path.join(self.path, time.strftime("%Y%m%d%H%M%S") + '_widefield_stack.tif')
            tf.imwrite(fd, np.asarray(data))
            self.viewer.plot_trace(y=pzs, x=zps)
            fp = ipr.peak_find(zps, pzs)
            if isinstance(fp, str):
                self.logg.error(fp)
            else:
                self.ctrl_panel.QDoubleSpinBox_stage_z_usb.setValue(fp)
            # time.sleep(0.06)
            # data_calib.append(self.devs.cam_set[self.cameras["focus_lock"]].get_last_image())
            # fd = os.path.join(self.path, time.strftime("%Y%m%d%H%M%S") + '_focus_calibration_stack.tif')
            # tf.imwrite(fd, np.asarray(data_calib), imagej=True, resolution=(
            #     1 / self.pixel_sizes[self.cameras["focus_lock"]], 1 / self.pixel_sizes[self.cameras["focus_lock"]]),
            #            metadata={'unit': 'um'})
            # self.p.foc_ctrl.calibrate(np.append(zps, fp), np.asarray(data_calib))
        except Exception as e:
            self.finish_focus_finding()
            self.logg.error(f"Error running focus finding: {e}")
            return
        self.finish_focus_finding()

    def finish_focus_finding(self):
        try:
            self.devs.daq.stop_triggers()
            self.devs.cam_set[self.cameras["imaging"]].stop_live()
            # self.devs.cam_set[self.cameras["focus_lock"]].stop_live()
            if self.slm_seq != "None":
                self.devs.slm.deactivate()
            self.lasers_off()
            self.reset_piezo_positions()
            self.logg.info("Focus finding stack acquired")
        except Exception as e:
            self.logg.error(f"Error stopping focus finding: {e}")

    @pyqtSlot()
    def run_focus_finding(self):
        self.vw.get_dialog(txt="Focus Finding")
        self.run_task(task=self.focus_finding)

    def prepare_widefield(self, tim):
        self.lasers = self.ctrl_panel.get_lasers()
        self.set_lasers(self.lasers)
        self.cameras["imaging"] = self.ctrl_panel.get_imaging_camera()
        self.set_camera_roi("imaging")
        self.slm_seq = self.ctrl_panel.get_slm_sequence()
        if self.slm_seq != "None":
            self.devs.slm.select_order(self.devs.slm.ord_dict[self.slm_seq])
        self.devs.cam_set[self.cameras["imaging"]].prepare_live()
        self.update_trigger_parameters("imaging")
        dtr, ptr, dch, pch, pos = self.trg.generate_piezo_scan(self.lasers, self.cameras["imaging"], self.slm_seq)
        self.devs.daq.set_piezo_position(pos=[ptr[0]], indices=[2])
        self.devs.cam_set[self.cameras["imaging"]].acq_num = pos
        self.devs.daq.write_triggers(piezo_sequences=ptr, piezo_channels=pch,
                                     digital_sequences=dtr, digital_channels=dch,
                                     finite=True)
        self.ctrl_panel.display_emccd_timings(clean=self.trg.initial_time,
                                              exposure=self.trg.exposure_time,
                                              standby=self.trg.standby_time)
        fd = os.path.join(self.path, tim + r"_widefield_triggers.npy")
        np.save(str(fd), np.vstack((ptr, dtr)))

    def widefield(self):
        tim = time.strftime("%Y%m%d%H%M%S")
        try:
            self.prepare_widefield(tim)
        except Exception as e:
            self.logg.error(f"Error preparing widefield: {e}")
            self.devs.daq.stop_triggers()
            self.lasers_off()
            return
        try:
            if self.slm_seq != "None":
                self.devs.slm.activate()
            self.devs.cam_set[self.cameras["imaging"]].start_data_acquisition()
            self.devs.daq.run_triggers()
            self.devs.cam_set[self.cameras["imaging"]].data.on_update(self.viewer.on_camera_update_from_thread)
            time.sleep(0.2)
            self.svd.emit(time.strftime("%Y%m%d%H%M%S") + '_widefield')
        except Exception as e:
            self.finish_widefield()
            self.logg.error(f"Error running widefield: {e}")
            return
        self.finish_widefield()

    def finish_widefield(self):
        try:
            self.devs.daq.stop_triggers()
            self.devs.cam_set[self.cameras["imaging"]].stop_data_acquisition()
            self.lasers_off()
            if self.slm_seq != "None":
                self.devs.slm.deactivate()
            self.reset_piezo_positions()
            self.logg.info("Widefield image stack acquired")
        except Exception as e:
            self.logg.error(f"Error stopping widefield: {e}")

    def run_widefield(self, n: int):
        self.vw.get_dialog(txt="Widefield Acquisition")
        self.run_task(task=self.widefield, iteration=n)

    def prepare_point_scan(self, tim):
        self.lasers = self.ctrl_panel.get_lasers()
        self.set_lasers(self.lasers)
        self.slm_seq = self.ctrl_panel.get_slm_sequence()
        if self.slm_seq != "None":
            self.devs.slm.select_order(self.devs.slm.ord_dict[self.slm_seq])
        self.update_trigger_parameters("imaging")
        ptr, pch, dtr, dch, dwl = self.trg.generate_piezo_point_scan_2d(self.lasers)
        self.devs.daq.write_triggers(piezo_sequences=ptr, piezo_channels=pch,
                                     digital_sequences=dtr, digital_channels=dch, finite=True)
        self.devs.daq.photon_counter_mode = 1
        self.devs.daq.psr = self.rec
        self.rec.point_scan_gate_mask = dtr[-1]
        self.rec.set_point_scan_params(n_lines=self.trg.piezo_scan_pos[1],
                                       n_pixels=self.trg.piezo_scan_pos[0],
                                       dwell_samples=dwl)
        self.rec.prepare_point_scan_live_recon()
        self.devs.daq.photon_counter_length = self.rec.gate_len
        self.devs.daq.prepare_photon_counter()
        fd = os.path.join(self.path, tim + r"_point_scanning_triggers.npy")
        np.save(str(fd), np.vstack((ptr, dtr)))

    def point_scan(self):
        tim = time.strftime("%Y%m%d%H%M%S")
        try:
            self.prepare_point_scan(tim)
        except Exception as e:
            self.logg.error(f"Error preparing point scanning: {e}")
            return
        try:
            self.devs.slm.activate()
            self.devs.daq.start_triggers()
            self.devs.daq.start_photon_count()
            self.devs.daq.run_triggers()
            time.sleep(0.2)
            self.psv.emit(tim + r"_point_scanning")
        except Exception as e:
            self.finish_point_scan()
            self.logg.error(f"Error running point scanning: {e}")
            return
        self.finish_point_scan()

    def finish_point_scan(self):
        try:
            self.devs.daq.stop_photon_count()
            self.devs.daq.stop_triggers()
            self.devs.slm.deactivate()
            self.lasers_off()
            self.reset_piezo_positions()
            self.logg.info("Point scanning image acquired")
        except Exception as e:
            self.logg.error(f"Error stopping point scanning: {e}")

    def run_point_scan(self, n: int):
        self.vw.get_dialog(txt="Point scanning Acquisition")
        self.run_task(task=self.point_scan, iteration=n)

    def prepare_sim_2d(self):
        self.lasers = self.ctrl_panel.get_lasers()
        self.set_lasers(self.lasers)
        self.cameras["imaging"] = self.ctrl_panel.get_imaging_camera()
        self.set_camera_roi("imaging")
        self.slm_seq = self.ctrl_panel.get_slm_sequence()
        if self.slm_seq == "None":
            raise ValueError("SLM sequence cannot be None.")
        else:
            self.devs.slm.select_order(self.devs.slm.ord_dict[self.slm_seq])
        self.devs.cam_set[self.cameras["imaging"]].prepare_data_acquisition()
        self.update_trigger_parameters("imaging")
        dtr, sw, dch = self.trg.generate_sim_triggers(self.lasers, self.cameras["imaging"], self.slm_seq, 2)
        self.devs.cam_set[self.cameras["imaging"]].acq_num = 3
        self.devs.daq.write_triggers(digital_sequences=dtr, digital_channels=dch, finite=True)
        self.ctrl_panel.display_emccd_timings(clean=self.trg.initial_time,
                                              exposure=self.trg.exposure_time,
                                              standby=self.trg.standby_time)

    def sim_2d(self):
        try:
            self.prepare_sim_2d()
        except Exception as e:
            self.logg.error(f"Error preparing 2D SIM stack: {e}")
            self.devs.daq.stop_triggers()
            self.lasers_off()
            return
        try:
            self.devs.slm.activate()
            self.devs.cam_set[self.cameras["imaging"]].start_data_acquisition()
            time.sleep(0.2)
            self.devs.daq.run_triggers()
            time.sleep(0.2)
            self.svd.emit(time.strftime("%Y%m%d%H%M%S") + '_sim_2d')
        except Exception as e:
            self.finish_sim_2d()
            self.logg.error(f"Error running 2D SIM stack: {e}")
            return
        self.finish_sim_2d()

    def finish_sim_2d(self):
        try:
            self.devs.daq.stop_triggers()
            self.devs.cam_set[self.cameras["imaging"]].stop_data_acquisition()
            self.lasers_off()
            self.devs.slm.deactivate()
            self.logg.info("2D SIM image stack acquired")
        except Exception as e:
            self.logg.error(f"Error stopping 2D SIM stack: {e}")

    def run_sim_2d(self, n: int):
        self.vw.get_dialog(txt="2D SIM Acquisition")
        self.run_task(task=self.sim_2d, iteration=n)

    def prepare_sim_3d(self):
        self.lasers = self.ctrl_panel.get_lasers()
        self.set_lasers(self.lasers)
        self.cameras["imaging"] = self.ctrl_panel.get_imaging_camera()
        self.set_camera_roi("imaging")
        self.slm_seq = self.ctrl_panel.get_slm_sequence()
        if self.slm_seq == "None":
            raise ValueError("SLM sequence cannot be None.")
        else:
            self.devs.slm.select_order(self.devs.slm.ord_dict[self.slm_seq])
        self.devs.cam_set[self.cameras["imaging"]].prepare_data_acquisition()
        self.update_trigger_parameters("imaging")
        dtr, ptr, dch, pch, pos = self.trg.generate_sim_3d(self.lasers, self.cameras["imaging"], self.slm_seq)
        self.devs.daq.set_piezo_position(pos=[ptr[0]], indices=[2])
        self.devs.cam_set[self.cameras["imaging"]].acq_num = pos * 5
        self.devs.daq.write_triggers(piezo_sequences=ptr, piezo_channels=pch,
                                     digital_sequences=dtr, digital_channels=dch,
                                     finite=True)
        self.ctrl_panel.display_emccd_timings(clean=self.trg.initial_time,
                                              exposure=self.trg.exposure_time,
                                              standby=self.trg.standby_time)

    def sim_3d(self):
        try:
            self.prepare_sim_3d()
        except Exception as e:
            self.logg.error(f"Error preparing 3D SIM stack: {e}")
            self.devs.daq.stop_triggers()
            self.lasers_off()
            return
        try:
            if self.slm_seq != "None":
                self.devs.slm.activate()
            self.devs.cam_set[self.cameras["imaging"]].start_data_acquisition()
            self.devs.daq.run_triggers()
            time.sleep(0.2)
            self.svd.emit(time.strftime("%Y%m%d%H%M%S") + '_sim_3d')
        except Exception as e:
            self.finish_sim_3d()
            self.logg.error(f"Error running 3D SIM stack: {e}")
            return
        self.finish_sim_3d()

    def finish_sim_3d(self):
        try:
            self.devs.daq.stop_triggers()
            self.devs.cam_set[self.cameras["imaging"]].stop_data_acquisition()
            self.lasers_off()
            if self.slm_seq != "None":
                self.devs.slm.deactivate()
            self.reset_piezo_positions()
            self.logg.info("3D SIM image stack acquired")
        except Exception as e:
            self.logg.error(f"Error stopping 3D SIM stack: {e}")

    def run_sim_3d(self, n: int):
        self.vw.get_dialog(txt="3D SIM Acquisition")
        self.run_task(task=self.sim_3d, iteration=n)

    def prepare_parallel_scan_2d(self):
        self.lasers = self.ctrl_panel.get_lasers()
        self.set_lasers(self.lasers)
        self.cameras["imaging"] = self.ctrl_panel.get_imaging_camera()
        self.set_camera_roi("imaging")
        self.slm_seq = self.ctrl_panel.get_slm_sequence()
        self.devs.slm.select_order(self.devs.slm.ord_dict[self.slm_seq])
        self.devs.cam_set[self.cameras["imaging"]].prepare_data_acquisition()
        self.update_trigger_parameters("imaging")
        dtr, ptr, dch, pch, pos = self.trg.generate_piezo_scan(self.lasers, self.cameras["imaging"],
                                                                   self.slm_seq)
        self.devs.daq.set_piezo_position(pos=list(np.swapaxes(ptr, 0, 1)[0]), indices=pch)
        self.devs.cam_set[self.cameras["imaging"]].acq_num = pos
        self.devs.daq.write_triggers(piezo_sequences=ptr, piezo_channels=pch,
                                     digital_sequences=dtr, digital_channels=dch)
        self.ctrl_panel.display_emccd_timings(clean=self.trg.initial_time,
                                              exposure=self.trg.exposure_time,
                                              standby=self.trg.standby_time)

    def parallel_scan_2d(self):
        try:
            self.prepare_parallel_scan_2d()
        except Exception as e:
            self.logg.error(f"Error preparing monalisa scanning: {e}")
            self.devs.daq.stop_triggers()
            self.lasers_off()
            return
        try:
            self.devs.cam_set[self.cameras["imaging"]].start_data_acquisition()
            time.sleep(0.02)
            self.devs.daq.run_triggers()
            time.sleep(1.)
            self.svd.emit(time.strftime("%Y%m%d%H%M%S") + '_parallel_scanning',
                          self.devs.cam_set[self.cameras["imaging"]].get_data(),
                          list(self.devs.cam_set[self.cameras["imaging"]].data.ind_list),
                          self.trg.piezo_scan_positions)
        except Exception as e:
            self.finish_parallel_scan()
            self.logg.error(f"Error running monalisa scanning: {e}")
            return
        self.finish_parallel_scan()

    def finish_parallel_scan(self):
        try:
            self.devs.cam_set[self.cameras["imaging"]].stop_data_acquisition()
            self.devs.daq.stop_triggers()
            self.lasers_off()
            self.reset_piezo_positions()
            self.logg.info("Monalisa scanning image acquired")
        except Exception as e:
            self.logg.error(f"Error stopping monalisa scanning: {e}")

    def run_parallel_scan(self, n: int):
        self.vw.get_dialog(txt="ParallelScan Acquisition")
        self.run_task(task=self.parallel_scan_2d, iteration=n)
