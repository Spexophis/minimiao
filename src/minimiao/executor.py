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
        self.path = path
        self.logg = logger or self.setup_logging()
        self._set_signal_executions()
        self._initial_setup()
        self.lasers = []
        self.cameras = {"imaging": 0, "wfs": 0}
        self.task_worker = None

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    def _set_signal_executions(self):
        # Galvo
        self.ctrl_panel.Signal_galvo_set.connect(self.set_galvo)
        # Piezo
        self.ctrl_panel.Signal_piezo_move.connect(self.set_piezo_positions)
        self.ctrl_panel.Signal_focus_finding.connect(self.run_focus_finding)
        # Lasers
        self.ctrl_panel.Signal_set_laser.connect(self.set_laser)
        # DAQ
        self.ctrl_panel.Signal_daq_reset.connect(self.reset_daq_channels)
        self.ctrl_panel.Signal_daq_update.connect(self.update_daq_sample_rate)
        # Acquisitions
        self.ctrl_panel.Signal_video.connect(self.video)
        self.ctrl_panel.Signal_fft.connect(self.fft)
        self.ctrl_panel.Signal_plot_profile.connect(self.profile_plot)
        self.ctrl_panel.Signal_add_profile.connect(self.plot_add)
        self.ctrl_panel.Signal_data_acquire.connect(self.data_acquisition)
        self.svd.connect(self.save_data)
        self.psv.connect(self.save_scan)

    def _initial_setup(self):
        try:
            self.reset_galvo_positions()
            self.reset_piezo_positions()
            self.laser_lists = list(self.devs.laser.lasers.keys())
            self.logg.info("Finish setting up controllers")
        except Exception as e:
            self.logg.error(f"Initial setup Error: {e}")

    def reset_galvo_positions(self):
        g_x, g_y = self.ctrl_panel.get_galvo_positions()
        try:
            self.devs.daq.set_galvo_position([g_x, g_y], [0, 1])
        except Exception as e:
            self.logg.error(f"Galvo Error: {e}")

    @pyqtSlot(float, float)
    def set_galvo(self, volt_x: float, volt_y: float):
        try:
            self.devs.daq.set_galvo_position([volt_x, volt_y], [0, 1])
        except Exception as e:
            self.logg.error(f"Galvo Error: {e}")

    @pyqtSlot()
    def update_galvo_scanner(self):
        galvo_positions, galvo_ranges, dot_pos, offset, ret = self.ctrl_panel.get_galvo_scan_parameters()
        self.trg.update_galvo_scan_parameters(origins=galvo_positions, ranges=galvo_ranges, foci=dot_pos,
                                              offsets=offset, returns=ret)

    def reset_piezo_positions(self):
        pos_z = self.ctrl_panel.get_piezo_positions()
        self.set_piezo_position_z(pos_z)

    @pyqtSlot(str, float)
    def set_piezo_positions(self, axis: str, value_z: float):
        if axis == "z":
            self.set_piezo_position_z(value_z)

    def set_piezo_position_z(self, pos_z):
        try:
            self.devs.daq.set_piezo_position([pos_z / 10.], [0])
            time.sleep(0.1)
        except Exception as e:
            self.logg.error(f"MCL Piezo Error: {e}")

    def update_piezo_scanner(self):
        axis_origins, axis_lengths, step_sizes = self.ctrl_panel.get_piezo_scan_parameters()
        return_time = self.ctrl_panel.get_piezo_scan_time()
        self.trg.update_piezo_scan_parameters(axis_lengths, step_sizes, axis_origins, return_time)

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
                expo = self.ctrl_panel.get_cmos_exposure()
                self.devs.cam_set[0].set_exposure(expo)
                x, y, nx, ny, bx, by = self.ctrl_panel.get_cmos_roi()
                self.devs.cam_set[0].set_roi(x, y, nx, ny)
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

    def update_digital_timings(self):
        digital_starts, digital_ends = self.ctrl_panel.get_digital_parameters()
        self.trg.update_digital_parameters(digital_starts, digital_ends)

    def update_trigger_parameters(self, cam_key):
        """Ensure that the camera acquisition is fully set up before executing this function."""
        try:
            self.update_digital_timings()
            self.update_galvo_scanner()
            self.update_piezo_scanner()
            if self.cameras[cam_key] == 0:
                self.trg.update_camera_parameters(initial_time=self.devs.cam_set[self.cameras[cam_key]].t_clean,
                                                  standby_time=self.devs.cam_set[self.cameras[cam_key]].t_readout)
            self.logg.info(f"Trigger Updated")
        except Exception as e:
            self.logg.error(f"Trigger Error: {e}")

    def prepare_video(self, vd_mod):
        self.lasers = self.ctrl_panel.get_lasers()
        self.set_lasers(self.lasers)
        self.cameras["imaging"] = self.ctrl_panel.get_imaging_camera()
        if self.cameras["imaging"] == 0:
            self.set_camera_roi("imaging")
            self.devs.cam_set[self.cameras["imaging"]].prepare_live()
            self.update_trigger_parameters("imaging")
            self.viewer.switch_camera(self.devs.cam_set[self.cameras["imaging"]].pixels_x,
                                      self.devs.cam_set[self.cameras["imaging"]].pixels_y)
            if vd_mod == "Wide Field":
                dtr, chs = self.trg.generate_digital_triggers(self.lasers, self.cameras["imaging"])
                self.devs.daq.write_triggers(digital_sequences=dtr, digital_channels=chs, finite=False)
        elif self.cameras["imaging"] > 0:
            self.update_trigger_parameters("imaging")
            if vd_mod == "Wide Field":
                dtr, dch = self.trg.generate_digital_triggers(self.lasers, self.cameras["imaging"])
                dwl = self.trg.exposure_samples
                self.devs.daq.write_triggers(digital_sequences=dtr, digital_channels=dch, finite=False)
                self.devs.daq.photon_counter_mode = 0
                self.viewer.psr_mode = False
            elif vd_mod == "Point Scan":
                ptr, pch, dtr, dch, dwl = self.trg.generate_galvo_point_scan_2d(self.lasers)
                self.devs.daq.write_triggers(piezo_sequences=ptr, piezo_channels=pch,
                                             digital_sequences=dtr, digital_channels=dch, finite=False)
                self.devs.daq.photon_counter_mode = 1
                self.devs.daq.psr = self.rec
                self.viewer.psr_mode = True
            else:
                raise Exception(f"Invalid video mode {vd_mod} for MPD")
            self.rec.point_scan_gate_mask = dtr[-1]
            self.rec.set_point_scan_params(n_lines=self.trg.galvo_scan_pos[1], n_pixels=self.trg.galvo_scan_pos[0],
                                           dwell_samples=dwl)
            self.rec.prepare_point_scan_live_recon()
            self.devs.daq.photon_counter_length = dtr.shape[1]
            self.devs.daq.prepare_photon_counter()
            self.viewer.photon_pool.reset_buffer(max_len=self.devs.daq.photon_counter_length,
                                                 dt_s=1 / self.devs.daq.sample_rate,
                                                 px=(self.trg.galvo_scan_pos[1], self.trg.galvo_scan_pos[0]))
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
            if self.cameras["imaging"] == 0:
                self.devs.cam_set[self.cameras["imaging"]].start_live()
                self.devs.cam_set[self.cameras["imaging"]].data.on_update(self.viewer.on_camera_update_from_thread)
                self.devs.daq.run_triggers()
            else:
                self.devs.daq.run_triggers()
                self.viewer.stream_trace(self.viewer.photon_pool.xt, self.viewer.photon_pool.buf_0, self.viewer.photon_pool.buf_1)
            self.logg.info("Live Video Started")
        except Exception as e:
            self.logg.error(f"Error starting imaging video: {e}")
            self.stop_video()
            return

    def stop_video(self):
        try:
            if self.cameras["imaging"] == 0:
                self.devs.cam_set[self.cameras["imaging"]].stop_live()
                self.devs.daq.stop_triggers()
            else:
                if getattr(self.viewer, "psr_worker", None) is not None:
                    self.viewer.psr_worker.stop()
                    self.viewer.psr_worker = None
                self.devs.daq.stop_photon_count()
                self.devs.daq.stop_triggers()
            self.lasers_off()
            self.reset_piezo_positions()
            self.logg.info(r"Live Video Stopped")
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
            self.viewer.plot_trace(ipr.get_profile(self.viewer.image_viewer._display_frame, ax, norm=True),
                                   overlay=True)
        except Exception as e:
            self.logg.error(f"Error plotting profile: {e}")

    @pyqtSlot()
    def plot_trigger(self):
        try:
            self.update_trigger_parameters("imaging")
            dtr, dch = self.trg.generate_digital_triggers(self.lasers, self.cameras["imaging"])
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
        self.devs.cam_set[self.cameras["imaging"]].prepare_live()
        self.update_trigger_parameters("imaging")
        dtr, chs = self.trg.generate_digital_triggers(self.lasers, self.cameras["imaging"])
        self.devs.daq.write_triggers(digital_sequences=dtr, digital_channels=chs, finite=True)

    def focus_finding(self):
        try:
            self.prepare_focus_finding()
        except Exception as e:
            self.logg.error(f"Error starting focus finding: {e}")
            self.devs.daq.stop_triggers()
            self.lasers_off()
            return
        try:
            pos_z = self.ctrl_panel.get_piezo_positions()
            center_pos, axis_length, step_size = pos_z[0], 0.96, 0.06
            start = center_pos - axis_length
            end = center_pos + axis_length
            zps = np.arange(start, end + step_size, step_size)
            data = []
            pzs = []
            self.devs.cam_set[self.cameras["imaging"]].start_live()
            for i, z in enumerate(zps):
                self.set_piezo_position_z(z)
                time.sleep(0.1)
                self.devs.daq.run_triggers()
                time.sleep(0.04)
                self.devs.daq.stop_triggers(_close=False)
                temp = self.devs.cam_set[self.cameras["imaging"]].get_last_image()
                data.append(temp)
                pzs.append(ipr.calculate_focus_measure_with_sobel(temp - temp.min()))
            fd = os.path.join(self.path, time.strftime("%Y%m%d%H%M%S") + '_widefield_zstack.tif')
            tf.imwrite(fd, np.asarray(data))
            self.viewer.update_plot(pzs, x=zps)
            fp = ipr.peak_find(zps, pzs)
            if isinstance(fp, str):
                self.logg.error(fp)
            else:
                self.ctrl_panel.QDoubleSpinBox_stage_z_usb.setValue(fp)
        except Exception as e:
            self.finish_focus_finding()
            self.logg.error(f"Error running focus finding: {e}")
            return
        self.finish_focus_finding()

    def finish_focus_finding(self):
        try:
            self.devs.daq.stop_triggers()
            self.devs.cam_set[self.cameras["imaging"]].stop_live()
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
        self.devs.cam_set[self.cameras["imaging"]].prepare_live()
        self.update_trigger_parameters("imaging")
        dtr, ptr, dch, pch, pos = self.trg.generate_piezo_scan(self.lasers, self.cameras["imaging"])
        self.devs.daq.set_piezo_position(pos=[ptr[0]], indices=[2])
        self.devs.cam_set[self.cameras["imaging"]].acq_num = pos
        self.devs.daq.write_triggers(piezo_sequences=ptr, piezo_channels=pch,
                                     digital_sequences=dtr, digital_channels=dch,
                                     finite=True)
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
            self.lasers_off()
            self.reset_piezo_positions()
            self.logg.info("Point scanning image acquired")
        except Exception as e:
            self.logg.error(f"Error stopping point scanning: {e}")

    def run_point_scan(self, n: int):
        self.vw.get_dialog(txt="Point Scanning Acquisition")
        self.run_task(task=self.point_scan, iteration=n)
