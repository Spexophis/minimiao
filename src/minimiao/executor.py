# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import os
import time

import numpy as np
import pandas as pd
import tifffile as tf
from PyQt6.QtCore import QObject, pyqtSlot, pyqtSignal, QTimer, Qt
from .utilities import image_processor as ipr
from .utilities import zernike_generator as tz
from . import run_threads, logger


class CommandExecutor(QObject):
    psv = pyqtSignal(str)
    zsv = pyqtSignal(list, object)

    def __init__(self, dev, cwd, cmp, path, config, logg=None, cf=None):
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
        self.config = config
        self.cfd = cf
        self.logg = logg or logger.setup_logging()
        self._set_signal_executions()
        self._initial_setup()
        self.lasers = []
        self.detector = {0: [0, 1], 1: [0]}
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
        # Lasers
        self.ctrl_panel.Signal_set_laser.connect(self.set_laser)
        # DAQ
        self.ctrl_panel.Signal_daq_reset.connect(self.reset_daq_channels)
        self.ctrl_panel.Signal_daq_update.connect(self.update_daq_sample_rate)
        self.ctrl_panel.Signal_plot_trigger.connect(self.plot_trigger)
        # Acquisitions
        self.ctrl_panel.Signal_video.connect(self.video)
        self.ctrl_panel.Signal_data_acquire.connect(self.data_acquisition)
        self.psv.connect(self.save_scan)
        # Deformable Mirror
        self.ao_panel.Signal_push_actuator.connect(self.push_actuator)
        self.ao_panel.Signal_set_zernike.connect(self.set_zernike)
        self.ao_panel.Signal_set_dm.connect(self.set_dm_current)
        self.ao_panel.Signal_set_dm_flat.connect(self.set_dm_flat)
        self.ao_panel.Signal_update_cmd.connect(self.update_dm)
        self.ao_panel.Signal_save_dm.connect(self.save_dm)
        self.ao_panel.Signal_influence_function.connect(self.run_influence_function)
        # WFS
        self.ao_panel.Signal_img_shwfs_base.connect(self.set_reference_wf)
        self.ao_panel.Signal_img_wfs.connect(self.wfs)
        self.ao_panel.Signal_img_shwfr_run.connect(self.run_img_wfr)
        self.ao_panel.Signal_img_shwfs_compute_wf.connect(self.run_wf_decomposition)
        self.zsv.connect(self.save_zernike_coeffs)
        self.ao_panel.Signal_img_shwfs_save_wf.connect(self.save_img_wf)
        # AO
        self.ao_panel.Signal_sensorlessAO_run.connect(self.run_sensorless_iteration)

    def _initial_setup(self):
        try:
            self.reset_galvo_positions()
            self.reset_piezo_positions()
            self.laser_lists = list(self.devs.laser.lasers.keys())
            self.ao_panel.QComboBox_dms.addItem(self.devs.dfm.dm_model)
            for i in range(len(self.devs.dfm.dm_cmd)):
                self.ao_panel.QComboBox_cmd.addItem(f"{i}")
            self.ao_panel.QComboBox_cmd.setCurrentIndex(self.devs.dfm.current_cmd)
            # self.dm_cmd_ind = self.devs.dfm.current_cmd
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
            QTimer.singleShot(100, lambda: self._update_piezo_display_z())
        except Exception as e:
            self.logg.error(f"MCL Piezo Error: {e}")

    def _update_piezo_display_z(self):
        """Update display after piezo has settled"""
        pass
        # try:
        #     position = self.devs.daq.read_piezo_position(2)
        #     self.ctrl_panel.display_piezo_position_x(position)
        # except Exception as e:
        #     self.logg.error(f"MCL Piezo Read Error: {e}")

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

    def set_camera_roi(self):
        try:
            expo = self.ao_panel.get_cmos_exposure()
            self.devs.camera.t_exposure = expo * 1000
            gain = self.ao_panel.get_cmos_gain()
            self.devs.camera.gain = gain
            x, y, nx, ny, bn = self.ao_panel.get_cmos_roi()
            self.devs.camera.pixels_x = nx
            self.devs.camera.start_h = x
            self.devs.camera.pixels_y = ny
            self.devs.camera.start_v = y
            self.devs.camera.bin_h = bn
            self.devs.camera.bin_v = bn
        except Exception as e:
            self.logg.error(f"Camera Error  : {e}")

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

    def update_trigger_parameters(self):
        """Ensure that the camera acquisition is fully set up before executing this function."""
        try:
            self.update_digital_timings()
            self.update_galvo_scanner()
            self.update_piezo_scanner()
            self.trg.update_camera_parameters(initial_time=self.devs.camera.t_clean,
                                              standby_time=self.devs.camera.t_readout)
            self.logg.info(f"Trigger Updated")
        except Exception as e:
            self.logg.error(f"Trigger Error: {e}")

    def prepare_video(self, vd_mod, finite=False):
        self.lasers = self.ctrl_panel.get_lasers()
        self.set_lasers(self.lasers)
        self.update_trigger_parameters()
        dn = self.ctrl_panel.get_detector()
        self.viewer.set_plot_1(dn)
        if vd_mod == "Point Scan":
            dtr, gtr, dch, gch, pos, pdw = self.trg.generate_galvo_scan(self.lasers, self.detector[dn])
            self.devs.daq.write_triggers(analog_sequences=gtr, analog_channels=gch,
                                         digital_sequences=dtr, digital_channels=dch, finite=finite)
            self.devs.daq.photon_counter_mode = 1
            self.devs.daq.psr = self.rec
            self.viewer.psr_mode = True
        elif vd_mod == "Static Point":
            dtr, dch, pdw = self.trg.generate_digital_triggers(self.lasers, self.detector[dn])
            self.devs.daq.write_triggers(digital_sequences=dtr, digital_channels=dch, finite=finite)
            self.devs.daq.photon_counter_mode = 0
            self.viewer.psr_mode = False
        else:
            raise Exception(f"Invalid video mode {vd_mod} for MPD")
        self.rec.point_scan_gate_mask = dtr[-1]
        self.rec.set_point_scan_params(n_lines=self.trg.galvo_scan_pos[1], n_pixels=self.trg.galvo_scan_pos[0],
                                       dwell_samples=pdw)
        self.rec.prepare_point_scan_live_recon()
        self.devs.daq.photon_counter_length = dtr.shape[1]
        self.devs.daq.prepare_photon_counter(2 - dn)
        if dn:
            self.devs.daq.prepare_pmt_reader()
        self.viewer.photon_pool.reset_buffer(max_len=self.devs.daq.photon_counter_length,
                                             dt_s=1 / self.devs.daq.sample_rate,
                                             px=(self.trg.galvo_scan_pos[1], self.trg.galvo_scan_pos[0]))
        if getattr(self.viewer, "psr_worker", None) is None:
            self.viewer.psr_worker = run_threads.PSLiveWorker(self.rec, self.devs.daq.mpd_data, self.devs.daq.pmt_data,
                                                              fps=10, parent=self.viewer)
            self.viewer.psr_worker.psr_ready.connect(self.viewer.photon_pool.new_acquire)
            self.viewer.psr_worker.psr_new.connect(self.viewer.on_psr_frame)
            self.viewer.psr_worker.start()

    @pyqtSlot(bool, str)
    def video(self, sw: bool, md: str):
        if sw:
            try:
                self.prepare_video(md)
                self.logg.info(f"Finish preparing video")
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
            self.devs.daq.run_triggers()
            self.viewer.stream_trace(self.viewer.photon_pool.xt, self.viewer.photon_pool.buf_0,
                                     self.viewer.photon_pool.buf_1)
            self.logg.info("Live Video Started")
        except Exception as e:
            self.logg.error(f"Error starting imaging video: {e}")
            self.stop_video()
            return

    def stop_video(self):
        try:
            if getattr(self.viewer, "psr_worker", None) is not None:
                self.viewer.psr_worker.stop()
                self.viewer.psr_worker = None
            self.devs.daq.stop_photon_count()
            self.devs.daq.stop_triggers()
            self.lasers_off()
            self.reset_galvo_positions()
            self.reset_piezo_positions()
            self.logg.info(r"Live Video Stopped")
        except Exception as e:
            self.logg.error(f"Error stopping imaging video: {e}")

    def _cleanup_psr_worker(self):
        """Properly cleanup psr worker"""
        worker = getattr(self.viewer, "psr_worker", None)
        if worker is not None:
            # Stop the worker thread
            worker.stop()

            # Disconnect all signals
            try:
                worker.psr_ready.disconnect()
                worker.psr_new.disconnect()
            except TypeError:
                pass  # Already disconnected

            # Clear data references
            worker.clear_data()

            # Delete worker
            worker.deleteLater()  # Qt will delete when safe
            self.viewer.psr_worker = None

    @pyqtSlot()
    def plot_trigger(self):
        try:
            self.update_trigger_parameters()
            dtr, dch, dwl = self.trg.generate_digital_triggers([0, 1, 2], [0, 1, 2])
            self.viewer.plot_trace(dtr[0], overlay=False)
            for i in range(dtr.shape[0] - 1):
                self.viewer.plot_trace(dtr[i + 1] + i + 1, overlay=True)
        except Exception as e:
            self.logg.error(f"Error plotting digital triggers: {e}")

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
        # elif acq_mod == "Static Point":
        #     self.run_static_point(acq_num)
        else:
            self.logg.error(f"Invalid video mode")

    @pyqtSlot(str)
    def save_scan(self, tm: str):
        fn = self.vw.get_file_dialog()
        if fn is not None:
            fd = os.path.join(self.path, tm + '_' + fn)
        else:
            fd = os.path.join(self.path, tm)
        img_res = np.array(self.rec.live_rec).astype(np.float16)
        tf.imwrite(str(fd + r"_recon_image.tif"), data=img_res, compression='zlib')
        try:
            res = np.zeros((4, self.rec.gate_len))
            res[0] = np.arange(self.rec.gate_len) / self.trg.sample_rate
            res[1] = self.rec.point_scan_gate_mask * 1
            res[2] = np.array(self.devs.daq.mpd_data.count_lists[0])
            res[3] = np.array(self.devs.daq.mpd_data.count_lists[1])
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
        self.devs.daq.mpd_data = None

    def prepare_point_scan(self, tim):
        self.lasers = self.ctrl_panel.get_lasers()
        self.set_lasers(self.lasers)
        self.update_trigger_parameters()
        dtr, gtr, dch, gch, pos, pdw = self.trg.generate_galvo_scan(self.lasers, [0, 1])
        self.devs.daq.write_triggers(analog_sequences=gtr, analog_channels=gch,
                                     digital_sequences=dtr, digital_channels=dch, finite=True)
        self.devs.daq.photon_counter_mode = 1
        self.devs.daq.psr = self.rec
        self.rec.point_scan_gate_mask = dtr[-1]
        self.rec.set_point_scan_params(n_lines=self.trg.galvo_scan_pos[1], n_pixels=self.trg.galvo_scan_pos[0],
                                       dwell_samples=pdw)
        self.rec.prepare_point_scan_live_recon()
        self.devs.daq.photon_counter_length = dtr.shape[1]
        self.devs.daq.prepare_photon_counter()
        fd = os.path.join(self.path, tim + r"_point_scanning_triggers.npy")
        np.save(str(fd), np.vstack((np.array(gtr), np.array(dtr))))

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

    @pyqtSlot(int, float)
    def push_actuator(self, n: int, a: float):
        try:
            values = [0.] * self.devs.dfm.n_actuator
            values[n] = a
            self.devs.dfm.set_dm(self.devs.dfm.cmd_add(values, self.devs.dfm.dm_cmd[self.devs.dfm.current_cmd]))
        except Exception as e:
            self.logg.error(f"DM Error: {e}")

    @pyqtSlot(str, int, float)
    def set_zernike(self, md: str, iz: int, amp: float, factory=False):
        try:
            if factory:
                self.devs.dfm.set_dm(
                    self.devs.dfm.cmd_add([i * amp for i in self.devs.dfm.z2c[iz]], self.devs.dfm.dm_cmd[self.devs.dfm.current_cmd]))
            else:
                self.devs.dfm.set_dm(
                    self.devs.dfm.cmd_add(self.devs.dfm.get_zernike_cmd(iz, amp, md), self.devs.dfm.dm_cmd[self.devs.dfm.current_cmd]))
        except Exception as e:
            self.logg.error(f"DM Error: {e}")

    @pyqtSlot(int)
    def set_dm_current(self, i: int):
        try:
            self.devs.dfm.set_dm(self.devs.dfm.dm_cmd[i])
            self.devs.dfm.current_cmd = i
        except Exception as e:
            self.logg.error(f"DM Error: {e}")

    @pyqtSlot()
    def set_dm_flat(self):
        if int(self.ao_panel.get_cmd_index()) == self.devs.dfm.current_cmd:
            self.devs.dfm.write_flat_cmd(t=time.strftime("%Y_%m_%d_%H_%M"), cmd=self.devs.dfm.dm_cmd[self.devs.dfm.current_cmd])

    @pyqtSlot()
    def update_dm(self):
        try:
            self.devs.dfm.dm_cmd.append(self.devs.dfm.temp_cmd[-1])
            self.ao_panel.update_cmd_index()
            self.devs.dfm.set_dm(self.devs.dfm.dm_cmd[-1])
        except Exception as e:
            self.logg.error(f"DM Error: {e}")

    @pyqtSlot()
    def save_dm(self):
        try:
            t = time.strftime("%Y%m%d_%H%M%S_")
            self.devs.dfm.write_cmd(self.path, t, flatfile=False)
            self.logg.info('DM cmd saved')
        except Exception as e:
            self.logg.error(f"DM Error: {e}")

    def set_img_wfs(self):
        parameters = self.ao_panel.get_parameters_foc()
        self.wfr.pixel_size = 3.45 / 1000
        self.wfr.update_parameters(parameters)
        self.logg.info('SHWFS parameter updated')

    def prepare_wfs(self):
        self.set_img_wfs()
        self.set_camera_roi()
        self.devs.camera.prepare_live()
        self.viewer.switch_camera(self.devs.camera.pixels_y, self.devs.camera.pixels_x)

    @pyqtSlot(bool)
    def wfs(self, sw: bool):
        if sw:
            try:
                self.prepare_wfs()
                self.logg.info(f"Finish preparing wfs")
            except Exception as e:
                self.logg.error(f"Error preparing wfs: {e}")
                return
            self.start_wfs()
        else:
            self.stop_wfs()

    def start_wfs(self):
        try:
            self.devs.camera.start_live()
            self.devs.camera.data.on_update(self.viewer.on_camera_update_from_thread)
            self.logg.info("WFS Started")
        except Exception as e:
            self.logg.error(f"Error starting wfs: {e}")
            self.stop_video()
            return

    def stop_wfs(self):
        try:
            self.devs.camera.stop_live()
            self.logg.info(r"WFS Stopped")
        except Exception as e:
            self.logg.error(f"Error stopping wfs: {e}")

    @pyqtSlot()
    def set_reference_wf(self):
        try:
            self.wfr.ref = self.devs.camera.get_last_image()
            self.logg.info('shwfs base set')
        except Exception as e:
            self.logg.error(f"Error setting shwfs base: {e}")

    @pyqtSlot(bool)
    def run_img_wfr(self, on: bool):
        self.wfr.method = self.ao_panel.get_gradient_method_img()
        if on:
            if getattr(self.viewer, "wfr_worker", None) is None:
                self.viewer.wfr_worker = run_threads.WFRWorker(fps=8, op=self.wfr, parent=self.viewer)
                self.viewer.wfr_worker.wfr_ready.connect(self.viewer.on_wfr_frame, Qt.ConnectionType.QueuedConnection)
                self.viewer.wfr_worker.wfr_ready.connect(self.show_wf_metric)
                self.viewer.wfr_worker.start()
            self.viewer.wfr_mode = True
        else:
            self.viewer.wfr_mode = False
            if getattr(self.viewer, "wfr_worker", None) is not None:
                self.viewer.wfr_worker.stop()
                self.viewer.wfr_worker = None

    def show_wf_metric(self, wf_img):
        try:
            self.ao_panel.display_img_wf_properties(ipr.img_properties(wf_img))
        except Exception as e:
            self.logg.error(f"SHWFS Wavefront Show Error: {e}")

    @pyqtSlot(bool)
    def run_wf_decomposition(self, on: bool):
        if on:
            self.viewer.wfr_decomp = True
        else:
            self.viewer.wfr_decomp = False

    @pyqtSlot(list, object)
    def save_zernike_coeffs(self, zdx: list, za: object):
        df = pd.DataFrame({'mods': zdx, 'amps': za})
        fn = self.vw.get_file_dialog()
        if fn is not None:
            file_path = fn + '_' + time.strftime("%Y%m%d%H%M%S")
        else:
            file_path = os.path.join(self.path, time.strftime("%Y%m%d%H%M%S"))
        df.to_excel(file_path + '_zernike_coefficients.xlsx', index=False)

    @pyqtSlot()
    def save_img_wf(self):
        fn = self.vw.get_file_dialog()
        if fn is not None:
            file_name = os.path.join(self.path, time.strftime("%Y%m%d%H%M%S") + "_" + fn)
        else:
            file_name = os.path.join(self.path, time.strftime("%Y%m%d%H%M%S"))
        self.wfr.save_wfs_results(file_name, self.devs.dfm)

    def influence_function(self):
        try:
            self.prepare_wfs()
        except Exception as e:
            self.logg.error(f"Error preparing influence function: {e}")
            return
        try:
            fd = os.path.join(self.path, time.strftime("%Y%m%d%H%M") + '_influence_function')
            os.makedirs(fd, exist_ok=True)
            self.logg.info(f'Directory {fd} has been created successfully.')
        except Exception as er:
            self.logg.error(f'Error creating influence function directory: {er}')
            return
        try:
            n, amp = self.ao_panel.get_actuator()
            self.devs.camera.start_live()
            time.sleep(0.02)
            for i in range(self.devs.dfm.n_actuator):
                shimg = []
                self.vw.dialog_text.setText(f"actuator {i}")
                values = [0.] * self.devs.dfm.n_actuator
                self.devs.dfm.set_dm(values)
                time.sleep(0.1)
                shimg.append(self.devs.camera.get_last_image())

                values[i] = amp
                self.devs.dfm.set_dm(values)
                time.sleep(0.1)
                shimg.append(self.devs.camera.get_last_image())

                values = [0.] * self.devs.dfm.n_actuator
                self.devs.dfm.set_dm(values)
                time.sleep(0.1)
                shimg.append(self.devs.camera.get_last_image())

                values[i] = - amp
                self.devs.dfm.set_dm(values)
                time.sleep(0.1)
                shimg.append(self.devs.camera.get_last_image())

                tf.imwrite(fd + r'/' + 'actuator_' + str(i) + '_push_' + str(amp) + '.tif', np.asarray(shimg))
        except Exception as e:
            self.logg.error(f"Error running influence function: {e}")
            self.stop_wfs()
            return
        try:
            self.vw.dialog_text.setText(f"computing influence function")
            dmn = self.ao_panel.QComboBox_dms.currentText()
            self.wfr.generate_influence_matrices(data_folder=fd, dm=self.devs.dfm, sv=self.config, cfd=self.cfd)
        except Exception as e:
            self.logg.error(f"Error computing influence function: {e}")
            self.stop_wfs()
            return
        self.stop_wfs()

    @pyqtSlot()
    def run_influence_function(self):
        self.vw.get_dialog(txt="Influence Function")
        self.run_task(self.influence_function)

    def sensorless_iteration(self, dms):
        ims = []
        for dmsp in dms:
            self.devs.dfm.set_dm(dmsp)
            time.sleep(0.016)
            self.devs.daq.run_triggers()
            time.sleep(0.032)
            self.devs.daq.stop_triggers(_close=False)
            ims.append(self.devs.camera.get_last_image())
        return ims

    def sensorless_iterations(self):
        try:
            lpr, hpr, slf, mf, err = self.ao_panel.get_ao_parameters()
            name = time.strftime("%Y%m%d_%H%M%S_") + self.devs.dfm.dm_serial + '_ao_iterations_' + mf
            new_folder = os.path.join(self.path, name)
            os.makedirs(new_folder, exist_ok=True)
            self.logg.info(f'Directory {new_folder} has been created successfully.')
        except Exception as e:
            self.logg.error(f'Error creating directory for sensorless iteration: {e}')
            return
        try:
            vd_mod = self.ctrl_panel.get_live_mode()
            self.prepare_video(vd_mod, True)
        except Exception as e:
            self.logg.error(f"Prepare sensorless iteration Error: {e}")
            return
        try:
            mode_start, mode_stop, amp_start, amp_step, amp_step_number = self.ao_panel.get_ao_iteration()
            md = self.ao_panel.get_img_wfs_method()
            amprange = [amp_start + step_number * amp_step for step_number in range(amp_step_number)]
            results = [('Mode', 'Amp', 'Metric')]
            za = []
            mv = []
            zp = [0] * self.devs.dfm.n_zernike
            cmd = self.devs.dfm.dm_cmd[self.devs.dfm.current_cmd]
            self.devs.camera.start_live()
            time.sleep(0.1)
            self.logg.info("Sensorless AO iterations start")
            self.devs.dfm.set_dm(cmd)
            time.sleep(0.016)
            if err:
                images = []
                for i in range(8):
                    self.devs.daq.run_triggers()
                    time.sleep(0.032)
                    self.devs.daq.stop_triggers(_close=False)
                    images.append(self.devs.camera.get_last_image())
                if mf == "Max(Intensity)":
                    mts = [img.max() for img in images]
                if mf == "Sum(Intensity)":
                    mts = [img.sum() for img in images]
                if mf == "SNR(FFT)":
                    mts = [ipr.snr(img, lpr, hpr, True) for img in images]
                if mf == "HighPass(FFT)":
                    mts = [ipr.hpf(img, hpr) for img in images]
                if mf == "Selected(FFT)":
                    mts = [ipr.selected_frequency(img, [slf, 2 * slf]) for img in images]
                std = np.std(mts)
                fn = new_folder + r"\original.tiff"
                tf.imwrite(str(fn), np.asarray(images))
            else:
                self.devs.daq.run_triggers()
                time.sleep(0.032)
                self.devs.daq.stop_triggers(_close=False)
                fn = new_folder + r"\original.tiff"
                tf.imwrite(str(fn), self.devs.camera.get_last_image())
            for mode in range(mode_start, mode_stop + 1):
                self.v.dialog_text.setText(f"Zernike mode #{mode}")
                labels = ["zm%0.2d_amp%.4f" % (mode, amp) for amp in amprange]
                cmds = [self.devs.dfm.cmd_add(self.devs.dfm.get_zernike_cmd(mode, amp, method=md), cmd) for amp in amprange]
                images = self.sensorless_iteration(cmds)
                if mf == "Max(Intensity)":
                    mts = [img.max() for img in images]
                if mf == "Sum(Intensity)":
                    mts = [img.sum() for img in images]
                if mf == "SNR(FFT)":
                    mts = [ipr.snr(img, lpr, hpr, True) for img in images]
                if mf == "HighPass(FFT)":
                    mts = [ipr.hpf(img, hpr) for img in images]
                if mf == "Selected(FFT)":
                    mts = [ipr.selected_frequency(img, [slf, 2 * slf]) for img in images]
                self.logg.info(f"zernike mode #{mode}, ({amprange}), ({mts})")
                self.sig_plt.emit(amprange, mts)
                if err:
                    mts_err = [std] * len(mts)
                    pm = ipr.peak_find(amprange, mts, mts_err)
                else:
                    pm = ipr.peak_find(amprange, mts)
                if isinstance(pm, str):
                    self.logg.error(f"zernike mode #{mode} " + pm)
                else:
                    zp[mode] = pm
                    cmd = self.devs.dfm.cmd_add(self.devs.dfm.get_zernike_cmd(mode, pm, method=md), cmd)
                    self.devs.dfm.set_dm(cmd)
                    self.logg.info("set mode %d at value of %.4f" % (mode, pm))
                for amp, mt in zip(amprange, mts):
                    results.append((mode, amp, mt))
                za.extend(amprange)
                mv.extend(mts)
                fn = os.path.join(str(new_folder), f"zernike mode #{mode}.tiff")
                with tf.TiffWriter(fn) as tif:
                    for img, label in zip(images, labels):
                        tif.write(img, description=label)
            self.devs.dfm.set_dm(cmd)
            time.sleep(0.016)
            self.devs.daq.run_triggers()
            time.sleep(0.032)
            self.devs.daq.stop_triggers(_close=False)
            fn = new_folder + r"\final.tiff"
            tf.imwrite(str(fn), self.devs.camera.get_last_image())
            self.devs.dfm.dm_cmd.append(cmd)
            self.ao_panel.update_cmd_index()
            i = int(self.ao_panel.get_cmd_index())
            self.devs.dfm.current_cmd = i
            self.devs.dfm.write_cmd(new_folder, '_')
            self.devs.dfm.save_sensorless_results(os.path.join(str(new_folder), 'results.xlsx'), za, mv, zp)
        except Exception as e:
            self.stop_video()
            self.logg.error(f"Sensorless AO Error: {e}")
            return
        self.stop_video()

    @pyqtSlot()
    def run_sensorless_iteration(self):
        self.vw.get_dialog(txt="Sensorless Iteration")
        self.run_task(task=self.sensorless_iterations)
