# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import os
import time

import numpy as np
import pandas as pd
import tifffile as tf
from PyQt6.QtCore import QObject, pyqtSlot, Qt, pyqtSignal, QTimer

try:
    from . import run_threads, logger
except ImportError as e:
    from minimiao import run_threads, logger

try:
    from .utilities import image_processor as ipr
except ImportError as e:
    from minimiao.utilities import image_processor as ipr


class CommandExecutor(QObject):
    svd = pyqtSignal(str)
    sig_plt = pyqtSignal(list, list)
    sig_auto_focus = pyqtSignal(float)

    def __init__(self, dev, cwd, cmp, path, logg=None):
        super().__init__()
        self.devs = dev
        self.vw = cwd
        self.ctrl_panel = self.vw.ctrl_panel
        self.viewer = self.vw.viewer
        self.ao_panel = self.vw.ao_panel
        self.trg = cmp.trg
        self.path = path
        self.logg = logg or logger.setup_logging()
        self._set_signal_executions()
        self._initial_setup()
        self.lasers = []
        self.slm_seq = ""
        self.task_worker = None

    def _set_signal_executions(self):
        # EMCCD
        self.ctrl_panel.Signal_check_emccd_temperature.connect(self.check_emdccd_temperature)
        self.ctrl_panel.Signal_switch_emccd_cooler.connect(self.switch_emdccd_cooler)
        # MCL Mad Deck
        self.ctrl_panel.Signal_deck_read_position.connect(self.deck_read_position)
        self.ctrl_panel.Signal_deck_zero_position.connect(self.deck_zero_position)
        self.ctrl_panel.Signal_deck_move_single_step.connect(self.move_deck_single_step)
        self.ctrl_panel.Signal_deck_move_continuous.connect(self.move_deck_continuous)
        # MCL Piezo
        self.ctrl_panel.Signal_piezo_move_usb.connect(self.set_piezo_positions_usb)
        self.ctrl_panel.Signal_piezo_move.connect(self.set_piezo_positions)
        self.ctrl_panel.Signal_focus_finding.connect(self.run_focus_finding)
        self.sig_auto_focus.connect(self.ctrl_panel.QDoubleSpinBox_stage_z_usb.setValue)
        # self.ctrl_panel.Signal_focus_locking.connect(self.run_focus_locking)
        # Cobolt Lasers
        self.ctrl_panel.Signal_set_laser.connect(self.set_laser)
        # Thorlabs Motor
        self.ctrl_panel.Signal_motor_home.connect(self.home_motor)
        self.ctrl_panel.Signal_motor_step.connect(self.set_motor_step)
        self.ctrl_panel.Signal_motor_move.connect(self.move_motor)
        # DAQ
        self.ctrl_panel.Signal_daq_reset.connect(self.reset_daq_channels)
        self.ctrl_panel.Signal_daq_update.connect(self.update_daq_sample_rate)
        # Acquisition
        self.ctrl_panel.Signal_video.connect(self.video)
        self.ctrl_panel.Signal_fft.connect(self.fft)
        self.ctrl_panel.Signal_dpc.connect(self.dpc)
        self.ctrl_panel.Signal_dpc_acquire.connect(self.dpc_acquisition)
        self.ctrl_panel.Signal_plot_profile.connect(self.profile_plot)
        self.ctrl_panel.Signal_add_profile.connect(self.plot_add)
        self.ctrl_panel.Signal_data_acquire.connect(self.acquisition)
        self.svd.connect(self.save_data)
        self.sig_plt.connect(self.viewer.plot_trace)
        # DM
        self.ao_panel.Signal_set_zernike.connect(self.set_zernike)
        self.ao_panel.Signal_set_dm.connect(self.set_dm_current)
        self.ao_panel.Signal_set_dm_flat.connect(self.set_dm_flat)
        self.ao_panel.Signal_update_cmd.connect(self.update_dm)
        self.ao_panel.Signal_save_dm.connect(self.save_dm)
        # AO
        self.ao_panel.Signal_sensorlessAO.connect(self.run_sensorless_iteration)

    def _initial_setup(self):
        try:
            self._update_deck_display()
        except Exception as e:
            self.logg.error(f"Initial setup Error: {e}")
        try:
            self.reset_piezo_positions()
        except Exception as e:
            self.logg.error(f"Initial setup Error: {e}")
        try:
            self.devs.motor.move_to(0)
            self.devs.motor.set_velocity(100)
            self.set_motor_step(14.5)
        except Exception as e:
            self.logg.error(f"Initial setup Error: {e}")
        try:
            for key in self.devs.slm.ord_dict.keys():
                self.ctrl_panel.QComboBox_slm_sequence.addItem(key)
        except Exception as e:
            self.logg.error(f"Initial setup Error: {e}")
        try:
            self.ao_panel.update_dm_display(self.devs.dfm.dpp_cmd[self.devs.dfm.current_cmd])
        except Exception as e:
            self.logg.error(f"Initial setup Error: {e}")
        self.logg.info("Finish setting up controllers")

    @pyqtSlot()
    def check_emdccd_temperature(self):
        try:
            self.devs.img_cam.get_temperature()
            self.ctrl_panel.display_emccd_temperature(self.devs.img_cam.temperature)
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
            self.devs.img_cam.cooler_on()
        except Exception as e:
            self.logg.error(f"CCD Camera Error: {e}")

    def switch_emdccd_cooler_off(self):
        try:
            self.devs.img_cam.cooler_off()
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
                self.devs.deck.move_relative(axis=3, distance=self.devs.deck.step_distance, velocity=self.devs.deck.velocity_min)
                QTimer.singleShot(200, lambda: self._update_deck_display())
        except Exception as e:
            self.logg.error(f"MadDeck Error: {e}")

    def move_deck_down(self):
        try:
            _moving = self.devs.deck.is_moving()
            if _moving:
                self.logg.info("MadDeck is moving")
            else:
                self.devs.deck.move_relative(axis=3, distance=-self.devs.deck.step_distance, velocity=self.devs.deck.velocity_min)
                QTimer.singleShot(200, lambda: self._update_deck_display())
        except Exception as e:
            self.logg.error(f"MadDeck Error: {e}")

    @pyqtSlot(bool, int, float)
    def move_deck_continuous(self, moving: bool, direction: int, velocity: float):
        if moving:
            self.devs.deck.move_deck(direction, velocity)
        else:
            self.devs.deck.stop_deck()
            QTimer.singleShot(200, lambda: self._update_deck_display())

    def _update_deck_display(self):
        self.ctrl_panel.display_deck_position(self.devs.deck.position)

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
            self.logg.error(f"MCL Piezo {port} Error: {e}")

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

    @pyqtSlot(float)
    def home_motor(self, hp: float):
        self.devs.motor.move_to(hp)

    @pyqtSlot(float)
    def set_motor_step(self, jog_step: float):
        self.devs.motor.set_jog_step(jog_step)

    @pyqtSlot(bool)
    def move_motor(self, direction: bool):
        self.devs.motor.jog(forward=direction)

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
        pw_405 = self.ctrl_panel.get_cobolt_laser_power("405")
        try:
            self.devs.laser.set_modulation_mode(["405"], pw_405)
        except Exception as e:
            self.logg.error(f"Cobolt Laser Error: {e}")
        pw_488 = self.ctrl_panel.get_cobolt_laser_power("488_1")
        try:
            self.devs.laser.set_modulation_mode(["488_1"], pw_488)
        except Exception as e:
            self.logg.error(f"Cobolt Laser Error: {e}")

    def lasers_off(self):
        try:
            self.devs.laser.laser_off("all")
        except Exception as e:
            self.logg.error(f"Cobolt Laser Error: {e}")

    def set_camera_roi(self):
        try:
            # x, y, nx, ny, bn = self.ctrl_panel.get_emccd_roi()
            # self.devs.img_cam.bin_h, self.devs.img_cam.bin_v = bn, bn
            # self.devs.img_cam.start_h, self.devs.img_cam.end_h = x, x + nx - 1
            # self.devs.img_cam.start_v, self.devs.img_cam.end_v = y, y + ny - 1
            # self.devs.img_cam.gain = self.ctrl_panel.get_emccd_gain()
            # self.devs.img_cam.t_exposure = self.ctrl_panel.get_emccd_exposure()
            x, y, nx, ny, bn = self.ctrl_panel.get_emccd_roi()
            self.devs.img_cam.bin_h, self.devs.img_cam.bin_v = bn, bn
            self.devs.img_cam.pixels_x, self.devs.img_cam.pixels_y = nx, ny
            self.devs.img_cam.start_h, self.devs.img_cam.end_h = x, x + nx - 1
            self.devs.img_cam.start_v, self.devs.img_cam.end_v = y, y + ny - 1
            self.devs.img_cam.t_exposure = self.ctrl_panel.get_emccd_exposure()
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

    def update_trigger_parameters(self):
        """Ensure that the camera acquisition is fully set up before executing this function."""
        try:
            self.update_digital_triggers()
            self.update_piezo_scanning()
            self.logg.info(f"Trigger Updated")
        except Exception as e:
            self.logg.error(f"Trigger Error: {e}")

    def prepare_camera(self):
        # self.devs.img_cam.set_roi()
        # self.devs.img_cam.set_acquisition_mode(3)
        # self.devs.img_cam.set_exposure_time()
        # self.devs.img_cam.set_gain()
        # self.devs.img_cam.set_kinetic_cycle_time(self.trg.cycle_time)
        # self.devs.img_cam.set_kinetics_num(20000)
        # self.devs.img_cam.get_acquisition_timings()
        self.devs.img_cam.set_roi()
        self.devs.img_cam.set_exposure_time()

    def prepare_video(self, vd_mod):
        self.update_trigger_parameters()
        self.lasers = self.ctrl_panel.get_lasers()
        self.set_lasers(self.lasers)
        self.slm_seq = self.ctrl_panel.get_slm_sequence()
        slm_total, slm_end, slm_on = self.devs.slm.select_order(self.devs.slm.ord_dict[self.slm_seq])
        self.trg.update_slm_parameters(total_time=slm_total, on_time=slm_on, end_time=slm_end)
        self.set_camera_roi()
        self.devs.img_cam.t_exposure = slm_on + 5e-6
        interval_t, transfer_t = self.ctrl_panel.get_camera_times()
        self.trg.update_camera_parameters(initial_time=self.devs.img_cam.t_clean,
                                          exposure_time=self.devs.img_cam.t_exposure,
                                          standby_time=self.devs.img_cam.t_readout,
                                          frame_rate=self.devs.img_cam.fps,
                                          interval_time=interval_t, transfer_time=transfer_t)
        if vd_mod == "Widefield":
            dtr, chs = self.trg.generate_digital_triggers(self.lasers, 0)
        elif vd_mod == "SIM":
            dtr, chs = self.trg.generate_sim_triggers(0)
        elif vd_mod == "NLSIM":
            dtr, chs = self.trg.generate_nlsim_triggers(0)
        else:
            raise Exception(f"Invalid Live Mode")
        self.prepare_camera()
        self.viewer.switch_camera(self.devs.img_cam.pixels_x, self.devs.img_cam.pixels_y)
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
            self.devs.img_cam.start_live()
            self.devs.img_cam.data.on_update(self.viewer.on_camera_update_from_thread)
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
            self.devs.img_cam.stop_live()
            self.logg.info(r"Live Video Stopped")
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
            self.viewer.plot_trace(ipr.get_profile(self.viewer.image_viewer._display_frame, ax, norm=True),
                                   overlay=True)
        except Exception as e:
            self.logg.error(f"Error plotting profile: {e}")

    @pyqtSlot()
    def plot_trigger(self):
        try:
            self.slm_seq = self.ctrl_panel.get_slm_sequence()
            self.devs.slm.select_order(self.devs.slm.ord_dict[self.slm_seq])
            self.update_trigger_parameters()
            dtr, dch = self.trg.generate_digital_triggers(self.lasers, 0, self.slm_seq)
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

    @pyqtSlot(bool)
    def dpc(self, on: bool):
        if on:
            try:
                self.prepare_dpc()
            except Exception as e:
                self.logg.error(f"Error preparing DPC: {e}")
                return
            self.start_dpc()
        else:
            self.stop_dpc()

    def setup_dpc_camera(self):
        x, y, nx, ny, bn = self.ctrl_panel.get_scmos_roi()
        self.devs.dpc_cam.bin_h, self.devs.dpc_cam.bin_v = bn, bn
        self.devs.dpc_cam.start_h, self.devs.dpc_cam.end_h = x, x + nx - 1
        self.devs.dpc_cam.start_v, self.devs.dpc_cam.end_v = y, y + ny - 1
        self.devs.dpc_cam.t_exposure, t_readout = self.ctrl_panel.get_scmos_exposure()
        return t_readout

    def write_dpc_illumination(self, t_readout):
        dpc_seqs = self.devs.led.dpc_sequences(1, 128, self.devs.dpc_cam.t_exposure, t_readout)
        self.devs.daq.write_dpc_sequences(dpc_seqs, finite=False)

    def start_dpc_worker(self):
        if getattr(self.viewer, "dpc_worker", None) is None:
            self.viewer.dpc_worker = run_threads.DPCWorker(fps=10)
            self.viewer.dpc_worker.dpc_ready.connect(self.viewer.on_dpc_frame, Qt.ConnectionType.QueuedConnection)
            self.viewer.dpc_worker.start()
        self.devs.dpc_cam.data.on_update(self.viewer.dpc_worker.push_frame)

    def stop_dpc_worker(self):
        if getattr(self.viewer, "dpc_worker", None) is not None:
            self.viewer.dpc_worker.stop()
            self.viewer.dpc_worker = None

    def prepare_dpc(self):
        t_readout = self.setup_dpc_camera()
        self.devs.dpc_cam.prepare_live()
        self.write_dpc_illumination(t_readout)

    def start_dpc(self):
        try:
            self.start_dpc_worker()
            self.devs.dpc_cam.start_live()
            self.devs.daq.run_dpc_sequences()
            self.logg.info("DPC live started")
        except Exception as e:
            self.logg.error(f"Error starting DPC: {e}")
            self.stop_dpc()

    def dpc_off(self):
        dpc_seqs = self.devs.led.led_ring()
        self.devs.daq.write_dpc_sequences(dpc_seqs, [0], finite=True)
        self.devs.daq.run_dpc_sequences()
        time.sleep(0.1)
        self.devs.daq.stop_dpc_sequences()

    def stop_dpc(self):
        try:
            self.devs.daq.stop_dpc_sequences()
            self.dpc_off()
            self.devs.dpc_cam.stop_live()
            self.stop_dpc_worker()
            self.logg.info("DPC live stopped")
        except Exception as e:
            self.logg.error(f"Error stopping DPC: {e}")

    @pyqtSlot(bool, int)
    def dpc_acquisition(self, sw: bool, acq_num: int):
        if sw:
            fn = self.vw.get_file_dialog()
            tim = time.strftime("%Y%m%d%H%M%S")
            if fn is not None:
                file_name = tim + "_DPC_" + fn
            else:
                file_name = tim + "_DPC"
            try:
                self.prepare_dpc_acquisition(file_name, acq_num)
            except Exception as e:
                self.logg.error(f"Error preparing DPC acquisition: {e}")
                self.stop_dpc_acquisition()
                return
            self.start_dpc_acquisition()
        else:
            self.stop_dpc_acquisition()

    def prepare_dpc_acquisition(self, labl: str, acq_num: int):
        t_readout = self.setup_dpc_camera()
        self.devs.dpc_cam.prepare_data_acquisition(n=acq_num, fd=self.path, fn=labl)
        self.write_dpc_illumination(t_readout)

    def start_dpc_acquisition(self):
        try:
            self.start_dpc_worker()
            self.devs.dpc_cam.start_data_acquisition()
            self.devs.daq.run_dpc_sequences()
            self.logg.info("DPC acquisition started")
        except Exception as e:
            self.logg.error(f"Error starting DPC acquisition: {e}")
            self.stop_dpc_acquisition()

    def stop_dpc_acquisition(self):
        try:
            self.devs.daq.stop_dpc_sequences()
            self.dpc_off()
            self.devs.dpc_cam.stop_data_acquisition()
            self.stop_dpc_worker()
            self.logg.info("DPC acquisition stopped")
        except Exception as e:
            self.logg.error(f"Error stopping DPC acquisition: {e}")

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
                pn = self.prepare_acquisition(acq_mod, acq_num)
            except Exception as e:
                self.logg.error(f"Error preparing widefield: {e}")
                self.devs.daq.stop_triggers()
                self.lasers_off()
                return
            self.start_acquisition(file_name, pn)
        else:
            self.stop_acquisition()

    def prepare_acquisition(self, aqm, aqn):
        self.update_trigger_parameters()
        self.lasers = self.ctrl_panel.get_lasers()
        self.set_lasers(self.lasers)
        self.slm_seq = self.ctrl_panel.get_slm_sequence()
        slm_total, slm_end, slm_on = self.devs.slm.select_order(self.devs.slm.ord_dict[self.slm_seq])
        self.trg.update_slm_parameters(total_time=slm_total, on_time=slm_on, end_time=slm_end)
        self.set_camera_roi()
        self.devs.img_cam.t_exposure = slm_on + 5e-6
        self.trg.update_camera_parameters(initial_time=self.devs.img_cam.t_clean,
                                          exposure_time=self.devs.img_cam.t_exposure,
                                          standby_time=self.devs.img_cam.t_readout,
                                          frame_rate=self.devs.img_cam.fps)
        if aqm == "2D_WideField":
            dtr, chs = self.trg.generate_digital_triggers(self.lasers, 0)
            self.devs.daq.write_triggers(digital_sequences=dtr, digital_channels=chs,
                                         finite=False, trg=False)
            pos = aqn
        elif aqm == "3D_WideField":
            dtr, ptr, dchs, pchs, pos = self.trg.generate_piezo_scan(1, self.lasers, 0)
            self.devs.daq.write_triggers(digital_sequences=dtr, digital_channels=dchs,
                                         analog_sequences=ptr, analog_channels=pchs,
                                         finite=False, trg=False)
        elif aqm == "2D_SIM":
            dtr, chs = self.trg.generate_sim_triggers(aqn)
            self.devs.daq.write_triggers(digital_sequences=dtr, digital_channels=chs,
                                         finite=False, trg=False)
            pos = aqn
        elif aqm == "3D_SIM":
            dtr, ptr, dchs, pchs, pos = self.trg.generate_piezo_scan(aqn, self.lasers, 0)
            self.devs.daq.write_triggers(digital_sequences=dtr, digital_channels=dchs,
                                         analog_sequences=ptr, analog_channels=pchs,
                                         finite=False, trg=False)
            pos *= aqn
        elif aqm == "2D_NLSIM":
            dtr, chs = self.trg.generate_nlsim_triggers(aqn)
            self.devs.daq.write_triggers(digital_sequences=dtr, digital_channels=chs,
                                         finite=False, trg=False)
            pos = aqn
        else:
            raise Exception(f"Invalid Acquisition Mode")
        self.prepare_camera()
        self.viewer.switch_camera(self.devs.img_cam.pixels_x, self.devs.img_cam.pixels_y)
        self.ctrl_panel.display_emccd_timings(exposure_time=self.trg.exposure_time, kinetic_time=self.trg.cycle_time)
        return pos

    def start_acquisition(self, labl: str, acq_num: int):
        try:
            self.devs.slm.activate()
            self.devs.img_cam.start_data_acquisition(n=acq_num, fd=self.path, fn=labl)
            self.devs.img_cam.data.on_update(self.viewer.on_camera_update_from_thread)
            self.devs.daq.run_triggers()
        except Exception as e:
            self.stop_acquisition()
            self.logg.error(f"Error start acquisition: {e}")
            return

    def stop_acquisition(self):
        try:
            self.devs.daq.stop_triggers()
            time.sleep(0.04)
            self.devs.img_cam.stop_data_acquisition()
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
        self.update_trigger_parameters()
        self.lasers = self.ctrl_panel.get_lasers()
        self.set_lasers(self.lasers)
        self.slm_seq = self.ctrl_panel.get_slm_sequence()
        slm_total, slm_end, slm_on = self.devs.slm.select_order(self.devs.slm.ord_dict[self.slm_seq])
        self.trg.update_slm_parameters(total_time=slm_total, on_time=slm_on, end_time=slm_end)
        self.set_camera_roi()
        self.devs.img_cam.t_exposure = slm_on
        self.devs.img_cam.prepare_live()
        self.trg.update_camera_parameters(initial_time=self.devs.img_cam.t_clean,
                                          exposure_time=self.devs.img_cam.t_exposure,
                                          standby_time=self.devs.img_cam.t_readout,
                                          frame_rate=self.devs.img_cam.fps)
        dtr, chs = self.trg.generate_digital_triggers(self.lasers, 0)
        self.devs.daq.write_triggers(digital_sequences=dtr, digital_channels=chs, finite=single, trg=False)
        self.prepare_camera()

    def finish_task(self):
        try:
            self.devs.daq.stop_triggers()
            time.sleep(0.04)
            self.devs.img_cam.stop_snap()
            self.devs.slm.deactivate()
            self.lasers_off()
            self.logg.info("Focus Finding Finish")
        except Exception as e:
            self.logg.error(f"Error Stopping Focus Finding: {e}")

    def focus_finding(self):
        try:
            pos_x, pos_y, pos_z = self.ctrl_panel.get_piezo_positions()
            center_pos, axis_length, step_size = pos_z[0], 2.0, 0.1
            start = center_pos - axis_length
            end = center_pos + axis_length
            zps = np.arange(start, end + step_size, step_size)
            data = []
            # data_calib = []
            self.devs.slm.activate()
            # self.devs.cam_set[self.cameras["focus_lock"]].start_live()
            self.devs.piezo.move_position(2, zps[0])
            time.sleep(0.06)
            self.devs.img_cam.start_snap()
            self.devs.daq.run_triggers()
            time.sleep(0.04)
            temp = self.devs.img_cam.get_last_snap()
            self.devs.img_cam.stop_snap()
            for i, z in enumerate(zps):
                self.devs.piezo.move_position(2, z)
                time.sleep(0.06)
                self.devs.img_cam.start_snap()
                self.devs.daq.run_triggers()
                time.sleep(0.06)
                temp = self.devs.img_cam.get_last_snap()
                self.devs.img_cam.stop_snap()
                if temp is not None:
                    data.append(temp)
                # data_calib.append(self.devs.cam_set[self.cameras["focus_lock"]].get_last_image())
            # self.devs.cam_set[self.cameras["focus_lock"]].stop_live()
            self.finish_task()
            fd = os.path.join(self.path, time.strftime("%Y%m%d%H%M%S") + '_widefield_stack.tif')
            tf.imwrite(fd, np.asarray(data))
            pzs = []
            for dat in data:
                pzs.append(ipr.calculate_focus_measure_with_sobel(dat - dat.min()))
            self.sig_plt.emit(list(pzs), list(zps))
            fp = ipr.peak_find(zps, pzs)
            if isinstance(fp, str):
                self.logg.error(fp)
            else:
                self.sig_auto_focus.emit(float(fp))
            # time.sleep(0.06)
            # data_calib.append(self.devs.cam_set[self.cameras["focus_lock"]].get_last_image())
            # fd = os.path.join(self.path, time.strftime("%Y%m%d%H%M%S") + '_focus_calibration_stack.tif')
            # tf.imwrite(fd, np.asarray(data_calib), imagej=True, resolution=(
            #     1 / self.pixel_sizes[self.cameras["focus_lock"]], 1 / self.pixel_sizes[self.cameras["focus_lock"]]),
            #            metadata={'unit': 'um'})
            # self.p.foc_ctrl.calibrate(np.append(zps, fp), np.asarray(data_calib))
        except Exception as e:
            # self.devs.cam_set[self.cameras["focus_lock"]].stop_live()
            self.finish_task()
            self.logg.error(f"Error Running Focus Finding: {e}")
            return

    @pyqtSlot()
    def run_focus_finding(self):
        self.vw.get_dialog(txt="Focus Finding")
        try:
            self.prepare_task()
        except Exception as e:
            self.logg.error(f"Error Starting Focus Finding: {e}")
            self.devs.daq.stop_triggers()
            self.lasers_off()
            return
        self.run_task(task=self.focus_finding)

    @pyqtSlot(int)
    def set_dm_current(self, i: int):
        try:
            self.devs.dfm.set_dpp(self.devs.dfm.dpp_cmd[i])
            self.devs.dfm.current_cmd = i
            self.ao_panel.update_dm_display(self.devs.dfm.dpp_cmd[i])
        except Exception as e:
            self.logg.error(f"DM Error: {e}")

    @pyqtSlot(int, float)
    def set_zernike(self, iz: int, amp: float):
        try:
            self.devs.dfm.set_zm(iz, amp)
            phase_temp = self.devs.dfm.dpp_cmd[self.devs.dfm.current_cmd].copy()
            phase_temp[iz] += amp
            self.ao_panel.update_dm_display(phase_temp)
        except Exception as e:
            self.logg.error(f"DM Error: {e}")

    @pyqtSlot()
    def set_dm_flat(self):
        if int(self.ao_panel.get_cmd_index()) == self.devs.dfm.current_cmd:
            self.devs.dfm.write_flat_cmd(t=time.strftime("%Y_%m_%d_%H_%M"),
                                         cmd=self.devs.dfm.dpp_cmd[self.devs.dfm.current_cmd])

    @pyqtSlot()
    def update_dm(self):
        try:
            self.devs.dfm.dpp_cmd.append(self.devs.dfm.temp_cmd[-1])
            self.ao_panel.update_cmd_index()
            self.devs.dfm.set_dpp(self.devs.dfm.dpp_cmd[-1])
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

    def sensorless_iteration(self, amps):
        ims = []
        for amp in amps:
            self.devs.dfm.set_dpp(amp)
            time.sleep(0.04)
            self.devs.img_cam.start_snap()
            self.devs.daq.run_triggers()
            time.sleep(0.04)
            temp = self.devs.img_cam.get_last_snap()
            self.devs.img_cam.stop_snap()
            if temp is not None:
                ims.append(temp)
        return ims

    @staticmethod
    def image_assessment(mf, images):
        if mf == "Max(Intensity)":
            return [img.max() for img in images]
        elif mf == "Sum(Intensity)":
            return [img.sum() for img in images]
        elif mf == "SNR(FFT)":
            return [ipr.snr(img) for img in images]
        elif mf == "HighPass(FFT)":
            return [ipr.hpf(img) for img in images]
        else:
            return []

    def sensorless_iterations(self):
        try:
            lpr, hpr, mf, err = self.ao_panel.get_sensorless_parameters()
            ipr.set_hpr(hpr)
            ipr.set_lpr(lpr)
            name = time.strftime("%Y%m%d_%H%M%S_") + self.devs.dfm.dpp_model + '_ao_iterations_' + mf
            new_folder = os.path.join(self.path, name)
            os.makedirs(new_folder, exist_ok=True)
            self.logg.info(f'Directory {new_folder} has been created successfully.')
        except Exception as e:
            self.logg.error(f'Error creating directory for sensorless iteration: {e}')
            return
        try:
            mode_start, mode_stop, amp_start, amp_step, amp_step_number = self.ao_panel.get_sensorless_iteration()
            amprange = [amp_start + step_number * amp_step for step_number in range(amp_step_number)]
            results = [('Mode', 'Amp', 'Metric')]
            za = []
            mv = []
            zp = [0] * self.devs.dfm.n_zernike
            cmd = self.devs.dfm.dpp_cmd[self.devs.dfm.current_cmd]
            self.devs.slm.activate()
            time.sleep(0.04)
            self.logg.info("Sensorless AO iterations start")
            self.devs.dfm.set_dpp(cmd)
            time.sleep(0.04)
            if err:
                images = []
                for i in range(8):
                    self.devs.img_cam.start_snap()
                    self.devs.daq.run_triggers()
                    time.sleep(0.04)
                    temp = self.devs.img_cam.get_last_snap()
                    self.devs.img_cam.stop_snap()
                    if temp is not None:
                        images.append(temp)
                mts = self.image_assessment(mf, images)
                std = np.std(mts)
                fn = new_folder + r"\original.tiff"
                tf.imwrite(str(fn), np.asarray(images, dtype=np.float16))
            else:
                self.devs.img_cam.start_snap()
                self.devs.daq.run_triggers()
                time.sleep(0.04)
                temp = self.devs.img_cam.get_last_snap()
                self.devs.img_cam.stop_snap()
                fn = new_folder + r"\original.tiff"
                tf.imwrite(str(fn), temp.astype(np.float16))
            for mode in range(mode_start, mode_stop + 1):
                self.vw.dialog_text.setText(f"Zernike mode #{mode}")
                labels = ["zm%0.2d_amp%.4f" % (mode, amp) for amp in amprange]
                cmds = []
                for amp in amprange:
                    temp = cmd.copy()
                    temp[mode] += amp
                    cmds.append(temp)
                images = self.sensorless_iteration(cmds)
                # mts = self.image_assessment(mf, images)
                # self.logg.info(f"zernike mode #{mode}, ({amprange}), ({mts})")
                # self.sig_plt.emit(mts, amprange)
                # if err:
                #     mts_err = [std] * len(mts)
                #     pm = ipr.peak_find(amprange, mts, mts_err)
                # else:
                #     pm = ipr.peak_find(amprange, mts)
                # if isinstance(pm, str):
                #     self.logg.error(f"zernike mode #{mode} " + pm)
                # else:
                #     zp[mode] = pm
                #     cmd[mode] += pm
                #     self.devs.dfm.set_dpp(cmd)
                #     self.logg.info("set mode %d at value of %.4f" % (mode, pm))
                # for amp, mt in zip(amprange, mts):
                #     results.append((mode, amp, mt))
                # za.extend(amprange)
                # mv.extend(mts)
                fn = os.path.join(str(new_folder), f"zernike mode #{mode}.tiff")
                with tf.TiffWriter(fn) as tif:
                    for img, label in zip(images, labels):
                        tif.write(img.astype(np.float16), description=label)
            self.devs.dfm.set_dpp(cmd)
            time.sleep(0.04)
            self.devs.img_cam.start_snap()
            self.devs.daq.run_triggers()
            time.sleep(0.04)
            fn = new_folder + r"\final.tiff"
            temp = self.devs.img_cam.get_last_snap()
            self.devs.img_cam.stop_snap()
            tf.imwrite(str(fn), temp.astype(np.float16))
            self.devs.dfm.dpp_cmd.append(cmd)
            self.ao_panel.update_cmd_index()
            i = int(self.ao_panel.get_cmd_index())
            self.devs.dfm.current_cmd = i
            self.devs.dfm.write_cmd(new_folder, '_')
            self.devs.dfm.save_sensorless_results(os.path.join(str(new_folder), 'results.xlsx'), za, mv, zp)
            self.finish_task()
        except Exception as e:
            self.finish_task()
            self.logg.error(f"Sensorless AO Error: {e}")
            return

    @pyqtSlot()
    def run_sensorless_iteration(self):
        self.vw.get_dialog(txt="Sensorless Iteration")
        try:
            self.prepare_task()
        except Exception as e:
            self.logg.error(f"Error Starting Sensorless Iteration: {e}")
            self.devs.daq.stop_triggers()
            self.lasers_off()
            return
        self.run_task(task=self.sensorless_iterations)
