import time

from PyQt6.QtCore import QObject, pyqtSlot

import run_threads


class CommandExecutor(QObject):

    def __init__(self, dev, cwd, pr, bus, path, logger=None):
        super().__init__()
        self.bus = bus
        self.devs = dev
        self.ctrl_panel = cwd.ctrl_panel
        self.viewer = cwd.viewer
        self.ao_panel = cwd.ao_panel
        self.trg = pr.trg
        self.wfr = pr.wfp
        self.flk = pr.flp
        self.acq_thread = None
        self.path = path
        self.logg = logger or self.setup_logging()
        self._set_signal_executions()

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    def _set_signal_executions(self):
        # MCL Mad Deck
        self.ctrl_panel.Signal_deck_read_position.connect(self.deck_read_position)
        self.ctrl_panel.Signal_deck_zero_position.connect(self.deck_zero_position)
        self.ctrl_panel.Signal_deck_move_single_step.connect(self.move_deck_single_step)
        self.ctrl_panel.Signal_deck_move_continuous.connect(self.move_deck_continuous)
        # MCL Piezo
        self.ctrl_panel.Signal_piezo_move_usb.connect(self.set_piezo_positions_usb)
        self.ctrl_panel.Signal_piezo_move.connect(self.set_piezo_positions)
        # self.ctrl_panel.Signal_focus_finding.connect(self.run_focus_finding)
        # self.ctrl_panel.Signal_focus_locking.connect(self.run_focus_locking)
        # Galvo Scanners
        self.ctrl_panel.Signal_galvo_set.connect(self.set_galvo)
        # self.ctrl_panel.Signal_galvo_scan_update.connect(self.update_galvo_scanner)
        self.ctrl_panel.Signal_galvo_path_switch.connect(self.set_switch)
        # Cobolt Lasers
        self.ctrl_panel.Signal_set_laser.connect(self.set_laser)
        # Main Image Control
        self.ctrl_panel.Signal_check_emccd_temperature.connect(self.check_emdccd_temperature)
        self.ctrl_panel.Signal_switch_emccd_cooler.connect(self.switch_emdccd_cooler)
        # self.ctrl_panel.Signal_plot_trigger.connect(self.plot_trigger)
        self.ctrl_panel.Signal_video.connect(self.video)
        # self.ctrl_panel.Signal_fft.connect(self.fft)
        # self.ctrl_panel.Signal_plot_profile.connect(self.plot_live)
        # self.ctrl_panel.Signal_add_profile.connect(self.plot_add)
        # self.ctrl_panel.Signal_set_mask.connect(self.set_array_mask)
        # NIDAQ
        self.ctrl_panel.Signal_daq_update.connect(self.update_daq_sample_rate)
        self.ctrl_panel.Signal_daq_reset.connect(self.reset_daq_channels)
        # Main Data Recording
        # self.ctrl_panel.Signal_focal_array_scan.connect(self.run_focal_array_scan)
        # self.ctrl_panel.Signal_grid_pattern_scan.connect(self.run_grid_pattern_scan)
        # self.ctrl_panel.Signal_alignment.connect(self.run_pattern_alignment)
        # self.ctrl_panel.Signal_data_acquire.connect(self.data_acquisition)

    # --- Camera ----------------------------------------------------------
    @pyqtSlot()
    def check_emdccd_temperature(self):
        try:
            self.devs.ccdcam.get_ccd_temperature()
            self.ctrl_panel.display_camera_temperature(self.devs.ccdcam.temperature)
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
            self.devs.ccdcam.cooler_on()
        except Exception as e:
            self.logg.error(f"CCD Camera Error: {e}")

    def switch_emdccd_cooler_off(self):
        try:
            self.devs.ccdcam.cooler_off()
        except Exception as e:
            self.logg.error(f"CCD Camera Error: {e}")

    def set_camera_roi(self, key="imaging"):
        try:
            if self.devs.cam_set[key] == 0:
                x, y, nx, ny, bx, by = self.ctrl_panel.get_emccd_roi()
                self.devs.cameras[0].bin_h, self.devs.cameras[0].bin_v = bx, by
                self.devs.cameras[0].start_h, self.devs.cameras[0].end_h = x, x + nx - 1
                self.devs.cameras[0].start_v, self.devs.cameras[0].end_v = y, y + ny - 1
                self.devs.cameras[0].gain = self.ctrl_panel.get_emccd_gain()
                self.devs.cameras[0].t_exposure = self.ctrl_panel.get_emccd_expo()
            if self.devs.cam_set[key] == 1:
                x, y, nx, ny, bx, by = self.ctrl_panel.get_scmos_roi()
                self.devs.cameras[1].set_roi(bx, by, x, nx, y, ny)
            if self.devs.cam_set[key] == 2:
                x, y, nx, ny, bx, by = self.ctrl_panel.get_thorcam_roi()
                self.devs.cameras[2].set_roi(x, y, x + nx - 1, y + ny - 1)
            if self.devs.cam_set[key] == 3:
                expo = self.ctrl_panel.get_tis_expo()
                self.devs.cameras[3].set_exposure(expo)
                x, y, nx, ny, bx, by = self.ctrl_panel.get_tis_roi()
                self.devs.cameras[3].set_roi(x, y, nx, ny)
        except Exception as e:
            self.logg.error(f"Camera Error: {e}")

    # --- Mad Deck ----------------------------------------------------------
    @pyqtSlot()
    def deck_read_position(self):
        self.ctrl_panel.display_deck_position(self.devs.md.position)

    @pyqtSlot()
    def deck_zero_position(self):
        self.devs.md.position = 0
        self.ctrl_panel.display_deck_position(self.devs.md.position)

    @pyqtSlot(bool)
    def move_deck_single_step(self, direction: bool):
        if direction:
            self.move_deck_up()
        else:
            self.move_deck_down()

    def move_deck_up(self):
        try:
            _moving = self.devs.md.is_moving()
            if _moving:
                self.logg.info("MadDeck is moving")
            else:
                self.devs.md.move_relative(3, 0.000762, velocity=0.8)
                self.ctrl_panel.display_deck_position(self.devs.md.position)
        except Exception as e:
            self.logg.error(f"MadDeck Error: {e}")

    def move_deck_down(self):
        try:
            _moving = self.devs.md.is_moving()
            if _moving:
                self.logg.info("MadDeck is moving")
            else:
                self.devs.md.move_relative(3, -0.000762, velocity=0.8)
                self.ctrl_panel.display_deck_position(self.devs.md.position)
        except Exception as e:
            self.logg.error(f"MadDeck Error: {e}")

    @pyqtSlot(bool, int, float)
    def move_deck_continuous(self, moving: bool, direction: int, velocity: float):
        if moving:
            self.devs.md.move_deck(direction, velocity)
        else:
            self.devs.md.stop_deck()

    # --- Piezo Stage ----------------------------------------------------------
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
                self.devs.pz.move_position(0, pos_x)
                time.sleep(0.1)
                self.ctrl_panel.display_piezo_position_x(self.devs.pz.read_position(0))
            else:
                self.devs.daq.set_piezo_position([pos_x / 10.], [0])
                time.sleep(0.1)
                self.ctrl_panel.display_piezo_position_x(self.devs.pz.read_position(0))
        except Exception as e:
            self.logg.error(f"MCL Piezo Error: {e}")

    def set_piezo_position_y(self, pos_y, port="analog"):
        try:
            if port == "software":
                self.devs.pz.move_position(1, pos_y)
                time.sleep(0.1)
                self.ctrl_panel.display_piezo_position_y(self.devs.pz.read_position(1))
            else:
                self.devs.daq.set_piezo_position([pos_y / 10.], [1])
                time.sleep(0.1)
                self.ctrl_panel.display_piezo_position_y(self.devs.pz.read_position(1))
        except Exception as e:
            self.logg.error(f"MCL Piezo Error: {e}")

    def set_piezo_position_z(self, pos_z, port="analog"):
        try:
            if port == "software":
                self.devs.pz.move_position(2, pos_z)
                time.sleep(0.1)
                self.ctrl_panel.display_piezo_position_z(self.devs.pz.read_position(2))
            else:
                self.devs.daq.set_piezo_position([pos_z / 10.], [2])
                time.sleep(0.1)
                self.ctrl_panel.display_piezo_position_z(self.devs.pz.read_position(2))
        except Exception as e:
            self.logg.error(f"MCL Piezo Error: {e}")

    def reset_piezo_positions(self):
        pos_x, pos_y, pos_z = self.ctrl_panel.get_piezo_positions()
        self.set_piezo_position_x(pos_x[0], port="software")
        self.set_piezo_position_y(pos_y[0], port="software")
        self.set_piezo_position_z(pos_z[0], port="software")
        self.set_piezo_position_x(pos_x[1], port="analog")
        self.set_piezo_position_y(pos_y[1], port="analog")
        self.set_piezo_position_z(pos_z[1], port="analog")
        self.ctrl_panel.display_piezo_position_x(self.devs.pz.read_position(0))
        self.ctrl_panel.display_piezo_position_y(self.devs.pz.read_position(1))
        self.ctrl_panel.display_piezo_position_z(self.devs.pz.read_position(2))

    # --- Galvo Mirros ----------------------------------------------------------
    @pyqtSlot(float, float)
    def set_galvo(self, voltx: float, volty: float):
        try:
            self.devs.daq.set_galvo_position([voltx, volty], [0, 1])
        except Exception as e:
            self.logg.error(f"Galvo Error: {e}")

    @pyqtSlot(float)
    def set_switch(self, volt: float):
        try:
            self.devs.daq.set_switch_position(volt)
        except Exception as e:
            self.logg.error(f"Galvo Error: {e}")

    def reset_galvo_positions(self):
        g_x, g_y = self.ctrl_panel.get_galvo_positions()
        try:
            self.devs.daq.set_galvo_position([g_x, g_y], [0, 1])
            self.devs.daq.set_switch_position(0.)
        except Exception as e:
            self.logg.error(f"Galvo Error: {e}")

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

    # --- Lasers ----------------------------------------------------------
    def set_lasers(self, lasers, pws):
        try:
            self.devs.laser.set_modulation_mode(lasers, pws)
            self.devs.laser.laser_on(lasers)
        except Exception as e:
            self.logg.error(f"Cobolt Laser Error: {e}")

    def lasers_off(self):
        try:
            self.devs.laser.laser_off("all")
        except Exception as e:
            self.logg.error(f"Cobolt Laser Error: {e}")

    # --- Live Video ----------------------------------------------------------
    @pyqtSlot(int)
    def update_daq_sample_rate(self, sr: int):
        self.trg.update_nidaq_parameters(sr * 1000)
        self.update_galvo_scanner()
        self.devs.daq.sample_rate = sr * 1000

    @pyqtSlot()
    def reset_daq_channels(self):
        self.devs.daq.stop_triggers()

    @pyqtSlot()
    def update_galvo_scanner(self):
        galvo_positions, galvo_ranges, dot_pos, offset, galvo_positions_act, galvo_ranges_act, dot_pos_act, offset_act, sws = self.ctrl_panel.get_galvo_scan_parameters()
        self.trg.update_galvo_scan_parameters(origins=galvo_positions, ranges=galvo_ranges,
                                              foci=dot_pos, offsets=offset,
                                              origins_act=galvo_positions_act, ranges_act=galvo_ranges_act,
                                              foci_act=dot_pos_act, offsets_act=offset_act, sws=sws)
        self.ctrl_panel.display_frequency(self.trg.frequency, self.trg.frequency_act)

    def update_trigger_parameters(self, cam_key):
        """Ensure that the camera acquisition is fully set up before executing this function."""
        try:
            digital_starts, digital_ends = self.ctrl_panel.get_digital_parameters()
            self.trg.update_digital_parameters(digital_starts, digital_ends)
            self.update_galvo_scanner()
            axis_lengths, step_sizes = self.ctrl_panel.get_piezo_scan_parameters()
            pos_x, pos_y, pos_z = self.ctrl_panel.get_piezo_positions()
            positions = [pos_x[1], pos_y[1], pos_z[1]]
            return_time = self.ctrl_panel.get_piezo_return_time()
            self.trg.update_piezo_scan_parameters(axis_lengths, step_sizes, positions, return_time)
            self.trg.update_camera_parameters(initial_time=self.devs.cameras[self.devs.cam_set[cam_key]].t_clean,
                                              standby_time=self.devs.cameras[self.devs.cam_set[cam_key]].t_readout,
                                              cycle_time=self.devs.cameras[self.devs.cam_set[cam_key]].t_kinetic)
            if self.devs.cam_set[cam_key] == 0:
                self.ctrl_panel.display_camera_timings(standby=self.devs.cameras[self.devs.cam_set[cam_key]].t_kinetic)
            self.logg.info(f"Trigger Updated")
        except Exception as e:
            self.logg.error(f"Trigger Error: {e}")

    def generate_live_triggers(self, lasers, cam_key):
        self.update_trigger_parameters(cam_key)
        return self.trg.generate_digital_triggers(lasers, self.devs.cam_set[cam_key])

    def prepare_video(self, vd_mod):
        lasers, pws = self.ctrl_panel.get_lasers()
        self.set_lasers(lasers, pws)
        self.devs.cam_set["imaging"] = self.ctrl_panel.get_imaging_camera()
        self.set_camera_roi("imaging")
        if self.devs.cam_set["imaging"] == 0:
            self.devs.cameras[self.devs.cam_set["imaging"]].prepare_live()
            self.update_trigger_parameters("imaging")
        if self.devs.cam_set["imaging"] == 1:
            self.devs.cameras[self.devs.cam_set["imaging"]].mode = self.ctrl_panel.get_scmos_mode()
            if self.devs.cameras[self.devs.cam_set["imaging"]].mode == "LightSheet":
                self.update_trigger_parameters("imaging")
                _, _, interval_lines = self.ctrl_panel.get_scmos_expo()
                line_exposure, line_interval = self.trg.update_lightsheet_rolling(interval_lines)
                self.devs.cameras[self.devs.cam_set["imaging"]].line_exposure = line_exposure
                self.devs.cameras[self.devs.cam_set["imaging"]].line_interval = line_interval
                self.ctrl_panel.display_cmos_rolling_timings(line_exposure, line_interval)
                self.devs.cameras[self.devs.cam_set["imaging"]].prepare_live()
            if self.devs.cameras[self.devs.cam_set["imaging"]].mode == "Normal":
                self.devs.cameras[self.devs.cam_set["imaging"]].prepare_live()
                self.update_trigger_parameters("imaging")
        if vd_mod == "Wide Field":
            self.set_switch(self.trg.galvo_sw_states[self.devs.cam_set["imaging"]])
            dtr, sw, chs = self.trg.generate_digital_triggers(lasers, self.devs.cam_set["imaging"])
            self.devs.daq.write_triggers(digital_sequences=dtr, digital_channels=chs, finite=False)
            self.ctrl_panel.display_camera_timings(exposure=self.trg.exposure_time,
                                                   clean=self.trg.initial_time,
                                                   standby=self.trg.standby_time)
        if vd_mod == "Dot Scan":
            if self.devs.cam_set["imaging"] == 1:
                if self.devs.cameras[self.devs.cam_set["imaging"]].mode == "LightSheet":
                    self.set_switch(self.trg.galvo_sw_states[self.devs.cam_set["imaging"]])
                    dtr, gtr, chs = self.trg.generate_digital_scanning_triggers_rolling(lasers,
                                                                                        self.devs.cam_set["imaging"])
                    self.devs.daq.write_triggers(galvo_sequences=gtr, galvo_channels=[0, 1],
                                                 digital_sequences=dtr, digital_channels=chs, finite=False)
                else:
                    dtr, gtr, chs = self.trg.generate_digital_scanning_triggers(lasers,
                                                                                self.devs.cam_set["imaging"])
                    self.devs.daq.write_triggers(galvo_sequences=gtr, galvo_channels=[0, 1, 2],
                                                 digital_sequences=dtr, digital_channels=chs, finite=False)
            else:
                dtr, gtr, chs = self.trg.generate_digital_scanning_triggers(lasers, self.devs.cam_set["imaging"])
                self.devs.daq.write_triggers(galvo_sequences=gtr, galvo_channels=[0, 1, 2],
                                             digital_sequences=dtr, digital_channels=chs, finite=False)
                self.ctrl_panel.display_camera_timings(exposure=self.trg.exposure_time,
                                                       clean=self.trg.initial_time,
                                                       standby=self.trg.standby_time)
        if vd_mod == "Scan Calib":
            self.set_switch(self.trg.galvo_sw_states[self.devs.cam_set["imaging"]])
            dtr, sw, ptr, dch, pch = self.trg.generate_piezo_line_scan(lasers, self.devs.cam_set["imaging"])
            self.devs.daq.write_triggers(piezo_sequences=ptr, piezo_channels=pch,
                                         digital_sequences=dtr, digital_channels=dch, finite=False)
        if vd_mod == "Focus Lock":
            self.logg.info(f"Focus Lock live")

    # --- Live Video ----------------------------------------------------------
    @pyqtSlot(bool, str)
    def video(self, sw: bool, md: str):
        if sw:
            self.start_video(md)
        else:
            self.stop_video()

    def start_video(self, vm):
        if self.acq_thread and self.acq_thread.isRunning():
            return
        try:
            self.prepare_video(vm)
        except Exception as e:
            self.logg.error(f"Error preparing imaging video: {e}")
            self.devs.daq.stop_triggers()
            self.lasers_off()
            return
        try:
            self.devs.cameras[self.devs.cam_set["imaging"]].start_live()
            if self.devs.cam_set["imaging"] != self.devs.cam_set["focus_lock"]:
                self.acq_thread = run_threads.LiveViewThread(self.devs.cameras[self.devs.cam_set["imaging"]],
                                                             self.bus, interval_ms=50)
                self.devs.daq.run_triggers()
                self.acq_thread.start()
                self.logg.info(r"Live Video Started")
        except Exception as e:
            self.logg.error(f"Error starting imaging video: {e}")
            self.stop_video()
            return

    def stop_video(self):
        try:
            if self.acq_thread:
                self.acq_thread.stop()
                self.acq_thread = None
            self.devs.daq.stop_triggers()
            self.devs.cameras[self.devs.cam_set["imaging"]].stop_live()
            self.lasers_off()
            self.reset_galvo_positions()
            self.reset_piezo_positions()
            self.logg.info(r"Live Video Stopped")
        except Exception as e:
            self.logg.error(f"Error stopping imaging video: {e}")

    # # --- FFT ----------------------------------------------------------
    # @QtCore.pyqtSlot(bool)
    # def fft(self, sw: bool):
    #     if sw:
    #         self.run_fft()
    #     else:
    #         self.stop_fft()
    #
    # def run_fft(self):
    #     try:
    #         self.thread_fft.start()
    #     except Exception as e:
    #         self.logg.error(f"Error starting fft: {e}")
    #
    # def stop_fft(self):
    #     try:
    #         if self.thread_fft.isRunning():
    #             self.thread_fft.quit()
    #             self.thread_fft.wait()
    #     except Exception as e:
    #         self.logg.error(f"Error stopping fft: {e}")
    #
    # # --- Data Acquisition ----------------------------------------------------------
    # @pyqtSlot(str, int)
    # def data_acquisition(self, acq_mod: str, acq_num: int):
    #     if acq_mod == "Wide Field 2D":
    #         self.run_widefield_zstack(acq_num)
    #     elif acq_mod == "Wide Field 3D":
    #         self.run_widefield_zstack(acq_num)
    #     elif acq_mod == "Monalisa Scan 2D":
    #         self.run_monalisa_scan(acq_num)
    #     elif acq_mod == "Dot Scan 2D":
    #         self.run_dot_scan(acq_num)
    #     elif acq_mod == "Point Scan 2D":
    #         self.run_point_scan(acq_num)
    #     else:
    #         self.logg.error(f"Invalid video mode")
    #
    # @pyqtSlot(str, np.ndarray, list, list)
    # def save_data(self, tm: str, d: np.ndarray, idx: list, pos: list):
    #     fn = self.v.get_file_dialog()
    #     if fn is not None:
    #         fd = os.path.join(self.data_folder, tm + '_' + fn)
    #     else:
    #         fd = os.path.join(self.data_folder, tm)
    #     pixel_size = self.pixel_sizes[self.cameras["imaging"]]
    #     tf.imwrite(str(fd + r".tif"), data=d, metadata={"pixel_size": (pixel_size, pixel_size)})
    #     with pd.ExcelWriter(str(fd + r"_metadata.xlsx"), engine="openpyxl") as writer:
    #         if idx is not None:
    #             df_idx = pd.DataFrame(idx, columns=["acquisition_sequence"])
    #             df_idx.to_excel(writer, sheet_name="acquisition_sequence", index=False)
    #         if pos is not None:
    #             for i, arr in enumerate(pos):
    #                 df_pos = pd.DataFrame(arr, columns=[f"axis_{i}"])
    #                 df_pos.to_excel(writer, sheet_name=f"axis_{i}", index=False)
