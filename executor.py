import time

from PyQt6.QtCore import QObject, pyqtSlot, Qt

from utilities import image_processor as ipr
import run_threads

class CommandExecutor(QObject):

    def __init__(self, dev, cwd, pr, path, logger=None):
        super().__init__()
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
        self._initial_setup()
        self.lasers = []
        self.slm_seq = ""
        self.cameras = {"imaging": 0, "wfs": 1, "focus_lock": 2}

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    def _set_signal_executions(self):
        # MCL Piezo
        self.ctrl_panel.Signal_piezo_move_usb.connect(self.set_piezo_positions_usb)
        self.ctrl_panel.Signal_piezo_move.connect(self.set_piezo_positions)
        # self.ctrl_panel.Signal_focus_finding.connect(self.run_focus_finding)
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

        self.ctrl_panel.Signal_video.connect(self.video)
        self.ctrl_panel.Signal_fft.connect(self.fft)
        self.ctrl_panel.Signal_plot_profile.connect(self.profile_plot)
        self.ctrl_panel.Signal_add_profile.connect(self.plot_add)

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

    @pyqtSlot()
    def check_emdccd_temperature(self):
        try:
            self.devs.ccdcam.get_ccd_temperature()
            self.ctrl_panel.display_emccd_temperature(self.devs.ccdcam.temperature)
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
        self.trg.update_nidaq_parameters(sr * 1000)
        self.devs.daq.sample_rate = sr * 1000

    @pyqtSlot()
    def reset_daq_channels(self):
        self.devs.daq.stop_triggers()

    def update_trigger_parameters(self, cam_key):
        """Ensure that the camera acquisition is fully set up before executing this function."""
        try:
            digital_starts, digital_ends = self.ctrl_panel.get_digital_parameters()
            self.trg.update_digital_parameters(digital_starts, digital_ends)
            axis_lengths, step_sizes = self.ctrl_panel.get_piezo_scan_parameters()
            pos_x, pos_y, pos_z = self.ctrl_panel.get_piezo_positions()
            positions = [pos_x[1], pos_y[1], pos_z[1]]
            return_time = self.ctrl_panel.get_piezo_return_time()
            self.trg.update_piezo_scan_parameters(axis_lengths, step_sizes, positions, return_time)
            self.trg.update_camera_parameters(initial_time=self.devs.cam_set[self.cameras[cam_key]].t_clean,
                                              standby_time=self.devs.cam_set[self.cameras[cam_key]].t_readout)
            self.logg.info(f"Trigger Updated")
        except Exception as e:
            self.logg.error(f"Trigger Error: {e}")

    def prepare_video(self, vd_mod):
        self.lasers = self.ctrl_panel.get_lasers()
        self.set_lasers(self.lasers)
        self.cameras["imaging"] = self.ctrl_panel.get_imaging_camera()
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
        if vd_mod == "Scan Calib":
            dtr, ptr, dch, pch = self.trg.generate_piezo_line_scan(self.lasers, self.cameras["imaging"])
            self.devs.daq.write_triggers(piezo_sequences=ptr, piezo_channels=pch,
                                         digital_sequences=dtr, digital_channels=dch, finite=False)
        if vd_mod == "Focus Lock":
            self.logg.info(f"Focus Lock live")

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
            self.devs.cam_set[self.cameras["imaging"]].start_live()
            if self.cameras["imaging"] != self.cameras["focus_lock"]:
                self.devs.daq.run_triggers()
            self.devs.cam_set[self.cameras["imaging"]].data.on_update(self.viewer.on_camera_update_from_thread)
            self.logg.info("Live Video Started")
        except Exception as e:
            self.logg.error(f"Error starting imaging video: {e}")
            self.stop_video()
            return

    def stop_video(self):
        try:
            self.devs.daq.stop_triggers()
            self.devs.cam_set[self.cameras["imaging"]].stop_live()
            self.logg.info(r"Live Video Stopped")
            if self.slm_seq != "None":
                self.devs.slm.deactivate()
            self.lasers_off()
        except Exception as e:
            self.logg.error(f"Error stopping imaging video: {e}")

    @pyqtSlot(bool)
    def fft(self, on: bool):
        if on:
            if getattr(self.viewer, "fft_worker", None) is None:
                self.viewer.fft_worker = run_threads.FFTWorker(fps=10)
                self.viewer.fft_worker.fftReady.connect(self.viewer.on_fft_frame, Qt.ConnectionType.QueuedConnection)
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
            self.viewer.update_plot(ipr.get_profile(self.viewer.image_viewer._display_frame, ax, norm=True))
        except Exception as e:
            self.logg.error(f"Error plotting profile: {e}")

    @pyqtSlot()
    def plot_add(self):
        try:
            ax = self.ctrl_panel.get_profile_axis()
            self.viewer.plot_profile(ipr.get_profile(self.viewer.image_viewer._display_frame, ax, norm=True))
        except Exception as e:
            self.logg.error(f"Error plotting profile: {e}")

    @pyqtSlot()
    def plot_trigger(self):
        try:
            self.slm_seq = self.ctrl_panel.get_slm_sequence()
            self.devs.slm.select_order(self.devs.slm.ord_dict[self.slm_seq])
            self.update_trigger_parameters("imaging")
            dtr, dch = self.trg.generate_digital_triggers(self.lasers, self.cameras["imaging"], self.slm_seq)
            self.viewer.update_plot(dtr[0])
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
