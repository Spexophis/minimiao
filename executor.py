from PyQt6.QtCore import QObject, pyqtSlot, Qt
import run_threads
import time

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

        self.ctrl_panel.Signal_video.connect(self.video)
        
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

    @pyqtSlot(int)
    def update_daq_sample_rate(self, sr: int):
        self.trg.update_nidaq_parameters(sr * 1000)
        self.devs.daq.sample_rate = sr * 1000

    @pyqtSlot()
    def reset_daq_channels(self):
        self.devs.daq.stop_triggers()

    @pyqtSlot(bool, str)
    def video(self, sw: bool, md: str):
        if sw:
            self.start_video(md)
        else:
            self.stop_video(md)

    def start_video(self, md):
        self.devs.camera.start_live()
        self.devs.camera.data.on_update(self.viewer.on_camera_update_from_thread)
        self.logg.info("Live Video Started")

    def stop_video(self, md):
        self.devs.camera.stop_live()
        self.logg.info(r"Live Video Stopped")
