from PyQt6.QtCore import QObject, pyqtSlot
import time
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
        self.ctrl_panel.Signal_video.connect(self.video)
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

    # --- Live Video ----------------------------------------------------------
    @pyqtSlot(bool, str)
    def video(self, sw: bool, md: str):
        if sw:
            self.start_video(md)
        else:
            self.stop_video(md)

    def start_video(self, md):
        if self.acq_thread and self.acq_thread.isRunning():
            return
        self.acq_thread = run_threads.LiveViewThread(self.devs.camera, self.bus, interval_ms=50)
        self.acq_thread.start()
        self.logg.info(r"Live Video Started")

    def stop_video(self, md):
        if self.acq_thread:
            self.acq_thread.stop()
            self.acq_thread = None
        self.logg.info(r"Live Video Stopped")
