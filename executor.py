from PyQt6.QtCore import QObject, pyqtSlot, Qt
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
        self.logger = logger or self.setup_logging()
        self._set_signal_executions()

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    def _set_signal_executions(self):
        self.ctrl_panel.Signal_video.connect(self.video)

    @pyqtSlot(bool, str)
    def video(self, sw: bool, md: str):
        if sw:
            self.start_video(md)
        else:
            self.stop_video(md)

    def start_video(self, md):
        self.devs.camera.start_live()
        self.devs.camera.data.on_update(self.viewer.on_camera_update_from_thread)
        self.logger.info("Live Video Started")

    def stop_video(self, md):
        self.devs.camera.stop_live()
        self.logger.info(r"Live Video Stopped")
