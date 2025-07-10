import run_threads


class CommandExecutor:
    def __init__(self, devices, ctrl_panel, viewer, path, logger):
        self.devs = devices
        self.ctrl_panel = ctrl_panel
        self.viewer = viewer
        self.acq_thread = None
        self.path = path
        self.logger = logger

    def start_acquisition(self):
        if self.acq_thread and self.acq_thread.isRunning():
            return
        interval = self.ctrl_panel.interval_spin.value()
        self.acq_thread = run_threads.AcquisitionThread(self.devs.camera, interval_ms=interval)
        self.acq_thread.new_frame.connect(self.viewer.update_image)
        self.acq_thread.start()
        self.ctrl_panel.start_btn.setEnabled(False)
        self.ctrl_panel.stop_btn.setEnabled(True)
        self.logger.info(r"Live Acquisition Started")

    def stop_acquisition(self):
        if self.acq_thread:
            self.acq_thread.stop()
            self.acq_thread = None
        self.ctrl_panel.start_btn.setEnabled(True)
        self.ctrl_panel.stop_btn.setEnabled(False)
        self.logger.info(r"Live Acquisition Stopped")
