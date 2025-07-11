import run_threads


class CommandExecutor:
    def __init__(self, dev, cwd, pr, path, logger):
        self.devs = dev
        self.ctrl_panel = cwd.ctrl_panel
        self.viewer = cwd.viewer
        self.ao_panel = cwd.ao_panel
        self.trg = pr.trg
        self.wfr = pr.wfp
        self.flk = pr.flp
        self.acq_thread = None
        self.path = path
        self.logger = logger

    def start_acquisition(self):
        if self.acq_thread and self.acq_thread.isRunning():
            return
        self.acq_thread = run_threads.LiveViewThread(self.devs.camera, interval_ms=50)
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
