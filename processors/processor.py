from processors import shwfs_processer, foclok_processor, trigger_generator


class ProcessorManager:
    def __init__(self, config=None, logg=None, path=None):
        self.config = config
        self.logg = logg or self.setup_logging()
        self.data_folder = path
        self.wfp = shwfs_processer.WavefrontSensing(logg=self.logg)
        self.flp = foclok_processor.FocusLocker()
        self.trg = trigger_generator.TriggerSequence(logg=self.logg)

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging
