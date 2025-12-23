# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


from . import image_reconstructions, shwfs_reconstruction, focus_lock_control, trigger_generator


class ComputationManager:
    def __init__(self, config=None, logg=None, path=None):
        self.config = config
        self.logg = logg or self.setup_logging()
        self.data_folder = path
        self.rec = image_reconstructions.ImgRecon(logg=self.logg)
        self.wfp = shwfs_reconstruction.WavefrontSensing(logg=self.logg)
        self.flp = focus_lock_control.FocusLocker()
        self.trg = trigger_generator.TriggerSequence(logg=self.logg)

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging
