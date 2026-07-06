# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


from minimiao import logger
from . import image_reconstructions, trigger_generator


class ComputationManager:
    def __init__(self, dev, config=None, logg=None, path=None):
        self.dev = dev
        self.config = config
        self.logg = logg or logger.setup_logging()
        self.data_folder = path
        self.rec = image_reconstructions.ImgRecon(logg=self.logg)
        self.trg = trigger_generator.TriggerSequence(galvo=self.dev.gvs,
                                                     piezo=self.dev.pz,
                                                     logg=self.logg)

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging
