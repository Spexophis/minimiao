# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


from minimiao import logger
from . import trigger_generator, pattern_generator


class ComputationManager:
    def __init__(self, config=None, logg=None, path=None):
        self.config = config
        self.logg = logg or logger.setup_logging()
        self.data_folder = path
        self.trg = trigger_generator.TriggerSequence(logg=self.logg)
        self.holo = pattern_generator.CGH(logg=self.logg)

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging
