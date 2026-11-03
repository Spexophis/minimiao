# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


from minimiao import logger

try:
    from . import trigger_generator
except ImportError as e:
    from minimiao.computations import trigger_generator


class ComputationManager:
    def __init__(self, dev, config=None, logg=None, path=None):
        self.dev = dev
        self.config = config
        self.logg = logg or logger.setup_logging()
        self.data_folder = path
        self.trg = trigger_generator.TriggerSequence(galvo=self.dev.gvs, logg=self.logg)
