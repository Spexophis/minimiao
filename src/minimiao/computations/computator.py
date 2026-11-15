# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


from minimiao import logger

try:
    from . import pattern_generator
except ImportError:
    from minimiao.computations import pattern_generator


class ComputationManager:
    def __init__(self, logg=None, path=None):

        self.logg = logg or logger.setup_logging()
        self.data_folder = path
        self.cgh = pattern_generator.CGH(logg=self.logg)


if __name__ == '__main__':
    computation_manager = ComputationManager()