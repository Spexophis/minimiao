# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.

from minimiao import logger
from . import alpao_dm
from . import cobolt_laser
from . import ni_daq
from . import galvo_mirror
from . import pi_piezo


class DeviceManager:
    def __init__(self, config=None, logg=None, path=None, cf=None):
        self.config = config
        self.logg = logg or logger.setup_logging()
        self.data_folder = path
        self.cf = cf
        try:
            self.laser = cobolt_laser.CoboltLaser(logg=self.logg, config=self.config)
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.daq = ni_daq.NIDAQ(logg=self.logg)
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.dfm = alpao_dm.DeformableMirror(logg=self.logg, config=self.config, path=self.data_folder, cfn=self.cf)
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.gvs = galvo_mirror.GalvoWaveform(logg=self.logg)
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.pz = pi_piezo.PiezoWaveform(logg=self.logg)
        except Exception as e:
            self.logg.error(f"{e}")
        self.logg.info("Finish initiating devices")

    def close(self):
        try:
            self.laser.close()
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.daq.close()
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.dfm.close()
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            del self.gvs
        except Exception as e:
            self.logg.error(f"{e}")
