# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


from minimiao import logger
from . import andor_emccd
from . import cobolt_laser
from . import fdd_slm
from . import mcl_deck
from . import mcl_piezo
from . import ni_daq
from . import phaseform_dpp
from . import thorlab_scmos
from . import thorlabs_motor
from . import neopixel_ring


class DeviceManager:
    def __init__(self, config=None, logg=None, path=None):
        self.config = config
        self.logg = logg or logger.setup_logging()
        self.data_folder = path
        try:
            self.img_cam = andor_emccd.EMCCDCamera(logg=self.logg)
        except Exception as e:
            from . import mock_cam
            self.img_cam = mock_cam.MockCamera()
            self.logg.error(f"{e}")
        try:
            self.dpc_cam = thorlab_scmos.ThorCMOS(logg=self.logg)
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.slm = fdd_slm.QXGA(logg=self.logg, config=self.config)
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.dfm = phaseform_dpp.DPP(logg=self.logg, config=self.config, path=self.data_folder)
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.laser = cobolt_laser.CoboltLaser(logg=self.logg, config=self.config)
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.daq = ni_daq.NIDAQ(logg=self.logg)
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.deck = mcl_deck.MCLMicroDrive(logg=self.logg)
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.piezo = mcl_piezo.MCLNanoDrive(logg=self.logg)
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.motor = thorlabs_motor.ELL14(logg=self.logg)
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.led = neopixel_ring.NeoPixel(logg=self.logg)
        except Exception as e:
            self.logg.error(f"{e}")
        self.logg.info("Finish initiating devices")

    def close(self):
        try:
            self.img_cam.close()
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.dpc_cam.close()
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.laser.close()
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.slm.close()
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.dfm.close()
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.daq.close()
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.deck.close()
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.piezo.close()
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.motor.close()
        except Exception as e:
            self.logg.error(f"{e}")


if __name__ == '__main__':
    import json

    with open(r"C:\Users\ruizhe.lin\Documents\data\config_files\microscope_configurations_20240426.json", 'r') as f:
        cfg = json.load(f)
    devs = DeviceManager(config=cfg)
    devs.close()
