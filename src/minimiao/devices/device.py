# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


from minimiao import logger

try:
    from . import andor_emccd, cobolt_laser, fdd_slm, mcl_deck, mcl_piezo, ni_daq, phaseform_dpp, thorlab_scmos, \
        thorlabs_motor, neopixel_ring
except ImportError as e:
    from minimiao.devices import andor_emccd, cobolt_laser, fdd_slm, mcl_deck, mcl_piezo, ni_daq, phaseform_dpp, \
        thorlab_scmos, thorlabs_motor, neopixel_ring


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
