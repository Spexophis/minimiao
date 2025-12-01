from devices import mock_cam
from devices import andor_emccd
from devices import cobolt_laser
from devices import fdd_slm
from devices import mcl_deck
from devices import mcl_piezo
from devices import ni_daq


class DeviceManager:
    def __init__(self, config=None, logg=None, path=None):
        # self.camera = mock_cam.MockCamera()
        self.config = config
        self.logg = logg or self.setup_logging()
        self.data_folder = path
        self.cam_set = {}
        try:
            self.camera = andor_emccd.EMCCDCamera(logg=self.logg)
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.slm = fdd_slm.QXGA(logg=self.logg, config=self.config)
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
        self.logg.info("Finish initiating devices")

    def close(self):
        pass
        try:
            for key in self.cam_set.keys():
                self.cam_set[key].close()
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

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging


if __name__ == '__main__':
    import json
    with open(r"C:\Users\ruizhe.lin\Documents\data\config_files\microscope_configurations_20240426.json", 'r') as f:
        cfg = json.load(f)
    devs = DeviceManager(config=cfg)
    devs.close()
