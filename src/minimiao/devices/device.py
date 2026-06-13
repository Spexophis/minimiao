# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


from minimiao import logger

from . import cobolt_laser
from . import teledyne_kinetix
from . import hamamatsu_slm


class DeviceManager:
    def __init__(self, config=None, logg=None, path=None):
        self.config = config
        self.logg = logg or logger.setup_logging()
        self.data_folder = path
        try:
            self.camera = teledyne_kinetix.KinetixCamera(logg=self.logg, config=self.config)
        except Exception as e:
            from . import mock_cam
            self.camera = mock_cam.MockCamera()
            self.logg.error(f"{e}")
        try:
            self.laser = cobolt_laser.CoboltLaser(logg=self.logg, config=self.config)
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.slm = hamamatsu_slm.HamamatsuSLM(logg=self.logg, config=self.config)
        except Exception as e:
            self.logg.error(f"{e}")
        self.logg.info("Finish initiating devices")

    def close(self):
        try:
            self.camera.close()
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


if __name__ == '__main__':
    import json
    with open(r"C:\Users\ruizhe.lin\Documents\data\config_files\microscope_configurations_20240426.json", 'r') as f:
        cfg = json.load(f)
    devs = DeviceManager(config=cfg)
    devs.close()
