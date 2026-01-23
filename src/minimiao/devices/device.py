# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


from . import cobolt_laser
from . import flir_cmos
from . import ni_daq
from . import alpao_dm

class DeviceManager:
    def __init__(self, config=None, logg=None, path=None, cf=None):
        self.config = config
        self.logg = logg or self.setup_logging()
        self.data_folder = path
        self.cf = cfg
        try:
            self.camera = flir_cmos.FLIRCamera(logg=self.logg)
        except Exception as e:
            from . import mock_cam
            self.camera = mock_cam.MockCamera()
            self.logg.error(f"{e}")
        try:
            self.laser = cobolt_laser.CoboltLaser(logg=self.logg, config=self.config)
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.daq = ni_daq.NIDAQ(logg=self.logg)
        except Exception as e:
            self.logg.error(f"{e}")
        self.logg.info("Finish initiating devices")
        try:
            self.dm = alpao_dm.DeformableMirror(logg=self.logg, config=self.config, path=self.data_folder, cf=self.cf)
        except Exception as e:
            self.logg.error(f"{e}")
        self.logg.info("Finish initiating devices")

    def close(self):
        pass
        try:
            self.camera.close()
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.laser.close()
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.daq.close()
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.dm.close()
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
