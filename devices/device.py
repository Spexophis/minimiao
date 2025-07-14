from devices import mock_cam
from devices import andor_ixon
from devices import cobolt_laser
from devices import deformable_mirror
from devices import fdd_slm
from devices import hamamatsu_orchflash
from devices import mcl_maddeck
from devices import mcl_piezo
from devices import ni_daq


class DeviceManager:
    def __init__(self, bus, config=None, logg=None, path=None):
        self.camera = mock_cam.MockCamera()
        self.bus = bus
        self.config = config
        self.logg = logg or self.setup_logging()
        self.data_folder = path
        self.cam_set = {}
        try:
            self.cam_set[0] = andor_ixon.EMCCDCamera(logg=self.logg)
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.cam_set[1] = hamamatsu_orchflash.HamamatsuCamera(logg=self.logg)
        except Exception as e:
            self.logg.error(f"{e}")
        self.dm = {}
        for key in self.config["Adaptive Optics"]["Deformable Mirrors"].keys():
            try:
                self.dm[key] = deformable_mirror.DeformableMirror(name=key, logg=self.logg,
                                                                  config=self.config, path=self.data_folder)
            except Exception as e:
                self.logg.error(f"{e}")
        self.slm = {}
        try:
            self.slm["Binary"] = fdd_slm.QXGA(logg=self.logg, config=self.config)
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
            self.md = mcl_maddeck.MCLMicroDrive(logg=self.logg)
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.pz = mcl_piezo.MCLNanoDrive(logg=self.logg)
        except Exception as e:
            self.logg.error(f"{e}")
        self.logg.info("Finish initiating devices")

    def close(self):
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
            for key in self.dm.keys():
                self.dm[key].close()
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            for key in self.slm.keys():
                self.slm[key].close()
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.daq.close()
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.md.close()
        except Exception as e:
            self.logg.error(f"{e}")
        try:
            self.pz.close()
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
