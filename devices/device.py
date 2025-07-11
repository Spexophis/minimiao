# import cam_sim

import andor_ixon
import cobolt_laser
import deformable_mirror
import fdd_slm
import hamamatsu_orchflash
import mcl_maddeck
import mcl_piezo
import ni_daq


class DeviceManager:
    def __init__(self, config, logg, path):
        # self.camera = cam_sim.SimulatedCamera()
        self.config = config
        self.logg = logg
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
