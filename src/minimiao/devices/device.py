# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


from minimiao import logger

try:
    from . import cobolt_laser, teledyne_kinetix, hamamatsu_slm
except ImportError:
    from minimiao.devices import cobolt_laser, teledyne_kinetix, hamamatsu_slm


class DeviceManager:

    def __init__(self, logg=None):

        self.logg = logg or logger.setup_logging()

        self.camera = None
        self.laser = None
        self.slm = None

        try:
            self.camera = teledyne_kinetix.KinetixCamera(logg=self.logg)
        except Exception as e:
            self.logg.error(f"Camera init failed: {e}")

        try:
            self.laser = cobolt_laser.CoboltLaser(logg=self.logg)
        except Exception as e:
            self.logg.error(f"Laser init failed: {e}")

        try:
            self.slm = hamamatsu_slm.HamamatsuSLM(logg=self.logg)
        except Exception as e:
            self.logg.error(f"SLM init failed: {e}")

        self.logg.info("Finish initiating devices")

    def close(self):

        if self.camera:
            try:
                self.camera.close()
            except Exception as e:
                self.logg.error(f"Camera close failed: {e}")

        if self.laser:
            try:
                self.laser.close()
            except Exception as e:
                self.logg.error(f"Laser close failed: {e}")

        if self.slm:
            try:
                self.slm.close()
            except Exception as e:
                self.logg.error(f"SLM close failed: {e}")


if __name__ == '__main__':
    device_manager = DeviceManager()
