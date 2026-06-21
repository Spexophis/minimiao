# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


from minimiao import logger

try:
    from . import cobolt_laser
except ImportError:
    from minimiao.devices import cobolt_laser

try:
    from . import teledyne_kinetix
except ImportError:
    from minimiao.devices import teledyne_kinetix

try:
    from . import hamamatsu_slm
except ImportError:
    from minimiao.devices import hamamatsu_slm


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

"""
INFO: Kinetix camera opened: PMPCIECam00
INFO: Kinetix Serial Number: A24F723013
INFO: Sensor size: 2400 (ser) x 2400 (par)
INFO: Temperature setpoint: -10.0 °C
ERROR: No responce recieved for sn?
INFO: 473 Laser Connected
INFO: Turning on laser
INFO: Connected to SLM: LSH0805629
INFO: Head Temperature: 30.25, Controller Temperature: 58.814001464843784
INFO: Finish initiating devices
INFO: Kinetix camera closed
INFO: Turning off laser
INFO: SLM Closed

Process finished with exit code -1073741819 (0xC0000005)"""


if __name__ == '__main__':
    device_manager = DeviceManager()
    device_manager.close()
