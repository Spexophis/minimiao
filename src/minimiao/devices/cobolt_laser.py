# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import pycobolt

from minimiao import logger


class CoboltLaser:

    def __init__(self, logg=None):
        self.logg = logg or logger.setup_logging()
        laser_dict = {"473": "32010"}
        self.lasers, self._h = self._initiate_lasers(laser_dict)
        self.laser_on("all")

    def _initiate_lasers(self, laser_dict):
        lasers = {}
        for laser, sn in laser_dict.items():
            try:
                lasers[laser] = pycobolt.Cobolt06MLD(port="COM4")
                self.logg.info("{} Laser Connected".format(laser))
            except Exception as e:
                self.logg.error(f"Laser Error: {e}")
        _h = {key: True for key in lasers.keys()}
        return lasers, _h

    def close(self):
        self.laser_off("all")
        for key, _l in self._h.items():
            if _l:
                del self.lasers[key]

    def laser_off(self, laser):
        if laser == "all":
            for key, _l in self.lasers.items():
                _l.turn_off()
        else:
            for ind, ln in enumerate(laser):
                if self._h.get(ln, False):
                    self.lasers[ln].turn_off()

    def laser_on(self, laser):
        if laser == "all":
            for key, _l in self.lasers.items():
                _l.turn_on()
        else:
            for ind, ln in enumerate(laser):
                if self._h.get(ln, False):
                    self.lasers[ln].turn_on()

    def set_constant_power(self, laser: list, power: list):
        for ind, ln in enumerate(laser):
            if self._h.get(ln, False):
                self.lasers[ln].constant_power(power[ind])

    def set_modulation_mode(self, laser: list, pws: list):
        for ind, ln in enumerate(laser):
            if self._h.get(ln, False):
                self.lasers[ln].digital_modulation(1)
                self.lasers[ln].modulation_mode(pws[ind])
