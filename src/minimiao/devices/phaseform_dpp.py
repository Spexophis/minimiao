# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import os
from pathlib import Path

import numpy as np
import pandas as pd
from dpp_ctrl import api_dpp

from minimiao import logger

zms = [(-1, 1), (1, 1),
       (0, 2), (-2, 2), (2, 2),
       (-1, 3), (1, 3), (-3, 3), (3, 3),
       (0, 4), (-2, 4), (2, 4), (-4, 4), (4, 4),
       (-1, 5), (1, 5), (-3, 5), (3, 5), (-5, 5), (5, 5),
       (0, 6), (-2, 6), (2, 6), (-4, 6), (4, 6), (-6, 6), (6, 6),
       (-1, 7), (1, 7), (-3, 7), (3, 7), (-5, 7), (5, 7), (-7, 7), (7, 7)]

vms = [(-2, 2), (2, 2),
       (-1, 1), (-1, 1), (-1, 1),
       (-0.5, 0.5), (-0.5, 0.5), (-0.6, 0.6), (-0.6, 0.6),
       (-0.25, 0.25), (-0.25, 0.25), (-0.25, 0.25), (-0.25, 0.25), (-0.25, 0.25),
       (-0.15, 0.15), (-0.15, 0.15), (-0.15, 0.15), (-0.15, 0.15), (-0.15, 0.15), (-0.15, 0.15),
       (-0.07, 0.07), (-0.07, 0.07), (-0.07, 0.07), (-0.07, 0.07), (-0.07, 0.07), (-0.1, 0.1), (-0.1, 0.1),
       (-0.07, 0.07), (-0.07, 0.07), (-0.07, 0.07), (-0.07, 0.07), (-0.07, 0.07), (-0.07, 0.07), (-0.07, 0.07),
       (-0.07, 0.07)]

assert len(zms) == len(vms), "zms and vms must have the same length"

nz = len(zms)
vm = 300


class DPP:

    def __init__(self, logg=None, config=None, path=None):
        self.logg = logg or logger.setup_logging()
        self.config = config or self.load_configs()
        self.dtp = path
        self.opm = 'v'
        self.dpp, self.opened_flag = self._initialize()
        if self.opened_flag:
            self.dpp_model = self.config["Adaptive Optics"]["Deformable Phase Plate"]["PhaseForm"]["Model"]
            self._configure_dpp()
            self.n_zernike = nz

    def __del__(self):
        pass

    def close(self):
        self.dpp.zero_outputs()
        self.dpp.close()

    @staticmethod
    def load_configs():
        import json
        config_file = input("Enter configuration file directory: ")
        with open(config_file, 'r') as f:
            cfg = json.load(f)
        return cfg

    def _initialize(self):
        dpp = api_dpp.initialize()
        cal_file_dir = self.config["Adaptive Optics"]["Deformable Phase Plate"]["PhaseForm"]["Control Calibration"]
        cal_file = Path(cal_file_dir).expanduser()
        port = self.config["Adaptive Optics"]["Deformable Phase Plate"]["PhaseForm"]["COM"]
        if dpp.connect_device(port_name=port):
            if dpp.load_infl_matrix(str(cal_file), operation_mode=self.opm):
                dpp.set_max_voltage(vm)
                return dpp, True
            else:
                self.logg.error("Cannot connect DPP!")
                return False, False
        else:
            self.logg.error("Cannot connect DPP!")
            return False, False

    def _configure_dpp(self):
        self.initial_flat = self.config["Adaptive Optics"]["Deformable Phase Plate"]["PhaseForm"]["Initial Flat"]
        self.dpp_cmd = [[0.] * nz]
        self.read_cmd(self.initial_flat)
        self.current_cmd = 1
        self.set_dpp(self.dpp_cmd[self.current_cmd])
        self.correction = []
        self.temp_cmd = []

    def set_dpp(self, amps):
        assert len(zms) == len(vms) == len(amps), "zms, vms, and amps must have the same length"

        phases = {}

        for zm, amp, (vmin, vmax) in zip(zms, amps, vms):
            if vmin <= amp <= vmax:
                phases[zm] = amp
            else:
                phases[zm] = 0

        self.dpp.apply_phases(phases)

    def set_zm(self, zm, amp):
        amps_temp = self.dpp_cmd[self.current_cmd].copy()
        vmin, vmax = vms[zm]
        amp_temp = amp + amps_temp[zm]
        if vmin <= amp_temp <= vmax:
            amps_temp[zm] = amp_temp
        self.set_dpp(amps_temp)
        self.logg.info(f"set zernike mode {zm} to amplitude {amp_temp}")

    def read_cmd(self, fnd):
        df = pd.read_excel(fnd, sheet_name=None)
        for key, cmd in df.items():
            self.dpp_cmd.append(df[key]['Amp'].tolist())

    def write_cmd(self, path, t, flatfile=False):
        if flatfile:
            filename = t + f"{self.dpp_model}_flat_file.xlsx"
            df = pd.DataFrame(self.dpp_cmd[self.current_cmd], index=np.arange(nz), columns=['Amp'])
            fd = os.path.join(path, filename)
            df.to_excel(str(fd), index_label="Mode")
        else:
            filename = t + f"{self.dpp_model}_cmd_file.xlsx"
            fd = os.path.join(path, filename)
            data = {f'cmd{i}': cmd for i, cmd in enumerate(self.dpp_cmd)}
            with pd.ExcelWriter(str(fd), engine='xlsxwriter') as writer:
                for sheet_name, list_data in data.items():
                    df = pd.DataFrame(list_data, index=np.arange(nz), columns=["Amp"])
                    df.to_excel(writer, sheet_name=sheet_name, index_label="Mode")

    def write_flat_cmd(self, t, cmd):
        path = self.config["Adaptive Optics"]["Deformable Phase Plate"]["PhaseForm"]["Calibration File Folder"]
        filename = f"flat_file_{self.dpp_model}_{t}.xlsx"
        fd = os.path.join(path, filename)
        df = pd.DataFrame(cmd, index=np.arange(nz), columns=["Amp"])
        df.to_excel(str(fd), index_label="Mode")
        self.config["Adaptive Optics"]["Deformable Mirrors"]["PhaseForm"]["Initial Flat"] = str(fd)
        self.config.write_config(self.config, self.config.cfd)

    def save_sensorless_results(self, fd, a, v, p):
        df1 = pd.DataFrame(v, index=a, columns=['Values'])
        df2 = pd.DataFrame(p, index=np.arange(nz), columns=['Amplitudes'])
        with pd.ExcelWriter(fd, engine='xlsxwriter') as writer:
            df1.to_excel(writer, sheet_name='Metric Values')
            df2.to_excel(writer, sheet_name='Peaks')
