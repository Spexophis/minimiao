# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import json
import os
import struct
import sys
import time

import numpy as np
import pandas as pd
import tifffile as tf

from minimiao import logger
from minimiao.utilities import zernike_generator as tz

sys.path.append(r'C:\Program Files\Alpao\SDK\Samples\Python3')
if (8 * struct.calcsize("P")) == 32:
    from Lib.asdk import DM
else:
    from Lib64.asdk import DM


class DeformableMirror:

    def __init__(self, name="ALPAO", logg=None, config=None, path=None, cfn=None):
        self.dtp = path
        self.cfn = cfn
        self.logg = logg or logger.setup_logging()
        self.config = config or self.load_configs()
        self.dm_name = name
        self.dm_serial = self.config["Adaptive Optics"]["Deformable Mirror"][self.dm_name]["Serial"]
        self.dm_model = self.dm_name + '_' + self.config["Adaptive Optics"]["Deformable Mirror"][self.dm_name]["Model"]
        self.dm, self.n_actuator = self._initialize_dm(self.dm_serial)
        if self.dm is not None:
            self._configure_dm()
            self._get_zernike()
        else:
            raise RuntimeError(f"Error Initializing DM {self.dm_name}")
        self.g = 0.5
        try:
            self.set_dm(self.dm_cmd[self.current_cmd])
        except Exception as e:
            self.logg.error(f"Error set dm {e}")

    def __del__(self):
        pass

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    @staticmethod
    def load_configs():
        import json
        config_file = input("Enter configuration file directory: ")
        with open(config_file, 'r') as f:
            cfg = json.load(f)
        return cfg

    def _initialize_dm(self, sn):
        try:
            dm = DM(sn)
            n_act = int(dm.Get('NBOfActuator'))
            self.logg.info("Number of actuator for " + sn + ": " + str(n_act))
            return dm, n_act
        except Exception as e:
            self.logg.error(f"Error Initializing DM {self.dm_name}: {e}")
            return None, None

    def _configure_dm(self):
        self.dm_cmd = []
        try:
            self.control_matrix_phase = tf.imread(
                self.config["Adaptive Optics"]["Deformable Mirror"][self.dm_name]["Phase Control Matrix"])
        except Exception as e:
            self.logg.error(f"Error Loading DM {self.dm_name} control file: {e}")
        try:
            self.control_matrix_zonal = tf.imread(
                self.config["Adaptive Optics"]["Deformable Mirror"][self.dm_name]["Zonal Control Matrix"])
        except Exception as e:
            self.logg.error(f"Error Loading DM {self.dm_name} control file: {e}")
        try:
            self.control_matrix_modal = tf.imread(
                self.config["Adaptive Optics"]["Deformable Mirror"][self.dm_name]["Modal Control Matrix"])
        except Exception as e:
            self.logg.error(f"Error Loading DM {self.dm_name} control file: {e}")
        try:
            self.ctrl_calib = self.config["Adaptive Optics"]["Deformable Mirror"][self.dm_name]["Control Calibration"]
        except Exception as e:
            self.logg.error(f"Error Loading DM {self.dm_name} control file: {e}")
        try:
            self.read_cmd(self.config["Adaptive Optics"]["Deformable Mirror"][self.dm_name]["Mirror Flat"])
        except Exception as e:
            self.dm_cmd = [[0.] * self.n_actuator]
            self.logg.error(f"Error Loading DM {self.dm_name} Mirror Flat: {e}\n")
        try:  
            self.read_cmd(self.config["Adaptive Optics"]["Deformable Mirror"][self.dm_name]["Initial Flat"])
            self.current_cmd = 1
        except Exception as e:
            self.current_cmd = 0
            self.logg.error(f"Error Loading DM {self.dm_name} Initial Flat: {e}\n Started with Null")
        self.correction = []
        self.temp_cmd = []
        self.amp = 0.1

    def _get_zernike(self):
        try:
            influence_function_images = tf.imread(
                self.config["Adaptive Optics"]["Deformable Mirror"][self.dm_name]["Influence Function Images"])
            nct, self.nly, self.nlx = influence_function_images.shape
            image = np.sum(influence_function_images, axis=0)
            msk = image != 0
            self.pupil_mask = msk
            self.nls = self.nly * self.nlx
            self.n_zernike = tz.num_znk
            self.az = None
            Z, dZdx, dZdy = tz.zernike_basis(self.nlx, self.nly, self.n_zernike, mask=msk, normalize_to="circle")
            self.zernike, dZdx_orth, dZdy_orth, T = tz.gs_orthogonalize(Z, msk, dZdx, dZdy)
            self.zslopes = np.zeros((2 * self.nlx * self.nly, self.n_zernike))
            for j in range(self.n_zernike):
                if j == 0:
                    self.zslopes[:self.nls, j] = dZdx_orth[j].flatten()
                    self.zslopes[self.nls:, j] = dZdy_orth[j].flatten()
                else:
                    self.zslopes[:self.nls, j] = (dZdx_orth[j] / np.std(dZdx_orth[j])).flatten()
                    self.zslopes[self.nls:, j] = (dZdy_orth[j] / np.std(dZdy_orth[j])).flatten()
            self._get_beam_zernike()
        except Exception as e:
            self.logg.error(f"Error Loading DM {self.dm_name} control file: {e}")

    def _get_beam_zernike(self):
        """
        Zernike basis restricted to the imaging beam.

        The influence function is calibrated with the beam filling the DM, but
        the imaging beam is smaller than the mirror. Commanding a full-pupil
        Zernike then gives the beam the *centre* of that mode, which is a
        mixture of lower-order modes, not the mode that was asked for. Instead
        the mode is written into the illuminated area only, blended smoothly to
        the flat command outside it: a sharp beam-sized crop is a step the
        mirror cannot make, and its fitting error ripples back into the beam.

        Config keys (under the DM entry, all optional):

            "Beam Diameter Ratio" : beam diameter / DM diameter, 0 < r <= 1.
                                    1.0 (default) keeps the full-pupil basis.
            "Beam Center Offset"  : [dx, dy] in lenslets, if the beam is not
                                    centred on the DM.
            "Beam Edge"           : "hermite" (default) blends outside the beam
                                    and keeps the mode exact inside it;
                                    "taper" apodizes inside the beam instead,
                                    for when there is no room around it.
            "Beam Edge Width"     : width of the blending ring, in units of the
                                    beam radius (default 0.25). Wider follows
                                    the mode more faithfully inside the beam and
                                    costs more stroke outside it.
            "Beam Edge Order"     : 1 (C¹ join) or 2 (C², default).
        """
        cfg = self.config["Adaptive Optics"]["Deformable Mirror"][self.dm_name]
        self.beam_ratio = float(cfg.get("Beam Diameter Ratio", 1.0))
        self.beam_mask = self.pupil_mask
        self.beam_support = self.pupil_mask
        if not 0.0 < self.beam_ratio <= 1.0:
            self.logg.error(f"DM {self.dm_name}: 'Beam Diameter Ratio' must be in (0, 1], "
                            f"got {self.beam_ratio} — using the full pupil")
            self.beam_ratio = 1.0
        if self.beam_ratio == 1.0:
            self.zernike_beam = self.zernike
            self.zslopes_beam = self.zslopes
            return
        try:
            cx, cy, r_pupil = tz.aperture_from_mask(self.pupil_mask)
            offset = cfg.get("Beam Center Offset", (0.0, 0.0))
            center = (cx + float(offset[0]), cy + float(offset[1]))
            modes = tz.zernike_sub_aperture(
                self.nlx, self.nly, self.n_zernike,
                radius=self.beam_ratio * r_pupil,
                center=center,
                edge=str(cfg.get("Beam Edge", "hermite")),
                edge_width=float(cfg.get("Beam Edge Width", 0.25)),
                blend_order=int(cfg.get("Beam Edge Order", 2)),
                pupil_mask=self.pupil_mask,
            )
            # Orthogonalize over the beam — the domain the beam actually sees.
            # Two details: gs_orthogonalize() rebuilds its modes from the masked
            # pixels only, which would chop the blending ring off, so only its
            # transformation matrix is used and applied to the full maps; and it
            # is fed mode shapes with their beam mean removed, so that the edge
            # piston deliberately left in defocus and spherical is not counted
            # as mode content (it would otherwise halve their amplitude).
            centred = modes.phase - np.where(modes.beam_mask, 1.0, 0.0) * np.array(
                [m[modes.beam_mask].mean() for m in modes.phase])[:, None, None]
            _, _, _, T = tz.gs_orthogonalize(centred, modes.beam_mask)
            self.zernike_beam = tz.apply_transform(modes.phase, T)
            self.zslopes_beam = tz.stack_slopes(tz.apply_transform(modes.dphase_dx, T),
                                                tz.apply_transform(modes.dphase_dy, T))
            self.beam_mask = modes.beam_mask
            self.beam_support = modes.support_mask
            self.logg.info(f"DM {self.dm_name}: Zernike modes restricted to a beam of "
                           f"{self.beam_ratio:.2f} x the DM diameter "
                           f"({int(self.beam_mask.sum())} of {int(self.pupil_mask.sum())} lenslets, "
                           f"{int(self.beam_support.sum())} including the blending ring)")
            ring = self.beam_support & ~self.beam_mask
            if np.any(ring):
                peak_beam = np.abs(self.zernike_beam[:, self.beam_mask]).max(axis=1)
                peak_ring = np.abs(self.zernike_beam[:, ring]).max(axis=1)
                cost = peak_ring[1:] / np.maximum(peak_beam[1:], 1e-12)
                self.logg.info(f"DM {self.dm_name}: blending ring needs up to "
                               f"{cost[:14].max():.1f} x the in-beam peak stroke over modes 1-15, "
                               f"{cost.max():.1f} x over all {self.n_zernike} modes — narrow "
                               f"'Beam Edge Width' or use 'taper' if stroke runs out")
            self.logg.info(f"DM {self.dm_name}: the 'modal' method stays full-pupil "
                           f"(its control matrix is calibrated per mode) — use 'zonal' or "
                           f"'phase' for beam-sized modes")
        except Exception as e:
            self.logg.error(f"Error building beam Zernike basis for DM {self.dm_name}: {e}\n"
                            f"Falling back to the full-pupil basis")
            self.beam_ratio = 1.0
            self.zernike_beam = self.zernike
            self.zslopes_beam = self.zslopes
            self.beam_mask = self.pupil_mask
            self.beam_support = self.pupil_mask

    def set_beam_aperture(self, ratio, offset=None, edge=None, edge_width=None, order=None):
        """
        Change the imaging beam aperture and rebuild the beam Zernike basis.

        Parameters mirror the "Beam ..." config keys; anything left at None
        keeps the configured value. Use this when the beam size changes with
        the objective / relay in use.
        """
        cfg = self.config["Adaptive Optics"]["Deformable Mirror"][self.dm_name]
        cfg["Beam Diameter Ratio"] = float(ratio)
        if offset is not None:
            cfg["Beam Center Offset"] = [float(offset[0]), float(offset[1])]
        if edge is not None:
            cfg["Beam Edge"] = str(edge)
        if edge_width is not None:
            cfg["Beam Edge Width"] = float(edge_width)
        if order is not None:
            cfg["Beam Edge Order"] = int(order)
        self._get_beam_zernike()

    def close(self):
        self.write_cmd(path=self.dtp, t=time.strftime("%Y%m%d%H%M%S") + '_')
        self.reset_dm()
        self.logg.info(f"DM {self.dm_name} Close")

    def reset_dm(self):
        self.dm.Reset()
        self.logg.info(f"DM {self.dm_name} Reset")

    def set_dm(self, values):
        if all(np.abs(v) < 1. for v in values):
            self.dm.Send(values)
            self.logg.info(f"DM {self.dm_name} set")
        else:
            raise ValueError("Some actuators exceed the DM push range!")

    def null_dm(self):
        self.dm.Send([0.] * self.n_actuator)
        self.logg.info(f"DM {self.dm_name} set to null")

    def get_zernike_cmd(self, j, a, method="modal", aperture="beam"):
        """
        Actuator commands (relative to the current flat) for one Zernike mode.

        Parameters
        ----------
        j : int
            Mode index, 0-based into the basis: j=0 is Noll Z1 (piston),
            j=1 is Noll Z2, and so on.
        a : float
            Mode amplitude. With the beam basis this is an RMS wavefront
            amplitude over the beam, in the units the control matrix was
            calibrated in.
        method : {"modal", "zonal", "phase"}
            "modal" uses the per-mode control matrix, "zonal" drives the
            measured x/y slopes, "phase" drives the wavefront map itself.
        aperture : {"beam", "full"}
            "beam" keeps the mode inside the illuminated area and leaves the
            rest of the mirror flat (see :meth:`_get_beam_zernike`); it is
            identical to "full" when no beam ratio is configured. The "modal"
            method is always full-pupil — its control matrix is calibrated per
            mode, so it cannot be restricted to a sub-aperture.
        """
        beam = aperture == "beam"
        if method == 'modal':
            voltages = self.control_matrix_modal[:, j] * a
            return voltages.tolist()
        if method == 'zonal':
            target = (self.zslopes_beam if beam else self.zslopes)[:, j] * a
            voltages = self.control_matrix_zonal @ target
            return voltages.tolist()
        if method == 'phase':
            target = (self.zernike_beam if beam else self.zernike)[j].ravel() * a
            voltages = self.control_matrix_phase @ target
            return voltages.tolist()
        return None

    def get_zernike_phase(self, coefficients, aperture="beam"):
        """
        Wavefront map a set of mode coefficients asks the DM for, on the
        influence-function grid. Useful to check what is being commanded
        before sending it, or to save it alongside a measurement.

        Parameters
        ----------
        coefficients : sequence of float — one amplitude per mode, or a
            (mode index, amplitude) mapping.
        aperture : {"beam", "full"}

        Returns
        -------
        phase : (nly, nlx) array
        """
        modes = self.zernike_beam if aperture == "beam" else self.zernike
        if isinstance(coefficients, dict):
            items = coefficients.items()
        else:
            items = enumerate(coefficients)
        phase = np.zeros((self.nly, self.nlx))
        for j, a in items:
            if not 0 <= j < self.n_zernike:
                raise IndexError(f"Zernike mode index {j} out of range (0 ... {self.n_zernike - 1})")
            phase = phase + a * modes[j]
        return phase

    @staticmethod
    def cmd_add(cmd_0, cmd_1):
        return list(np.asarray(cmd_0) + np.asarray(cmd_1))

    def read_cmd(self, fnd):
        df = pd.read_excel(fnd, sheet_name=None)
        for key, cmd in df.items():
            self.dm_cmd.append(df[key]['Push'].tolist())

    def write_cmd(self, path, t, flatfile=False):
        if flatfile:
            filename = t + f"{self.dm_serial}_flat_file.xlsx"
            df = pd.DataFrame(self.dm_cmd[-1], index=np.arange(self.n_actuator), columns=['Push'])
            fd = os.path.join(path, filename)
            df.to_excel(str(fd), index_label='Actuator')
        else:
            filename = t + f"{self.dm_serial}_cmd_file.xlsx"
            fd = os.path.join(path, filename)
            data = {f'cmd{i}': cmd for i, cmd in enumerate(self.dm_cmd)}
            with pd.ExcelWriter(str(fd), engine='xlsxwriter') as writer:
                for sheet_name, list_data in data.items():
                    df = pd.DataFrame(list_data, index=np.arange(self.n_actuator), columns=['Push'])
                    df.to_excel(writer, sheet_name=sheet_name, index_label='Actuator')

    def write_flat_cmd(self, t, cmd):
        path = self.config["Adaptive Optics"]["Deformable Mirror"][self.dm_name]["Calibration File Folder"]
        filename = f"flat_file_{self.dm_serial}_{t}.xlsx"
        fd = os.path.join(path, filename)
        df = pd.DataFrame(cmd, index=np.arange(self.n_actuator), columns=['Push'])
        df.to_excel(str(fd), index_label='Actuator')
        self.config["Adaptive Optics"]["Deformable Mirror"][self.dm_name]["Initial Flat"] = str(fd)
        with open(self.cfn, 'w') as f:
            json.dump(self.config, f, indent=4)

    def save_sensorless_results(self, fd, a, v, p):
        df1 = pd.DataFrame(v, index=a, columns=['Values'])
        df2 = pd.DataFrame(p, index=np.arange(self.n_zernike), columns=['Amplitudes'])
        with pd.ExcelWriter(fd, engine='xlsxwriter') as writer:
            df1.to_excel(writer, sheet_name='Metric Values')
            df2.to_excel(writer, sheet_name='Peaks')
