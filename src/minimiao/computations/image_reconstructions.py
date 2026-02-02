# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.

import numpy as np


class ImgRecon:

    def __init__(self, logg=None):
        self.logg = logg or logger.setup_logging()
        self._gate_len = 256
        self._point_scan_gate_mask = np.zeros(self._gate_len, dtype=bool)

        self.point_scan_n_pixels = 0
        self.point_scan_n_lines = 0
        self.point_scan_dwell_samples = 0
        self._expected = 0

        self.live_counts = None
        self.live_rec = None

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    @property
    def point_scan_gate_mask(self) -> np.ndarray:
        return self._point_scan_gate_mask

    @property
    def gate_len(self) -> int:
        return self._gate_len

    @point_scan_gate_mask.setter
    def point_scan_gate_mask(self, gate_mask) -> None:
        gate_mask = np.asarray(gate_mask, dtype=bool)
        self._point_scan_gate_mask = np.roll(gate_mask, 1)
        self._rebuild_gate_cache()

    def set_point_scan_params(self, n_lines: int, n_pixels: int, dwell_samples: int) -> None:
        self.point_scan_n_lines = int(n_lines)
        self.point_scan_n_pixels = int(n_pixels)
        self.point_scan_dwell_samples = int(dwell_samples)
        self._rebuild_expected()

    def _rebuild_gate_cache(self) -> None:
        m = self._point_scan_gate_mask
        self._gate_len = int(m.shape[0])

    def _rebuild_expected(self) -> None:
        self._expected = int(self.point_scan_n_lines) * int(self.point_scan_n_pixels) * int(
            self.point_scan_dwell_samples)

    def point_scan_img_recon(self, photon_counts, bi_direction: bool = False):
        if self._expected <= 0:
            raise ValueError("Scan params not set. Call set_point_scan_params(n_lines, n_pixels, dwell_samples).")

        photon_counts = np.array(photon_counts)
        gate = self._point_scan_gate_mask
        per_on = photon_counts[gate]

        if per_on.size != self._expected:
            raise ValueError(f"Gate-on samples = {per_on.size}, expected {self._expected}. ")

        img = per_on.reshape(
            self.point_scan_n_lines,
            self.point_scan_n_pixels,
            self.point_scan_dwell_samples,
        ).sum(axis=2)

        if bi_direction:
            img[1::2] = img[1::2, ::-1]

        return img

    def prepare_point_scan_live_recon(self):
        self.live_counts = np.zeros(self._gate_len, dtype=np.uint16)
        self.live_rec = np.zeros((self.point_scan_n_lines, self.point_scan_n_pixels), dtype=np.uint16)

    def point_scan_live_recon(self, photon_counts, ind, bi_direction: bool = False):
        if len(photon_counts) == len(ind):
            photon_counts = np.array(photon_counts)
            self.live_counts[ind] = photon_counts
            gate = self._point_scan_gate_mask
            per_on = self.live_counts[gate]

            if per_on.size != self._expected:
                self.logg.error(f"Gate-on samples = {per_on.size}, expected {self._expected}. ")
            else:
                img = per_on.reshape(
                    self.point_scan_n_lines,
                    self.point_scan_n_pixels,
                    self.point_scan_dwell_samples,
                ).sum(axis=2)

                if bi_direction:
                    img[1::2] = img[1::2, ::-1]

                self.live_rec = img
        else:
            self.logg.error(f"photon counts = {len(photon_counts)}, indices {len(ind)}. ")
