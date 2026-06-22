# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.

import threading

import numpy as np

from minimiao import logger


class ImgRecon:

    def __init__(self, logg=None):
        self.logg = logg or logger.setup_logging()

        self._scan_gate_len = 1024
        self._point_scan_gate_mask = np.zeros(self._scan_gate_len, dtype=bool)
        self._off_gate_len = 1024
        self._switch_off_gate_mask = np.zeros(self._off_gate_len, dtype=bool)

        self.point_scan_n_pixels = 0
        self.point_scan_n_lines = 0
        self.point_scan_dwell_samples = 0
        self.point_scan_off_samples = 0
        self._expected = 0

        self.live_counts = []
        self.live_rec = []
        self._reshape_buffer = []

        self.lock = threading.Lock()

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    @property
    def point_scan_gate_mask(self) -> np.ndarray:
        return self._point_scan_gate_mask

    @property
    def switch_off_gate_mask(self) -> np.ndarray:
        return self._switch_off_gate_mask

    @property
    def scan_gate_len(self) -> int:
        return self._scan_gate_len

    @point_scan_gate_mask.setter
    def point_scan_gate_mask(self, gate_mask) -> None:
        gate_mask = np.asarray(gate_mask, dtype=bool)
        self._point_scan_gate_mask = np.roll(gate_mask, 1)
        self._rebuild_scan_gate_cache()

    @switch_off_gate_mask.setter
    def switch_off_gate_mask(self, gate_mask) -> None:
        gate_mask = np.asarray(gate_mask, dtype=bool)
        self._switch_off_gate_mask = np.roll(gate_mask, 1)
        self._rebuild_off_gate_cache()

    def set_point_scan_params(self, n_lines: int, n_pixels: int, dwell_samples: int) -> None:
        self.point_scan_n_lines = int(n_lines)
        self.point_scan_n_pixels = int(n_pixels)
        self.point_scan_dwell_samples = int(dwell_samples)
        self._rebuild_expected()

    def _rebuild_scan_gate_cache(self) -> None:
        m = self._point_scan_gate_mask
        self._scan_gate_len = int(m.shape[0])
        
    def _rebuild_off_gate_cache(self) -> None:
        n = self._switch_off_gate_mask
        self._off_gate_len = int(n.shape[0])

    def _rebuild_expected(self) -> None:
        self._expected = int(self.point_scan_n_lines) * int(self.point_scan_n_pixels) * int(
            self.point_scan_dwell_samples)

    def prepare_point_scan_live_recon(self, n=2):
        self.live_counts = [np.zeros(self._scan_gate_len, dtype=np.uint32) for _ in range(n)]
        self.live_rec = [np.zeros((self.point_scan_n_lines, self.point_scan_n_pixels), dtype=np.uint32) for _ in
                         range(n)]
        self._reshape_buffer = [
            np.zeros((self.point_scan_n_lines, self.point_scan_n_pixels, self.point_scan_dwell_samples),
                     dtype=np.uint32) for _ in range(n)]

    def point_scan_live_recon(self, photon_counts, ind_list, ind, bi_direction: bool = False):
        with self.lock:
            if photon_counts.shape[0] == len(ind_list):
                self.live_counts[ind][ind_list] = photon_counts
                gate = self._point_scan_gate_mask
                per_on = self.live_counts[ind][gate]

                if per_on.size != self._expected:
                    self.logg.error(f"Gate-on samples = {per_on.size}, expected {self._expected}. ")
                else:
                    self._reshape_buffer[ind][:] = per_on.reshape(
                        self.point_scan_n_lines,
                        self.point_scan_n_pixels,
                        self.point_scan_dwell_samples
                    )
                    np.sum(self._reshape_buffer[ind], axis=2, out=self.live_rec[ind])

                    if bi_direction:
                        self.live_rec[ind][1::2] = self.live_rec[ind][1::2, ::-1]

            else:
                self.logg.error(f"photon counts = {len(photon_counts)}, indices {len(ind_list)}. ")

    @staticmethod
    def full_recon(gate, counts, ny, nx):
        n_pixels = ny * nx

        on_gate = np.roll(np.asarray(gate[2], dtype=bool), 1)
        readout_gate = np.roll(np.asarray(gate[4], dtype=bool), 1)
        off_gate = np.roll(np.asarray(gate[3], dtype=bool), 1)

        per_on = counts[on_gate]
        per_readout = counts[readout_gate]
        per_off = counts[off_gate]

        assert len(per_on) % n_pixels == 0, f"Length is not divisible by {ny}*{nx}"
        dwell_on = len(per_on) // n_pixels

        assert len(per_off) % n_pixels == 0, f"Length is not divisible by {ny}*{nx}"
        dwell_off = len(per_off) // n_pixels

        assert len(per_readout) % n_pixels == 0, f"Length is not divisible by {ny}*{nx}"
        dwell_readout = len(per_readout) // n_pixels

        on_buf = per_on.reshape(ny, nx, dwell_on)
        on_img = np.sum(on_buf, axis=2)

        off_buf = per_off.reshape(ny, nx, dwell_off)
        off_img = np.sum(off_buf, axis=2)

        readout_buf = per_readout.reshape(ny, nx, dwell_readout)
        readout_img = np.sum(readout_buf, axis=2)

        return on_img, off_img, readout_img
