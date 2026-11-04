# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


from dataclasses import dataclass

import numpy as np
from PIL import Image
import tifffile as tf
from pathlib import Path

from minimiao import logger


@dataclass
class HoloSystem:
    wavelength: float = 473e-9  # laser wavelength

    _magnification: float = 1.0

    _f_fresnel: float = 320e-3
    slm_pixel_pitch: float = 12.5e-6

    image_center_x: int = 600
    image_center_y: int = 395
    slm_nx: int = 1024  # number of pixels in x
    slm_ny: int = 1272  # number of pixels in y
    slm_offset_x: int = 0
    slm_offset_y: int = 0
    phase_nx: int = 850
    phase_ny: int = 850
    correction_value: int = 137
    _correction_pattern: np.ndarray = None

    trigger_value: int = 128

    def __post_init__(self):
        self._correction_pattern = np.zeros((self.slm_ny, self.slm_nx), dtype=np.uint8)

    @property
    def magnification(self) -> float:
        return self._magnification

    @magnification.setter
    def magnification(self, value: float):
        self._magnification = value

    @property
    def f_fresnel(self) -> float:
        return self._f_fresnel

    @f_fresnel.setter
    def f_fresnel(self, value: float):
        self._f_fresnel = value

    @property
    def correction_pattern(self) -> np.ndarray:
        return self._correction_pattern

    @correction_pattern.setter
    def correction_pattern(self, pattern: np.ndarray):
        self._correction_pattern = pattern


class CGH:

    def __init__(self, logg=None):
        self.logg = logg or logger.setup_logging()
        self.holo_sys = HoloSystem()
        self.spots_xy = None
        self.cell_mask = None
        self.itn = 64
        self.phase_slm = None
        self.holograms = []
        self.affine = None        # 2x3 camera -> hologram-plane matrix (slm_calibration)
        self.use_affine = False
        self.phase_total = np.zeros((self.holo_sys.phase_ny, self.holo_sys.phase_nx), dtype=np.uint8)

    def load_correction_pattern(self, file):
        correct_pattern = Image.open(file)
        self.holo_sys.correction_pattern = np.array(correct_pattern, dtype=np.uint8)

    def update_parameters(self, itn, mag, fl, cnt, offset):
        self.itn = itn
        self.holo_sys.magnification = mag
        self.holo_sys.f_fresnel = fl
        self.holo_sys.image_center_x, self.holo_sys.image_center_y = cnt
        self.holo_sys.slm_offset_x, self.holo_sys.slm_offset_y = offset

    def load_mask(self, fnd):
        ext = Path(fnd).suffix.lower()
        if ext == ".bmp":
            img = Image.open(fnd)
            if img.mode != "1":
                img = img.convert("L")
        elif ext in (".tif", ".tiff"):
            img = tf.imread(fnd)
        else:
            self.logg.error("Invalid file type")
            return
        self.cell_mask = np.asarray(img)

    def load_spots_picked(self, spots_picked):
        self.spots_xy = spots_picked

    def build_target_from_spots(self, spots_sample):
        target_amp = np.zeros((self.holo_sys.phase_ny, self.holo_sys.phase_nx), dtype=np.float32)
        for (xx, yy, zz, ii) in spots_sample:
            target_amp[self.holo_sys.phase_ny - round(yy), self.holo_sys.phase_nx - round(xx)] = ii
        return target_amp

    def gerchberg_saxton_multiplane(self, targets, defocus_phases, plane_weights=None,
                                    weighted_gs=True, initial_phase=None):
        """
        Multi-plane weighted Gerchberg-Saxton phase retrieval.

        A single common SLM phase pattern is optimized to reproduce M target
        intensity patterns simultaneously, each at its own axial plane. Each
        iteration: propagate the common phase to every plane, apply that
        plane's (weighted) amplitude constraint, undo the per-plane defocus,
        and coherently sum the M corrected fields back at the SLM plane.

        Parameters
        ----------
        targets : list of np.ndarray
            Target intensity for each plane, shape (phase_ny, phase_nx).
        defocus_phases : list of np.ndarray
            Per-plane defocus phase (radians), e.g. generate_defocus_phase(z_m).
            Use a zero array for the nominal in-focus plane.
        plane_weights : list of float, optional
            Relative energy budget per plane before summation (default: equal).
        weighted_gs : bool
            Apply per-pixel weighting within each plane's target mask.
        initial_phase : ndarray, optional
            Initial SLM phase guess.

        Returns
        -------
        phase_slm : ndarray
            Common SLM phase (radians, wrapped) reproducing all M planes.
        """
        assert len(targets) == len(defocus_phases)
        M = len(targets)
        size = targets[0].shape
        source = np.ones(size, dtype=complex)

        if plane_weights is None:
            plane_weights = [1.0] * M

        sqrt_targets = [np.sqrt(t) for t in targets]
        target_masks = [t != 0 for t in targets]

        if initial_phase is not None:
            phase_slm = initial_phase
        else:
            np.random.seed(1)
            phase_slm = np.random.uniform(-np.pi, np.pi, size)

        if weighted_gs:
            weights = [np.ones(size, dtype=float) for _ in range(M)]

        for k in range(self.itn):
            u_merged = np.zeros(size, dtype=complex)

            for m in range(M):
                # forward: SLM phase + this plane's defocus, then propagate
                u_m = source * np.exp(1j * (phase_slm + defocus_phases[m]))
                v_m = np.fft.fftshift(np.fft.fft2(u_m))

                amplitude = np.abs(v_m)
                phase = np.angle(v_m)
                signal = self.normalize(amplitude ** 2)

                mask = target_masks[m]
                if weighted_gs:
                    weights[m][mask] *= np.sqrt(targets[m][mask] / np.maximum(signal[mask], 1e-10))
                    v_corrected = weights[m] * sqrt_targets[m] * np.exp(1j * phase)
                else:
                    v_corrected = sqrt_targets[m] * np.exp(1j * phase)

                # back-propagate to SLM plane and undo this plane's defocus
                u_back = np.fft.ifft2(np.fft.ifftshift(v_corrected))
                u_back = u_back * np.exp(-1j * defocus_phases[m])

                u_merged += plane_weights[m] * u_back

                self.logg.info((k, m, self.eval_performances(signal, targets[m])))

            phase_slm = np.angle(u_merged)

        return phase_slm

    CGH_MODES = ("Sequential", "Simultaneous", "Hybrid")

    def compute_cgh(self, trg=True):
        self.phase_slm = self.compute_cgh_multiplane(self.spots_xy, trg=trg)

    def compute_holograms(self, mode="Simultaneous", trg=True):
        """
        Build the hologram list for the selected mode from self.spots_xy.

        Sequential   : one single-spot hologram per picked spot
        Simultaneous : one hologram containing all picked spots
        Hybrid       : the all-spot hologram first, then one hologram per spot

        Returns (and stores in self.holograms) a list of dicts
        {"label": str, "spots": [spot indices], "pattern": uint8 SLM array};
        list position i is meant for SLM memory slot i + 1 (slot 0 holds the
        correction pattern).
        """
        if mode not in self.CGH_MODES:
            raise ValueError(f"Unknown CGH mode {mode!r}; expected one of {self.CGH_MODES}")
        if not self.spots_xy:
            raise ValueError("No spots picked")
        holograms = []
        if mode in ("Simultaneous", "Hybrid"):
            pattern = self.compute_cgh_multiplane(self.spots_xy, trg=trg)
            holograms.append({"label": "all", "spots": list(range(len(self.spots_xy))), "pattern": pattern})
        if mode in ("Sequential", "Hybrid"):
            for i, spot in enumerate(self.spots_xy):
                pattern = self.compute_single_spot(spot, trg=trg)
                if pattern is None:
                    self.logg.warning(f"Spot {i} {tuple(spot[:2])} is outside the hologram field; skipped")
                    continue
                holograms.append({"label": f"spot {i}", "spots": [i], "pattern": pattern})
        self.holograms = holograms
        self.phase_slm = holograms[0]["pattern"] if holograms else None
        return holograms

    def set_affine(self, M):
        """Set (or clear with None) the 2x3 camera -> hologram-plane affine."""
        self.affine = None if M is None else np.asarray(M, dtype=float).reshape(2, 3)

    def _map_spot(self, x0, y0):
        """Camera pixel -> hologram-plane pixel, or None if outside the field.
        Uses the affine calibration when enabled, else magnification + center."""
        if self.use_affine and self.affine is not None:
            x1, y1 = self.affine @ np.array([x0, y0, 1.0])
        else:
            x1 = (x0 - self.holo_sys.image_center_x) * self.holo_sys.magnification + self.holo_sys.phase_nx / 2
            y1 = (y0 - self.holo_sys.image_center_y) * self.holo_sys.magnification + self.holo_sys.phase_ny / 2
        return self._in_field(x1, y1)

    def _in_field(self, x1, y1):
        # target index is phase_n - round(coord), so round(coord) must be in 1..phase_n
        if 1 <= round(x1) <= self.holo_sys.phase_nx and 1 <= round(y1) <= self.holo_sys.phase_ny:
            return x1, y1
        return None

    def _finalize(self, phase_spots, trg=True):
        """Add lens, quantize, embed in the correction pattern, add trigger patch."""
        phase_fresnel_lens = self.generate_fresnel_lens_phase()
        phase_total = (phase_spots + phase_fresnel_lens) % (2.0 * np.pi)
        phase_uint = self.phase_to_uint(phase_total, self.holo_sys.correction_value)
        pattern = self.add_arrays(phase_uint, self.holo_sys.correction_pattern.copy(),
                                  self.holo_sys.slm_offset_x, self.holo_sys.slm_offset_y)
        if trg:
            self.add_trigger_pattern(pattern)
        return pattern

    def compute_cgh_plane(self, trg=True):
        spots_sample = []
        for (x0, y0, z0, I0) in self.spots_xy:
            xy = self._map_spot(x0, y0)
            if xy is not None:
                spots_sample.append((*xy, z0, I0))
        target_amp = self.build_target_from_spots(spots_sample)
        self.phase_slm = self._finalize(self.gerchberg_saxton(target_amp), trg=trg)

    def compute_single_spot(self, spot, z_bin_um=1.0, trg=True):
        """
        Exact hologram for one spot: the inverse FFT of a single point is a
        pure grating, so no GS iterations are needed. Uses the same mapping,
        z convention and forward model as compute_cgh_multiplane.
        Returns None if the spot falls outside the hologram field.
        """
        x0, y0, z0, I0 = spot
        xy = self._map_spot(x0, y0)
        if xy is None:
            return None
        return self.compute_hologram_point(*xy, z_um=round(z0 / z_bin_um) * z_bin_um, trg=trg)

    def compute_hologram_point(self, x1, y1, z_um=0.0, trg=True):
        """Exact single-spot hologram at hologram-plane pixel (x1, y1), z in um.
        Returns None if the point is outside the field."""
        if self._in_field(x1, y1) is None:
            return None
        target = self.build_target_from_spots([(x1, y1, 0, 1.0)])
        phase_spot = np.angle(np.fft.ifft2(np.fft.ifftshift(target)))
        z_offset = z_um * 1e-6
        if z_offset:
            phase_spot = phase_spot - self.generate_defocus_phase(z_offset)
        return self._finalize(phase_spot, trg=trg)

    def compute_cgh_multiplane(self, spots, z_bin_um=1.0, plane_weights=None, trg=True):
        """
        Multi-plane hologram for all given spots. Groups spots by z
        (spot z in um, binned to z_bin_um) and runs iterative multi-plane weighted GS.
        Returns the uint8 SLM pattern.
        """
        groups = {}
        for (x0, y0, z0, I0) in spots:
            key = round(z0 / z_bin_um)
            groups.setdefault(key, []).append((x0, y0, I0))

        targets, defocus_phases = [], []
        for key, group in groups.items():
            z_offset = key * z_bin_um * 1e-6  # m
            spots_sample = []
            for (x0, y0, I0) in group:
                xy = self._map_spot(x0, y0)
                if xy is not None:
                    spots_sample.append((*xy, 0, I0))
            if not spots_sample:
                continue
            targets.append(self.build_target_from_spots(spots_sample))
            defocus_phases.append(self.generate_defocus_phase(z_offset))
        if not targets:
            raise ValueError("All spots are outside the hologram field")

        phase_spots = self.gerchberg_saxton_multiplane(targets, defocus_phases, plane_weights)
        return self._finalize(phase_spots, trg=trg)

    def generate_defocus_phase(self, delta_z: float) -> np.ndarray:
        """
        Exact (non-paraxial) defocus phase for an axial offset delta_z from the
        nominal focal plane (referenced to f_fresnel), on the same SLM-plane
        grid as generate_fresnel_lens_phase.
        """
        x = (np.arange(self.holo_sys.phase_nx) - self.holo_sys.phase_nx / 2 + 0.5) * self.holo_sys.slm_pixel_pitch
        y = (np.arange(self.holo_sys.phase_ny) - self.holo_sys.phase_ny / 2 + 0.5) * self.holo_sys.slm_pixel_pitch
        X, Y = np.meshgrid(x, y)
        k = 2.0 * np.pi / self.holo_sys.wavelength
        f = self.holo_sys.f_fresnel
        rho2 = X ** 2 + Y ** 2
        phi = k * delta_z * (np.sqrt(np.clip(1.0 - rho2 / f ** 2, 0.0, None)) - 1.0)
        return np.mod(phi, 2.0 * np.pi)

    def generate_fresnel_lens_phase(self):
        """
        Generate a Fresnel lens phase pattern

        Returns
        -------
        phase : 2D np.ndarray
            Phase map in radians, wrapped to [0, 2π)
        """

        x = (np.arange(self.holo_sys.phase_nx) - self.holo_sys.phase_nx / 2 + 0.5) * self.holo_sys.slm_pixel_pitch
        y = (np.arange(self.holo_sys.phase_ny) - self.holo_sys.phase_ny / 2 + 0.5) * self.holo_sys.slm_pixel_pitch
        X, Y = np.meshgrid(x, y)
        k = 2.0 * np.pi / self.holo_sys.wavelength
        phi = -k * (X ** 2 + Y ** 2) / (2.0 * self.holo_sys.f_fresnel)
        phase = np.mod(phi, 2.0 * np.pi)
        return phase

    def gerchberg_saxton(self, target, weighted_gs=True, initial_phase=None):
        """
        Gerchberg-Saxton phase retrieval algorithm
        Parameters:
        -----------
        target : ndarray
            Target intensity pattern
        weighted_gs : bool
            Whether to use weighted GS algorithm
        initial_phase : ndarray, optional
            Initial phase guess
        Returns:
        --------
        field_slm : ndarray
            Complex field at SLM plane
        """
        size = target.shape
        source = np.ones(size, dtype=complex)
        sqrt_target = np.sqrt(target)
        target_mask = target != 0

        if initial_phase is not None:
            phase_slm = initial_phase
        else:
            np.random.seed(1)
            phase_slm = np.random.uniform(-np.pi, np.pi, size)

        if weighted_gs:
            weights = np.ones(size, dtype=float)

        for k in range(self.itn):

            u = source * np.exp(1j * phase_slm)
            v = np.fft.fftshift(np.fft.fft2(u))

            amplitude = np.abs(v)
            phase = np.angle(v)

            I = amplitude ** 2
            signal = self.normalize(I)

            if weighted_gs:

                weights[target_mask] *= np.sqrt(target[target_mask] / np.maximum(signal[target_mask], 1e-10))
                v = weights * sqrt_target * np.exp(1j * phase)
            else:
                v = sqrt_target * np.exp(1j * phase)

            u = np.fft.ifft2(np.fft.ifftshift(v))
            phase_slm = np.angle(u)

            self.logg.info(self.eval_performances(signal, target))
        return phase_slm

    def save_cgh(self, fnd):
        """Save every hologram as <fnd>_slotNN_8bit.bmp plus a <fnd>_slots.json slot map."""
        import json
        if not self.holograms:
            if self.phase_slm is not None:
                self.save_to_bmp(fnd, self.phase_slm)
            return
        slot_map = []
        for slot, h in enumerate(self.holograms, start=1):
            self.save_to_bmp(f"{fnd}_slot{slot:02d}", h["pattern"])
            slot_map.append({"slot": slot, "label": h["label"],
                             "spots": [list(map(float, self.spots_xy[i])) for i in h["spots"]]})
        with open(fnd + "_slots.json", "w") as f:
            json.dump(slot_map, f, indent=2)

    @staticmethod
    def phase_to_uint(phase: np.ndarray, max_level: int) -> np.ndarray:
        phase = phase % (2.0 * np.pi)
        return np.round(phase * max_level / (2.0 * np.pi)).astype(np.uint8)

    @staticmethod
    def save_to_bmp(fn, data):
        img = Image.fromarray(data, mode='L')
        img.save(fn + r"_8bit.bmp", format='BMP')

    @staticmethod
    def normalize(array):
        """Normalize array between [0, 1]"""
        min_val = np.min(array)
        max_val = np.max(array)
        return (array - min_val) / (max_val - min_val)

    @staticmethod
    def eval_performances(signal, target):
        """Evaluate the performances of GS algorithm"""
        mask = target != 0
        I = signal[mask]
        efficiency = np.sum(I) / np.sum(signal)
        Imax = np.max(I)
        Imin = np.min(I)
        uniformity = 1 - (Imax - Imin) / (Imax + Imin)
        mean_I = np.mean(I)
        std = np.std(I) / mean_I
        return efficiency, uniformity, std

    @staticmethod
    def add_arrays(a: np.ndarray, b: np.ndarray, offset_x: int = 0, offset_y: int = 0):
        """Add array a into array b, centered with offset. """
        a_h, a_w = a.shape[:2]
        b_h, b_w = b.shape[:2]

        center_y = (b_h - a_h) // 2 + offset_y
        center_x = (b_w - a_w) // 2 + offset_x

        src_y_start = max(0, -center_y)
        src_x_start = max(0, -center_x)
        src_y_end = min(a_h, b_h - center_y)
        src_x_end = min(a_w, b_w - center_x)

        if src_y_start >= src_y_end or src_x_start >= src_x_end:
            return b

        dst_y_start = max(0, center_y)
        dst_x_start = max(0, center_x)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)

        b[dst_y_start:dst_y_end, dst_x_start:dst_x_end] += a[src_y_start:src_y_end, src_x_start:src_x_end]

        return b

    def add_trigger_pattern(self, slm_pattern=None, ny: int = 1024, nx: int = 200):
        if slm_pattern is None:
            slm_pattern = self.phase_slm
        i, j = np.indices((ny, nx))
        pattern = (j // 2) % 2
        slm_pattern[-ny:, -nx:] = pattern * self.holo_sys.trigger_value
        return slm_pattern
