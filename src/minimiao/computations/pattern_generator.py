# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


from dataclasses import dataclass

import numpy as np
from PIL import Image

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
    phase_nx: int = 800
    phase_ny: int = 800
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
        self.phase_total = np.zeros((512, 512), dtype=np.uint8)

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
        img = Image.open(fnd)
        if img.mode != "1":
            img = img.convert("L")
        self.cell_mask = np.asarray(img)

    def load_spots_picked(self, spots_picked):
        self.spots_xy = spots_picked

    def build_target_from_spots(self, spots_sample):
        target_amp = np.zeros((self.holo_sys.phase_ny, self.holo_sys.phase_nx), dtype=np.float32)
        for (xx, yy, zz, ii) in spots_sample:
                target_amp[round(yy), round(xx)] = ii
        return target_amp

    def compute_cgh(self, trg=True):
        spots_sample = []
        for (x0, y0, z0, I0) in self.spots_xy:
            x1 = (x0 - self.holo_sys.image_center_x) * self.holo_sys.magnification + self.holo_sys.phase_nx / 2
            y1 = (y0 - self.holo_sys.image_center_y) * self.holo_sys.magnification + self.holo_sys.phase_ny / 2
            if self.holo_sys.phase_nx > x1 > 0 and self.holo_sys.phase_ny > y1 > 0:
                spots_sample.append((x1, y1, z0, I0))
        target_amp = self.build_target_from_spots(spots_sample)
        phase_spots = self.gerchberg_saxton(target_amp)
        phase_fresnel_lens = self.generate_fresnel_lens_phase()
        phase_total = (phase_spots + phase_fresnel_lens) % (2.0 * np.pi)
        phase_uint = self.phase_to_uint(phase_total, self.holo_sys.correction_value)
        self.phase_slm = self.add_arrays(phase_uint, self.holo_sys.correction_pattern.copy(),
                                         self.holo_sys.slm_offset_x, self.holo_sys.slm_offset_y)
        if trg:
            self.add_trigger_pattern()

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
        performances : list
            Performance metrics for each iteration
        """
        size = target.shape
        performances = []
        # Pre-allocate arrays
        source = np.ones(size, dtype=complex)
        sqrt_target = np.sqrt(target)
        target_mask = target != 0
        # Initialize phase
        if initial_phase is not None:
            phase_slm = initial_phase
        else:
            np.random.seed(1)
            phase_slm = np.random.uniform(-np.pi, np.pi, size)
        # Initialize weights for weighted GS
        if weighted_gs:
            weights = np.ones(size, dtype=float)
        # Main iteration loop
        for k in range(self.itn):
            # Forward propagation
            u = source * np.exp(1j * phase_slm)
            v = np.fft.fftshift(np.fft.fft2(u))
            # Extract amplitude and phase
            amplitude = np.abs(v)
            phase = np.angle(v)
            # Calculate normalized intensity
            I = amplitude ** 2
            signal = self.normalize(I)
            # Apply constraint in Fourier plane
            if weighted_gs:
                # Update weights only where target is non-zero
                weights[target_mask] *= np.sqrt(target[target_mask] / np.maximum(signal[target_mask], 1e-10))
                v = weights * sqrt_target * np.exp(1j * phase)
            else:
                v = sqrt_target * np.exp(1j * phase)
            # Backward propagation
            u = np.fft.ifft2(np.fft.ifftshift(v))
            phase_slm = np.angle(u)
            # Store performance metrics
            self.logg.info(self.eval_performances(signal, target))
        return phase_slm

    def save_cgh(self, fnd):
        self.save_to_bmp(fnd, self.phase_slm)

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

        # Calculate center position in b
        center_y = (b_h - a_h) // 2 + offset_y
        center_x = (b_w - a_w) // 2 + offset_x

        # Source (a) range
        src_y_start = max(0, -center_y)
        src_x_start = max(0, -center_x)
        src_y_end = min(a_h, b_h - center_y)
        src_x_end = min(a_w, b_w - center_x)

        # Check if there's anything to copy
        if src_y_start >= src_y_end or src_x_start >= src_x_end:
            return b

        # Destination (b) range
        dst_y_start = max(0, center_y)
        dst_x_start = max(0, center_x)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)

        # Add a to b
        b[dst_y_start:dst_y_end, dst_x_start:dst_x_end] += a[src_y_start:src_y_end, src_x_start:src_x_end]

        return b

    def add_trigger_pattern(self, ny: int = 1024, nx: int = 200):
        i, j = np.indices((ny, nx))
        pattern = (j // 2) % 2
        self.phase_slm[-ny:, -nx:] = pattern * self.holo_sys.trigger_value
