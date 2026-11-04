# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.

"""
Helpers for the labelling measurement:

1. a quick readout-only RESOLFT pre-scan gives an image,
2. point sources are detected in that image and located in galvo voltages,
3. the beam is parked on each point source in turn and dithered inside the
   PSF while the photon count trace is recorded.

Everything here is pure computation (no hardware access). ``ComputationManager``
creates one ``LabellingMeasurement`` instance (``cmp.lbl``) that holds the settings.
"""

import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max

from minimiao import logger


class LabellingMeasurement:
    """Settings and computations of the labelling measurement."""

    def __init__(self, logg=None):
        self.logg = logg or logger.setup_logging()
        # pre-scan: readout-only frame, dwell time per pixel
        self.prescan_dwell_s = 50e-6
        # point source detection
        self.spot_sigma = 1.0  # px, smoothing before peak finding
        self.spot_min_distance = 2  # px between two point sources
        self.spot_threshold = 5.0  # x noise above background
        # dither inside the PSF during the readout
        self.dither_radius_px = 0.5  # half-width, in pre-scan pixel steps
        self.dither_freq_hz = 2000.0  # reduced automatically to the galvo limits

    @staticmethod
    def find_point_sources(img, sigma=1.0, min_distance=2, threshold_sigma=5.0, exclude_border=False):
        """
        Detect point sources in an image.

        The image is smoothed with a Gaussian of ``sigma`` pixels; local maxima at
        least ``min_distance`` pixels apart and brighter than
        ``median + threshold_sigma * noise`` are kept (noise from the median
        absolute deviation, with a Poisson floor for sparse photon images).
        Each maximum is refined to sub-pixel precision with the intensity-weighted
        centroid of its 3x3 neighbourhood in the background-subtracted raw image.

        Returns a list of dicts in raster order (row, then column) with keys
        ``iy``, ``ix`` (integer pixel indices), ``y``, ``x`` (sub-pixel indices)
        and ``peak`` (smoothed peak value).
        """
        img = np.asarray(img, dtype=np.float64)
        if img.ndim != 2 or img.size == 0:
            return []
        sm = ndimage.gaussian_filter(img, sigma) if sigma and sigma > 0 else img
        med = float(np.median(sm))
        noise = 1.4826 * float(np.median(np.abs(sm - med)))
        poisson = float(np.sqrt(max(med, 0.0) + 1.0))
        threshold = med + threshold_sigma * max(noise, poisson / max(sigma * 2.0, 1.0))
        peaks = peak_local_max(sm, min_distance=int(min_distance), threshold_abs=threshold,
                               exclude_border=exclude_border)
        if len(peaks) == 0:
            return []
        peaks = sorted((int(p[0]), int(p[1])) for p in peaks)
        bg = float(np.median(img))
        ny, nx = img.shape
        out = []
        for iy, ix in peaks:
            y0, y1 = max(iy - 1, 0), min(iy + 2, ny)
            x0, x1 = max(ix - 1, 0), min(ix + 2, nx)
            w = np.clip(img[y0:y1, x0:x1] - bg, 0, None)
            s = w.sum()
            if s > 0:
                yy, xx = np.mgrid[y0:y1, x0:x1]
                y, x = float((w * yy).sum() / s), float((w * xx).sum() / s)
            else:
                y, x = float(iy), float(ix)
            out.append({"iy": iy, "ix": ix, "y": y, "x": x, "peak": float(sm[iy, ix])})
        return out


    @staticmethod
    def pixel_voltage_maps(galvo_sequences, gate, ny, nx, dwell):
        """
        Galvo command voltage at every image pixel of a galvo point scan.

        ``galvo_sequences`` is the (n_channels, n_samples) analog output of the
        scan, ``gate`` the detection gate (not shifted) whose on-samples are the
        pixel readouts in raster order. Returns (n_channels, ny, nx) with the mean
        command voltage of each channel during each pixel's readout, or None if
        the gate does not match the image size.
        """
        gv = np.atleast_2d(np.asarray(galvo_sequences, dtype=np.float64))
        idx = np.flatnonzero(np.asarray(gate) > 0)
        if idx.size != ny * nx * dwell or idx.size == 0 or idx[-1] >= gv.shape[1]:
            return None
        idx = idx.reshape(ny, nx, dwell)
        return gv[:, idx].mean(axis=-1)


    @staticmethod
    def voltages_at(vmaps, y, x):
        """Bilinear interpolation of the (n_channels, ny, nx) voltage maps at
        sub-pixel position (y, x). Returns one voltage per channel."""
        return [float(ndimage.map_coordinates(m, [[y], [x]], order=1, mode="nearest")[0]) for m in vmaps]


    @staticmethod
    def _lissajous(t_us, w, r):
        """Position and velocity of a per-channel sine dither of amplitude r."""
        return r * np.sin(w * t_us), r * w * np.cos(w * t_us)


    @staticmethod
    def build_dither_sequences(targets, start, galvo, sample_rate, readout_samples, radius,
                               freq_hz, min_move_samples, tail_samples=4, freq_ratio=0.83):
        """
        Galvo waveforms that visit each target, one after another, and dither the
        beam around it during a readout window.

        For every target the sequence holds ``move`` samples of a minimum-jerk
        move (lasers off) followed by ``readout_samples`` samples of dither:
        channel k follows ``target_k + radius * sin(w_k t)`` with
        ``w_1 = freq_ratio * w_0`` so the pattern fills the square of half-width
        ``radius`` around the target. The move length is the same for all targets
        (the shortest one that keeps every move inside the galvo velocity and
        acceleration limits), so each target occupies a segment of equal length.
        The dither frequency is reduced if needed to respect those limits.

        Parameters
        ----------
        targets : (n_points, n_channels) target voltages, per DAQ galvo channel.
        start   : (n_channels,) command voltage before the sequence starts.
        galvo   : waveform object providing ``get_trajectory`` and ``gs`` limits.

        Returns (sequences (n_channels, n_samples), move_samples, freqs_hz).
        """
        targets = np.atleast_2d(np.asarray(targets, dtype=np.float64))
        n_pts, n_ch = targets.shape
        start = np.asarray(start, dtype=np.float64)
        dt_us = 1e6 / sample_rate
        v_max = galvo.gs.max_command_velocity_V_per_us
        a_max = galvo.gs.max_command_accel_V_per_us2

        # dither angular frequencies [rad/us], kept at half the galvo limits
        w0 = 2 * np.pi * float(freq_hz) * 1e-6
        ws = np.array([w0 * (freq_ratio ** k) for k in range(n_ch)])
        if radius > 0:
            w_lim = min(0.5 * v_max / radius, np.sqrt(0.5 * a_max / radius))
            ws = np.minimum(ws, w_lim)
        ws = np.minimum(ws, 2 * np.pi * (sample_rate / 8.0) * 1e-6)

        t_dither = (np.arange(readout_samples) + 1) * dt_us
        d_pos = np.empty((n_ch, readout_samples))
        d_vel_start = np.empty(n_ch)
        d_vel_end = np.empty(n_ch)
        for k in range(n_ch):
            d_pos[k], vel = LabellingMeasurement._lissajous(t_dither, ws[k], radius)
            d_vel_start[k] = radius * ws[k]  # velocity at t = 0
            d_vel_end[k] = vel[-1]

        def moves(n):
            out = []
            p_prev, v_prev = start, np.zeros(n_ch)
            t = np.arange(n + 1) * dt_us
            for i in range(n_pts):
                seg = np.empty((n_ch, n))
                for k in range(n_ch):
                    pos, _, _ = galvo.get_trajectory(t, p0=p_prev[k], v0=v_prev[k],
                                                     p1=targets[i, k], v1=d_vel_start[k])
                    seg[k] = pos[1:]
                out.append(seg)
                p_prev = targets[i] + d_pos[:, -1]
                v_prev = d_vel_end
            return out

        n_move = max(int(min_move_samples), 2)
        for _ in range(60):
            try:
                mv = moves(n_move)
                break
            except ValueError:
                n_move = int(np.ceil(n_move * 1.25))
        else:
            raise ValueError("Cannot find galvo moves within the velocity/acceleration limits")

        seg_len = n_move + readout_samples
        seq = np.empty((n_ch, n_pts * seg_len + tail_samples))
        for i in range(n_pts):
            b = i * seg_len
            seq[:, b:b + n_move] = mv[i]
            seq[:, b + n_move:b + seg_len] = targets[i][:, None] + d_pos
        seq[:, n_pts * seg_len:] = seq[:, n_pts * seg_len - 1][:, None]
        return seq, n_move, ws / (2 * np.pi * 1e-6)


    @staticmethod
    def build_dither_triggers(n_points, n_digital, move_samples, readout_samples, tail_samples=4):
        """Digital triggers (all channels identical: off during the move, on during
        the readout) and the matching detection gate, for ``n_points`` segments."""
        seg = np.r_[np.zeros(move_samples, dtype=np.uint8), np.ones(readout_samples, dtype=np.uint8)]
        one = np.r_[np.tile(seg, n_points), np.zeros(tail_samples, dtype=np.uint8)]
        return np.tile(one, (n_digital, 1)), one.copy()


    @staticmethod
    def split_traces(counts, n_points, move_samples, readout_samples):
        """Cut a full count trace into (n_points, readout_samples) readout traces.
        ``counts[k]`` holds the photons of trigger sample k-1 (counter latency)."""
        seg = move_samples + readout_samples
        aligned = np.roll(np.asarray(counts), -1)[:n_points * seg]
        return aligned.reshape(n_points, seg)[:, move_samples:]
