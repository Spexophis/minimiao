# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.

"""
Camera -> SLM hologram-plane affine calibration.

Hologram-plane coordinates (x1, y1) are the pixel coordinates fed to
CGH.build_target_from_spots (0..phase_nx, 0..phase_ny). The calibration is

    [x1, y1]^T = A @ [x_cam, y_cam]^T + b,    M = [A | b]  (2 x 3)

which covers translation, rotation, scale, shear and mirror flips.

Procedure: one single-spot hologram per known hologram-plane point is
projected (one SLM slot each), the camera records one frame per slot, the
spot is located in each frame and M is fitted by least squares.
"""

import json
import time

import numpy as np


def make_grid(phase_nx, phase_ny, n=4, span=0.5):
    """
    n x n grid of hologram-plane points spanning `span` of the field,
    centred on the field. Use an even n so no point falls on the field
    centre (zero spatial frequency = zero order).
    """
    if n < 2:
        raise ValueError("Grid needs n >= 2")
    fx = np.linspace(-span / 2, span / 2, n) * phase_nx + phase_nx / 2
    fy = np.linspace(-span / 2, span / 2, n) * phase_ny + phase_ny / 2
    xx, yy = np.meshgrid(fx, fy)
    return np.column_stack([xx.ravel(), yy.ravel()])


def _box_blur(img, r=2):
    """Separable box blur (r = half width) without scipy."""
    k = 2 * r + 1
    out = img
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r + 1, r)
        c = np.cumsum(np.pad(out, pad, mode="edge"), axis=axis)
        if axis == 0:
            out = (c[k:] - c[:-k]) / k
        else:
            out = (c[:, k:] - c[:, :-k]) / k
    return out


def detect_spots(frames, window=7, min_snr=5.0):
    """
    Locate the single calibration spot in each frame.

    frames : (K, H, W) one frame per calibration hologram. The per-pixel
             median over frames is used as background, which removes the
             zero order, the trigger patch and any static light, since each
             calibration spot appears in only one frame (needs K >= 3).
    Returns (points (K, 2) as camera (x=col, y=row) with NaN for failed
    detections, snr (K,), background-subtracted max projection (H, W)).
    """
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim != 3 or frames.shape[0] < 3:
        raise ValueError("Need at least 3 frames of shape (K, H, W)")
    bg = np.median(frames, axis=0)
    diff = np.clip(frames - bg, 0, None)
    h, w = bg.shape
    yy, xx = np.mgrid[-window:window + 1, -window:window + 1]
    pts = np.full((frames.shape[0], 2), np.nan)
    snr = np.zeros(frames.shape[0])
    for k, d in enumerate(diff):
        sm = _box_blur(d)
        r, c = np.unravel_index(np.argmax(sm), sm.shape)
        noise = np.std(d) + 1e-12
        snr[k] = sm[r, c] / noise
        if snr[k] < min_snr:
            continue
        r0, r1 = max(r - window, 0), min(r + window + 1, h)
        c0, c1 = max(c - window, 0), min(c + window + 1, w)
        patch = d[r0:r1, c0:c1]
        tot = patch.sum()
        if tot <= 0:
            continue
        sub_y = yy[r0 - r + window:r1 - r + window, c0 - c + window:c1 - c + window]
        sub_x = xx[r0 - r + window:r1 - r + window, c0 - c + window:c1 - c + window]
        pts[k] = (c + (patch * sub_x).sum() / tot, r + (patch * sub_y).sum() / tot)
    return pts, snr, diff.max(axis=0)


def fit_affine(cam_pts, holo_pts):
    """
    Least-squares fit holo = A @ cam + b over valid (non-NaN) pairs.
    Returns (M (2x3), residuals_cam_px (K,) with NaN for unused points).
    Residuals are expressed in camera pixels.
    """
    cam_pts = np.asarray(cam_pts, float)
    holo_pts = np.asarray(holo_pts, float)
    ok = np.all(np.isfinite(cam_pts), axis=1)
    if ok.sum() < 3:
        raise ValueError(f"Need >= 3 detected spots for an affine fit, got {int(ok.sum())}")
    X = np.column_stack([cam_pts[ok], np.ones(ok.sum())])
    sol, *_ = np.linalg.lstsq(X, holo_pts[ok], rcond=None)
    M = sol.T  # 2 x 3
    if abs(np.linalg.det(M[:, :2])) < 1e-12:
        raise ValueError("Degenerate affine fit (points collinear?)")
    res = np.full(len(cam_pts), np.nan)
    pred_cam = holo_to_cam(M, holo_pts[ok])
    res[ok] = np.linalg.norm(pred_cam - cam_pts[ok], axis=1)
    return M, res


def cam_to_holo(M, pts):
    pts = np.atleast_2d(np.asarray(pts, float))
    return pts @ M[:, :2].T + M[:, 2]


def holo_to_cam(M, pts):
    pts = np.atleast_2d(np.asarray(pts, float))
    A_inv = np.linalg.inv(M[:, :2])
    return (pts - M[:, 2]) @ A_inv.T


def describe(M):
    """Scale (holo px per cam px), rotation (deg), mirror flag, shear, for logging.
    If mirrored, the decomposition is: flip camera x first, then rotate/scale."""
    A = M[:, :2]
    det = np.linalg.det(A)
    if det < 0:
        A = A @ np.diag([-1.0, 1.0])
        det = -det
    sx = np.hypot(A[0, 0], A[1, 0])
    rot = np.degrees(np.arctan2(A[1, 0], A[0, 0]))
    sy = det / sx
    shear = (A[0, 0] * A[0, 1] + A[1, 0] * A[1, 1]) / det
    return {"scale_x": float(sx), "scale_y": float(abs(sy)), "rotation_deg": float(rot),
            "mirrored": bool(np.linalg.det(M[:, :2]) < 0), "shear": float(shear)}


def save_calibration(path, M, cam_pts, holo_pts, residuals, meta=None):
    data = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "holo = A @ cam + b; M = [A | b]; cam = (x=col, y=row) px; holo = CGH target px",
        "M": M.tolist(),
        "summary": describe(M),
        "rms_cam_px": float(np.sqrt(np.nanmean(residuals ** 2))),
        "cam_pts": [[None if not np.isfinite(v) else float(v) for v in p] for p in cam_pts],
        "holo_pts": np.asarray(holo_pts, float).tolist(),
        "residuals_cam_px": [None if not np.isfinite(r) else float(r) for r in residuals],
        "meta": meta or {},
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return data


def load_calibration(path):
    with open(path) as f:
        data = json.load(f)
    M = np.asarray(data["M"], float)
    if M.shape != (2, 3):
        raise ValueError(f"Calibration matrix has shape {M.shape}, expected (2, 3)")
    return M, data
