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


def camera_grid_to_holo(M, cam_shape, n=5, margin=0.1, phase_n=(850, 850), zero_order_radius=0.06):
    """
    Refinement grid: n x n points evenly covering the camera frame (minus a
    `margin` fraction on each side), mapped to hologram-plane coordinates
    with the coarse affine M. Points outside the hologram field or within
    `zero_order_radius` (fraction of the field) of the field centre (zero
    order) are dropped.
    Returns (holo_pts (K, 2), cam_pts_expected (K, 2)).
    """
    h, w = cam_shape
    xs = np.linspace(margin, 1 - margin, n) * (w - 1)
    ys = np.linspace(margin, 1 - margin, n) * (h - 1)
    xx, yy = np.meshgrid(xs, ys)
    cam = np.column_stack([xx.ravel(), yy.ravel()])
    holo = cam_to_holo(M, cam)
    nx, ny = phase_n
    r0 = zero_order_radius * min(nx, ny)
    keep = ((np.round(holo[:, 0]) >= 1) & (np.round(holo[:, 0]) <= nx) &
            (np.round(holo[:, 1]) >= 1) & (np.round(holo[:, 1]) <= ny) &
            (np.hypot(holo[:, 0] - nx / 2, holo[:, 1] - ny / 2) > r0))
    return holo[keep], cam[keep]


def equivalent_simple_params(M, phase_n=(850, 850)):
    """
    Closest magnification / center parameters for the non-affine mapping
    x1 = (x - cx) * mag + nx/2. Center = camera point that maps to the
    hologram field centre; mag = mean absolute scale. Rotation, shear and
    mirroring cannot be represented and are only kept by the affine.
    """
    nx, ny = phase_n
    cx, cy = holo_to_cam(M, [[nx / 2, ny / 2]])[0]
    d = describe(M)
    return {"magnification": 0.5 * (d["scale_x"] + d["scale_y"]),
            "center_x": float(cx), "center_y": float(cy),
            "exact": abs(d["rotation_deg"]) < 0.5 and not d["mirrored"] and abs(d["shear"]) < 0.01
            and abs(d["scale_x"] - d["scale_y"]) < 0.005 * d["scale_x"]}


# ---- simple model: magnification + center (the hologram panel parameters) -----

def simple_to_affine(magnification, center_x, center_y, phase_n=(850, 850)):
    """CGH._map_spot without affine: holo = m * (cam - c) + n / 2, as a 2x3 matrix."""
    nx, ny = phase_n
    m = float(magnification)
    return np.array([[m, 0.0, nx / 2 - m * center_x],
                     [0.0, m, ny / 2 - m * center_y]])


def fit_simple(cam_pts, holo_pts, phase_n=(850, 850)):
    """
    Least-squares fit of magnification m and center (cx, cy) in
        holo_x - nx/2 = m * (cam_x - cx),   holo_y - ny/2 = m * (cam_y - cy)
    over valid (non-NaN) pairs. m may be negative (180 deg rotation).
    Returns ({magnification, center_x, center_y}, residuals in camera px (NaN for unused)).
    """
    cam_pts = np.asarray(cam_pts, float)
    holo_pts = np.asarray(holo_pts, float)
    nx, ny = phase_n
    ok = np.all(np.isfinite(cam_pts), axis=1)
    k = int(ok.sum())
    if k < 2:
        raise ValueError(f"Need >= 2 detected spots to fit magnification/center, got {k}")
    c, h = cam_pts[ok], holo_pts[ok]
    # unknowns: m, bx, by  with holo - n/2 = m * cam + b
    A = np.zeros((2 * k, 3))
    A[:k, 0], A[:k, 1] = c[:, 0], 1.0
    A[k:, 0], A[k:, 2] = c[:, 1], 1.0
    rhs = np.concatenate([h[:, 0] - nx / 2, h[:, 1] - ny / 2])
    (m, bx, by), *_ = np.linalg.lstsq(A, rhs, rcond=None)
    if abs(m) < 1e-9:
        raise ValueError("Degenerate fit (magnification ~ 0)")
    params = {"magnification": float(m), "center_x": float(-bx / m), "center_y": float(-by / m)}
    pred_cam = (h - np.array([nx / 2, ny / 2])) / m + np.array([params["center_x"], params["center_y"]])
    res = np.full(len(cam_pts), np.nan)
    res[ok] = np.linalg.norm(pred_cam - c, axis=1)
    return params, res


def model_check(cam_pts, holo_pts):
    """
    Fit the full affine on the same data and report what the simple model cannot
    represent (mirror, rotation other than 0/180 deg, unequal x/y scale, shear).
    """
    M, _ = fit_affine(cam_pts, holo_pts)
    d = describe(M)
    rot = d["rotation_deg"]
    rot_off = min(abs(rot), abs(abs(rot) - 180.0))
    problems = []
    if d["mirrored"]:
        problems.append("mirror flip")
    if rot_off > 1.0:
        problems.append(f"rotation {rot:.1f} deg")
    if abs(d["scale_x"] - d["scale_y"]) > 0.01 * max(d["scale_x"], d["scale_y"]):
        problems.append(f"unequal x/y scale ({d['scale_x']:.4f} / {d['scale_y']:.4f})")
    if abs(d["shear"]) > 0.01:
        problems.append(f"shear {d['shear']:.3f}")
    return {"affine": d, "simple_model_ok": not problems, "problems": problems}
