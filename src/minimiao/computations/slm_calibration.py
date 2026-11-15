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


def fit_affine_robust(cam_pts, holo_pts, thresh_px=5.0, n_trials=3000, seed=0):
    """
    Affine fit that ignores wrong detections (RANSAC): affines through every (or n_trials random)
    triple of detected points are scored by how many points they predict within thresh_px camera
    px; the largest consistent set is refitted by least squares (twice, re-selecting inliers).
    Returns (M, residuals_cam_px (NaN = not detected), inlier mask).
    """
    from itertools import combinations
    cam = np.asarray(cam_pts, float)
    holo = np.asarray(holo_pts, float)
    valid = np.all(np.isfinite(cam), axis=1)
    idx = np.flatnonzero(valid)
    if len(idx) < 3:
        raise ValueError(f"Need >= 3 detected spots for an affine fit, got {len(idx)}")

    def residuals(M):
        r = np.full(len(cam), np.nan)
        r[valid] = np.linalg.norm(holo_to_cam(M, holo[valid]) - cam[valid], axis=1)
        return r

    def solve(sel):
        X = np.column_stack([cam[sel], np.ones(len(sel))])
        sol, *_ = np.linalg.lstsq(X, holo[sel], rcond=None)
        M = sol.T
        return M if abs(np.linalg.det(M[:, :2])) > 1e-9 else None

    rng = np.random.default_rng(seed)
    n_comb = len(idx) * (len(idx) - 1) * (len(idx) - 2) // 6
    triples = (combinations(idx, 3) if n_comb <= n_trials
               else (rng.choice(idx, 3, replace=False) for _ in range(n_trials)))
    best, best_n = None, -1
    for t in triples:
        M = solve(np.asarray(t))
        if M is None:
            continue
        with np.errstate(invalid="ignore"):
            n_in = int(np.sum(residuals(M) < thresh_px))
        if n_in > best_n:
            best, best_n = M, n_in
    if best is None:
        raise ValueError("Degenerate affine fit (points collinear?)")
    inl = valid & (residuals(best) < thresh_px)
    for _ in range(2):
        M = solve(np.flatnonzero(inl)) if inl.sum() >= 3 else best
        if M is None:
            break
        new = valid & (residuals(M) < thresh_px)
        if (new == inl).all():
            break
        inl = new
    M = solve(np.flatnonzero(inl)) if inl.sum() >= 3 else best
    return M, residuals(M), inl


def calibration_quality(residuals, inliers, min_inliers=8, min_fraction=0.6, max_rms_px=3.0):
    """(ok, rms over inliers, list of reasons a calibration should not be used)."""
    det = np.isfinite(np.asarray(residuals, float))
    inl = np.asarray(inliers, bool)
    n_in, n_det = int(inl.sum()), int(det.sum())
    rms = float(np.sqrt(np.mean(np.asarray(residuals)[inl] ** 2))) if n_in else float("nan")
    why = []
    if n_in < min_inliers:
        why.append(f"only {n_in} consistent spots (need >= {min_inliers})")
    if n_det and n_in / n_det < min_fraction:
        why.append(f"only {n_in} of {n_det} detected spots are consistent (need >= {min_fraction:.0%})")
    if not np.isfinite(rms) or rms > max_rms_px:
        why.append(f"fit error {rms:.2f} px (need <= {max_rms_px:g} px)")
    return not why, rms, why


def holo_grid_in_camera(M, cam_shape, n=5, phase_n=(850, 850), span=0.9, margin=0.05, zero_order_radius=0.025):
    """
    Refinement grid in hologram space: n x n points over `span` of the hologram field, kept if
    the model M puts them inside the camera image (minus `margin`) and away from the zero order.
    Covers the part of the hologram field the camera actually sees, whichever is larger.
    Returns (holo_pts (K, 2), cam_pts_expected (K, 2)).
    """
    nx, ny = phase_n
    h, w = cam_shape
    holo = make_grid(nx, ny, n, span)
    cam = holo_to_cam(M, holo)
    r0 = zero_order_radius * min(nx, ny)
    keep = ((cam[:, 0] >= margin * w) & (cam[:, 0] <= (1 - margin) * (w - 1)) &
            (cam[:, 1] >= margin * h) & (cam[:, 1] <= (1 - margin) * (h - 1)) &
            (np.hypot(holo[:, 0] - nx / 2, holo[:, 1] - ny / 2) > r0))
    return holo[keep], cam[keep]


def save_calibration(path, M, cam_pts, holo_pts, residuals, meta=None, inliers=None):
    residuals = np.asarray(residuals, float)
    if inliers is None:
        inliers = np.isfinite(residuals)
    inliers = np.asarray(inliers, bool)
    rms = float(np.sqrt(np.mean(residuals[inliers] ** 2))) if inliers.any() else float("nan")
    data = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "holo = A @ cam + b; M = [A | b]; cam = (x=col, y=row) px; holo = CGH target px",
        "M": M.tolist(),
        "summary": describe(M),
        "rms_cam_px": rms,
        "n_inliers": int(inliers.sum()),
        "outliers": [int(i) for i in np.flatnonzero(np.isfinite(residuals) & ~inliers)],
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


def camera_grid_to_holo(M, cam_shape, n=5, margin=0.1, phase_n=(850, 850), zero_order_radius=0.025):
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


def find_peaks(diff, n, window=7, min_snr=5.0):
    """
    Up to n separate spots in a background-subtracted frame (brightest first).
    Returns (points (k, 2) as camera (x=col, y=row), snr (k,)), k <= n; stops below min_snr.
    """
    d = np.clip(np.asarray(diff, dtype=np.float64), 0, None)
    sm = _box_blur(d)
    noise = np.std(d) + 1e-12
    h, w = d.shape
    yy, xx = np.mgrid[-window:window + 1, -window:window + 1]
    pts, snrs = [], []
    for _ in range(int(n)):
        r, c = np.unravel_index(np.argmax(sm), sm.shape)
        snr = sm[r, c] / noise
        if snr < min_snr:
            break
        r0, r1, c0, c1 = max(r - window, 0), min(r + window + 1, h), max(c - window, 0), min(c + window + 1, w)
        patch = d[r0:r1, c0:c1]
        tot = patch.sum()
        sub_y = yy[r0 - r + window:r1 - r + window, c0 - c + window:c1 - c + window]
        sub_x = xx[r0 - r + window:r1 - r + window, c0 - c + window:c1 - c + window]
        pts.append((c + (patch * sub_x).sum() / tot, r + (patch * sub_y).sum() / tot) if tot > 0 else (c, r))
        snrs.append(snr)
        sm[max(r - 2 * window, 0):r + 2 * window + 1, max(c - 2 * window, 0):c + 2 * window + 1] = 0  # suppress
    return np.array(pts, dtype=float).reshape(-1, 2), np.array(snrs)


def check_point_set(cam_pts, n_min=6, min_spread=0.15):
    """
    Raise ValueError unless at least n_min points are detected and they are not nearly
    collinear (smallest / largest singular value of the centred points >= min_spread).
    Three points always fit an affine exactly, so a fit from too few or badly spread points
    can report a tiny error and still be wrong. Returns the number of detected points.
    """
    pts = np.asarray(cam_pts, float)
    ok = np.all(np.isfinite(pts), axis=1)
    k = int(ok.sum())
    if k < n_min:
        raise ValueError(f"only {k} of {len(pts)} calibration spots detected (need >= {n_min})")
    sv = np.linalg.svd(pts[ok] - pts[ok].mean(axis=0), compute_uv=False)
    if sv[0] <= 0 or sv[-1] / sv[0] < min_spread:
        raise ValueError(f"the {k} detected spots lie almost on a line; they must cover the field in 2D")
    return k
