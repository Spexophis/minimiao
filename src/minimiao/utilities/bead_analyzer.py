# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.

"""
Single-bead PSF quality from widefield 3D stacks, used as the image metric of the DM system-flat loop.
"""

from dataclasses import dataclass, field, asdict

import numpy as np
from scipy import ndimage

METRICS = ("Peak/Flux", "1/FWHM3D", "Peak")
BEAD_CLASSES = ("All", "100 nm", "200 nm")


@dataclass
class BeadParams:
    dx: float = 0.069  # lateral pixel size, um
    dz: float = 0.1  # z step, um
    wl: float = 0.505  # emission wavelength, um
    na: float = 1.3
    threshold: float = 8.0  # detection threshold, in robust noise sigmas of the band-passed projection
    min_separation: float = 2.0  # um, minimum distance to any other detected spot
    lateral_half: float = 0.8  # um, half-width of the lateral crop used for the metric
    axial_half: float = 1.0  # um, half-height of the axial crop used for the metric
    search: float = 0.4  # um, lateral re-centring search radius (axial search is the full stack)
    max_beads: int = 100
    bead_class: str = "All"
    doublet_ratio: float = 1.8  # reject beads brighter than this times their class median flux
    saturation: float | None = None  # raw value at/above which a bead is rejected; None: dtype max for ints

    @property
    def fwhm_lateral(self):
        return 0.51 * self.wl / self.na

    @property
    def h(self):
        return max(3, int(np.ceil(self.lateral_half / self.dx)))

    @property
    def hz(self):
        return max(2, int(np.ceil(self.axial_half / self.dz)))


@dataclass
class Bead:
    z: int
    y: int
    x: int
    flux: float = 0.0
    snr: float = 0.0
    size_class: str = "?"
    info: dict = field(default_factory=dict)


def _robust_sigma(a):
    a = np.asarray(a, dtype=np.float64).ravel()
    med = np.median(a)
    return 1.4826 * np.median(np.abs(a - med)) + 1e-12


def _fwhm_1d(profile, spacing):
    """FWHM of a background-subtracted 1D profile by linear interpolation of the half-max crossings around its
    maximum. NaN if the profile does not drop below half max on both sides."""
    p = np.asarray(profile, dtype=np.float64)
    i = int(np.argmax(p))
    pk = p[i]
    if pk <= 0:
        return np.nan
    half = pk / 2.0
    left = np.where(p[:i] < half)[0]
    right = np.where(p[i + 1:] < half)[0]
    if left.size == 0 or right.size == 0:
        return np.nan
    il = left[-1]
    ir = i + 1 + right[0]
    xl = il + (half - p[il]) / (p[il + 1] - p[il])
    xr = ir - 1 + (p[ir - 1] - half) / (p[ir - 1] - p[ir])
    return (xr - xl) * spacing


def _crop(stack, z, y, x, hz, h):
    return stack[z - hz:z + hz + 1, y - h:y + h + 1, x - h:x + h + 1]


def _background(crop):
    """Median of the lateral border of the crop (all planes): local background incl. out-of-focus haze."""
    border = np.concatenate([crop[:, 0, :].ravel(), crop[:, -1, :].ravel(),
                             crop[:, 1:-1, 0].ravel(), crop[:, 1:-1, -1].ravel()])
    return float(np.median(border))


def _bead_quantities(crop, p: BeadParams):
    """All per-bead quantities of one background-subtracted-able 3D crop centred on the bead."""
    c = crop.astype(np.float64) - _background(crop)
    cs = ndimage.gaussian_filter(c, sigma=(0.7, 0.7, 0.7))
    zp, yp, xp = np.unravel_index(int(np.argmax(cs)), cs.shape)
    peak = float(cs[zp, yp, xp])
    z0, z1 = max(0, zp - 1), min(c.shape[0], zp + 2)
    flux = float(c[z0:z1].sum(axis=(1, 2)).mean())
    fx = _fwhm_1d(cs[zp, yp, :], p.dx)
    fy = _fwhm_1d(cs[zp, :, xp], p.dx)
    y0, y1 = max(0, yp - 1), min(c.shape[1], yp + 2)
    x0, x1 = max(0, xp - 1), min(c.shape[2], xp + 2)
    fz = _fwhm_1d(cs[:, y0:y1, x0:x1].mean(axis=(1, 2)), p.dz)
    return dict(peak=peak, flux=flux, fwhm_x=fx, fwhm_y=fy, fwhm_z=fz, off_z=zp - crop.shape[0] // 2,
                off_y=yp - crop.shape[1] // 2, off_x=xp - crop.shape[2] // 2)


def bead_metric(q, metric):
    if metric == "Peak/Flux":
        return q["peak"] / q["flux"] if q["flux"] > 0 else np.nan
    if metric == "1/FWHM3D":
        v = q["fwhm_x"] * q["fwhm_y"] * q["fwhm_z"]
        return 1.0 / v if np.isfinite(v) and v > 0 else np.nan
    if metric == "Peak":
        return q["peak"]
    raise ValueError(f"Unknown bead metric {metric!r}")


def _otsu_1d(v):
    v = np.sort(np.asarray(v, dtype=np.float64))
    best, thr = -1.0, v[0]
    for k in range(1, v.size):
        w0, w1 = k / v.size, 1 - k / v.size
        s = w0 * w1 * (v[:k].mean() - v[k:].mean()) ** 2
        if s > best:
            best, thr = s, (v[k - 1] + v[k]) / 2
    return thr


def find_single_beads(stack, p: BeadParams, log=None):
    """
    Detect single, isolated beads in a widefield z-stack (nz, ny, nx).
    Returns (beads, report) where report counts the rejections per reason.
    """
    stack = np.asarray(stack)
    if stack.ndim != 3:
        raise ValueError("stack must be (nz, ny, nx)")
    nz, ny, nx = stack.shape
    h, hz = p.h, p.hz
    if nz < 2 * hz + 3:
        raise ValueError(f"stack has {nz} planes; at least {2 * hz + 3} are needed for an axial crop of "
                         f"+/-{p.axial_half} um at {p.dz} um steps")
    sat = p.saturation
    if sat is None and np.issubdtype(stack.dtype, np.integer):
        sat = np.iinfo(stack.dtype).max

    s_psf = p.fwhm_lateral / 2.355 / p.dx
    proj = stack.max(axis=0).astype(np.float64)
    dog = ndimage.gaussian_filter(proj, s_psf) - ndimage.gaussian_filter(proj, 3 * s_psf)
    noise = _robust_sigma(dog)
    size = max(3, int(2 * np.ceil(2 * s_psf) + 1))
    is_max = dog == ndimage.maximum_filter(dog, size=size)
    # Low threshold: everything that could spoil a neighbour; high threshold: bead candidates.
    cy, cx = np.nonzero(is_max & (dog > 0.5 * p.threshold * noise))
    report = dict(candidates=0, ring=0, border=0, crowded=0, axial_edge=0, saturated=0, shape=0,
                  doublet=0, size_class=0, kept=0)
    if cy.size == 0:
        return [], report
    smooth = ndimage.gaussian_filter(stack.astype(np.float32), sigma=(1.0, s_psf, s_psf))
    cz = np.argmax(smooth[:, cy, cx], axis=0)
    amp = dog[cy, cx]

    # Defocus rings of bright beads show up as local maxima of the max-projection. A maximum is a ring if it is
    # much dimmer than a brighter maximum within the largest defocus blur radius of the stack, and its own best
    # focus is axially far from that one's (a real neighbour on the same coverslip focuses in the same plane).
    n_imm = 1.518
    r_ring = (nz * p.dz / 2) * np.tan(np.arcsin(min(p.na / n_imm, 0.99))) / p.dx + 3 * s_psf
    dz_ring = max(2, int(np.ceil(0.5 * (2 * n_imm * p.wl / p.na ** 2) / p.dz)))
    ring = np.zeros(cy.size, dtype=bool)
    for j in np.argsort(amp):
        brighter = (amp > 2 * amp[j]) & ~ring
        if not brighter.any():
            continue
        d = np.hypot(cy[brighter] - cy[j], cx[brighter] - cx[j])
        far_z = np.abs(cz[brighter] - cz[j]) >= dz_ring
        if np.any((d < r_ring) & far_z):
            ring[j] = True
    cy, cx, cz, amp = cy[~ring], cx[~ring], cz[~ring], amp[~ring]
    strong = amp > p.threshold * noise
    report["candidates"] = int(strong.sum())
    report["ring"] = int(ring.sum())

    pts = np.stack([cy, cx], axis=1).astype(np.float64)
    min_sep_px = p.min_separation / p.dx

    beads = []
    for k in np.nonzero(strong)[0]:
        y, x = int(cy[k]), int(cx[k])
        if y < h or y >= ny - h or x < h or x >= nx - h:
            report["border"] += 1
            continue
        d = np.hypot(pts[:, 0] - y, pts[:, 1] - x)
        d[k] = np.inf
        if d.min() < min_sep_px:
            report["crowded"] += 1
            continue
        z = int(cz[k])
        if z < hz or z >= nz - hz:
            report["axial_edge"] += 1
            continue
        crop = _crop(stack, z, y, x, hz, h)
        if sat is not None and crop.max() >= sat:
            report["saturated"] += 1
            continue
        q = _bead_quantities(crop, p)
        fl = np.nanmax([q["fwhm_x"], q["fwhm_y"]])
        # loose: the system is not corrected yet, only reject things that are clearly not a point source
        if not np.isfinite(fl) or fl > 4 * p.fwhm_lateral or q["flux"] <= 0:
            report["shape"] += 1
            continue
        beads.append(Bead(z=z + q["off_z"], y=y + q["off_y"], x=x + q["off_x"], flux=q["flux"],
                          snr=float(dog[y, x] / noise), info=q))

    if not beads:
        return [], report

    # 100 / 200 nm populations from focal-plane flux (two-class Otsu on log flux)
    lf = np.log10([b.flux for b in beads])
    separated = False
    if len(beads) >= 4:
        thr = _otsu_1d(lf)
        lo, hi = lf[lf <= thr], lf[lf > thr]
        # 200 nm beads should be ~8x (0.9 decades) brighter; accept a separation of > 0.5 decades
        separated = lo.size > 0 and hi.size > 0 and (np.median(hi) - np.median(lo)) > 0.5
    for b, v in zip(beads, lf):
        b.size_class = ("200 nm" if v > thr else "100 nm") if separated else "?"
    if log is not None and not separated:
        log.warning("Bead flux is not bimodal; 100 nm / 200 nm beads are not separated")

    # doublets / aggregates: much brighter than the median of their class
    kept = []
    for cls in {b.size_class for b in beads}:
        group = [b for b in beads if b.size_class == cls]
        med = np.median([b.flux for b in group])
        for b in group:
            if b.flux > p.doublet_ratio * med:
                report["doublet"] += 1
            else:
                kept.append(b)

    if p.bead_class != "All":
        if not separated:
            if log is not None:
                log.warning(f"Cannot select {p.bead_class} beads; using all beads")
        else:
            n0 = len(kept)
            kept = [b for b in kept if b.size_class == p.bead_class]
            report["size_class"] = n0 - len(kept)

    kept.sort(key=lambda b: b.snr, reverse=True)
    kept = kept[:p.max_beads]
    report["kept"] = len(kept)
    return kept, report


def measure_beads(stack, beads, p: BeadParams, metric="Peak/Flux"):
    """
    Quality of every bead in one stack. Each bead is re-centred within +/-search laterally and over the whole
    usable z range on a smoothed copy of the stack. Returns (values, quantities); values[i] is NaN when the
    bead could not be measured (moved out of the stack, profile not resolved).
    """
    stack = np.asarray(stack)
    nz, ny, nx = stack.shape
    h, hz = p.h, p.hz
    r = max(1, int(np.ceil(p.search / p.dx)))
    s_psf = p.fwhm_lateral / 2.355 / p.dx
    vals = np.full(len(beads), np.nan)
    qs = []
    for i, b in enumerate(beads):
        y0, y1 = max(h, b.y - r), min(ny - h, b.y + r + 1)
        x0, x1 = max(h, b.x - r), min(nx - h, b.x + r + 1)
        if y0 >= y1 or x0 >= x1:
            qs.append(None)
            continue
        win = ndimage.gaussian_filter(stack[:, y0:y1, x0:x1].astype(np.float32), sigma=(1.0, s_psf, s_psf))
        z, y, x = np.unravel_index(int(np.argmax(win)), win.shape)
        y, x = y + y0, x + x0
        if z < hz or z >= nz - hz:
            qs.append(None)
            continue
        q = _bead_quantities(_crop(stack, z, y, x, hz, h), p)
        q.update(z=int(z), y=int(y), x=int(x))
        qs.append(q)
        vals[i] = bead_metric(q, metric)
    return vals, qs


def aggregate(values):
    """
    values: (n_amps, n_beads) bead metrics for one mode scan. Beads with any NaN are dropped, each remaining bead
    is divided by its own mean over the scan. Returns (median per amp, robust SEM per amp, n_beads used).
    """
    v = np.asarray(values, dtype=np.float64)
    ok = np.all(np.isfinite(v), axis=0) & (np.nanmean(v, axis=0) > 0)
    v = v[:, ok]
    n = v.shape[1]
    if n == 0:
        return np.full(v.shape[0], np.nan), np.full(v.shape[0], np.nan), 0
    v = v / v.mean(axis=0, keepdims=True)
    med = np.median(v, axis=1)
    mad = 1.4826 * np.median(np.abs(v - med[:, None]), axis=1)
    sem = 1.2533 * mad / np.sqrt(n)  # SE of the median for ~normal scatter
    return med, sem, n


def beads_table(beads):
    rows = []
    for i, b in enumerate(beads):
        d = asdict(b)
        info = d.pop("info")
        d.update({k: v for k, v in info.items() if k.startswith("fwhm") or k == "peak"})
        d["id"] = i
        rows.append(d)
    return rows
