# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


"""
Computes Zernike polynomials and their analytical x/y derivatives on a rectangular grid,
using the Noll indexing convention.
"""

import warnings

import numpy as np
from math import factorial
from typing import NamedTuple, Optional, Tuple, List


num_znk = 64


def zernike_names(nz: int) -> List[str]:
    """
    Return standard optical names for the Zernike modes up to j=36 (5th radial order).
    """
    standard_names = {
        1: "Piston",
        2: "Tilt Y",
        3: "Tilt X",
        4: "Defocus",
        5: "Astigmatism 45°",
        6: "Astigmatism 0°",
        7: "Coma Y",
        8: "Coma X",
        9: "Trefoil Y",
        10: "Trefoil X",
        11: "Primary Spherical",
        12: "2nd Astigmatism 0°",
        13: "2nd Astigmatism 45°",
        14: "2nd Coma X",
        15: "2nd Coma Y",
        16: "Tetrafoil X",
        17: "Tetrafoil Y",
        18: "2nd Spherical",
        19: "3rd Astigmatism 45°",
        20: "3rd Astigmatism 0°",
        21: "3rd Coma Y",
        22: "3rd Coma X",
        23: "2nd Trefoil Y",
        24: "2nd Trefoil X",
        25: "Pentafoil Y",
        26: "Pentafoil X",
        27: "3rd Spherical",
        28: "4th Astigmatism 0°",
        29: "4th Astigmatism 45°",
        30: "4th Coma X",
        31: "4th Coma Y",
        32: "3rd Trefoil X",
        33: "3rd Trefoil Y",
        34: "2nd Tetrafoil X",
        35: "2nd Tetrafoil Y",
        36: "Hexafoil X",
    }

    names = []
    for j in range(1, nz + 1):
        if j in standard_names:
            n, m = noll_to_nm(j)
            names.append(f"Z{j} ({n},{m:+d}) {standard_names[j]}")
        else:
            n, m = noll_to_nm(j)
            names.append(f"Z{j} ({n},{m:+d})")
    return names


def noll_to_nm(j: int) -> Tuple[int, int]:
    """
    Convert Noll single index j to radial order n and azimuthal frequency m.

    Parameters
    ----------
    j : int   — Noll index (j >= 1)

    Returns
    -------
    n : int   — radial order (n >= 0)
    m : int   — azimuthal frequency (-n <= m <= n, same parity as n)
    """
    if j < 1:
        raise ValueError(f"Noll index must be >= 1, got {j}")

    # Find radial order n
    n = 0
    while (n + 1) * (n + 2) // 2 < j:
        n += 1

    # Remainder within this order
    k = j - n * (n + 1) // 2 - 1  # 0-based position within order n

    # Azimuthal frequency m
    m_values = []
    for m_candidate in range(-n, n + 1, 2):  # same parity as n
        m_values.append(m_candidate)
    m_values.sort(key=lambda x: (abs(x), -x))  # Noll ordering: 0, -1, +1, -2, +2, ...

    # Reconstruct m from j
    m_abs = 2 * ((k + (n % 2 == 0)) // 2) + (n % 2)  # |m|
    if m_abs == 0:
        m = 0
    elif j % 2 == 0:
        m = m_abs
    else:
        m = -m_abs

    return n, m


def nm_to_noll(n: int, m: int) -> int:
    """
    Convert (n, m) to Noll index j.

    Parameters
    ----------
    n : int — radial order
    m : int — azimuthal frequency

    Returns
    -------
    j : int — Noll index (j >= 1)
    """
    abs_m = abs(m)

    # Base index for order n
    j_base = n * (n + 1) // 2 + 1

    if m == 0:
        return j_base
    elif m > 0:
        return j_base + 2 * abs_m - 1 + (1 if n % 2 == 1 else 0)
    else:
        return j_base + 2 * abs_m - (0 if n % 2 == 1 else 0)


def _radial_polynomial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """
    Compute the radial Zernike polynomial R_n^|m|(ρ).
    """
    abs_m = abs(m)
    R = np.zeros_like(rho)
    num_terms = (n - abs_m) // 2 + 1

    for s in range(num_terms):
        coeff = ((-1) ** s * factorial(n - s) /
                 (factorial(s) *
                  factorial((n + abs_m) // 2 - s) *
                  factorial((n - abs_m) // 2 - s)))
        R = R + coeff * rho ** (n - 2 * s)

    return R


def _radial_derivative(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """
    Compute dR_n^|m|/dρ analytically.
    """
    abs_m = abs(m)
    dR = np.zeros_like(rho)
    num_terms = (n - abs_m) // 2 + 1

    for s in range(num_terms):
        power = n - 2 * s
        if power == 0:
            continue  # derivative of constant = 0
        coeff = ((-1) ** s * factorial(n - s) /
                 (factorial(s) *
                  factorial((n + abs_m) // 2 - s) *
                  factorial((n - abs_m) // 2 - s)))
        # Avoid 0^(-1) by masking
        with np.errstate(divide='ignore', invalid='ignore'):
            term = coeff * power * rho ** (power - 1)
            term = np.where(np.isfinite(term), term, 0.0)
        dR = dR + term

    return dR


def _zernike_single(n: int, m: int,
                    rho: np.ndarray, theta: np.ndarray,
                    x_norm: np.ndarray, y_norm: np.ndarray,
                    pupil: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a single Zernike mode and its Cartesian derivatives dZ/dx, dZ/dy on the normalized pupil.

    The Zernike polynomial in polar form:
        Z = N · R(ρ) · Θ(θ)

    where Θ(θ) = cos(|m|θ) for m > 0
                 sin(|m|θ) for m < 0
                 1         for m = 0

    Derivatives use the chain rule:
        dZ/dx = dZ/dρ · dρ/dx + dZ/dθ · dθ/dx
        dZ/dy = dZ/dρ · dρ/dy + dZ/dθ · dθ/dy

    where:
        dρ/dx = x/ρ = cos(θ)
        dρ/dy = y/ρ = sin(θ)
        dθ/dx = -sin(θ)/ρ = -y/ρ²
        dθ/dy =  cos(θ)/ρ =  x/ρ²

    Returns
    -------
    Z, dZdx, dZdy : 2D arrays
    """
    abs_m = abs(m)

    # Normalization factor
    if m == 0:
        norm = np.sqrt(n + 1.0)
    else:
        norm = np.sqrt(2.0 * (n + 1.0))

    # Radial polynomial and its derivative
    R = _radial_polynomial(n, m, rho)
    dR = _radial_derivative(n, m, rho)

    # Angular part and its derivative
    if m > 0:
        Theta = np.cos(abs_m * theta)
        dTheta = -abs_m * np.sin(abs_m * theta)
    elif m < 0:
        Theta = np.sin(abs_m * theta)
        dTheta = abs_m * np.cos(abs_m * theta)
    else:
        Theta = np.ones_like(theta)
        dTheta = np.zeros_like(theta)

    # Zernike value
    Z = norm * R * Theta * pupil

    # --- Analytical Cartesian derivatives ---
    # dZ/dx = N · [dR/dρ · (dρ/dx) · Θ  +  R · dΘ/dθ · (dθ/dx)]
    # dZ/dy = N · [dR/dρ · (dρ/dy) · Θ  +  R · dΘ/dθ · (dθ/dy)]
    #
    # dρ/dx = x/ρ,  dρ/dy = y/ρ
    # dθ/dx = -y/ρ²,  dθ/dy = x/ρ²

    # Avoid division by zero at origin
    rho_safe = np.where(rho > 1e-12, rho, 1e-12)
    rho2_safe = rho_safe ** 2

    cos_t = x_norm / rho_safe  # = cos(theta), safe at origin
    sin_t = y_norm / rho_safe  # = sin(theta), safe at origin

    # dZ/dρ and dZ/dθ (in polar)
    dZ_drho = norm * dR * Theta
    dZ_dtheta = norm * R * dTheta

    # Chain rule to Cartesian
    dZdx = (dZ_drho * cos_t - dZ_dtheta * sin_t / rho_safe) * pupil
    dZdy = (dZ_drho * sin_t + dZ_dtheta * cos_t / rho_safe) * pupil

    # Fix origin (ρ=0): derivatives should be finite
    # For most modes, the derivative at ρ=0 is 0 (or a well-defined limit)
    origin = rho < 1e-12
    if n == 1 and abs_m == 1:
        # Tip/tilt: dZ/dx or dZ/dy is a constant at origin
        # Z = N·ρ·cos(θ) = N·x  →  dZ/dx = N, dZ/dy = 0  (for m=+1)
        # Z = N·ρ·sin(θ) = N·y  →  dZ/dx = 0, dZ/dy = N  (for m=-1)
        if m > 0:
            dZdx = np.where(origin, norm, dZdx) * pupil
            dZdy = np.where(origin, 0.0, dZdy) * pupil
        else:
            dZdx = np.where(origin, 0.0, dZdx) * pupil
            dZdy = np.where(origin, norm, dZdy) * pupil
    else:
        dZdx = np.where(origin, 0.0, dZdx) * pupil
        dZdy = np.where(origin, 0.0, dZdy) * pupil

    return Z, dZdx, dZdy


def zernike_basis(
        nx: int,
        ny: int,
        nz: int,
        mask: Optional[np.ndarray] = None,
        normalize_to: str = "circle",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Zernike polynomials and their x/y derivatives on a (ny, nx) grid.

    Parameters
    ----------
    nx : int
        Number of grid points in x (columns).
    ny : int
        Number of grid points in y (rows).
    nz : int
        Number of Zernike modes (Noll index j = 1 to nz).
    mask : 2D bool array (ny, nx), optional
        Pupil mask. If None, uses a circular aperture inscribed in the
        smaller dimension.
    normalize_to : str
        "circle" — normalize coordinates to unit circle inscribed in
                    min(nx, ny) (standard for circular pupils)
        "rect"   — normalize x to [-1,1] over nx, y to [-1,1] over ny
                    (for rectangular pupils)

    Returns
    -------
    Z    : (nz, ny, nx) — Zernike polynomial values
    dZdx : (nz, ny, nx) — dZ/dx in normalized pupil coordinates
    dZdy : (nz, ny, nx) — dZ/dy in normalized pupil coordinates

    Notes
    -----
    Coordinates are normalized so that the pupil edge is at ρ = 1.
    Derivatives are with respect to these normalized coordinates.

    To convert derivatives to physical units:
        dZ/dx_physical = dZ/dx_normalized / R_pupil
    where R_pupil is the physical pupil radius.
    """
    # Build coordinate grids
    # Pixel centers at 0, 1, ..., N-1
    x = np.arange(nx, dtype=np.float64)
    y = np.arange(ny, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)  # xx: (ny, nx), yy: (ny, nx)

    # Center
    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0

    # Normalize
    if normalize_to == "circle":
        radius = min(nx, ny) / 2.0
        x_norm = (xx - cx) / radius
        y_norm = (yy - cy) / radius
    elif normalize_to == "rect":
        x_norm = (xx - cx) / (nx / 2.0)
        y_norm = (yy - cy) / (ny / 2.0)
    else:
        raise ValueError(f"normalize_to must be 'circle' or 'rect', got '{normalize_to}'")

    rho = np.sqrt(x_norm ** 2 + y_norm ** 2)
    theta = np.arctan2(y_norm, x_norm)

    # Pupil mask
    if mask is None:
        pupil = (rho <= 1.0).astype(np.float64)
    else:
        pupil = mask.astype(np.float64)

    # Compute all modes
    Z = np.zeros((nz, ny, nx))
    dZdx = np.zeros((nz, ny, nx))
    dZdy = np.zeros((nz, ny, nx))

    for j in range(1, nz + 1):
        n, m = noll_to_nm(j)
        Z[j - 1], dZdx[j - 1], dZdy[j - 1] = _zernike_single(
            n, m, rho, theta, x_norm, y_norm, pupil
        )

    return Z, dZdx, dZdy


# ══════════════════════════════════════════════════════════════════════
#  GRAM-SCHMIDT ORTHOGONALIZATION
# ══════════════════════════════════════════════════════════════════════

def gs_orthogonalize(
        Z: np.ndarray,
        mask: np.ndarray,
        dZdx: Optional[np.ndarray] = None,
        dZdy: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """
    Gram-Schmidt orthogonalization of Zernike modes on a discrete pupil.

    On a finite, discrete grid, the analytically-defined Zernike polynomials
    are only approximately orthogonal. This function produces a basis that
    is **exactly orthonormal** on the sampled pupil:

        <Zi_orth, Zj_orth>_mask = δ_ij

    The same linear transformation is applied to the derivatives so that
    they remain consistent with the orthogonalized modes.

    Parameters
    ----------
    Z : (nz, ny, nx) array
        Zernike polynomial values on the grid.
    mask : (ny, nx) bool array
        Pupil mask defining the valid region.
    dZdx : (nz, ny, nx) array, optional
        x-derivatives. If provided, will be orthogonalized consistently.
    dZdy : (nz, ny, nx) array, optional
        y-derivatives. If provided, will be orthogonalized consistently.

    Returns
    -------
    Z_orth    : (nz, ny, nx) — orthonormalized modes
    dZdx_orth : (nz, ny, nx) or None — consistent derivatives
    dZdy_orth : (nz, ny, nx) or None — consistent derivatives
    T         : (nz, nz) — transformation matrix
                Z_orth[i] = Σ_j T[i,j] · Z[j]
                (useful for converting coefficients between bases)

    Notes
    -----
    The inner product is defined as:
        <f, g> = (1/N) Σ_{mask} f · g
    where N is the number of valid pixels. This makes the norm
    independent of grid resolution.

    Modes are processed in order (j=1,2,...), so lower-order modes
    retain more of their original character. The resulting modes are
    close to the original Zernikes but exactly orthogonal.

    To convert coefficients from the orthogonal basis back to the
    original Zernike basis:
        a_original = T^T · a_orth    (since T is orthogonal-ish)
    Or more precisely:
        a_original = inv(T) · a_orth
    """
    nz, ny, nx = Z.shape
    mask_f = mask.astype(np.float64)
    n_pixels = mask_f.sum()

    if n_pixels == 0:
        raise ValueError("Pupil mask has no valid pixels.")

    # Work with flattened vectors over the pupil for efficiency
    valid = mask.ravel()
    n_valid = valid.sum()

    # Extract valid pixels for each mode
    Z_vecs = np.zeros((nz, n_valid))
    for i in range(nz):
        Z_vecs[i] = Z[i].ravel()[valid]

    has_dx = dZdx is not None
    has_dy = dZdy is not None

    if has_dx:
        dZdx_vecs = np.zeros((nz, n_valid))
        for i in range(nz):
            dZdx_vecs[i] = dZdx[i].ravel()[valid]

    if has_dy:
        dZdy_vecs = np.zeros((nz, n_valid))
        for i in range(nz):
            dZdy_vecs[i] = dZdy[i].ravel()[valid]

    # Transformation matrix: Z_orth = T · Z
    T = np.zeros((nz, nz))

    # Gram-Schmidt
    Q = np.zeros((nz, n_valid))  # orthonormalized vectors

    for i in range(nz):
        v = Z_vecs[i].copy()

        # Subtract projections onto all previous orthonormal vectors
        for k in range(i):
            proj = np.dot(Q[k], v) / n_valid
            v -= proj * Q[k]

        # Normalize
        norm = np.sqrt(np.dot(v, v) / n_valid)
        if norm < 1e-15:
            # Mode is linearly dependent on previous modes (degenerate)
            Q[i] = 0.0
            continue

        Q[i] = v / norm

    # Build transformation matrix T such that Q = T · Z_vecs
    # Q[i] = Σ_j T[i,j] · Z_vecs[j]
    # Since Q is orthonormal: T[i,j] = <Q[i], Z_vecs[j]> / <Q[i], Q[i]>
    # But it's cleaner to reconstruct T from the GS steps.
    # Re-derive T by expressing each Q[i] in terms of original Z_vecs.

    # More stable: solve T from Q = T · Z_vecs via least squares
    # T · Z_vecs^T = Q^T  →  each row of T is solved independently
    # But since Z_vecs may not be square, use:  T = Q · Z_vecs^T · inv(Z_vecs · Z_vecs^T)
    # Or simply redo GS tracking coefficients:

    T = np.zeros((nz, nz))
    Q2 = np.zeros((nz, n_valid))

    for i in range(nz):
        # Start with original mode
        coeffs = np.zeros(nz)
        coeffs[i] = 1.0
        v = Z_vecs[i].copy()

        # Subtract projections
        for k in range(i):
            proj = np.dot(Q2[k], v) / n_valid
            v -= proj * Q2[k]
            coeffs -= proj * T[k]

        # Normalize
        norm = np.sqrt(np.dot(v, v) / n_valid)
        if norm < 1e-15:
            Q2[i] = 0.0
            T[i] = 0.0
            continue

        Q2[i] = v / norm
        T[i] = coeffs / norm

    # Build output arrays
    Z_orth = np.zeros_like(Z)
    for i in range(nz):
        mode = np.zeros(ny * nx)
        mode[valid] = Q2[i]
        Z_orth[i] = mode.reshape(ny, nx)

    # Apply same transformation to derivatives
    dZdx_orth = None
    dZdy_orth = None

    if has_dx:
        dZdx_orth = np.zeros_like(dZdx)
        for i in range(nz):
            # dZdx_orth[i] = Σ_j T[i,j] · dZdx[j]
            for j in range(nz):
                if abs(T[i, j]) > 1e-15:
                    dZdx_orth[i] += T[i, j] * dZdx[j]

    if has_dy:
        dZdy_orth = np.zeros_like(dZdy)
        for i in range(nz):
            for j in range(nz):
                if abs(T[i, j]) > 1e-15:
                    dZdy_orth[i] += T[i, j] * dZdy[j]

    return Z_orth, dZdx_orth, dZdy_orth, T


def convert_coefficients(coeffs_orth: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Convert coefficients from the orthogonalized basis back to the
    original Zernike basis.

        wavefront = Σ a_orth[i] · Z_orth[i]
                  = Σ a_orth[i] · Σ T[i,j] · Z[j]
                  = Σ (T^T · a_orth)[j] · Z[j]

    Parameters
    ----------
    coeffs_orth : 1D array (nz,)
        Coefficients in the orthogonalized basis.
    T : (nz, nz) array
        Transformation matrix from gs_orthogonalize().

    Returns
    -------
    coeffs_original : 1D array (nz,)
        Coefficients in the original Zernike basis.
    """
    return T.T @ coeffs_orth


# ══════════════════════════════════════════════════════════════════════
#  SUB-APERTURE (BEAM-SIZED) ZERNIKE MODES
# ══════════════════════════════════════════════════════════════════════

class SubApertureModes(NamedTuple):
    """
    Zernike modes confined to a sub-aperture of the DM.

    Attributes
    ----------
    phase        : (nz, ny, nx) — mode shapes, zero outside ``support_mask``
    dphase_dx    : (nz, ny, nx) — dφ/dx
    dphase_dy    : (nz, ny, nx) — dφ/dy
    beam_mask    : (ny, nx) bool — the sub-aperture itself (ρ ≤ 1)
    support_mask : (ny, nx) bool — every pixel the modes are allowed to move
                   (beam + blending ring for ``edge="hermite"``)
    scale        : (nz,) — factor each raw mode was divided by (see ``normalize``)

    Notes
    -----
    Derivatives are with respect to coordinates normalized to the *sub-aperture*
    radius (ρ = 1 at the sub-aperture edge), the same convention as
    :func:`zernike_basis`. To convert to per-pixel slopes divide by ``radius``.
    """
    phase: np.ndarray
    dphase_dx: np.ndarray
    dphase_dy: np.ndarray
    beam_mask: np.ndarray
    support_mask: np.ndarray
    scale: np.ndarray


def _zernike_norm(n: int, m: int) -> float:
    """Noll normalization factor (unit RMS over the unit circle)."""
    return np.sqrt(n + 1.0) if m == 0 else np.sqrt(2.0 * (n + 1.0))


def _angular_part(m: int, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Angular factor Θ(θ) and its derivative dΘ/dθ."""
    abs_m = abs(m)
    if m > 0:
        return np.cos(abs_m * theta), -abs_m * np.sin(abs_m * theta)
    if m < 0:
        return np.sin(abs_m * theta), abs_m * np.cos(abs_m * theta)
    return np.ones_like(theta), np.zeros_like(theta)


def _radial_second_derivative(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """
    Compute d²R_n^|m|/dρ² analytically.
    """
    abs_m = abs(m)
    d2R = np.zeros_like(rho)
    num_terms = (n - abs_m) // 2 + 1

    for s in range(num_terms):
        power = n - 2 * s
        if power < 2:
            continue  # second derivative of a constant or of ρ is 0
        coeff = ((-1) ** s * factorial(n - s) /
                 (factorial(s) *
                  factorial((n + abs_m) // 2 - s) *
                  factorial((n - abs_m) // 2 - s)))
        d2R = d2R + coeff * power * (power - 1) * rho ** (power - 2)

    return d2R


def _hermite_blend(t: np.ndarray, order: int = 2
                   ) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
    """
    Hermite blend from a given (value, slope, curvature) at t=0 down to
    (0, 0, 0) at t=1:

        f(t) = p·H0(t) + v·H1(t) + a·H2(t)

    order=1 : cubic   — matches value and slope   (C¹ join, ~7 % less stroke)
    order=2 : quintic — also matches curvature    (C² join, nothing for the
              mirror to ring against)

    Every basis function and its derivatives vanish at t = 1, so the blend
    reaches the surrounding flat area with no step, no kink and — for
    order=2 — no curvature jump.
    """
    t2 = t * t
    t3 = t2 * t

    if order == 1:
        H0 = 1.0 - 3.0 * t2 + 2.0 * t3
        H1 = t - 2.0 * t2 + t3
        H2 = np.zeros_like(t)
        dH0 = -6.0 * t + 6.0 * t2
        dH1 = 1.0 - 4.0 * t + 3.0 * t2
        dH2 = np.zeros_like(t)
        return (H0, H1, H2), (dH0, dH1, dH2)

    if order != 2:
        raise ValueError(f"blend order must be 1 or 2, got {order}")

    t4 = t3 * t
    t5 = t4 * t

    H0 = 1.0 - 10.0 * t3 + 15.0 * t4 - 6.0 * t5
    H1 = t - 6.0 * t3 + 8.0 * t4 - 3.0 * t5
    H2 = 0.5 * t2 - 1.5 * t3 + 1.5 * t4 - 0.5 * t5

    dH0 = -30.0 * t2 + 60.0 * t3 - 30.0 * t4
    dH1 = 1.0 - 18.0 * t2 + 32.0 * t3 - 15.0 * t4
    dH2 = t - 4.5 * t2 + 6.0 * t3 - 2.5 * t4

    return (H0, H1, H2), (dH0, dH1, dH2)


def _smoothstep(s: np.ndarray, order: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Smoothstep S(s) rising 0 → 1 on s ∈ [0, 1], and its derivative.

    order=1 : cubic  (C¹ at both ends)
    order=2 : quintic (C² at both ends) — preferred, a DM cannot reproduce a
              curvature discontinuity and would ring around it.
    """
    if order == 1:
        return 3.0 * s ** 2 - 2.0 * s ** 3, 6.0 * s * (1.0 - s)
    if order == 2:
        return (6.0 * s ** 5 - 15.0 * s ** 4 + 10.0 * s ** 3,
                30.0 * s ** 2 * (1.0 - s) ** 2)
    raise ValueError(f"smoothstep order must be 1 or 2, got {order}")


def aperture_from_mask(mask: np.ndarray) -> Tuple[float, float, float]:
    """
    Estimate the centre and radius of a (roughly circular) illuminated area.

    Parameters
    ----------
    mask : (ny, nx) bool array — e.g. the lenslets lit during the full-pupil
           influence-function calibration.

    Returns
    -------
    cx, cy : float — centroid in pixel coordinates
    radius : float — equal-area radius, sqrt(N_lit / π), in pixels
    """
    mask = np.asarray(mask, dtype=bool)
    n_lit = int(mask.sum())
    if n_lit == 0:
        raise ValueError("Mask has no valid pixels.")

    yy, xx = np.nonzero(mask)
    return float(xx.mean()), float(yy.mean()), float(np.sqrt(n_lit / np.pi))


def zernike_sub_aperture(
        nx: int,
        ny: int,
        nz: int,
        radius: float,
        center: Optional[Tuple[float, float]] = None,
        edge: str = "hermite",
        edge_width: float = 0.25,
        piston: str = "edge",
        pupil_mask: Optional[np.ndarray] = None,
        normalize: str = "rms",
        blend_order: int = 2,
) -> SubApertureModes:
    """
    Zernike modes defined on a sub-aperture of the DM, blended smoothly to a
    flat surface outside it.

    Use this when the influence function / control matrix was calibrated with
    the beam filling the whole DM, but the beam actually used for imaging is
    smaller: the aberration has to be written into the illuminated area only,
    while the rest of the mirror stays at its flat command. Simply cropping a
    Zernike to the beam leaves a step at the beam edge; the DM cannot reproduce
    a step, so the fit spills ripples back into the beam and wastes stroke.
    This function removes the step.

    Parameters
    ----------
    nx, ny : int
        Grid size (columns, rows) — the influence-function / lenslet grid.
    nz : int
        Number of Zernike modes, Noll index j = 1 … nz.
    radius : float
        Sub-aperture (beam) radius **in grid pixels**.
    center : (cx, cy), optional
        Sub-aperture centre in pixels. Defaults to the grid centre.
    edge : {"hermite", "taper", "hard"}
        How the mode reaches the flat surroundings:

        * ``"hermite"`` — the mode is *exactly* the Zernike inside the beam,
          and a ring of width ``edge_width`` **outside** the beam carries a
          quintic-Hermite blend down to flat (C² continuous). Requires free
          mirror area around the beam, which is exactly what is available when
          the beam is smaller than the DM. Preferred: the wavefront seen by
          the beam is undistorted.
        * ``"taper"`` — the mode is apodized to zero over a ring of width
          ``edge_width`` **inside** the beam. Needs no free area outside the
          beam and costs no extra stroke, but the mode is attenuated near the
          beam edge, so it is no longer a pure Zernike over the beam.
        * ``"hard"`` — plain crop at the beam edge (discontinuous). Provided
          for comparison / diagnostics only.
    edge_width : float
        Width of the blending ring, in units of ``radius``.
    piston : {"edge", "mean", "none"}
        Constant removed from the mode before blending:

        * ``"edge"`` — the mode's value at the beam edge (nonzero only for the
          rotationally symmetric modes: defocus, spherical, …). This makes the
          mode meet the surrounding flat area at zero, so the blending ring
          only has to absorb the slope, not a step — a large saving in stroke
          and in fitting error for defocus and spherical.
        * ``"mean"`` — the mean over the beam (the usual Zernike convention).
        * ``"none"`` — nothing removed.

        Piston is not an aberration, so removing it changes nothing optically.
        Note that with ``"edge"`` (or ``"mean"``) the piston mode itself, j=1,
        becomes identically zero.
    pupil_mask : (ny, nx) bool array, optional
        Area the DM can actually control (e.g. the lenslets lit during
        calibration). The modes are forced to zero outside it; a warning is
        issued if that clips a non-negligible part of the blending ring.
    normalize : {"rms", "none"}
        ``"rms"`` scales every mode to unit RMS about its mean **over the
        beam**, so a coefficient is an RMS wavefront amplitude in the beam and
        amplitudes are comparable between modes. ``"none"`` keeps the Noll
        normalization.
    blend_order : {1, 2}
        Continuity of the join with the flat area: 1 gives a C¹ join (value and
        slope), 2 a C² join (also curvature). 2 is the safer default; 1 asks
        for slightly less stroke in the blending ring.

    Returns
    -------
    SubApertureModes — see that class.

    Notes
    -----
    Confining a mode to a sub-aperture costs stroke in the blending ring, and
    the cost climbs with radial order and with ``edge_width``. Measured for a
    beam of 0.7 × the DM diameter, as the peak in the ring over the peak in the
    beam (unit-RMS modes):

        radial order n   1     2     3     4     5     6     7
        edge_width 0.15  1.0   0.7   1.1   1.1   1.4   1.1   1.9
        edge_width 0.25  1.0   0.7   1.2   1.3   1.8   1.5   2.8
        edge_width 0.40  1.0   0.8   1.5   1.6   2.5   2.2   4.4

    So a wider ring buys fidelity inside the beam, and low-order modes (what
    sensorless AO usually scans) pay almost nothing for it. If high-order modes
    run out of stroke, narrow the ring or switch to ``edge="taper"``.

    Examples
    --------
    >>> modes = zernike_sub_aperture(28, 28, 15, radius=10.0)   # doctest: +SKIP
    >>> phase = 0.2 * modes.phase[3]        # 0.2 RMS of defocus in the beam
    """
    if radius <= 0:
        raise ValueError(f"radius must be > 0, got {radius}")
    if edge not in ("hermite", "taper", "hard"):
        raise ValueError(f"edge must be 'hermite', 'taper' or 'hard', got '{edge}'")
    if piston not in ("edge", "mean", "none"):
        raise ValueError(f"piston must be 'edge', 'mean' or 'none', got '{piston}'")
    if normalize not in ("rms", "none"):
        raise ValueError(f"normalize must be 'rms' or 'none', got '{normalize}'")
    if edge != "hard" and not 0.0 < edge_width <= 1.0:
        raise ValueError(f"edge_width must be in (0, 1], got {edge_width}")
    if blend_order not in (1, 2):
        raise ValueError(f"blend_order must be 1 or 2, got {blend_order}")

    # --- coordinates normalized to the sub-aperture: ρ = 1 at the beam edge --
    xx, yy = np.meshgrid(np.arange(nx, dtype=np.float64),
                         np.arange(ny, dtype=np.float64))
    if center is None:
        cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    else:
        cx, cy = float(center[0]), float(center[1])

    u = (xx - cx) / radius
    v = (yy - cy) / radius
    rho = np.sqrt(u ** 2 + v ** 2)
    theta = np.arctan2(v, u)
    rho_safe = np.where(rho > 1e-12, rho, 1e-12)

    w = float(edge_width)
    beam = rho <= 1.0
    if edge == "hermite":
        ring = (rho > 1.0) & (rho <= 1.0 + w)
        support = beam | ring
    else:
        ring = np.zeros_like(beam)
        support = beam.copy()

    if pupil_mask is not None:
        pupil_mask = np.asarray(pupil_mask, dtype=bool)
        if pupil_mask.shape != (ny, nx):
            raise ValueError(f"pupil_mask shape {pupil_mask.shape} != ({ny}, {nx})")
        if not np.any(beam & pupil_mask):
            raise ValueError("Sub-aperture does not overlap the pupil mask — "
                             "check radius/center against the calibration mask.")
        outside = ~pupil_mask
    else:
        outside = np.zeros_like(beam)

    beam_valid = beam & ~outside      # domain for RMS normalization
    beam_f = beam.astype(np.float64)

    phase = np.zeros((nz, ny, nx))
    dphase_dx = np.zeros((nz, ny, nx))
    dphase_dy = np.zeros((nz, ny, nx))
    scale = np.ones(nz)

    # Apodization window, shared by all modes (edge="taper")
    if edge == "taper":
        s = np.clip((rho - (1.0 - w)) / w, 0.0, 1.0)
        S, dS = _smoothstep(s, order=blend_order)
        window = (1.0 - S) * beam_f
        dwindow_drho = np.where(beam & (rho > 1.0 - w), -dS / w, 0.0)
        dwindow_dx = dwindow_drho * u / rho_safe
        dwindow_dy = dwindow_drho * v / rho_safe
    if edge == "hermite":
        t = np.clip((rho - 1.0) / w, 0.0, 1.0)
        (H0, H1, H2), (dH0, dH1, dH2) = _hermite_blend(t, order=blend_order)
        ring_f = ring.astype(np.float64)
        cos_t = u / rho_safe
        sin_t = v / rho_safe

    clipped = 0.0
    for j in range(1, nz + 1):
        n, m = noll_to_nm(j)
        norm = _zernike_norm(n, m)

        # --- the Zernike itself, over the beam -------------------------------
        z, dzdx, dzdy = _zernike_single(n, m, rho, theta, u, v, beam_f)

        if piston == "edge":
            # value of the mode on the beam edge, averaged over θ:
            # N·R(1) for m = 0, and exactly 0 for every m ≠ 0
            c = norm * float(_radial_polynomial(n, m, np.ones(1))[0]) if m == 0 else 0.0
        elif piston == "mean":
            c = float(z[beam_valid].mean()) if np.any(beam_valid) else 0.0
        else:
            c = 0.0
        z = (z - c) * beam_f

        # --- join it to the surrounding flat area ----------------------------
        if edge == "hard":
            p, px, py = z, dzdx, dzdy

        elif edge == "taper":
            p = z * window
            px = dzdx * window + z * dwindow_dx
            py = dzdy * window + z * dwindow_dy

        else:  # hermite: continue the mode into the ring outside the beam
            Theta, dTheta = _angular_part(m, theta)
            one = np.ones(1)
            r1 = float(_radial_polynomial(n, m, one)[0])
            dr1 = float(_radial_derivative(n, m, one)[0])
            d2r1 = float(_radial_second_derivative(n, m, one)[0])

            # boundary value / radial slope / radial curvature at ρ = 1
            P = norm * r1 * Theta - c
            M = w * norm * dr1 * Theta
            A = w * w * norm * d2r1 * Theta
            dP = norm * r1 * dTheta
            dM = w * norm * dr1 * dTheta
            dA = w * w * norm * d2r1 * dTheta

            f = P * H0 + M * H1 + A * H2
            f_rho = (P * dH0 + M * dH1 + A * dH2) / w
            f_theta = dP * H0 + dM * H1 + dA * H2

            p = z + f * ring_f
            px = dzdx + (f_rho * cos_t - f_theta * sin_t / rho_safe) * ring_f
            py = dzdy + (f_rho * sin_t + f_theta * cos_t / rho_safe) * ring_f

        # --- nothing outside the controllable pupil --------------------------
        if np.any(outside):
            clipped = max(clipped, float(np.abs(p[outside]).max(initial=0.0)))
            p = np.where(outside, 0.0, p)
            px = np.where(outside, 0.0, px)
            py = np.where(outside, 0.0, py)

        if normalize == "rms":
            if np.any(beam_valid):
                # RMS about the mean: a piston offset is not an aberration and
                # must not count towards the amplitude, otherwise removing the
                # edge piston from defocus would halve the defocus it delivers
                rms = float(np.std(p[beam_valid]))
                if rms <= 1e-12:
                    rms = float(np.sqrt(np.mean(p[beam_valid] ** 2)))
            else:
                rms = 0.0
            if rms > 1e-12:
                scale[j - 1] = rms
                p, px, py = p / rms, px / rms, py / rms
            else:
                # degenerate (e.g. piston with piston="edge") — leave it at zero
                scale[j - 1] = 1.0

        phase[j - 1], dphase_dx[j - 1], dphase_dy[j - 1] = p, px, py

    if clipped > 0.01:
        warnings.warn(
            f"The sub-aperture blending ring reaches outside the pupil mask and was "
            f"clipped (max clipped amplitude {clipped:.3g}, i.e. a step of that size "
            f"at the pupil edge). Reduce 'radius' or 'edge_width', or use edge='taper'.",
            RuntimeWarning, stacklevel=2)

    support = support & ~outside
    return SubApertureModes(phase, dphase_dx, dphase_dy, beam & ~outside, support, scale)


def apply_transform(modes: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Apply a mode-mixing matrix to a stack of maps: ``out[i] = Σ_j T[i,j]·modes[j]``.

    Unlike :func:`gs_orthogonalize`, which rebuilds its output from the pixels
    inside the mask only, this keeps whatever the modes carry outside the mask —
    which is what sub-aperture modes need, since their blending ring lies
    outside the beam used as the inner-product domain.
    """
    return np.tensordot(T, modes, axes=(1, 0))


def stack_slopes(dZdx: np.ndarray, dZdy: np.ndarray,
                 normalize: bool = True) -> np.ndarray:
    """
    Pack x/y derivatives into the (2·npix, nz) slope matrix a zonal control
    matrix consumes: the x slopes of every pixel, then the y slopes.

    Parameters
    ----------
    dZdx, dZdy : (nz, ny, nx) arrays
    normalize : bool
        Scale each mode by the RMS of its **combined** (x, y) slope vector, so
        that one unit of coefficient means a comparable slope amplitude for
        every mode. The two components must be scaled by the *same* factor:
        normalizing x and y separately rewrites the direction of the mode, and
        blows numerical noise up to full scale for modes whose slope is purely
        along one axis (tilt).

    Returns
    -------
    slopes : (2·ny·nx, nz)
    """
    nz, ny, nx = dZdx.shape
    npix = ny * nx
    slopes = np.zeros((2 * npix, nz))

    for j in range(nz):
        gx = dZdx[j].ravel()
        gy = dZdy[j].ravel()
        if normalize:
            rms = np.sqrt(np.mean(gx ** 2 + gy ** 2) / 2.0)
            if rms > 1e-12:
                gx, gy = gx / rms, gy / rms
            else:
                gx, gy = np.zeros_like(gx), np.zeros_like(gy)
        slopes[:npix, j] = gx
        slopes[npix:, j] = gy

    return slopes


def verify_orthogonality(Z: np.ndarray, mask: np.ndarray, nz: int = None):
    """
    Verify orthogonality of the Zernike basis: <Zi, Zj> ≈ δij.

    Parameters
    ----------
    Z : (nz, ny, nx) array
    mask : (ny, nx) bool array

    Returns
    -------
    cross : (nz, nz) cross-correlation matrix
    """
    if nz is None:
        nz = Z.shape[0]

    n_pixels = mask.sum()
    cross = np.zeros((nz, nz))

    for i in range(nz):
        for j in range(nz):
            cross[i, j] = np.sum(Z[i] * Z[j]) / n_pixels

    return cross


def verify_derivatives(Z, dZdx, dZdy, mask, dx=1e-5):
    """
    Verify analytical derivatives against numerical finite differences.

    Returns max relative error for each mode.
    """
    nz, ny, nx = Z.shape
    errors_x = []
    errors_y = []

    # Recompute with shifted grids
    x = np.arange(nx, dtype=np.float64)
    y = np.arange(ny, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    cx_c = (nx - 1) / 2.0
    cy_c = (ny - 1) / 2.0
    radius = min(nx, ny) / 2.0

    for j_idx in range(nz):
        # Numerical dZ/dx via central difference on the mode values
        # This checks internal consistency
        Z_mode = Z[j_idx]
        dZdx_mode = dZdx[j_idx]
        dZdy_mode = dZdy[j_idx]

        # Numerical x-derivative (central difference)
        dZdx_num = np.zeros((ny, nx))
        dZdx_num[:, 1:-1] = (Z_mode[:, 2:] - Z_mode[:, :-2]) / 2.0 * radius

        # Numerical y-derivative
        dZdy_num = np.zeros((ny, nx))
        dZdy_num[1:-1, :] = (Z_mode[2:, :] - Z_mode[:-2, :]) / 2.0 * radius

        # Compare only interior valid points (exclude edges)
        interior = mask.copy()
        interior[0, :] = False
        interior[-1, :] = False
        interior[:, 0] = False
        interior[:, -1] = False

        valid = interior & (np.abs(dZdx_mode) > 1e-6)
        if valid.sum() > 0:
            rel_err_x = np.abs(dZdx_mode[valid] - dZdx_num[valid]) / (np.abs(dZdx_mode[valid]) + 1e-10)
            errors_x.append(np.median(rel_err_x))
        else:
            errors_x.append(0.0)

        valid = interior & (np.abs(dZdy_mode) > 1e-6)
        if valid.sum() > 0:
            rel_err_y = np.abs(dZdy_mode[valid] - dZdy_num[valid]) / (np.abs(dZdy_mode[valid]) + 1e-10)
            errors_y.append(np.median(rel_err_y))
        else:
            errors_y.append(0.0)

    return np.array(errors_x), np.array(errors_y)


if __name__ == "__main__":

    nx, ny, nz = 28, 28, 36

    # Noll table
    print(f"\nNoll index table:")
    names = zernike_names(nz)
    print(f"{'j':>4}  {'n':>3}  {'m':>4}  {'Name'}")
    print("-" * 50)
    for j in range(1, nz + 1):
        n, m = noll_to_nm(j)
        print(f"{j:>4}  {n:>3}  {m:>+4d}  {names[j - 1]}")

    Z, dZdx, dZdy = zernike_basis(nx=nx, ny=ny, nz=nz)
    print(f"  Z    shape: {Z.shape}")
    print(f"  dZdx shape: {dZdx.shape}")
    print(f"  dZdy shape: {dZdy.shape}")

    # Build mask
    yy, xx = np.mgrid[:ny, :nx]
    center = ((nx - 1) / 2.0, (ny - 1) / 2.0)
    radius = min(nx, ny) / 2.0
    rho = np.sqrt(((xx - center[0]) / radius) ** 2 + ((yy - center[1]) / radius) ** 2)
    mask = rho <= 1.0

    # ── Orthogonality BEFORE GS ──────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  Orthogonality BEFORE Gram-Schmidt")
    print(f"{'─' * 60}")
    cross_before = verify_orthogonality(Z, mask)
    diag = np.diag(cross_before)
    off_diag = cross_before - np.diag(diag)
    print(f"  Diagonal mean (should be 1.0: {diag.mean():.4f}")
    print(f"  Diagonal std:                            {diag.std():.4f}")
    print(f"  Max |off-diagonal|:                      {np.abs(off_diag).max():.6f}")

    # ── Gram-Schmidt ─────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  Applying Gram-Schmidt orthogonalization...")
    print(f"{'─' * 60}")
    Z_orth, dZdx_orth, dZdy_orth, T = gs_orthogonalize(Z, mask, dZdx, dZdy)

    # ── Orthogonality AFTER GS ───────────────────────────────────────
    print(f"\n  Orthogonality AFTER Gram-Schmidt")
    cross_after = verify_orthogonality(Z_orth, mask)
    diag_a = np.diag(cross_after)
    off_diag_a = cross_after - np.diag(diag_a)
    print(f"  Diagonal mean (should be 1.0: {diag_a.mean():.4f}")
    print(f"  Diagonal std:                            {diag_a.std():.6f}")
    print(f"  Max |off-diagonal|:                      {np.abs(off_diag_a).max():.2e}")

    # ── Derivative consistency ───────────────────────────────────────
    print(f"\n  Derivative consistency (analytical vs numerical on orthogonalized modes)")
    err_x, err_y = verify_derivatives(Z_orth, dZdx_orth, dZdy_orth, mask)
    print(f"  Median relative error dZ/dx: {np.median(err_x):.4f}")
    print(f"  Median relative error dZ/dy: {np.median(err_y):.4f}")

    # ── Transformation matrix properties ─────────────────────────────
    print(f"\n  Transformation matrix T:")
    print(f"  Shape: {T.shape}")
    print(f"  Max |off-diagonal| of T: {np.abs(T - np.diag(np.diag(T))).max():.4f}")
    print(f"  (small off-diagonal → original Zernikes were nearly orthogonal)")

    # ── Coefficient conversion test ──────────────────────────────────
    print(f"\n  Coefficient conversion test:")
    a_orth = np.random.randn(nz)
    a_orig = convert_coefficients(a_orth, T)
    # Reconstruct wavefront both ways — should be identical
    wf_orth = np.sum(a_orth[:, None, None] * Z_orth, axis=0)
    wf_orig = np.sum(a_orig[:, None, None] * Z, axis=0)
    diff = np.abs(wf_orth - wf_orig)[mask].max()
    print(f"  Max wavefront difference: {diff:.2e} (should be ~0)")

    # ── Rectangular grid test ────────────────────────────────────────
    print(f"\n  Testing rectangular grid (40×28)...")
    Z_rect, dZdx_rect, dZdy_rect = zernike_basis(nx=40, ny=28, nz=15)
    mask_rect = np.ones((28, 40), dtype=bool)
    rr = np.sqrt(((np.arange(40) - 19.5) / 14) ** 2 +
                 ((np.arange(28)[:, None] - 13.5) / 14) ** 2)
    mask_rect = rr <= 1.0
    Z_r_orth, _, _, _ = gs_orthogonalize(Z_rect, mask_rect, dZdx_rect, dZdy_rect)
    cross_rect = verify_orthogonality(Z_r_orth, mask_rect)
    print(f"  Max |off-diagonal| after GS: {np.abs(cross_rect - np.diag(np.diag(cross_rect))).max():.2e}")

    # ── Sub-aperture (beam-sized) modes ──────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"  Sub-aperture modes — beam smaller than the DM")
    print(f"{'─' * 60}")
    beam_ratio = 0.7
    cx_p, cy_p, r_pupil = aperture_from_mask(mask)
    print(f"  Pupil: centre ({cx_p:.1f}, {cy_p:.1f}), equal-area radius {r_pupil:.2f} px")
    print(f"  Beam : {beam_ratio:.2f} x the DM diameter\n")
    print(f"  {'edge':10s} {'beam px':>8s} {'support px':>11s} {'max step / px':>14s} {'peak |phase|':>13s}")
    sub = {}
    low = slice(1, 12)  # modes 2-12, the ones sensorless AO usually scans
    for edge_mode in ("hard", "taper", "hermite"):
        sub[edge_mode] = zernike_sub_aperture(nx, ny, nz, radius=beam_ratio * r_pupil,
                                              center=(cx_p, cy_p), edge=edge_mode,
                                              edge_width=0.25, pupil_mask=mask)
        s = sub[edge_mode]
        step = max(np.abs(np.diff(s.phase[low], axis=1)).max(),
                   np.abs(np.diff(s.phase[low], axis=2)).max())
        print(f"  {edge_mode:10s} {int(s.beam_mask.sum()):8d} {int(s.support_mask.sum()):11d} "
              f"{step:14.4f} {np.abs(s.phase[low]).max():13.4f}")
    print("  (step and peak over modes 2-12; a coarse 28x28 grid undersamples the high orders)")
    s = sub["hermite"]
    print(f"\n  Outside the support the modes are flat to "
          f"{np.abs(s.phase[:, ~s.support_mask]).max():.1e}")
    print(f"  RMS over the beam (modes 2-10): "
          f"{np.round([np.std(s.phase[j][s.beam_mask]) for j in range(1, 10)], 4)}")

    # ── Plots ────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        save_path = "zernike_modes_orthogonalized.png"

        nz_plot = min(15, Z_orth.shape[0])
        names = zernike_names(nz_plot)

        fig, axes = plt.subplots(3, nz_plot, figsize=(2.5 * nz_plot, 7))

        for j in range(nz_plot):
            vmax = max(np.abs(Z_orth[j]).max(), 1e-10)
            axes[0, j].imshow(Z_orth[j], cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="none")
            axes[0, j].set_title(f"Z{j + 1}", fontsize=8)
            axes[0, j].axis("off")

            vmax_d = max(np.abs(dZdx_orth[j]).max(), 1e-10)
            axes[1, j].imshow(dZdx_orth[j], cmap="RdBu_r", vmin=-vmax_d, vmax=vmax_d, interpolation="none")
            axes[1, j].axis("off")

            vmax_d = max(np.abs(dZdy_orth[j]).max(), 1e-10)
            axes[2, j].imshow(dZdy_orth[j], cmap="RdBu_r", vmin=-vmax_d, vmax=vmax_d, interpolation="none")
            axes[2, j].axis("off")

        axes[0, 0].set_ylabel("Z", fontsize=12)
        axes[1, 0].set_ylabel("dZ/dx", fontsize=12)
        axes[2, 0].set_ylabel("dZ/dy", fontsize=12)

        plt.suptitle("Zernike Polynomials and Derivatives", fontsize=14, y=1.01)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {save_path}")
        plt.show()

        # Comparison: before vs after orthogonality matrices
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

        vmax = max(np.abs(cross_before).max(), np.pi + 0.5)
        im0 = axes[0].imshow(cross_before, cmap="RdBu_r", vmin=-0.5, vmax=vmax, interpolation="none")
        axes[0].set_title("Before GS\n(cross-correlation)")
        fig.colorbar(im0, ax=axes[0], shrink=0.8)

        im1 = axes[1].imshow(cross_after, cmap="RdBu_r", vmin=-0.5, vmax=vmax, interpolation="none")
        axes[1].set_title("After GS\n(cross-correlation)")
        fig.colorbar(im1, ax=axes[1], shrink=0.8)

        # Transformation matrix
        im2 = axes[2].imshow(T, cmap="RdBu_r", vmin=-np.abs(T).max(), vmax=np.abs(T).max(), interpolation="none")
        axes[2].set_title("Transformation matrix T\n(Z_orth = T · Z)")
        fig.colorbar(im2, ax=axes[2], shrink=0.8)

        for ax in axes:
            ax.set_xlabel("Mode j")
            ax.set_ylabel("Mode j")

        plt.tight_layout()
        plt.savefig("zernike_gs_comparison.png", dpi=150, bbox_inches="tight")
        print("\nSaved to zernike_gs_comparison.png")
        plt.show()

        # Sub-aperture modes: hard crop vs taper vs blended ring
        show = [1, 3, 5, 7, 10]
        fig, axes = plt.subplots(3, len(show) + 1, figsize=(3.0 * (len(show) + 1), 8.5))
        for row, edge_mode in enumerate(("hard", "taper", "hermite")):
            s_modes = sub[edge_mode]
            for col, j in enumerate(show):
                v = np.abs(s_modes.phase[j]).max()
                axes[row, col].imshow(s_modes.phase[j], cmap="RdBu_r", vmin=-v, vmax=v,
                                      interpolation="none")
                axes[row, col].contour(s_modes.beam_mask, [0.5], colors="k", linewidths=0.8)
                axes[row, col].axis("off")
                if row == 0:
                    axes[row, col].set_title(zernike_names(nz)[j].split(") ")[-1], fontsize=9)
            # radial cut through defocus
            cut = s_modes.phase[3][int(round(cy_p))]
            axes[row, -1].plot(cut, lw=1.2)
            axes[row, -1].axvline(cx_p - beam_ratio * r_pupil, ls=":", c="k", lw=0.8)
            axes[row, -1].axvline(cx_p + beam_ratio * r_pupil, ls=":", c="k", lw=0.8)
            axes[row, -1].set_title(f"{edge_mode}: cut through defocus", fontsize=9)
            axes[row, -1].set_xlabel("lenslet")
        plt.suptitle(f"Zernike modes on a beam {beam_ratio:.2f} x the DM diameter "
                     f"(dotted lines / black contour = beam edge)", fontsize=12)
        plt.tight_layout()
        plt.savefig("zernike_sub_aperture.png", dpi=150, bbox_inches="tight")
        print("Saved to zernike_sub_aperture.png")
        plt.show()

    except Exception as e:
        print(f"\n(Plotting skipped: {e})")
