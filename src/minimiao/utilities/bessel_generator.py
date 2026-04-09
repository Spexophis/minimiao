# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


"""
Generate real-valued weighted-Bessel modes on a circular pupil.
"""

import math
from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.special import jn_zeros, jv


@dataclass(frozen=True)
class ModeIndex:
    n: int
    m: int


def valid_nm(n: int, m: int) -> bool:
    """Return True if (n, m) is a valid real weighted-Bessel mode index."""
    return n >= 0 and abs(m) <= n and ((n - abs(m)) % 2 == 0)


def nm_list(n_max: int) -> List[ModeIndex]:
    """List all valid mode indices up to radial order n_max."""
    out: List[ModeIndex] = []
    for n in range(n_max + 1):
        for m in range(-n, n + 1, 2):
            if valid_nm(n, m):
                out.append(ModeIndex(n=n, m=m))
    return out


def make_grid(grid_size: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a square Cartesian grid spanning [-1, 1] in both x and y."""
    x = np.linspace(-1.0, 1.0, grid_size, dtype=np.float64)
    y = np.linspace(-1.0, 1.0, grid_size, dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.sqrt(X ** 2 + Y ** 2)
    Theta = np.arctan2(Y, X)
    return X, Y, R, Theta


def pupil_mask(R: np.ndarray) -> np.ndarray:
    """Boolean mask for the unit disk."""
    return R <= 1.0


def gaussian_pupil_intensity(R: np.ndarray, waist: float = 0.8) -> np.ndarray:
    """
    Normalized Gaussian pupil intensity I_hat(r).

    Parameters
    ----------
    R : array
        Radial coordinate on the pupil.
    waist : float
        Gaussian 1/e^2 intensity radius, expressed in units of pupil radius.
        Smaller values represent stronger underfilling.
    """
    if waist <= 0:
        raise ValueError("waist must be > 0")
    I = np.exp(-2.0 * (R / waist) ** 2)
    return I


def angular_factor(m: int, theta: np.ndarray) -> np.ndarray:
    """Real-valued angular factor."""
    if m == 0:
        return np.ones_like(theta)
    if m > 0:
        return np.cos(m * theta)
    return np.sin(abs(m) * theta)


def bessel_zero_for_nm(n: int, m: int) -> float:
    """
    Return alpha = j_{|m|, k} with k = (n - |m|)//2 + 1.
    """
    if not valid_nm(n, m):
        raise ValueError(f"Invalid mode indices (n={n}, m={m})")
    absm = abs(m)
    k = (n - absm) // 2 + 1
    return float(jn_zeros(absm, k)[-1])


def weighted_inner_product(
        A: np.ndarray,
        B: np.ndarray,
        I_hat: np.ndarray,
        mask: np.ndarray,
        dx: float,
        dy: float,
) -> float:
    """
    Approximate weighted inner product:
        <A, B> = (1/pi) * ∬ A B I_hat dA
    over the unit pupil.
    """
    integrand = np.where(mask, A * B * I_hat, 0.0)
    return float(np.sum(integrand) * dx * dy / math.pi)


def weighted_bessel_mode(
        n: int,
        m: int,
        R: np.ndarray,
        Theta: np.ndarray,
        I_hat: np.ndarray,
        mask: np.ndarray,
        dx: float,
        dy: float,
        normalize: bool = True,
) -> np.ndarray:
    """
    Generate one weighted-Bessel mode on a grid.

    The returned mode is zero outside the unit pupil.
    """
    if not valid_nm(n, m):
        raise ValueError(f"Invalid mode indices (n={n}, m={m})")

    alpha = bessel_zero_for_nm(n, m)
    radial = jv(abs(m), alpha * R)
    ang = angular_factor(m, Theta)

    mode = np.zeros_like(R, dtype=np.float64)
    denom = np.sqrt(np.clip(I_hat, 1e-300, None))
    mode_inside = radial[mask] * ang[mask] / denom[mask]
    mode[mask] = mode_inside

    if normalize:
        norm2 = weighted_inner_product(mode, mode, I_hat, mask, dx, dy)
        if norm2 <= 0:
            raise RuntimeError(f"Non-positive norm encountered for mode (n={n}, m={m})")
        mode /= math.sqrt(norm2)

    return mode


def generate_weighted_bessel_basis(
        n_max: int,
        grid_size: int = 512,
        waist: float = 0.65,
        normalize: bool = True,
) -> tuple[list[ModeIndex], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate all weighted-Bessel modes up to radial order n_max.

    Returns
    -------
    modes_nm : list[ModeIndex]
        List of mode indices.
    modes : ndarray, shape (num_modes, grid_size, grid_size)
        Mode stack.
    X, Y, R, Theta, I_hat : ndarrays
        Grid and pupil intensity arrays.
    """
    X, Y, R, Theta = make_grid(grid_size=grid_size)
    mask = pupil_mask(R)
    I_hat = gaussian_pupil_intensity(R, waist=waist)

    x = X[0, :]
    y = Y[:, 0]
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])

    modes_nm = nm_list(n_max)
    modes = np.empty((len(modes_nm), grid_size, grid_size), dtype=np.float64)

    for i, idx in enumerate(modes_nm):
        modes[i] = weighted_bessel_mode(
            idx.n, idx.m, R, Theta, I_hat, mask, dx, dy, normalize=normalize
        )

    return modes_nm, modes, X, Y, R, Theta, I_hat


def gram_matrix(
        modes: np.ndarray,
        I_hat: np.ndarray,
        mask: np.ndarray,
        dx: float,
        dy: float,
) -> np.ndarray:
    """
    Compute the weighted Gram matrix G_ij = <mode_i, mode_j>.
    """
    n_modes = modes.shape[0]
    G = np.empty((n_modes, n_modes), dtype=np.float64)
    for i in range(n_modes):
        for j in range(i, n_modes):
            gij = weighted_inner_product(modes[i], modes[j], I_hat, mask, dx, dy)
            G[i, j] = gij
            G[j, i] = gij
    return G


if __name__ == "__main__":
    n_max = 4  # maximum radial order
    grid_size = 64  # grid size in pixels
    waist = 0.8  # Gaussian 1/e^2 intensity radius relative to pupil radius
    max_modes_to_plot = None
    save_npz = True
    check_gram = True

    modes_nm, modes, X, Y, R, Theta, I_hat = generate_weighted_bessel_basis(
        n_max=n_max,
        grid_size=grid_size,
        waist=waist,
        normalize=True,
    )

    bpp = np.pad(modes, ((0, 0), (96, 96), (96, 96)), 'constant', constant_values=(0, 0))
    mask = bg.pupil_mask(R)
    dx = float(X[0, 1] - X[0, 0])
    dy = float(Y[1, 0] - Y[0, 0])
    msk = np.pad(mask * 1, ((96, 96), (96, 96)), 'constant', constant_values=(0, 0))

    imgs = np.zeros((15, 256, 256))
    for i in range(15):
        imgs[i] = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(msk * np.exp(1j * bpp[i]))))) ** 2

