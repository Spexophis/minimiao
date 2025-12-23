# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import numpy as np
from scipy.special import factorial

num_znk = 16

indices = [[1, 0, 0],
           [2, 1, 1], [3, 1, -1],
           [4, 2, 0], [5, 2, -2], [6, 2, 2],
           [7, 3, -1], [8, 3, 1], [9, 3, -3], [10, 3, 3],
           [11, 4, 0], [12, 4, 2], [13, 4, -2],
           [16, 5, 1], [17, 5, -1],
           [22, 6, 0]]

modes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 22]

rms = [1.,
       2., 2.,
       np.sqrt(3), np.sqrt(6), np.sqrt(6),
       np.sqrt(8), np.sqrt(8), np.sqrt(8), np.sqrt(8),
       np.sqrt(5), np.sqrt(10), np.sqrt(10),
       np.sqrt(12), np.sqrt(12),
       np.sqrt(7)]


def _polar_grid(x, y):
    return np.sqrt(x ** 2 + y ** 2), np.arctan2(y, x)


def _cartesian_grid(nx, ny):
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    return np.meshgrid(y, x, indexing='ij')


def _elliptical_mask(radius, size):
    coord_x = np.arange(0.5, size[0], 1.0)
    coord_y = np.arange(0.5, size[1], 1.0)
    y, x = np.meshgrid(coord_y, coord_x)
    x -= size[0] / 2.
    y -= size[1] / 2.
    return (x * x / (radius[0] * radius[0])) + (y * y / (radius[1] * radius[1])) <= 1


def _circular_mask(radius, size):
    coords = np.arange(0.5, size, 1.0)
    x, y = np.meshgrid(coords, coords)
    x -= size / 2.
    y -= size / 2.
    return x * x + y * y <= radius * radius


def _zernike_j_nm(j):
    n = int((-1. + np.sqrt(8 * (j - 1) + 1)) / 2.)
    p = (j - (n * (n + 1)) / 2.)
    k = n % 2
    m = int((p + k) / 2.) * 2 - k
    if m != 0:
        if j % 2 == 0:
            s = 1
        else:
            s = -1
        m *= s
    return n, m


def _zernike(n, m, rho, phi):
    if (n < 0) or (n < abs(m)) or (n % 2 != abs(m) % 2):
        raise ValueError("n and m are not valid Zernike indices")
    kmax = int((n - abs(m)) / 2)
    _R = 0
    _O = 0
    _C = 0
    if m == 0:
        _C = np.sqrt(n + 1)
    else:
        _C = np.sqrt(2 * n + 1)
    for k in range(kmax + 1):
        _R += (-1) ** k * factorial(n - k) / (
                factorial(k) * factorial(0.5 * (n + abs(m)) - k) * factorial(0.5 * (n - abs(m)) - k)) * rho ** (
                      n - 2 * k)
    if m >= 0:
        _O = np.cos(m * phi)
    if m < 0:
        _O = - np.sin(m * phi)
    return _C * _R * _O


def _zernike_derivatives(n, m, rho, phi):
    if (n < 0) or (n < abs(m)) or (n % 2 != abs(m) % 2):
        raise ValueError("n and m are not valid Zernike indices")
    kmax = int((n - abs(m)) / 2)
    _R = 0
    _dR = 0
    _O = 0
    _dO = 0
    for k in range(kmax + 1):
        _R += (-1) ** k * factorial(n - k) / (
                factorial(k) * factorial(0.5 * (n + abs(m)) - k) * factorial(0.5 * (n - abs(m)) - k)) * rho ** (
                      n - 2 * k)
        _dR += (-1) ** k * factorial(n - k) / (
                factorial(k) * factorial(0.5 * (n + abs(m)) - k) * factorial(0.5 * (n - abs(m)) - k)) * (
                       n - 2 * k) * rho ** (n - 2 * k - 1)
    if m >= 0:
        _O = np.cos(m * phi)
        _dO = - m * np.sin(m * phi)
    if m < 0:
        _O = - np.sin(m * phi)
        _dO = - m * np.cos(m * phi)
    zdx = _dR * _O * np.cos(phi) - np.divide(_R, rho, out=np.zeros_like(_R, dtype=float),
                                             where=rho != 0.) * _dO * np.sin(phi)
    zdy = _dR * _O * np.sin(phi) + np.divide(_R, rho, out=np.zeros_like(_R, dtype=float),
                                             where=rho != 0.) * _dO * np.cos(phi)
    return zdx, zdy


def zernike_polynomials(size=None):
    if size is None:
        size = (16, 16)
    y, x = size
    yv, xv = _cartesian_grid(x, y)
    rho, phi = _polar_grid(xv, yv)
    phi = np.pi / 2 - phi
    phi = np.mod(phi, 2 * np.pi)
    msk = _elliptical_mask((y / 2, x / 2), (y, x))
    zls = [np.zeros_like(rho) + 1.,
           rho * np.cos(phi),
           rho * np.sin(phi),
           2 * (rho ** 2) - 1,
           (rho ** 2) * np.sin(2 * phi),
           (rho ** 2) * np.cos(2 * phi),
           (3 * (rho ** 3) - 2 * rho) * np.sin(phi),
           (3 * (rho ** 3) - 2 * rho) * np.cos(phi),
           (rho ** 3) * np.sin(3 * phi),
           (rho ** 3) * np.cos(3 * phi),
           6 * (rho ** 4) - 6 * (rho ** 2) + 1,
           (4 * (rho ** 4) - 3 * (rho ** 2)) * np.cos(2 * phi),
           (4 * (rho ** 4) - 3 * (rho ** 2)) * np.sin(2 * phi),
           (10 * (rho ** 5) - 12 * (rho ** 3) + 3 * rho) * np.cos(phi),
           (10 * (rho ** 5) - 12 * (rho ** 3) + 3 * rho) * np.sin(phi),
           20 * (rho ** 6) - 30 * (rho ** 4) + 12 * (rho ** 2) - 1]
    zls = [zk * rm * msk for zk, rm in zip(zls, rms)]
    return _gs_orthogonalisation(np.asarray(zls))


def zernike_derivatives(size=None):
    if size is None:
        size = (16, 16)
    y, x = size
    yv, xv = _cartesian_grid(x, y)
    msk = _elliptical_mask((y / 2, x / 2), (y, x))
    zdx = [np.zeros(size),
           np.zeros(size),
           2 * np.ones(size),
           4 * np.sqrt(3) * xv,
           2 * np.sqrt(6) * yv,
           - 2 * np.sqrt(6) * xv,
           2 * np.sqrt(2) * (9 * (xv ** 2) + 3 * (yv ** 2) - 2),
           12 * np.sqrt(2) * xv * yv,
           6 * np.sqrt(2) * ((yv ** 2) - (xv ** 2)),
           - 12 * np.sqrt(2) * xv * yv,
           12 * np.sqrt(5) * xv * (2 * ((yv ** 2) + (xv ** 2)) - 1),
           2 * np.sqrt(10) * xv * (8 * (xv ** 2) - 3),
           2 * np.sqrt(10) * yv * (12 * (xv ** 2) + 4 * (yv ** 2) - 3),
           16 * np.sqrt(3) * xv * yv * (5 * (xv ** 2) + 5 * (yv ** 2) - 3),
           2 * np.sqrt(3) * (
                   50 * (xv ** 4) + 12 * (xv ** 2) * (5 * (yv ** 2) - 3) + 2 * (yv ** 2) * (5 * (yv ** 2) - 6) + 3),
           24 * np.sqrt(7) * xv * (1 + 5 * ((xv ** 2) + (yv ** 2)) * ((xv ** 2) + (yv ** 2) - 1))
           ]
    zdy = [np.zeros(size),
           2 * np.ones(size),
           np.zeros(size),
           4 * np.sqrt(3) * yv,
           2 * np.sqrt(6) * xv,
           2 * np.sqrt(6) * yv,
           12 * np.sqrt(2) * xv * yv,
           2 * np.sqrt(2) * (9 * (yv ** 2) + 3 * (xv ** 2) - 2),
           12 * np.sqrt(2) * xv * yv,
           6 * np.sqrt(2) * ((yv ** 2) - (xv ** 2)),
           12 * np.sqrt(5) * yv * (2 * ((yv ** 2) + (xv ** 2)) - 1),
           2 * np.sqrt(10) * yv * (3 - 8 * (yv ** 2)),
           2 * np.sqrt(10) * xv * (12 * (yv ** 2) + 4 * (xv ** 2) - 3),
           2 * np.sqrt(3) * (
                   50 * (xv ** 4) + 12 * (xv ** 2) * (5 * (yv ** 2) - 3) + 2 * (yv ** 2) * (5 * (yv ** 2) - 6) + 3),
           16 * np.sqrt(3) * xv * yv * (5 * (xv ** 2) + 5 * (yv ** 2) - 3),
           24 * np.sqrt(7) * yv * (1 + 5 * ((xv ** 2) + (yv ** 2)) * ((xv ** 2) + (yv ** 2) - 1))
           ]
    zdx = [zd * msk for zd in zdx]
    zdy = [zd * msk for zd in zdy]
    zs = np.zeros((2 * x * y, num_znk))
    for j in range(num_znk):
        zs[:x * y, j] = zdx[j].flatten()
        zs[x * y:, j] = zdy[j].flatten()
    return zs


def zernike_primes(size=None):
    if size is None:
        size = (16, 16)
    y, x = size
    yv, xv = _cartesian_grid(x, y)
    rho, phi = _polar_grid(xv, yv)
    phi = np.pi / 2 - phi
    phi = np.mod(phi, 2 * np.pi)
    msk = _elliptical_mask((y / 2, x / 2), (y, x))
    R = [np.zeros_like(rho),
         rho,
         rho,
         2 * (rho ** 2) - 1,
         rho ** 2,
         rho ** 2,
         3 * (rho ** 3) - 2 * rho,
         3 * (rho ** 3) - 2 * rho,
         rho ** 3,
         rho ** 3,
         6 * (rho ** 4) - 6 * (rho ** 2) + 1,
         4 * (rho ** 4) - 3 * (rho ** 2),
         4 * (rho ** 4) - 3 * (rho ** 2),
         10 * (rho ** 5) - 12 * (rho ** 3) + 3 * rho,
         10 * (rho ** 5) - 12 * (rho ** 3) + 3 * rho,
         20 * (rho ** 6) - 30 * (rho ** 4) + 12 * (rho ** 2) - 1]
    dR = [np.zeros_like(rho),
          np.zeros_like(phi) + 1.,
          np.zeros_like(phi) + 1.,
          4 * rho,
          rho * 2,
          rho * 2,
          9 * (rho ** 2) - 2,
          9 * (rho ** 2) - 2,
          3 * rho ** 2,
          3 * rho ** 2,
          24 * (rho ** 3) - 12 * rho,
          16 * (rho ** 3) - 6 * rho,
          16 * (rho ** 3) - 6 * rho,
          50 * (rho ** 4) - 36 * (rho ** 2) + 3,
          50 * (rho ** 4) - 36 * (rho ** 2) + 3,
          120 * (rho ** 5) - 120 * (rho ** 4) + 24 * rho]
    A = [np.zeros_like(phi) + 1.,
         np.cos(phi),
         np.sin(phi),
         np.zeros_like(phi) + 1,
         np.sin(2 * phi),
         np.cos(2 * phi),
         np.sin(phi),
         np.cos(phi),
         np.sin(3 * phi),
         np.cos(3 * phi),
         np.zeros_like(phi) + 1.,
         np.cos(2 * phi),
         np.sin(2 * phi),
         np.cos(phi),
         np.sin(phi),
         np.zeros_like(phi) + 1.]
    dA = [np.zeros_like(phi),
          -np.sin(phi),
          np.cos(phi),
          np.zeros_like(phi),
          2 * np.cos(2 * phi),
          - 2 * np.sin(2 * phi),
          np.cos(phi),
          - np.sin(phi),
          3 * np.cos(3 * phi),
          - 3 * np.sin(3 * phi),
          np.zeros_like(phi),
          - 2 * np.sin(2 * phi),
          2 * np.cos(2 * phi),
          -np.sin(phi),
          np.cos(phi),
          np.zeros_like(phi)]
    zdx = []
    zdy = []
    for j in range(num_znk):
        N = np.sqrt(2 * (indices[j][1] + 1)) if indices[j][2] != 0 else np.sqrt(indices[j][1] + 1)
        if indices[j][2] == 0:
            O = 1
            dO = 0
        else:
            if indices[j][2] % 2 == 0:
                O = np.cos(np.abs(indices[j][2]) * phi)
                dO = - np.abs(indices[j][2]) * np.sin(np.abs(indices[j][2]) * phi)
            else:
                O = np.sin(np.abs(indices[j][2]) * phi)
                dO = np.abs(indices[j][2]) * np.cos(np.abs(indices[j][2]) * phi)
        zdx.append(msk * rms[j] * (dR[j] * O * np.cos(phi) - np.divide(R[j], rho, out=np.zeros_like(R[j], dtype=float),
                                                                       where=rho != 0.) * dO * np.sin(phi)))
        zdy.append(msk * rms[j] * (dR[j] * O * np.sin(phi) + np.divide(R[j], rho, out=np.zeros_like(R[j], dtype=float),
                                                                       where=rho != 0.) * dO * np.cos(phi)))
    return zdx, zdy


def _gs_orthogonalisation(arrays):
    nz, ny, nx = arrays.shape
    ortharray = np.zeros((nz, ny, nx))
    ortharray[0] = arrays[0] / np.linalg.norm(arrays[0])
    for ii in range(1, nz):
        ortharray[ii] = arrays[ii]
        for jj in range(ii):
            inner_product = np.einsum('ij,ij->', np.conjugate(ortharray[jj]), ortharray[ii])
            ortharray[ii] = ortharray[ii] - inner_product * ortharray[jj]
        norm = np.linalg.norm(ortharray[ii], axis=(0, 1))
        ortharray[ii] = ortharray[ii] / norm
    return ortharray


def get_zernike_polynomials(nz=60, size=None):
    if size is None:
        size = [16, 16]
    y, x = size
    yv, xv = _cartesian_grid(x, y)
    rho, phi = _polar_grid(xv, yv)
    phi = np.pi / 2 - phi
    phi = np.mod(phi, 2 * np.pi)
    zernike = np.zeros((nz, y, x))
    msk = _elliptical_mask((y / 2, x / 2), (y, x))
    for j in range(nz):
        n, m = _zernike_j_nm(j + 1)
        zernike[j, :, :] = msk * _zernike(n, m, rho, phi)
    return _gs_orthogonalisation(zernike)


def get_zernike_slopes(nz=58, size=None):
    if size is None:
        size = [64, 64]
    x, y = size
    xv, yv = _cartesian_grid(x, y)
    rho, phi = _polar_grid(xv, yv)
    phi = np.pi / 2 - phi
    phi = np.mod(phi, 2 * np.pi)
    msk = _elliptical_mask((y / 2, x / 2), (y, x))
    zs = np.zeros((2 * x * y, nz))
    for j in range(nz):
        n, m = _zernike_j_nm(j + 1)
        zdx, zdy = msk * _zernike_derivatives(n, m, rho, phi)
        zs[:x * y, j] = zdx.flatten()
        zs[x * y:, j] = zdy.flatten()
    return zs
