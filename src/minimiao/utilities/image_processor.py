# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


from functools import lru_cache
import numpy as np
from numpy.fft import fft2, fftshift
from scipy.optimize import curve_fit, minimize
from skimage import filters
import matplotlib.pyplot as plt

wl = 0.5  # wavelength in microns
na = 1.3  # numerical aperture
dx = 0.06  # pixel size in microns
fs = 1 / dx  # Spatial sampling frequency, inverse microns


def rms(data):
    _nx, _ny = data.shape
    _n = _nx * _ny
    m = np.mean(data, dtype=np.float64)
    a = (data - m) ** 2
    r = np.sqrt(np.sum(a) / _n)
    return r


def img_properties(img):
    return img.min(), img.max(), rms(img)


def fourier_transform(data):
    return np.log(np.abs(fftshift(fft2(data))))


def find_center_of_mass(image):
    height, width = image.shape
    # row_indices, col_indices = np.indices((height, width))
    # total_mass = np.sum(image)
    # row_mass = np.sum(row_indices * image) / total_mass
    # col_mass = np.sum(col_indices * image) / total_mass
    row_indices = np.arange(0, height)[:, np.newaxis]
    col_indices = np.arange(0, width)
    total_mass = np.sum(image)
    row_mass = np.sum(image * row_indices) / total_mass
    col_mass = np.sum(image * col_indices) / total_mass
    return row_mass, col_mass


def find_valley_2d(image, coordinates=None):
    data = image - image.min()
    data = data / data.max()
    mx = np.average(data, axis=0)
    my = np.average(data, axis=1)
    if coordinates is not None:
        cdx, cdy = coordinates
        l_ = valley_find(cdy, my)
        k_ = valley_find(cdx, mx)
        return k_, l_
    else:
        l_ = np.where(my == my.min())
        k_ = np.where(mx == mx.min())
        return k_[0][0], l_[0][0]


def find_peak_2d(image, coordinates=None):
    data = image - image.min()
    data = data / data.max()
    mx = np.average(data, axis=0)
    my = np.average(data, axis=1)
    if coordinates is not None:
        cdx, cdy = coordinates
        l_ = peak_find(cdy, my)
        k_ = peak_find(cdx, mx)
        return k_, l_
    else:
        l_ = np.where(my == my.max())
        k_ = np.where(mx == mx.max())
        return k_[0][0], l_[0][0]


def gaussian_beam(r, bg, I0, r0, w0):
    return bg + I0 * np.exp(-2 * ((r - r0) / w0) ** 2)


def fit_gaussian(image, verbose=False, plot=False, bounds=None):
    y_px, x_px = image.shape
    x, y = range(x_px), range(y_px)
    x_max = np.max(image, axis=0)
    y_max = np.max(image, axis=1)
    if bounds is None:
        bg_min, bg_max = 0, 10 * np.min(image)  # background
        I0_min, I0_max = np.min(image), 2 * np.max(image)  # peak intensity
        mean_min, mean_max = 0, max(x_px, y_px)  # mean
        w0_min, w0_max = 0, max(x_px, y_px)  # beam 1/e^2 radius
        bounds = ((bg_min, I0_min, mean_min, w0_min),
                  (bg_max, I0_max, mean_max, w0_max))
    xp = curve_fit(gaussian_beam, x, x_max, bounds=bounds)[0]  # x parameters
    yp = curve_fit(gaussian_beam, y, y_max, bounds=bounds)[0]  # y parameters
    xp = np.append(xp, (2 * np.log(2)) ** 0.5 * xp[3])  # add x FWHM
    yp = np.append(yp, (2 * np.log(2)) ** 0.5 * yp[3])  # add y FWHM
    if verbose:
        print('x: bg=%0.2f, I0=%0.2f, r0=%0.2f, w0=%0.2f, FWHM=%0.2f' % tuple(xp))
        print('y: bg=%0.2f, I0=%0.2f, r0=%0.2f, w0=%0.2f, FWHM=%0.2f' % tuple(yp))
    if plot:
        x_crv = gaussian_beam(x, *xp[:-1])
        y_crv = gaussian_beam(y, *yp[:-1])
        fig, ax = plt.subplots()
        ax.set_title('Gaussian fit')
        ax.set_ylabel('intensity')
        ax.set_xlabel('pixels')
        ax.plot(x, x_max, color='g', label='x_max', linestyle='--')
        ax.plot(x, x_crv, color='g', label='x_curve: (FWHM=%0.1f)' % xp[4])
        ax.plot(y, y_max, color='b', label='y_max', linestyle='--')
        ax.plot(y, y_crv, color='b', label='y_curve: (FWHM=%0.1f)' % yp[4])
        ax.legend(loc="upper right")
        fig.savefig('guassian_fit', dpi=150)
        fig.show()
    return xp, yp


def disc_array(shape=(128, 128), radi=64.0, origin=None, dtp=np.float64):
    _nx = shape[0]
    _ny = shape[1]
    ox = _nx / 2
    oy = _ny / 2
    x = np.linspace(-ox, ox - 1, _nx)
    y = np.linspace(-oy, oy - 1, _ny)
    xv, yv = np.meshgrid(x, y)
    rho = np.sqrt(xv ** 2 + yv ** 2)
    disc = (rho < radi).astype(dtp)
    if origin is not None:
        s0 = origin[0] - int(_nx / 2)
        s1 = origin[1] - int(_ny / 2)
        disc = np.roll(np.roll(disc, int(s0), 0), int(s1), 1)
    return disc


def gaussian_filter(shape, sigma, pv, orig=None):
    _nx, _ny = shape
    if orig is None:
        ux = _nx / 2.
        uy = _ny / 2.
    else:
        ux, uy = orig
    g = np.fromfunction(lambda i, j: np.exp(-((i - ux) ** 2. + (j - uy) ** 2.) / (2. * sigma ** 2.)), (_nx, _ny))
    return pv * g


def fft_frequency_2dmap(rows, cols, psy, psx):
    freq_x = np.fft.fftfreq(cols, psx)
    freq_y = np.fft.fftfreq(rows, psy)
    fx, fy = np.meshgrid(freq_x, freq_y)
    fxy = np.sqrt(fx ** 2 + fy ** 2)
    frequency_map = np.divide(1.0, fxy, where=fxy != 0, out=np.zeros_like(fxy))
    return fftshift(frequency_map)


def selected_frequency(img, freqs, relative=True):
    _ny, _nx = img.shape
    df = fs / _nx
    radius = (na / wl) / df
    freq_x = fftshift(np.fft.fftfreq(_nx, dx))
    freq_y = fftshift(np.fft.fftfreq(_ny, dx))
    freq_x = np.divide(1.0, freq_x, where=freq_x != 0, out=np.zeros_like(freq_x))
    freq_y = np.divide(1.0, freq_y, where=freq_y != 0, out=np.zeros_like(freq_y))
    freq_coords = []
    for freq in freqs:
        horizontal_indices = np.argsort(np.abs(freq_x - freq))[:2]
        horizontal_coords = [(x, _ny // 2) for x in horizontal_indices]
        vertical_indices = np.argsort(np.abs(freq_y - freq))[:2]
        vertical_coords = [(_nx // 2, y) for y in vertical_indices]
        freq_coords += horizontal_coords + vertical_coords
    msk = disc_array(shape=(_ny, _nx), radi=0.9 * radius)
    g = np.zeros((_ny, _nx))
    for freq_coord in freq_coords:
        g += disc_array(shape=(_ny, _nx), radi=9, origin=freq_coord)
    wft = np.fft.fftshift(np.fft.fft2(img))
    if relative:
        return (np.abs(wft * g)).sum() / (np.abs(wft * msk)).sum()
    else:
        return (np.abs(wft * g)).sum()


def snr(img, lpr, hpr, relative=True, gau=True):
    _ny, _nx = img.shape
    df = fs / _nx
    radius = (na / wl) / df
    msk = disc_array(shape=(_nx, _ny), radi=0.9 * radius)
    if gau:
        lp = msk * gaussian_filter(shape=(_nx, _ny), sigma=lpr * radius, pv=1, orig=None)
        hp = (1 - gaussian_filter(shape=(_nx, _ny), sigma=hpr * radius, pv=1, orig=None)) * msk
    else:
        lp = disc_array(shape=(_nx, _ny), radi=lpr * radius)
        hp = msk - disc_array(shape=(_nx, _ny), radi=hpr * radius)
    wft = np.fft.fftshift(np.fft.fft2(img))
    if relative:
        num = (np.abs(hp * wft)).sum() / (np.abs(wft * msk)).sum()
        den = (np.abs(lp * wft)).sum() / (np.abs(wft * msk)).sum()
        return num / den
    else:
        return (np.abs(hp * wft)).sum() / (np.abs(lp * wft)).sum()


def hpf(img, hpr, relative=True, gau=True):
    _nx, _ny = img.shape
    df = fs / _nx
    radius = (na / wl) / df
    msk = disc_array(shape=(_nx, _ny), radi=0.9 * radius)
    if gau:
        hp = (1 - gaussian_filter(shape=(_nx, _ny), sigma=hpr * radius, pv=1, orig=None)) * msk
    else:
        hp = msk - disc_array(shape=(_nx, _ny), radi=hpr * radius)
    wft = np.fft.fftshift(np.fft.fft2(img))
    if relative:
        return (np.abs(wft * hp)).sum() / (np.abs(wft * msk)).sum()
    else:
        return (np.abs(wft * hp)).sum()


def binomial_model(x, a, b, c):
    return a * x**2 + b * x + c


def peak_find(x_data, y_data, y_std=None):
    x = np.asarray(x_data)
    y = np.asarray(y_data)
    if y_std is not None:
        y_err = np.asarray(y_std)
        p_opt, p_cov = curve_fit(binomial_model, x, y, sigma=y_err, absolute_sigma=True)
        a, b, c = p_opt
        x_peak = -b / (2 * a)
        if a > 0:
            return "No peak"
        elif x_peak >= x.max():
            return "Peak above maximum"
        elif x_peak <= x.min():
            return "Peak below minimum"
        else:
            sigma_a, sigma_b, sigma_c = np.sqrt(np.diag(p_cov))
            x_peak_err = np.sqrt((b / (2 * a ** 2) * sigma_a) ** 2 + (-1 / (2 * a) * sigma_b) ** 2)
            print(x_peak_err)
            return x_peak
    else:
        a, b, c = np.polyfit(x, y, 2)
        x_peak = -b / (2 * a)
        if a > 0:
            return "No peak"
        elif x_peak >= x.max():
            return "Peak above maximum"
        elif x_peak <= x.min():
            return "Peak below minimum"
        else:
            return x_peak


def valley_find(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    a, b, c = np.polyfit(x, y, 2)
    v = -1 * b / a / 2.0
    if a < 0:
        raise ValueError("no minimum")
    elif (v >= x.max()) or (v <= x.min()):
        raise ValueError("minimum exceeding range")
    else:
        return v


def get_profile(data, ax, norm=False):
    data = data - data.min()
    data = data / data.max()
    if ax == 'Y':
        m = data.mean(0)
    elif ax == 'X':
        m = data.mean(1)
    else:
        raise ValueError("invalid axis")
    if norm:
        return m / m.max()
    else:
        return m


def pseudo_inverse(A, n=32):
    u, s, vt = np.linalg.svd(A)
    s_inv = np.zeros_like(A.T)
    if n is None:
        s_inv[:min(A.shape), :min(A.shape)] = np.diag(1 / s[:min(A.shape)])
    else:
        s_inv[:n, :n] = np.diag(1 / s[:n])
    return vt.T @ s_inv @ u.T


def get_eigen_coefficients(mta, mtb, ng=32):
    mp = pseudo_inverse(mtb, n=ng)
    return np.matmul(mp, mta)


def calculate_focus_measure_with_sobel(image):
    edges = filters.sobel(image)
    focus_measure = np.var(edges)
    return focus_measure


def calculate_focus_measure_with_laplacian(image):
    laplacian_image = filters.laplace(image)
    focus_measure = np.var(laplacian_image)
    return focus_measure


def meshgrid(nx_, ny_):
    x_ = np.arange(-nx_ / 2, nx_ / 2)
    y_ = np.arange(-ny_ / 2, ny_ / 2)
    xv_, yv_ = np.meshgrid(x_, y_, indexing='ij', sparse=True)
    return np.roll(xv_, int(nx_ / 2)), np.roll(yv_, int(ny_ / 2))


def shift(arr, shifts=None):
    if shifts is None:
        shifts = np.array(arr.shape) / 2
    if len(arr.shape) == len(shifts):
        for m, p in enumerate(shifts):
            arr = np.roll(arr, int(p), m)
    return arr


@lru_cache(maxsize=32)
def get_pupil(nx_, rd_):
    """Compute pupil function (cached)"""
    msk = shift(disc_array(shape=(nx_, nx_), radi=rd_)) / np.sqrt(np.pi * rd_ ** 2) / nx_
    phi = np.zeros((nx_, nx_))
    return msk * np.exp(1j * phi)


@lru_cache(maxsize=32)
def get_psf(nx_, rd_):
    """
    Compute Point Spread Function (cached)

    Args:
        nx_: Grid size (int)
        rd_: Radius (float)

    Returns:
        Normalized PSF array
    """
    bpp = get_pupil(nx_, rd_)
    psf_ = np.abs((fft2(fftshift(bpp)))) ** 2
    return psf_ / psf_.sum()


def get_correlation(image_to_be_computed, shift_orientation, shift_spacing, xv, yv, psf):
    kx = dx * np.cos(shift_orientation) / shift_spacing
    ky = dx * np.sin(shift_orientation) / shift_spacing
    ysh = np.exp(2j * np.pi * (kx * xv + ky * yv))
    otf0 = fft2(psf)
    imgf0 = fft2(image_to_be_computed)
    otf = fft2(psf * ysh)
    imgf = fft2(image_to_be_computed * ysh)
    a = np.sum(imgf0 * np.conj(otf0) * np.conj(imgf * np.conj(otf)))
    return np.abs(a), np.angle(a)


def find_pattern_coarse_to_fine(data, angle=0., spacing=0.268,
                                nps_coarse=5, nps_fine=3,
                                r_ang=0.005, r_sp=0.005, verbose=False):
    """
    Two-stage search: coarse grid, then fine grid around best
    """
    nx, ny = data.shape
    df = fs / nx
    radius = (na / wl) / df
    xv_, yv_ = meshgrid(nx, nx)
    psf_ = get_psf(nx, radius)  # Cached!

    # Coarse grid (6×6 = 36 evaluations)
    d_ang_coarse = 2 * r_ang / nps_coarse
    d_sp_coarse = 2 * r_sp / nps_coarse
    ang_iter_coarse = np.linspace(angle - r_ang, angle + r_ang, nps_coarse + 1)
    sp_iter_coarse = np.linspace(spacing - r_sp, spacing + r_sp, nps_coarse + 1)

    mags_coarse = np.zeros((nps_coarse + 1, nps_coarse + 1))
    for m, ang in enumerate(ang_iter_coarse):
        for n, sp in enumerate(sp_iter_coarse):
            mag, _ = get_correlation(data, ang, sp, xv_, yv_, psf_)
            mags_coarse[m, n] = mag

    # Find coarse maximum
    m_best, n_best = np.unravel_index(np.argmax(mags_coarse), mags_coarse.shape)
    ang_best_coarse = ang_iter_coarse[m_best]
    sp_best_coarse = sp_iter_coarse[n_best]

    # Fine grid around best (4×4 = 16 evaluations)
    r_ang_fine = d_ang_coarse
    r_sp_fine = d_sp_coarse
    ang_iter_fine = np.linspace(ang_best_coarse - r_ang_fine,
                                ang_best_coarse + r_ang_fine, nps_fine + 1)
    sp_iter_fine = np.linspace(sp_best_coarse - r_sp_fine,
                               sp_best_coarse + r_sp_fine, nps_fine + 1)

    mags_fine = np.zeros((nps_fine + 1, nps_fine + 1))
    phs_fine = np.zeros((nps_fine + 1, nps_fine + 1))
    for m, ang in enumerate(ang_iter_fine):
        for n, sp in enumerate(sp_iter_fine):
            mag, phase = get_correlation(data, ang, sp, xv_, yv_, psf_)
            mags_fine[m, n] = mag
            phs_fine[m, n] = phase

    # Find fine maximum
    m_best, n_best = np.unravel_index(np.argmax(mags_fine), mags_fine.shape)

    return ang_iter_fine[m_best], sp_iter_fine[n_best], phs_fine[m_best, n_best]


def find_pattern_optimized(data, angle=0., spacing=0.268,
                           r_ang=0.005, r_sp=0.005, verbose=False):
    """
    Use Nelder-Mead simplex optimization instead of grid search
    """
    nx, ny = data.shape
    df = fs / nx
    radius = (na / wl) / df
    xv_, yv_ = meshgrid(nx, nx)
    psf_ = get_psf(nx, radius)

    # Objective function (negative because we want to maximize)
    def objective(params):
        ang, sp = params
        mag, _ = get_correlation(data, ang, sp, xv_, yv_, psf_)
        return -mag  # Negative for minimization

    # Optimization with Nelder-Mead (typically 15-30 evaluations)
    result = minimize(
        objective,
        x0=[angle, spacing],  # Initial guess
        method='Nelder-Mead',
        bounds=[(angle - r_ang, angle + r_ang),
                (spacing - r_sp, spacing + r_sp)],
        options={'maxiter': 50, 'xatol': 1e-6}
    )

    # Get phase at optimum
    _, phase_best = get_correlation(data, result.x[0], result.x[1], xv_, yv_, psf_)

    if verbose:
        print(f"Converged in {result.nfev} evaluations")
        print(f"Angle: {result.x[0]:.6f}, Spacing: {result.x[1]:.6f}")

    return result.x[0], result.x[1], phase_best


def find_pattern_hybrid(data, angle=0., spacing=0.268,
                        r_ang=0.005, r_sp=0.005, verbose=False):
    """
    Coarse grid + optimization refinement
    """
    # 5×5 coarse grid (25 evaluations)
    ang_coarse, sp_coarse, _ = find_pattern_coarse_to_fine(
        data, angle, spacing,
        nps_coarse=4, nps_fine=0,  # Only coarse
        r_ang=r_ang, r_sp=r_sp
    )

    # Stage 2: Optimize from best grid point (~10 evaluations)
    ang_final, sp_final, phase_final = find_pattern_optimized(
        data, ang_coarse, sp_coarse,
        r_ang=r_ang / 5, r_sp=r_sp / 5  # Smaller search region
    )

    return ang_final, sp_final, phase_final
