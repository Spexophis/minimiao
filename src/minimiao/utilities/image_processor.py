# -*- coding: utf-8 -*-
# Copyright (c) 2025 Ruizhe Lin
# Licensed under the MIT License.


import numpy as np
from numpy.fft import fft2, fftshift
from scipy.optimize import curve_fit
from skimage import filters


def rms(data):
    _nx, _ny = data.shape
    _n = _nx * _ny
    m = np.mean(data, dtype=np.float64)
    a = (data - m) ** 2
    r = np.sqrt(np.sum(a) / _n)
    return r


def img_properties(img):
    return img.min(), img.max(), rms(img)


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


def fourier_transform(data):
    return np.log(np.abs(fftshift(fft2(data))))


def binomial_model(x, a, b, c):
    return a * x ** 2 + b * x + c


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


FWHM_FACTOR = np.sqrt(2 * np.log(2))


def fit_gaussian_2d(image, verbose=False, plot=False, allow_rotation=True,
                    bounds=None, p0=None):
    """
    Fit 2D Gaussian directly to image (more accurate than 1D projections)

    Args:
        image: 2D numpy array
        verbose: Print fit parameters
        plot: Show fit visualization
        allow_rotation: Allow ellipse rotation (slower but more accurate)
        bounds: Custom bounds for curve_fit
        p0: Initial guess (auto-computed if None)

    Returns:
        params: Fitted parameters
            If allow_rotation=False: [bg, I0, x0, y0, wx, wy]
            If allow_rotation=True:  [bg, I0, x0, y0, wx, wy, theta]
        residual: RMS fitting residual
    """
    # Input validation
    if image.ndim != 2:
        raise ValueError(f"Image must be 2D, got shape {image.shape}")

    y_px, x_px = image.shape

    # Create coordinate arrays
    x = np.arange(x_px, dtype=np.float64)
    y = np.arange(y_px, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)

    # Flatten for fitting
    coords = np.vstack((xx.ravel(), yy.ravel()))
    data = image.ravel()

    # Compute statistics for bounds and initial guess
    img_min = np.min(image)
    img_max = np.max(image)

    #  Compute smart initial guess from image moments
    if p0 is None:
        p0 = _compute_2d_initial_guess(image, allow_rotation)

    #  Set bounds
    if bounds is None:
        bounds = _compute_2d_bounds(x_px, y_px, img_min, img_max, allow_rotation)

    #  Choose fitting function
    if allow_rotation:
        fit_func = lambda coords, bg, I0, x0, y0, wx, wy, theta: \
            gaussian_2d_rotated(coords, bg, I0, x0, y0, wx, wy, theta)
    else:
        fit_func = lambda coords, bg, I0, x0, y0, wx, wy: \
            gaussian_2d_simple(coords, bg, I0, x0, y0, wx, wy)

    #  Fit with error handling
    try:
        popt, pcov = curve_fit(fit_func, coords, data, p0=p0, bounds=bounds, maxfev=10000)
    except ValueError as e:
        raise ValueError(f"2D Gaussian fit failed: {e}") from e

    # Calculate residual
    fitted_data = fit_func(coords, *popt)
    residual = np.sqrt(np.mean((data - fitted_data) ** 2))

    if verbose:
        _print_2d_params(popt, residual, allow_rotation)

    if plot:
        _plot_2d_fit(image, xx, yy, popt, residual, allow_rotation)

    return popt, residual, fitted_data


def gaussian_2d_simple(coords, bg, I0, x0, y0, wx, wy):
    """
    2D Gaussian without rotation (axis-aligned ellipse)

    Args:
        coords: (2, N) array of [x, y] coordinates
        bg: Background intensity
        I0: Peak intensity
        x0, y0: Center position
        wx, wy: Width in x and y (1/e^2 radius)

    Returns:
        Intensity values at each coordinate
    """
    x, y = coords
    return bg + I0 * np.exp(-2 * ((x - x0) / wx) ** 2 - 2 * ((y - y0) / wy) ** 2)


def gaussian_2d_rotated(coords, bg, I0, x0, y0, wx, wy, theta):
    """
    2D Gaussian with rotation (full ellipse)

    Args:
        coords: (2, N) array of [x, y] coordinates
        bg: Background intensity
        I0: Peak intensity
        x0, y0: Center position
        wx, wy: Width in x and y (1/e^2 radius)
        theta: Rotation angle in radians

    Returns:
        Intensity values at each coordinate
    """
    x, y = coords

    # Rotation coefficients
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    cos_t_sq = cos_t ** 2
    sin_t_sq = sin_t ** 2

    # Rotated Gaussian formula
    a = cos_t_sq / (2 * wx ** 2) + sin_t_sq / (2 * wy ** 2)
    b = -np.sin(2 * theta) / (4 * wx ** 2) + np.sin(2 * theta) / (4 * wy ** 2)
    c = sin_t_sq / (2 * wx ** 2) + cos_t_sq / (2 * wy ** 2)

    dx = x - x0
    dy = y - y0

    return bg + I0 * np.exp(-(a * dx ** 2 + 2 * b * dx * dy + c * dy ** 2))


def _compute_2d_initial_guess(image, allow_rotation):
    """
    Compute smart initial guess from image moments

    Args:
        image: 2D numpy array
        allow_rotation: Include rotation parameter

    Returns:
        p0: Initial guess array
    """
    y_px, x_px = image.shape

    # Background: percentile of lowest values
    bg = np.percentile(image, 10)

    # Peak intensity
    I0 = np.max(image) - bg
    I0 = max(I0, 1.0)

    # Center of mass
    img_shifted = image - bg
    img_shifted = np.maximum(img_shifted, 0)

    total = np.sum(img_shifted)
    if total > 0:
        x = np.arange(x_px)
        y = np.arange(y_px)
        xx, yy = np.meshgrid(x, y)

        x0 = np.sum(xx * img_shifted) / total
        y0 = np.sum(yy * img_shifted) / total

        # Second moments for width estimation
        var_x = np.sum((xx - x0) ** 2 * img_shifted) / total
        var_y = np.sum((yy - y0) ** 2 * img_shifted) / total

        wx = max(1.0, np.sqrt(var_x))
        wy = max(1.0, np.sqrt(var_y))

        # Estimate rotation from covariance
        if allow_rotation:
            cov_xy = np.sum((xx - x0) * (yy - y0) * img_shifted) / total
            theta = 0.5 * np.arctan2(2 * cov_xy, var_x - var_y)
        else:
            theta = None
    else:
        # Fallback for empty/uniform image
        x0 = x_px / 2
        y0 = y_px / 2
        wx = x_px / 4
        wy = y_px / 4
        theta = 0.0 if allow_rotation else None

    if allow_rotation:
        return [bg, I0, x0, y0, wx, wy, theta]
    else:
        return [bg, I0, x0, y0, wx, wy]


def _compute_2d_bounds(x_px, y_px, img_min, img_max, allow_rotation):
    """
    Compute bounds for 2D Gaussian fit

    Args:
        x_px, y_px: Image dimensions
        img_min, img_max: Image intensity range
        allow_rotation: Include rotation bounds

    Returns:
        bounds: ((lower_bounds), (upper_bounds))
    """
    # Background
    bg_min = min(0, 1.5 * img_min)
    bg_max = max(10, img_min * 10) if img_min > 0 else 10

    # Peak intensity
    I0_min = max(0, img_min)
    I0_max = max(I0_min + 1, img_max * 2)

    # Position
    x0_min, x0_max = -x_px * 0.2, x_px * 1.2  # Allow slightly outside
    y0_min, y0_max = -y_px * 0.2, y_px * 1.2

    # Width
    wx_min, wy_min = 0.5, 0.5
    wx_max, wy_max = x_px * 2, y_px * 2

    if allow_rotation:
        # Rotation angle: -π/2 to π/2 (sufficient due to symmetry)
        theta_min, theta_max = -np.pi / 2, np.pi / 2

        lower = [bg_min, I0_min, x0_min, y0_min, wx_min, wy_min, theta_min]
        upper = [bg_max, I0_max, x0_max, y0_max, wx_max, wy_max, theta_max]
    else:
        lower = [bg_min, I0_min, x0_min, y0_min, wx_min, wy_min]
        upper = [bg_max, I0_max, x0_max, y0_max, wx_max, wy_max]

    return [lower, upper]


def _print_2d_params(params, residual, allow_rotation):
    """Print fitted parameters"""
    bg, I0, x0, y0, wx, wy = params[:6]

    print(f"\n2D Gaussian Fit Results:")
    print(f"  Background: {bg:.2f}")
    print(f"  Peak intensity: {I0:.2f}")
    print(f"  Center: ({x0:.2f}, {y0:.2f})")
    print(f"  Width X: {wx:.2f} px (FWHM: {wx * FWHM_FACTOR:.2f})")
    print(f"  Width Y: {wy:.2f} px (FWHM: {wy * FWHM_FACTOR:.2f})")
    print(f"  Ellipticity: {max(wx, wy) / min(wx, wy):.3f}")

    if allow_rotation and len(params) > 6:
        theta = params[6]
        print(f"  Rotation: {np.degrees(theta):.2f}°")

    print(f"  RMS residual: {residual:.2f}")


def _plot_2d_fit(image, xx, yy, params, residual, allow_rotation):
    """
    Plot 2D Gaussian fit results

    Args:
        image: Original image
        xx, yy: Coordinate meshgrids
        params: Fitted parameters
        residual: RMS residual
        allow_rotation: Whether rotation was used
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
    except ImportError:
        print("Warning: matplotlib not available, skipping plot")
        return

    # Generate fitted image
    coords = np.vstack((xx.ravel(), yy.ravel()))
    if allow_rotation:
        fitted_flat = gaussian_2d_rotated(coords, *params)
    else:
        fitted_flat = gaussian_2d_simple(coords, *params)

    fitted = fitted_flat.reshape(image.shape)
    residual_img = image - fitted

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original image
    im0 = axes[0, 0].imshow(image, cmap='viridis', origin='lower')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].set_xlabel('X (pixels)')
    axes[0, 0].set_ylabel('Y (pixels)')
    plt.colorbar(im0, ax=axes[0, 0])

    # Fitted image
    im1 = axes[0, 1].imshow(fitted, cmap='viridis', origin='lower')
    axes[0, 1].set_title('Fitted Gaussian')
    axes[0, 1].set_xlabel('X (pixels)')
    axes[0, 1].set_ylabel('Y (pixels)')
    plt.colorbar(im1, ax=axes[0, 1])

    # Residual
    im2 = axes[0, 2].imshow(residual_img, cmap='RdBu', origin='lower')
    axes[0, 2].set_title(f'Residual (RMS={residual:.2f})')
    axes[0, 2].set_xlabel('X (pixels)')
    axes[0, 2].set_ylabel('Y (pixels)')
    plt.colorbar(im2, ax=axes[0, 2])

    # X projection
    x_proj_orig = np.max(image, axis=0)
    x_proj_fit = np.max(fitted, axis=0)
    axes[1, 0].plot(xx[0, :], x_proj_orig, 'o', alpha=0.5, label='Original')
    axes[1, 0].plot(xx[0, :], x_proj_fit, '-', linewidth=2, label='Fitted')
    axes[1, 0].set_title('X Projection')
    axes[1, 0].set_xlabel('X (pixels)')
    axes[1, 0].set_ylabel('Intensity')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Y projection
    y_proj_orig = np.max(image, axis=1)
    y_proj_fit = np.max(fitted, axis=1)
    axes[1, 1].plot(yy[:, 0], y_proj_orig, 'o', alpha=0.5, label='Original')
    axes[1, 1].plot(yy[:, 0], y_proj_fit, '-', linewidth=2, label='Fitted')
    axes[1, 1].set_title('Y Projection')
    axes[1, 1].set_xlabel('Y (pixels)')
    axes[1, 1].set_ylabel('Intensity')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Contour plot with ellipse
    bg, I0, x0, y0, wx, wy = params[:6]
    theta = params[6] if allow_rotation and len(params) > 6 else 0

    axes[1, 2].imshow(image, cmap='gray', origin='lower', alpha=0.5)

    # Draw contours of fitted Gaussian
    levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    contours = axes[1, 2].contour(xx, yy, fitted,
                                  levels=[bg + I0 * l for l in levels],
                                  colors='red', linewidths=2)

    # Draw ellipse at 1/e^2 level
    ellipse = Ellipse(
        xy=(x0, y0),
        width=2 * wx,
        height=2 * wy,
        angle=np.degrees(theta),
        fill=False,
        edgecolor='cyan',
        linewidth=2,
        linestyle='--',
        label='1/e² radius'
    )
    axes[1, 2].add_patch(ellipse)

    # Mark center
    axes[1, 2].plot(x0, y0, 'r+', markersize=15, markeredgewidth=2)

    axes[1, 2].set_title('Contours & Ellipse')
    axes[1, 2].set_xlabel('X (pixels)')
    axes[1, 2].set_ylabel('Y (pixels)')
    axes[1, 2].legend()

    plt.tight_layout()
    plt.show()


def gaussian_integral_fwhm_2d(params, allow_rotation=False, bg=False):
    """
    Compute integral of 2D Gaussian within FWHM ellipse (analytical)

    Args:
        params: Fitted parameters from fit_gaussian_2d
                [bg, I0, x0, y0, wx, wy] or [bg, I0, x0, y0, wx, wy, theta]
        allow_rotation: Whether rotation parameter is included

    Returns:
        integral: Total intensity within FWHM ellipse
    """
    bg, I0, x0, y0, wx, wy = params[:6]
    theta = params[6] if allow_rotation and len(params) > 6 else 0

    # Total integral of 2D Gaussian (infinite extent): I0 * π * wx * wy / 2
    total_gaussian_integral = I0 * np.pi * wx * wy / 2

    # Fraction of power within FWHM ellipse (derived from error function)
    # For 1/e^2 definition, FWHM corresponds to ~1.177 * w0
    # Integral within FWHM ellipse is ~0.5 of total (50% power)
    fwhm_fraction = 1 - np.exp(-2 * np.log(2))  # ≈ 0.75 or 75%

    gaussian_within_fwhm = total_gaussian_integral * fwhm_fraction

    if bg:
        # Background contribution (area of FWHM ellipse)
        fwhm_x = wx * np.sqrt(2 * np.log(2))
        fwhm_y = wy * np.sqrt(2 * np.log(2))
        fwhm_area = np.pi * fwhm_x * fwhm_y
        background_contribution = bg * fwhm_area
        total_integral = gaussian_within_fwhm + background_contribution
        return total_integral
    else:
        return gaussian_within_fwhm


def gauss_metric(img, s=True):
    try:
        params, residual, ftg = fit_gaussian_2d(img, verbose=False, allow_rotation=True, plot=False)
        if s:
            ss = gaussian_integral_fwhm_2d(params, True)
            return ss
        else:
            return params[1]
    except Exception as e:
        return f"Gaussian Fitting Error: {e}"

